from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from dataset import save_emb

# 【8.10.1修改】实现 RMSNorm，用于替换 LayerNorm（只做均方根归一化 + 可学习缩放，不引入偏置）
class RMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps: float = 1e-8):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(self.normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(-len(self.normalized_shape), 0))
        rms = torch.sqrt(torch.mean(x.float().pow(2), dim=dims, keepdim=True) + self.eps)
        x_norm = x / rms
        return x_norm * self.weight

# 【8.12.2修改】新增 MLPBlock（4H + GELU + Dropout + 残差 + RMSNorm）
class MLPBlock(torch.nn.Module):
    def __init__(self, in_dim, hidden_units, dropout_rate, eps: float = 1e-8):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, hidden_units * 4)
        self.act = torch.nn.GELU()
        self.drop = torch.nn.Dropout(p=dropout_rate)
        self.fc2 = torch.nn.Linear(hidden_units * 4, hidden_units)
        self.skip = torch.nn.Identity() if in_dim == hidden_units else torch.nn.Linear(in_dim, hidden_units)
        self.norm = RMSNorm(hidden_units, eps=eps)

    def forward(self, x):
        y = self.fc2(self.drop(self.act(self.fc1(x))))
        out = self.norm(self.skip(x) + y)
        return out

class FlashMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate, pos_enc="abs",
                 use_action_gate: bool = False, action_vocab_size: int = 3, action_emb_dim: int = 16,
                 use_td_attn_bias: bool = False, time_bucket_count: int = 0):  # 【8.16.2修改】新增位置编码与动作门控开关；【8.18.1修改】新增时间差注意力偏置开关与桶数
        super(FlashMultiHeadAttention, self).__init__()

        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate
        self.pos_enc = pos_enc  # 【8.16.2修改】'abs'（默认）|'rope'
        self.use_action_gate = use_action_gate
        self.action_vocab_size = action_vocab_size
        self.action_emb_dim = action_emb_dim
        # 【TimeDelta-ATTN】按桶构造的注意力加性偏置（与 RoPE 兼容）  【8.18.1修改】
        self.use_td_attn_bias = use_td_attn_bias  # 【8.18.1修改】
        self.time_bucket_count = time_bucket_count  # 【8.18.1修改】
        if self.use_td_attn_bias and self.time_bucket_count > 0:  # 【8.18.1修改】
            # 每个时间桶 -> 每个注意力头的偏置值（列偏置，作用在 key 维度）  【8.18.1修改】
            self.td_bias_emb = torch.nn.Embedding(self.time_bucket_count, self.num_heads, padding_idx=0)  # 【8.18.1修改】
            # 可学习门控，控制偏置强度，初始接近 0 以避免过拟合  【8.18.1修改】
            self.td_bias_gate = torch.nn.Parameter(torch.tensor(0.0))  # 【8.18.1修改】
            with torch.no_grad():
                self.td_bias_emb.weight[0].zero_()  # 确保 padding=0 的偏置为 0  【8.18.1修改】

        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"

        self.q_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.k_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.v_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.out_linear = torch.nn.Linear(hidden_units, hidden_units)
        # 【8.14.2修改】新增 U 分支，用于对 V 进行门控重标定（捕获 U·V 的二阶交互）
        self.u_linear = torch.nn.Linear(hidden_units, hidden_units)

        # 【ActionGate】动作门控参数：动作嵌入 + 线性映射到 hidden_units
        if self.use_action_gate:
            self.action_emb = torch.nn.Embedding(self.action_vocab_size, self.action_emb_dim, padding_idx=0)
            if self.action_emb_dim != hidden_units:
                self.action_proj = torch.nn.Linear(self.action_emb_dim, hidden_units)
            else:
                self.action_proj = torch.nn.Identity()

    def forward(self, query, key, value, attn_mask=None, action_ids: torch.Tensor | None = None,
                time_deltas: torch.Tensor | None = None):  # 【8.18.1修改】
        batch_size, seq_len, _ = query.size()

        # 计算Q, K, V
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        # 【8.14.2修改】计算 U 分支
        U = self.u_linear(query)

        # 【ActionGate】可选的动作门控：g = proj(Emb(action))；V' = V ⊙ σ(U + g)
        if self.use_action_gate and action_ids is not None:
            # action_ids: [B, L]，0 用于 padding/user 位置
            g = self.action_emb(action_ids.long())
            g = self.action_proj(g)  # [B, L, H]
            U = U + g

        # reshape为multi-head格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 【8.14.2修改】将 U reshape 为 multi-head，并对 V 做门控
        U = U.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 源代码：未门控的 V
        # V = V
        # 【8.14.2修改】用 U 对 V 做逐元素门控：V' = V ⊙ σ(U)
        V = V * torch.sigmoid(U)

        # 【8.16.2修改】RoPE：对 Q/K 应用旋转位置编码（仅当选择 'rope'）
        if getattr(self, 'pos_enc', 'abs') == 'rope':
            def _rotate_half(x):
                D = x.size(-1)
                x1 = x[..., : D // 2]
                x2 = x[..., D // 2 : D // 2 * 2]
                return torch.cat((-x2, x1), dim=-1)

            B, H, L, D = Q.shape
            device = Q.device
            dtype = Q.dtype
            dim = D
            assert D % 2 == 0, "RoPE requires even head_dim; please set hidden_units/num_heads to make head_dim even"  # 【8.16.2修改】
            inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
            t = torch.arange(L, device=device, dtype=torch.float32)
            freqs = torch.einsum("l,d->ld", t, inv_freq)  # [L, D/2]
            cos = torch.cos(freqs)
            sin = torch.sin(freqs)
            cos = torch.repeat_interleave(cos, 2, dim=-1)[None, None, :, :]  # [1,1,L,D]
            sin = torch.repeat_interleave(sin, 2, dim=-1)[None, None, :, :]  # [1,1,L,D]
            cos = cos[..., :D].to(dtype)
            sin = sin[..., :D].to(dtype)
            Q = (Q * cos) + (_rotate_half(Q) * sin)
            K = (K * cos) + (_rotate_half(K) * sin)

        # 【TimeDelta-ATTN】构造按 key 位置的加性偏置（B,H,1,L），在 softmax 前加到 QK^T  【8.18.1修改】
        td_mask_float = None  # 【8.18.1修改】
        if self.use_td_attn_bias and (time_deltas is not None) and (self.time_bucket_count > 0):  # 【8.18.1修改】
            td = time_deltas.to(Q.device).long()  # [B, L]  【8.18.1修改】
            td = td.clamp(min=0, max=self.time_bucket_count - 1)  # 【8.18.1修改】
            bias_ph = self.td_bias_emb(td)  # [B, L, H]  【8.18.1修改】
            bias_ph = bias_ph.permute(0, 2, 1).unsqueeze(2)  # [B, H, 1, L]  【8.18.1修改】
            gate = torch.sigmoid(self.td_bias_gate)  # 标量门控 in (0,1)  【8.18.1修改】
            td_mask_float = gate * bias_ph  # 延 query 维广播为 [B,H,L,L]  【8.18.1修改】

        if hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ 使用内置的Flash Attention
            # 注意：PyTorch sdpa 中 bool 掩码语义为 True=屏蔽；本工程 attn_mask 为 True=允许
            # 统一改为加性浮点掩码：允许=0，不允许=-inf
            mask_float = torch.where(attn_mask, 0.0, float('-inf')).unsqueeze(1).to(Q.dtype)  # 【8.16.2修改】【8.18.1修改】
            if td_mask_float is not None:  # 【8.18.1修改】
                mask_float = mask_float + td_mask_float.to(Q.dtype)  # 【8.18.1修改】
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, dropout_p=self.dropout_rate if self.training else 0.0, attn_mask=mask_float
            )  # 【8.16.2修改】
        else:
            # 降级到标准注意力机制
            scale = (self.head_dim) ** -0.5
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
            if td_mask_float is not None:  # 【8.18.1修改】
                scores = scores + td_mask_float.to(scores.dtype)  # 【8.18.1修改】

            if attn_mask is not None:
                scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)
            attn_output = torch.matmul(attn_weights, V)

        # reshape回原来的格式
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)

        # 最终的线性变换
        output = self.out_linear(attn_output)

        return output, None

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate, ffn_type: str = "gelu"):
        super(PointWiseFeedForward, self).__init__()

        self.ffn_type = ffn_type  # 【8.16.2修改】新增 FFN 类型开关：'gelu'（默认）|'swiglu'
        if self.ffn_type == "swiglu":
            # 【8.16.2修改】SwiGLU：2H 投影 -> 分块门控 -> H 回投影
            self.inner = hidden_units * 2
            self.fc1 = torch.nn.Linear(hidden_units, self.inner * 2)
            self.act = torch.nn.SiLU()
            self.dropout = torch.nn.Dropout(p=dropout_rate)
            self.fc2 = torch.nn.Linear(self.inner, hidden_units)
        else:
            # 源实现（GELU FFN）
            self.linear1 = torch.nn.Linear(hidden_units, hidden_units * 4)
            self.activation = torch.nn.GELU()
            self.dropout = torch.nn.Dropout(p=dropout_rate)
            self.linear2 = torch.nn.Linear(hidden_units * 4, hidden_units)

    def forward(self, inputs):
        if self.ffn_type == "swiglu":
            u = self.fc1(inputs)
            a, b = torch.split(u, self.inner, dim=-1)
            y = self.act(a) * b
            y = self.dropout(y)
            y = self.fc2(y)
            return y
        else:
            outputs = self.linear1(inputs)
            outputs = self.activation(outputs)
            outputs = self.dropout(outputs)
            outputs = self.linear2(outputs)
            return outputs

class BaselineModel(torch.nn.Module):
    """
    Args:
        user_num: 用户数量
        item_num: 物品数量
        feat_statistics: 特征统计信息，key为特征ID，value为特征数量
        feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表，包括user和item的sparse, array, emb, continual类型
        args: 全局参数

    Attributes:
        user_num: 用户数量
        item_num: 物品数量
        dev: 设备
        norm_first: 是否先归一化
        maxlen: 序列最大长度
        item_emb: Item Embedding Table
        user_emb: User Embedding Table
        sparse_emb: 稀疏特征Embedding Table
        emb_transform: 多模态特征的线性变换
        userdnn: 用户特征拼接后经过的全连接层
        itemdnn: 物品特征拼接后经过的全连接层
    """

    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):  #
        super(BaselineModel, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.pos_enc = getattr(args, 'pos_enc', 'abs')  # 【8.16.2修改】'abs'|'rope'
        self.ffn_type = getattr(args, 'ffn', 'gelu')    # 【8.16.2修改】'gelu'|'swiglu'
        self.maxlen = args.maxlen
        # 【梯度检查点】启用梯度检查点以减少显存占用
        self.use_gradient_checkpointing = getattr(args, 'use_gradient_checkpointing', False)
        # 【InfoNCE显存优化】分块计算参数
        self.infonce_row_chunk = getattr(args, 'infonce_row_chunk', 512)
        # 【TimeDelta】时间差分桶配置
        self.time_bucket_count = getattr(args, 'time_bucket_count', 7)  # 【8.17.2修改】
        # 【TimeDelta-ATTN】是否启用时间差注意力偏置  【8.18.1修改】
        self.use_td_attn_bias = getattr(args, 'use_td_attn_bias', False)  # 【8.18.1修改】
        # 【ActionGate】动作门控配置
        self.use_action_gate = getattr(args, 'use_action_gate', False)
        self.action_vocab_size = getattr(args, 'action_vocab_size', 3)
        self.action_emb_dim = getattr(args, 'action_emb_dim', 16)
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        
        # 【9.2.2显存优化】启用稀疏梯度：只为实际访问的物品计算梯度，大幅减少显存占用
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0, sparse=True)
        self.user_emb = torch.nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0, sparse=True)
        self.pos_emb = torch.nn.Embedding(2 * args.maxlen + 1, args.hidden_units, padding_idx=0)
        # 【TimeDelta】时间差嵌入（与位置嵌入同维度），0 为 padding
        self.time_delta_emb = torch.nn.Embedding(self.time_bucket_count, args.hidden_units, padding_idx=0)  # 【8.17.2修改】
        self.emb_dropout = torch.nn.Dropout(p=args.emb_dropout)
        self.sparse_emb = torch.nn.ModuleDict()
        self.emb_transform = torch.nn.ModuleDict()

        # # 【8.28.2新增】为训练步采样添加一个可配置的超参数
        # self.max_samples_per_batch = getattr(args, 'max_samples_per_batch', 2048)
        # 【新增】从批内正样本中选取作为困难负样本的数量K
        self.use_in_batch_pos_as_neg = getattr(args, 'use_in_batch_pos_as_neg', False)  # 【新增开关】是否启用批内正样本作为负样本
        self.num_in_batch_pos_neg = getattr(args, 'num_in_batch_pos_neg', 4)
        self.hard_negative_weight = getattr(args, 'hard_negative_weight', 0.5)
        self.sampling_range_start = getattr(args, 'sampling_range_start', 10)
        self.sampling_range_end = getattr(args, 'sampling_range_end', 100)
        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        
        self._init_feat_info(feat_statistics, feat_types)

        userdim = args.hidden_units * (len(self.USER_SPARSE_FEAT) + 1 + len(self.USER_ARRAY_FEAT)) + len(
            self.USER_CONTINUAL_FEAT
        )
        itemdim = (
            args.hidden_units * (len(self.ITEM_SPARSE_FEAT) + 1 + len(self.ITEM_ARRAY_FEAT))
            + len(self.ITEM_CONTINUAL_FEAT)
            + args.hidden_units * len(self.ITEM_EMB_FEAT)
        )

        # 【8.12.2修改】将 userdnn 和 itemdnn 升级为 4H + GELU + Dropout + 残差 + RMSNorm 的 Block
        self.userdnn = MLPBlock(userdim, args.hidden_units, args.ffn_dropout)
        self.itemdnn = MLPBlock(itemdim, args.hidden_units, args.ffn_dropout)

        self.last_layernorm = RMSNorm(args.hidden_units, eps=1e-8)  # 【8.10.1修改】替换 LayerNorm 为 RMSNorm

        for _ in range(args.num_blocks):
            new_attn_layernorm = RMSNorm(args.hidden_units, eps=1e-8)  # 【8.10.1修改】替换 LayerNorm 为 RMSNorm
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = FlashMultiHeadAttention(
                args.hidden_units,
                args.num_heads,
                args.attn_dropout,
                pos_enc=self.pos_enc,
                use_action_gate=self.use_action_gate,
                action_vocab_size=self.action_vocab_size,
                action_emb_dim=self.action_emb_dim,
                use_td_attn_bias=self.use_td_attn_bias,  # 【8.18.1修改】
                time_bucket_count=self.time_bucket_count,  # 【8.18.1修改】
            )  # 优化：用FlashAttention替代标准Attention；【8.16.2修改】加入位置编码 + 动作门控
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = RMSNorm(args.hidden_units, eps=1e-8)  # 【8.10.1修改】替换 LayerNorm 为 RMSNorm
            self.forward_layernorms.append(new_fwd_layernorm)

            # 【8.16.2修改】可切换 FFN：'gelu' 或 'swiglu'
            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.ffn_dropout, ffn_type=self.ffn_type)
            self.forward_layers.append(new_fwd_layer)

        for k in self.USER_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.USER_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_EMB_FEAT:
            self.emb_transform[k] = torch.nn.Linear(self.ITEM_EMB_FEAT[k], args.hidden_units)
        
        # 【ENHANCED】InfoNCE 优化：温度参数与点击权重
        self.temp = getattr(args, 'tau', 0.05)  # 【ENHANCED】确保与main.py中的默认值一致
        self.click_weight = 2.5  # 【ENHANCED】默认点击权重，可通过外部设置覆盖

    def _chunked_similarity_matrix(self, query_embs, key_embs, chunk_size=None):
        """
        【显存优化】分块计算相似度矩阵，避免OOM

        Args:
            query_embs: [N, H] 查询向量
            key_embs: [M, H] 键向量
            chunk_size: 分块大小，默认使用self.infonce_row_chunk

        Returns:
            similarity_matrix: [N, M] 相似度矩阵
        """
        if chunk_size is None:
            chunk_size = self.infonce_row_chunk

        N, H = query_embs.shape
        M = key_embs.shape[0]

        # 如果矩阵较小或chunk_size很大，直接计算
        if N <= chunk_size or chunk_size <= 0:
            return torch.matmul(query_embs, key_embs.t())

        # 分块计算
        similarity_chunks = []
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            query_chunk = query_embs[i:end_i]  # [chunk_size, H]
            sim_chunk = torch.matmul(query_chunk, key_embs.t())  # [chunk_size, M]
            similarity_chunks.append(sim_chunk)

        return torch.cat(similarity_chunks, dim=0)  # [N, M]

    def _init_feat_info(self, feat_statistics, feat_types):
        """
        将特征统计信息（特征数量）按特征类型分组产生不同的字典，方便声明稀疏特征的Embedding Table

        Args:
            feat_statistics: 特征统计信息，key为特征ID，value为特征数量
            feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表，包括user和item的sparse, array, emb, continual类型
        """
        self.USER_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = feat_types['user_continual']
        self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['item_sparse']}
        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['item_array']}
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in feat_types['item_emb']}  # 记录的是不同多模态特征的维度

    # =============================== 新版字典接口实现 ===============================
    def feat2emb_v2(self, ids: torch.Tensor, features: dict, mask: torch.Tensor | None = None,
                    include_user: bool = False):  # 【8.18.2修改】
        """
        【8.18.2修改】基于批字典 Schema 将多种特征融合为时序 Embedding。
        - 当 include_user=True 时：序列同时包含 user/item 两类 token，通过 mask 区分后分别嵌入并与各自侧特征融合，最终逐元素相加。
        - 当 include_user=False 时：仅构建 item 分支（用于 pos/neg）。

        参数:
        - ids: Long [B, L]，离散 token ID 序列（与 mask 对齐）
        - features: dict，与 `dataset.collate_fn` 对齐；features['seq'|'pos'|'neg'] 下包含：
          - item_sparse: Dict[str -> LongTensor[B, L]]
          - item_array: Dict[str -> LongTensor[B, L, A]]（变长集合，0 为 padding）
          - item_continual: Dict[str -> FloatTensor[B, L]]
          - mm_emb: Dict[str -> FloatTensor[B, L, D_mm]]（多模态向量，经线性投影到 H）
          - 当 include_user=True 还可包含：
            - user_sparse: Dict[str -> LongTensor[B, L]]
            - user_array: Dict[str -> LongTensor[B, L, A]]
            - user_continual: Dict[str -> FloatTensor[B, L]]
        - mask: Long [B, L]，1=item，2=user；仅在 include_user=True 时必需
        - include_user: 是否融合用户侧特征

        返回:
        - seqs_emb: Float [B, L, H]

        实现要点:
        - array 特征：按 tf!=0 作为掩码做加权求和与计数，再做长度无关的平均聚合
        - continual 特征：小批量 z-score 标准化后以标量通道拼接
        - mm_emb：使用 `self.emb_transform[k]` 线性变换到 H 维
        - 用户侧与物品侧分别通过 `userdnn`/`itemdnn`（MLPBlock）后相加 
        - 所有张量在 `self.dev` 上计算
        """
        ids = ids.to(self.dev).long()

        if include_user:
            assert mask is not None, "feat2emb_v2 requires mask when include_user=True"
            user_mask = (mask.to(self.dev) == 2)
            item_mask = (mask.to(self.dev) == 1)
            user_embedding = self.user_emb((user_mask * ids).long())
            item_embedding = self.item_emb((item_mask * ids).long())
            item_feat_list = [item_embedding]
            user_feat_list = [user_embedding]
        else:
            item_embedding = self.item_emb(ids)
            item_feat_list = [item_embedding]

        # 取出各类特征映射（若不存在则为空字典）
        user_sparse = (features.get('user_sparse', {}) if include_user else {})
        user_array = (features.get('user_array', {}) if include_user else {})
        user_cont = (features.get('user_continual', {}) if include_user else {})
        item_sparse = features.get('item_sparse', {})
        item_array = features.get('item_array', {})
        item_cont = features.get('item_continual', {})
        mm_emb = features.get('mm_emb', {})

        # 用户侧特征
        if include_user:
            for k, tf in user_sparse.items():
                tf = tf.to(self.dev).long()
                user_feat_list.append(self.sparse_emb[k](tf))
            for k, tf in user_array.items():
                tf = tf.to(self.dev).long()  # [B, L, A]
                emb = self.sparse_emb[k](tf)  # [B, L, A, H]
                mask_array = (tf != 0).unsqueeze(-1)  # [B, L, A, 1]
                emb_sum = (emb * mask_array).sum(2)  # [B, L, H]
                lengths = mask_array.sum(2).clamp(min=1).to(emb.dtype)  # [B, L, 1]
                user_feat_list.append(emb_sum / lengths)
            for k, tf in user_cont.items():
                tf = tf.to(self.dev).float()  # [B, L]
                mean = tf.mean()
                std = tf.std(unbiased=False) + 1e-6
                tf = (tf - mean) / std
                user_feat_list.append(tf.unsqueeze(2))  # 标量拼接

        # 物品侧特征
        for k, tf in item_sparse.items():
            tf = tf.to(self.dev).long()
            item_feat_list.append(self.sparse_emb[k](tf))
        for k, tf in item_array.items():
            tf = tf.to(self.dev).long()
            emb = self.sparse_emb[k](tf)
            mask_array = (tf != 0).unsqueeze(-1)
            emb_sum = (emb * mask_array).sum(2)
            lengths = mask_array.sum(2).clamp(min=1).to(emb.dtype)
            item_feat_list.append(emb_sum / lengths)
        for k, tf in item_cont.items():
            tf = tf.to(self.dev).float()
            mean = tf.mean()
            std = tf.std(unbiased=False) + 1e-6
            tf = (tf - mean) / std
            item_feat_list.append(tf.unsqueeze(2))
        for k, tf in mm_emb.items():
            tf = tf.to(self.dev).float()  # [B, L, D]
            item_feat_list.append(self.emb_transform[k](tf))  # -> [B, L, H]

        # 融合并通过 MLPBlock
        all_item_emb = torch.cat(item_feat_list, dim=2)
        all_item_emb = self.itemdnn(all_item_emb)
        if include_user:
            all_user_emb = torch.cat(user_feat_list, dim=2)
            all_user_emb = self.userdnn(all_user_emb)
            seqs_emb = all_item_emb + all_user_emb
        else:
            seqs_emb = all_item_emb
        return seqs_emb

    def log2feats_v2(self, batch: dict):  # 【8.18.2修改】
        """
        【8.18.2修改】将批字典转换为时序表征（位置/时间差/动作门控/注意力堆叠）。
        输入 Schema（与 `dataset.MyTestDataset.collate_fn` 一致）：
        - batch['ids']：
          - seq: Long [B, L]
          - action_type: 可选 Long [B, L]，与 seq 对齐（仅在 `use_action_gate=True` 时使用）
        - batch['masks']：
          - token_type: Long [B, L]，1=item，2=user（0 为 padding）
        - batch['features']['seq']：同 `feat2emb_v2(features)` 的结构
        - batch['time_deltas']：可选 Long [B, L]，范围 [0, time_bucket_count-1]，0 为 padding

        处理流程：
        1) 通过 `feat2emb_v2` 构建序列嵌入，并乘以 sqrt(H)
        2) 当 `pos_enc=='abs'` 时，加绝对位置 `pos_emb` 与时间差 `time_delta_emb`
        3) Dropout
        4) 堆叠多层注意力：
           - 因果掩码：下三角
           - pad 掩码：`token_type!=0`
           - 动作门控：当 `use_action_gate=True` 且提供 `action_type`
           - 时间差注意力偏置：当 `use_td_attn_bias=True` 且提供 `time_deltas`
        5) 最后经 `RMSNorm`

        返回：`log_feats` Float [B, L, H]
        """
        ids = batch['ids']
        masks = batch['masks']
        features = batch['features']
        time_deltas = batch.get('time_deltas', None)

        seq_ids = ids['seq']
        token_type = masks['token_type']

        seqs = self.feat2emb_v2(seq_ids, features['seq'], mask=token_type, include_user=True)
        seqs *= self.item_emb.embedding_dim ** 0.5

        B = seqs.size(0)
        L = seqs.size(1)
        poss = torch.arange(1, L + 1, device=self.dev).unsqueeze(0).expand(B, -1).clone()
        poss = poss * (seq_ids.to(self.dev) != 0)
        if getattr(self, 'pos_enc', 'abs') == 'abs':
            seqs = seqs + self.pos_emb(poss)
            if time_deltas is not None:
                td = time_deltas.to(self.dev).long()
                td = td.clamp(min=0, max=self.time_bucket_count - 1)
                seqs = seqs + self.time_delta_emb(td)
        seqs = self.emb_dropout(seqs)

        ones_matrix = torch.ones((L, L), dtype=torch.bool, device=self.dev)
        attention_mask_tril = torch.tril(ones_matrix)
        attention_mask_pad = (token_type.to(self.dev) != 0)
        attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)

        act_type = ids.get('action_type', None)

        # 【梯度检查点】定义检查点函数
        def create_custom_forward(layer_idx):
            def custom_forward(hidden_states):
                if self.norm_first:
                    x = self.attention_layernorms[layer_idx](hidden_states)
                    mha_outputs, _ = self.attention_layers[layer_idx](
                        x, x, x, attn_mask=attention_mask,
                        action_ids=(act_type.to(self.dev) if (self.use_action_gate and act_type is not None) else None),
                        time_deltas=(time_deltas.to(self.dev) if (time_deltas is not None) else None),
                    )
                    hidden_states = hidden_states + mha_outputs
                    hidden_states = hidden_states + self.forward_layers[layer_idx](self.forward_layernorms[layer_idx](hidden_states))
                else:
                    mha_outputs, _ = self.attention_layers[layer_idx](
                        hidden_states, hidden_states, hidden_states, attn_mask=attention_mask,
                        action_ids=(act_type.to(self.dev) if (self.use_action_gate and act_type is not None) else None),
                        time_deltas=(time_deltas.to(self.dev) if (time_deltas is not None) else None),
                    )
                    hidden_states = self.attention_layernorms[layer_idx](hidden_states + mha_outputs)
                    hidden_states = self.forward_layernorms[layer_idx](hidden_states + self.forward_layers[layer_idx](hidden_states))
                return hidden_states
            return custom_forward

        for i in range(len(self.attention_layers)):
            if self.use_gradient_checkpointing and self.training:
                # 【梯度检查点】使用检查点减少显存占用
                seqs = checkpoint(create_custom_forward(i), seqs, use_reentrant=False)
            else:
                # 原始实现
                if self.norm_first:
                    x = self.attention_layernorms[i](seqs)
                    mha_outputs, _ = self.attention_layers[i](
                        x, x, x, attn_mask=attention_mask,
                        action_ids=(act_type.to(self.dev) if (self.use_action_gate and act_type is not None) else None),
                        time_deltas=(time_deltas.to(self.dev) if (time_deltas is not None) else None),
                    )
                    seqs = seqs + mha_outputs
                    seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
                else:
                    mha_outputs, _ = self.attention_layers[i](
                        seqs, seqs, seqs, attn_mask=attention_mask,
                        action_ids=(act_type.to(self.dev) if (self.use_action_gate and act_type is not None) else None),
                        time_deltas=(time_deltas.to(self.dev) if (time_deltas is not None) else None),
                    )
                    seqs = self.attention_layernorms[i](seqs + mha_outputs)
                    seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward(self, batch: dict):
        """
        【模型训练核心】新版训练入口函数（基于批字典 Schema）。
        
        该函数实现了基于 InfoNCE (Noise-Contrastive Estimation) 的对比学习损失计算。
        其核心思想是：对于每个有效的用户序列历史表征，模型需要将其与“下一个真实交互的物品（正样本）”的相似度最大化，
        同时将其与“批次内所有其他的负样本”的相似度最小化。

        Args:
            batch (dict): 从 `dataset.MyDataset.collate_fn` 输出的批字典，包含模型所需的所有数据，
                          如序列ID、特征、掩码、动作类型等。

        Returns:
            tuple:
                - pos_logits (None): 在此 InfoNCE 实现中不计算，返回 None。
                - neg_logits (None): 在此 InfoNCE 实现中不计算，返回 None。
                - infonce_loss (torch.Tensor): 计算出的标量 InfoNCE 损失值，用于反向传播。
        """
        # 1. --- 数据提取 ---
        # 从输入的批字典中解构出需要用到的各个部分。
        ids = batch['ids']
        masks = batch['masks']
        features = batch['features']
        # time_deltas = batch.get('time_deltas', None) # 此版本未使用时间差，但保留以备将来扩展

        # 2. --- 特征编码 ---
        # a. 将用户行为序列（包含用户、物品及其特征）编码成高维向量表示。
        #    log2feats_v2 是模型的序列编码器，通常是一个 Transformer 结构。
        #    输出的 log_feats 形状为 [B, L, H]，B=批大小, L=序列长度, H=隐藏维度。
        log_feats = self.log2feats_v2(batch)
        
        # b. 独立地将正样本（下一个真实交互的物品）和负样本（随机采样的物品）也编码成相同维度的高维向量。
        #    feat2emb_v2 是模型的物品编码器，include_user=False 表示只使用物品侧特征。
        #    输出的 pos_embs 和 neg_embs 形状均为 [B, L, H]。
        pos_embs = self.feat2emb_v2(ids['pos'], features['pos'], include_user=False)
        neg_embs = self.feat2emb_v2(ids['neg'], features['neg'], include_user=False)
        
        # c. 定义损失计算的有效位置掩码。只有当序列中的“下一个token”是物品时 (token_type==1)，
        #    该位置的预测才有意义，才需要计算损失。
        loss_mask = (masks['next_token_type'].to(self.dev) == 1)

        # 3. --- 向量归一化 ---
        # 为了使用余弦相似度（通过点积实现）并稳定训练，将所有向量进行 L2 归一化。
        # 归一化后，向量的 L2 范数为 1，此时向量点积等价于余弦相似度。
        seq_embs_norm = F.normalize(log_feats, p=2, dim=-1)
        pos_embs_norm = F.normalize(pos_embs, p=2, dim=-1)
        neg_embs_norm = F.normalize(neg_embs, p=2, dim=-1)

        # 4. --- 构造 InfoNCE Logits ---
        # a. 计算正样本相似度。
        #    逐元素计算序列表示和其对应的正样本表示的余弦相似度。
        pos_sims = F.cosine_similarity(seq_embs_norm, pos_embs_norm, dim=-1)  # 形状: [B, L]

        # b. 构造批内负样本 (In-Batch Negatives)。
        #    这是 InfoNCE 的关键，它利用了批次内其他样本的负例，极大地扩充了负样本数量。
        B, L, H = neg_embs_norm.shape
        # 将序列表示和负样本表示从 [B, L, H] 展平为 [B*L, H]。
        seq_flat = seq_embs_norm.view(B * L, H)
        neg_flat = neg_embs_norm.view(B * L, H)
        
        # 计算每个序列位置的表示与批内所有负样本表示的相似度矩阵。
        # 【显存优化】使用分块计算避免大矩阵OOM
        # `torch.matmul(seq_flat, neg_flat.t())` 的结果是一个 [B*L, B*L] 的矩阵，
        # 其中 `mat[i, j]` 表示第 i 个序列位置的向量与第 j 个负样本向量的相似度。
        neg_sims_mat = self._chunked_similarity_matrix(seq_flat, neg_flat)

        # c. 屏蔽自身配对的负样本。在构造的相似度矩阵中，对角线上的元素
        #    `neg_sims_mat[i, i]` 代表一个序列表示与其“配对的随机负样本”的相似度。
        #    在纯粹的批内负采样策略中，我们希望模型主要从批内“不相关”的样本中学习，
        #    因此有时会屏蔽掉这个配对的负样本，以避免它对损失产生过大的影响。
        #    这里通过一个单位矩阵 `eye_mask` 将对角线元素的值设置为一个极小的数（-1e4），
        #    使其在 softmax 计算中概率接近于 0。
        eye_mask = torch.eye(B * L, device=neg_sims_mat.device).bool()
        neg_sims_mat = neg_sims_mat.masked_fill(eye_mask, -1e4)

        # --- 新增代码开始 ---
        # d. [修改] 额外补充 K 个在指定范围内的困难正样本作为负例
        hard_pos_neg_sims = torch.empty(B * L, 0, device=self.dev)
        k = self.num_in_batch_pos_neg

        # 只有在K>0且有足够候选样本时才执行
        if k > 0 and B * L - 1 > self.sampling_range_start:
            # 同样需要展平的正样本向量和相似度矩阵
            pos_flat = pos_embs_norm.view(B * L, H)
            in_batch_pos_sims_mat = torch.matmul(seq_flat, pos_flat.t())
            in_batch_pos_sims_mat = in_batch_pos_sims_mat.masked_fill(eye_mask, -1e9)

            # 确定采样池的范围，并进行鲁棒性检查
            range_end = min(self.sampling_range_end, B * L - 1)
            range_start = min(self.sampling_range_start, range_end - 1)
            
            # 仅当候选池大小大于0时继续
            if range_end > range_start:
                # 1. 获取范围内的候选池索引
                _, top_indices = torch.topk(in_batch_pos_sims_mat, k=range_end, dim=1)
                candidate_indices = top_indices[:, range_start:range_end]

                # 2. 从候选池中随机采样K个 (通过随机排序实现无放回抽样)
                num_candidates = candidate_indices.shape[1]
                k_to_sample = min(k, num_candidates)
                
                rand_perms = torch.rand(candidate_indices.shape, device=self.dev).argsort(dim=1)
                shuffled_candidate_indices = torch.gather(candidate_indices, 1, rand_perms)
                sampled_indices = shuffled_candidate_indices[:, :k_to_sample]
                
                # 3. 根据采样到的索引，获取它们真实的相似度分数
                hard_pos_neg_sims = torch.gather(in_batch_pos_sims_mat, 1, sampled_indices)
        # --- 替换部分结束 ---

        # d. 组装最终的 logits。
        #    将正样本的相似度（形状变为 [B*L, 1]）和负样本相似度矩阵（形状 [B*L, B*L]）拼接起来。
        #    最终的 `logits_infonce` 形状为 [B*L, 1 + B*L]，
        #    对于每个序列位置，第 0 列是其与正样本的相似度，后面 B*L 列是其与所有批内负样本的相似度。
        pos_logits_infonce = pos_sims.view(B * L, 1)
        # 【开关控制】根据开关决定是否加入困难正样本作为负样本
        if self.use_in_batch_pos_as_neg and hard_pos_neg_sims.size(1) > 0:
            logits_infonce = torch.cat([pos_logits_infonce, neg_sims_mat, hard_pos_neg_sims * self.hard_negative_weight], dim=1) # [8.28.2修改]
        else:
            logits_infonce = torch.cat([pos_logits_infonce, neg_sims_mat], dim=1)  # 不使用批内正样本作为负样本
        
        # e. 应用温度系数 (temperature)，这是一个超参数，用于调节 softmax 的平滑程度。
        #    较低的温度会使模型更关注区分那些困难的负样本。
        logits_infonce = logits_infonce / self.temp

        # 5. --- 计算加权损失 ---
        # a. 筛选有效位置的 logits。
        #    使用之前定义的 `loss_mask`，只保留那些需要计算损失的 logits。
        valid_mask = loss_mask.view(B * L)
        logits_infonce = logits_infonce[valid_mask.bool()]

        # b. 创建目标标签。
        #    因为我们把正样本的相似度放在了 logits 的第 0 列，所以对于所有有效的样本，
        #    其正确的目标类别索引都是 0。
        labels_infonce = torch.zeros(logits_infonce.size(0), device=logits_infonce.device, dtype=torch.long)

        # c. 【优化】对点击行为进行加权。
        #    为了让模型更关注那些能带来转化的“点击”行为（action_type==1），
        #    我们为这部分样本的损失赋予更高的权重（`click_weight`）。
        #    这有助于提升模型预测高价值行为的能力。
        click_weight = getattr(self, 'click_weight', 1.5)
        next_action_type = ids['next_action_type']
        # 根据下一个行为是否为点击，生成权重张量。
        flat_weights = torch.where(next_action_type.to(self.dev) == 1,
                                torch.as_tensor(click_weight, device=self.dev, dtype=torch.float32),
                                torch.tensor(1.0, device=self.dev))
        flat_weights = flat_weights * loss_mask # 确保只在有效位置应用权重
        weights_infonce = flat_weights.view(B * L)[valid_mask.bool()].to(logits_infonce.dtype)
        
        # d. 计算最终的交叉熵损失。
        #    `reduction='none'` 表示计算每个样本的损失，而不是直接求平均。
        per_sample_loss = F.cross_entropy(logits_infonce, labels_infonce, reduction='none')
        # 将每个样本的损失与其对应的权重相乘，然后求加权平均，得到最终的标量损失。
        infonce_loss = (per_sample_loss * weights_infonce).sum() / (weights_infonce.sum() + 1e-8)
    
        return infonce_loss

    def predict_batch(self, batch: dict):  # 【8.18.2修改】
        """
        【8.18.2修改】新版推理入口：仅接受推理批字典（`dataset.MyTestDataset.collate_fn` 输出）。
        返回：
        - final_feat: Float [B, H]，取最后一个位置的时序表征。
        注意：本函数不做 L2 归一化；推理脚本 `infer.py` 在写出 `query.fbin` 前会进行归一化。
        """
        log_feats = self.log2feats_v2(batch)
        final_feat = log_feats[:, -1, :]
        return final_feat

    # def save_item_emb(self, item_ids, retrieval_ids, all_items_features, save_path, batch_size=1024):
    #     """
    #     【v2版-已重构】生成候选库 item embedding，并以二进制写盘。
    #     使用现代的 v2 数据接口，与训练流程保持一致，高效且易于维护。
    #     """
    #     all_embs = []
        
    #     # 关闭梯度计算，进入评估模式
    #     self.eval()
    #     with torch.no_grad():
    #         for i in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings (v2)"):
    #             end_i = min(i + batch_size, len(item_ids))
                
    #             # 1. 获取当前批次的数据
    #             item_ids_batch = item_ids[i:end_i]
    #             features_list_batch = all_items_features[i:end_i]
                
    #             # 2. 手动执行一个“迷你 Collate”过程
    #             # 将 list of dicts 转换为 dict of tensors
    #             # 这是从旧数据格式到新 v2 接口的关键桥梁
    #             collated_features = {}
    #             feature_keys = self.ITEM_SPARSE_FEAT.keys() | self.ITEM_ARRAY_FEAT.keys() | self.ITEM_CONTINUAL_FEAT.keys() | self.ITEM_EMB_FEAT.keys()

    #             for k in feature_keys:
    #                 # 从每个样本的特征字典中提取特定key的值
    #                 feature_values = [d.get(k, self.sparse_emb[k].padding_idx if k in self.sparse_emb else 0.0) for d in features_list_batch]
                    
    #                 # 根据特征类型进行不同的处理和堆叠
    #                 if k in self.ITEM_SPARSE_FEAT or k in self.ITEM_CONTINUAL_FEAT:
    #                     # 稀疏和连续特征可以直接堆叠
    #                     tensor = torch.tensor(feature_values, device=self.dev)
    #                     if k in self.ITEM_SPARSE_FEAT:
    #                         collated_features.setdefault('item_sparse', {})[k] = tensor.unsqueeze(1) # 增加伪序列长度维度
    #                     else:
    #                         collated_features.setdefault('item_continual', {})[k] = tensor.unsqueeze(1)
    #                 elif k in self.ITEM_ARRAY_FEAT:
    #                     # 数组特征需要填充
    #                     max_len = max(len(v) for v in feature_values if isinstance(v, list)) if feature_values else 0
    #                     padded_values = [v + [0] * (max_len - len(v)) for v in feature_values]
    #                     collated_features.setdefault('item_array', {})[k] = torch.tensor(padded_values, device=self.dev).unsqueeze(1)
    #                 elif k in self.ITEM_EMB_FEAT:
    #                     # 多模态特征已经是numpy数组，直接堆叠
    #                     collated_features.setdefault('mm_emb', {})[k] = torch.from_numpy(np.stack(feature_values)).to(self.dev).unsqueeze(1)

    #             # 3. 使用现代化的 v2 接口进行编码
    #             ids_tensor = torch.tensor(item_ids_batch, device=self.dev).unsqueeze(1) # 增加伪序列长度维度
    #             batch_emb = self.feat2emb_v2(
    #                 ids=ids_tensor,
    #                 features=collated_features, 
    #                 include_user=False
    #             ) # 输出形状: [batch_size, 1, H]
                
    #             # 移除伪序列长度维度，并进行归一化
    #             batch_emb = batch_emb.squeeze(1) # -> [batch_size, H]
    #             batch_emb = F.normalize(batch_emb, p=2, dim=-1)
                
    #             all_embs.append(batch_emb.cpu().numpy().astype(np.float32))

    #     # 合并所有批次的结果并保存
    #     final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
    #     final_embs = np.concatenate(all_embs, axis=0)
    #     save_emb(final_embs, Path(save_path, 'embedding.fbin'))
    #     save_emb(final_ids, Path(save_path, 'id.u64bin'))
    def feat2tensor(self, seq_feature, k):
        """
        Args:
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典，形状为 [batch_size, maxlen]
            k: 特征ID

        Returns:
            batch_data: 特征值的tensor，形状为 [batch_size, maxlen, max_array_len(if array)]
        """
        batch_size = len(seq_feature)

        if k in self.ITEM_ARRAY_FEAT or k in self.USER_ARRAY_FEAT:
            # 如果特征是Array类型，需要先对array进行padding，然后转换为tensor
            max_array_len = 0
            max_seq_len = 0

            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                max_seq_len = max(max_seq_len, len(seq_data))
                max_array_len = max(max_array_len, max(len(item_data) for item_data in seq_data))

            batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                for j, item_data in enumerate(seq_data):
                    actual_len = min(len(item_data), max_array_len)
                    batch_data[i, j, :actual_len] = item_data[:actual_len]

            return torch.from_numpy(batch_data).to(self.dev)
        else:
            # 如果特征是Sparse类型，直接转换为tensor
            max_seq_len = max(len(seq_feature[i]) for i in range(batch_size))
            batch_data = np.zeros((batch_size, max_seq_len), dtype=np.int64)

            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                batch_data[i] = seq_data

            return torch.from_numpy(batch_data).to(self.dev)

    def feat2emb(self, seq, feature_array, mask=None, include_user=False):
        """
        Args:
            seq: 序列ID
            feature_array: 特征list，每个元素为当前时刻的特征字典
            mask: 掩码，1表示item，2表示user
            include_user: 是否处理用户特征，在两种情况下不打开：1) 训练时在转换正负样本的特征时（因为正负样本都是item）;2) 生成候选库item embedding时。

        Returns:
            seqs_emb: 序列特征的Embedding
        """
        seq = seq.to(self.dev).long()
        # pre-compute embedding
        if include_user:
            user_mask = (mask == 2).to(self.dev)
            item_mask = (mask == 1).to(self.dev)
            user_embedding = self.user_emb(user_mask * seq)
            item_embedding = self.item_emb(item_mask * seq)
            item_feat_list = [item_embedding]
            user_feat_list = [user_embedding]
        else:
            item_embedding = self.item_emb(seq)
            item_feat_list = [item_embedding]

        # batch-process all feature types
        all_feat_types = [
            (self.ITEM_SPARSE_FEAT, 'item_sparse', item_feat_list),
            (self.ITEM_ARRAY_FEAT, 'item_array', item_feat_list),
            (self.ITEM_CONTINUAL_FEAT, 'item_continual', item_feat_list),
        ]

        if include_user:
            all_feat_types.extend(
                [
                    (self.USER_SPARSE_FEAT, 'user_sparse', user_feat_list),
                    (self.USER_ARRAY_FEAT, 'user_array', user_feat_list),
                    (self.USER_CONTINUAL_FEAT, 'user_continual', user_feat_list),
                ]
            )

        # batch-process each feature type
        for feat_dict, feat_type, feat_list in all_feat_types:
            if not feat_dict:
                continue

            for k in feat_dict:
                tensor_feature = self.feat2tensor(feature_array, k)

                if feat_type.endswith('sparse'):
                    feat_list.append(self.sparse_emb[k](tensor_feature))
                elif feat_type.endswith('array'):
                    # 【8.12.2修改】对可变长 array 特征做长度无关的平均聚合
                    emb = self.sparse_emb[k](tensor_feature)  # [B, T, A, H]
                    mask_array = (tensor_feature != 0).unsqueeze(-1)  # [B, T, A, 1]
                    emb_sum = (emb * mask_array).sum(2)  # [B, T, H]
                    lengths = mask_array.sum(2).clamp(min=1).to(emb.dtype)  # [B, T, 1]
                    feat_list.append(emb_sum / lengths)
                elif feat_type.endswith('continual'):
                    # 【8.12.2修改】对连续特征做小批量标准化（z-score）
                    tf = tensor_feature.float()
                    mean = tf.mean()
                    std = tf.std(unbiased=False) + 1e-6
                    tf = (tf - mean) / std
                    feat_list.append(tf.unsqueeze(2))

        for k in self.ITEM_EMB_FEAT:
            # collect all data to numpy, then batch-convert
            batch_size = len(feature_array)
            emb_dim = self.ITEM_EMB_FEAT[k]
            seq_len = len(feature_array[0])

            # pre-allocate tensor
            batch_emb_data = np.zeros((batch_size, seq_len, emb_dim), dtype=np.float32)

            for i, seq in enumerate(feature_array):
                for j, item in enumerate(seq):
                    if k in item:
                        batch_emb_data[i, j] = item[k]

            # batch-convert and transfer to GPU
            tensor_feature = torch.from_numpy(batch_emb_data).to(self.dev)
            item_feat_list.append(self.emb_transform[k](tensor_feature))

        # merge features
        all_item_emb = torch.cat(item_feat_list, dim=2)
        # 【8.12.2修改】移除外部 ReLU（Block 内已含激活与归一化）
        all_item_emb = self.itemdnn(all_item_emb)
        if include_user:
            all_user_emb = torch.cat(user_feat_list, dim=2)
            all_user_emb = self.userdnn(all_user_emb)
            seqs_emb = all_item_emb + all_user_emb
        else:
            seqs_emb = all_item_emb
        return seqs_emb

    def save_item_emb(self, item_ids, retrieval_ids, feat_dict, save_path, batch_size=1024):
        """
        生成候选库 item embedding（用于 ANN 检索），并以二进制写盘。  # 【8.18.2修改】

        Args:
            item_ids: 候选 item 的 re-id
            retrieval_ids: 候选 item 的检索 ID（从 0 开始，检索阶段用此 ID 回传）
            feat_dict: 训练集所有 item 的特征字典（按索引对齐）
            save_path: 保存目录
            batch_size: 批次大小

        行为：
        - 通过旧接口 `feat2emb` 批量生成 item 表征
        - 对每个向量做 L2 归一化（与训练/检索保持一致）
        - 保存为两类二进制文件（`dataset.save_emb` 写入，头部均为两个 uint32：N 与 D）：
          - embedding.fbin：float32，形状 [N, H]
          - id.u64bin：uint64，形状 [N, 1]（读取时 squeeze 为 [N]）
        """
        all_embs = []

        for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
            end_idx = min(start_idx + batch_size, len(item_ids))

            item_seq = torch.tensor(item_ids[start_idx:end_idx], device=self.dev).unsqueeze(0)
            batch_feat = []
            for i in range(start_idx, end_idx):
                batch_feat.append(feat_dict[i])

            batch_feat = np.array(batch_feat, dtype=object)

            batch_emb = self.feat2emb(item_seq, [batch_feat], include_user=False).squeeze(0)
            # 【ENHANCED】候选库向量 L2 归一化（与 InfoNCE 训练保持一致）
            batch_emb = torch.nn.functional.normalize(batch_emb, p=2, dim=-1)
            all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))

        # 合并所有批次的结果并保存
        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)
        save_emb(final_embs, Path(save_path, 'embedding.fbin'))
        save_emb(final_ids, Path(save_path, 'id.u64bin'))
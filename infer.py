import argparse
import json
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import struct
import random
from pathlib import Path

import numpy as np
import torch
torch.set_float32_matmul_precision('high')
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MyTestDataset, save_emb
from model import BaselineModel


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_ckpt_path():
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if ckpt_path is None:
        raise ValueError("MODEL_OUTPUT_PATH is not set")  
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--maxlen', default=101, type=int)
    parser.add_argument('--device', default='cuda', type=str)

    # 【9.2.2显存优化】与训练保持一致的显存优化参数
    parser.add_argument('--infonce_row_chunk', default=512, type=int, help='InfoNCE相似度矩阵分块大小，减少显存占用')
    parser.add_argument('--use_gradient_checkpointing', action='store_true', help='启用梯度检查点以减少显存占用')

    # --- 模型结构参数 (必须与训练时完全一致) ---
    parser.add_argument('--hidden_units', default=128, type=int)
    parser.add_argument('--num_blocks', default=8, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--emb_dropout', default=0.3, type=float)
    parser.add_argument('--attn_dropout', default=0.1, type=float)
    parser.add_argument('--ffn_dropout', default=0.1, type=float)
    parser.add_argument('--norm_first', action='store_true')
    parser.add_argument('--pos_enc', default='rope', choices=['abs', 'rope'])
    parser.add_argument('--ffn', default='swiglu', choices=['gelu', 'swiglu'])
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str)
    
    # --- 模型行为开关 (必须与训练时完全一致) ---
    parser.add_argument('--use_action_gate', default=True, type=bool)
    parser.add_argument('--action_vocab_size', default=3, type=int)
    parser.add_argument('--action_emb_dim', default=16, type=int)
    parser.add_argument('--time_bucket_count', default=7, type=int)
    parser.add_argument('--use_td_attn_bias', default=True, type=bool)  
    
    # --- 【修改】结果融合（Ensemble）相关参数 ---
    # 添加 'load_and_rrf' 选项
    parser.add_argument('--ensemble_mode', default='load_and_rrf', type=str, choices=['none', 'save', 'load_and_merge', 'load_and_rrf'], 
                        help="结果融合模式: 'none'-常规推理; 'save'-保存当前模型结果; 'load_and_merge'-加载并直接合并; 'load_and_rrf'-加载并使用RRF融合.")
    parser.add_argument('--ensemble_cache_file', default='ensemble_cache.json', type=str,
                        help="用于保存或加载融合结果的缓存文件名.")
    # 新增 RRF k 参数
    parser.add_argument('--rrf_k', default=60, type=int,
                        help="RRF融合中的平滑参数k.")
         
    args = parser.parse_args()

    return args


def read_result_ids(file_path):
    with open(file_path, 'rb') as f:
        # Read the header (num_points_query and FLAGS_query_ann_top_k)
        num_points_query = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes
        query_ann_top_k = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes

        print(f"num_points_query: {num_points_query}, query_ann_top_k: {query_ann_top_k}")

        # Calculate how many result_ids there are (num_points_query * query_ann_top_k)
        num_result_ids = num_points_query * query_ann_top_k

        # Read result_ids (uint64_t, 8 bytes per value)
        result_ids = np.fromfile(f, dtype=np.uint64, count=num_result_ids)

        return result_ids.reshape((num_points_query, query_ann_top_k))


# 【8.18.2修改】读取 .fbin 与 .u64bin（与 baseline/infer.py 对齐）
def read_fbin(file_path):
    with open(file_path, 'rb') as f:
        num_points = struct.unpack('I', f.read(4))[0]
        dim = struct.unpack('I', f.read(4))[0]
        data = np.fromfile(f, dtype=np.float32, count=num_points * dim)
        return data.reshape((num_points, dim))


def read_u64bin(file_path):
    with open(file_path, 'rb') as f:
        num_points = struct.unpack('I', f.read(4))[0]
        dim = struct.unpack('I', f.read(4))[0]
        data = np.fromfile(f, dtype=np.uint64, count=num_points * dim)
        return data.reshape((num_points, dim)).squeeze()


# 【8.18.2修改】基于 PyTorch 的 ANN Top‑K 检索（分块候选 + 查询分批，支持 GPU/CPU）
def torch_ann_topk(
    dataset_vecs,
    query_vecs,
    topk: int = 10,
    device: str = 'cpu',
    chunk_size: int = 200_000,
    q_batch_size: int = 128,
):
    """
    Args:
        dataset_vecs: np.ndarray 或 torch.Tensor，形状 [N, D]，建议已 L2 归一化
        query_vecs: np.ndarray 或 torch.Tensor，形状 [Q, D]，建议已 L2 归一化
        topk: 检索的 Top-K
        device: 'cuda' 或 'cpu'
        chunk_size: 候选库分块大小（例如 N=660k 时可取 200k）
        q_batch_size: 查询分批大小
    Returns:
        np.ndarray，形状 [Q, topk]，每行是候选的全局下标
    """
    # 转 tensor
    if isinstance(dataset_vecs, np.ndarray):
        dvecs = torch.from_numpy(dataset_vecs)
    else:
        dvecs = dataset_vecs
    if isinstance(query_vecs, np.ndarray):
        qvecs = torch.from_numpy(query_vecs)
    else:
        qvecs = query_vecs

    dvecs = dvecs.float().contiguous()
    qvecs = qvecs.float().contiguous()

    N, D = dvecs.shape
    Q = qvecs.shape[0]

    # 结果容器（放在 CPU）
    final_indices = torch.empty((Q, topk), dtype=torch.long)

    # 查询分批处理
    for q_start in range(0, Q, q_batch_size):
        q_end = min(q_start + q_batch_size, Q)
        qb = qvecs[q_start:q_end].to(device, non_blocking=True)  # [B, D]

        # 维持当前已见候选的 topk（在 device 上）
        B = qb.size(0)
        prev_scores = torch.full((B, topk), -1e9, device=device, dtype=torch.float32)
        prev_indices = torch.full((B, topk), -1, device=device, dtype=torch.long)

        # 候选库分块
        for d_start in range(0, N, chunk_size):
            d_end = min(d_start + chunk_size, N)
            dc = dvecs[d_start:d_end].to(device, non_blocking=True)  # [C, D]
            # [B, C] = [B, D] @ [D, C]
            scores = torch.matmul(qb, dc.t())

            # 取分块内的 topk（若 C < topk，则取 C）
            k_in = min(topk, scores.size(1))
            chunk_top_scores, chunk_top_idx = torch.topk(scores, k=k_in, dim=1)
            # 转全局下标
            chunk_top_idx = chunk_top_idx + d_start

            # 合并先前 topk 与当前分块 topk，保留全局 topk
            merged_scores = torch.cat([prev_scores, chunk_top_scores], dim=1)  # [B, k+k_in]
            merged_indices = torch.cat([prev_indices, chunk_top_idx], dim=1)   # [B, k+k_in]
            merged_top_scores, merged_top_pos = torch.topk(merged_scores, k=topk, dim=1)
            merged_top_indices = torch.gather(merged_indices, 1, merged_top_pos)

            prev_scores = merged_top_scores
            prev_indices = merged_top_indices

            # 显式释放块以便更好地复用显存
            del dc, scores, chunk_top_scores, chunk_top_idx, merged_scores, merged_indices, merged_top_scores, merged_top_pos, merged_top_indices
            torch.cuda.empty_cache() if torch.cuda.is_available() and 'cuda' in device else None

        final_indices[q_start:q_end] = prev_indices.cpu()

        del qb, prev_scores, prev_indices
        torch.cuda.empty_cache() if torch.cuda.is_available() and 'cuda' in device else None

    return final_indices.numpy()


def process_cold_start_feat(feat):
    """
    处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.可设计替换为更好的方法。
    """
    processed_feat = {}
    for feat_id, feat_value in feat.items():
        if type(feat_value) == list:
            value_list = []
            for v in feat_value:
                if type(v) == str:
                    value_list.append(0)
                else:
                    value_list.append(v)
            processed_feat[feat_id] = value_list
        elif type(feat_value) == str:
            processed_feat[feat_id] = 0
        else:
            processed_feat[feat_id] = feat_value
    return processed_feat


def get_candidate_emb(indexer, feat_types, feat_default_value, mm_emb_dict, model):
    """
    生产候选库item的id和embedding

    Args:
        indexer: 索引字典
        feat_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feature_default_value: 特征缺省值
        mm_emb_dict: 多模态特征字典
        model: 模型
    Returns:
        retrieve_id2creative_id: 索引id->creative_id的dict
    """
    EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    candidate_path = Path(os.environ.get('EVAL_DATA_PATH'), 'predict_set.jsonl')
    item_ids, creative_ids, retrieval_ids, features = [], [], [], []
    retrieve_id2creative_id = {}

    with open(candidate_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            # 读取item特征，并补充缺失值
            feature = line['features']
            creative_id = line['creative_id']
            retrieval_id = line['retrieval_id']
            item_id = indexer[creative_id] if creative_id in indexer else 0
            missing_fields = set(
                feat_types['item_sparse'] + feat_types['item_array'] + feat_types['item_continual']
            ) - set(feature.keys())
            feature = process_cold_start_feat(feature)
            for feat_id in missing_fields:
                feature[feat_id] = feat_default_value[feat_id]
            for feat_id in feat_types['item_emb']:
                if creative_id in mm_emb_dict[feat_id]:
                    feature[feat_id] = mm_emb_dict[feat_id][creative_id]
                else:
                    feature[feat_id] = np.zeros(EMB_SHAPE_DICT[feat_id], dtype=np.float32)

            item_ids.append(item_id)
            creative_ids.append(creative_id)
            retrieval_ids.append(retrieval_id)
            features.append(feature)
            retrieve_id2creative_id[retrieval_id] = creative_id

    # 保存候选库的embedding和sid
    model.save_item_emb(item_ids, retrieval_ids, features, os.environ.get('EVAL_RESULT_PATH'))
    with open(Path(os.environ.get('EVAL_RESULT_PATH'), "retrive_id2creative_id.json"), "w") as f:
        json.dump(retrieve_id2creative_id, f)
    return retrieve_id2creative_id


def make_collate_with_uid(base_collate):  # 【8.18.2修改】包装 collate，确保 ids.user_id 可用
    """
    【8.18.2修改】在不修改 dataset.py 的前提下，确保批字典中包含用户ID。
    - 若底层 collate 已包含 ids.user_id，则直接返回。
    - 否则从原始样本列表的第6个元素提取 user_id 并注入到 batch_dict['ids']['user_id']。
    """
    def collate_with_uid(batch):
        batch_dict = base_collate(batch)
        try:
            ids = batch_dict.get('ids', {})
            if not isinstance(ids, dict) or ('user_id' not in ids):
                user_ids = [sample[-1] for sample in batch]
                if not isinstance(ids, dict):
                    ids = {}
                ids['user_id'] = list(user_ids)
                batch_dict['ids'] = ids
        except Exception:
            # 若注入失败，保持原样返回，避免影响推理流程
            pass
        return batch_dict

    return collate_with_uid


def extract_ckpt_metrics(ckpt_dir):
    """从 ckpt 目录名提取训练验证指标 (仅作参考)"""
    import re
    name = os.path.basename(ckpt_dir) if ckpt_dir else ""
    ndcg = re.search(r'NDCG=([0-9.]+)', name)
    hr = re.search(r'HR=([0-9.]+)', name)
    ndcg = float(ndcg.group(1)) if ndcg else None
    hr = float(hr.group(1)) if hr else None
    score = ndcg * 0.85 + hr * 0.15 if (ndcg is not None and hr is not None) else None
    return ndcg, hr, score


def load_ground_truth(data_path):
    """
    从 predict_seq.jsonl 提取每个用户的 ground truth (最后一个点击 item 的 creative_id)。
    返回 dict: {user_id_str: creative_id_int}
    """
    import pickle as _pkl
    try:
        import orjson as _json
        def _loads(x): return _json.loads(x)
    except ImportError:
        import json as _json
        def _loads(x): return _json.loads(x)

    data_path = Path(data_path)
    with open(data_path / 'indexer.pkl', 'rb') as f:
        indexer = _pkl.load(f)
    i_rev = {v: k for k, v in indexer['i'].items()}  # rid -> oid
    u_rev = {v: k for k, v in indexer['u'].items()}  # rid -> user_id_str

    ground_truth = {}
    with open(data_path / 'predict_seq.jsonl', 'rb') as f:
        for line in f:
            if not line.strip():
                continue
            records = _loads(line)
            # 找 user_id: 第一条记录的 user_id (rid)
            user_rid = records[0][0]
            user_str = u_rev.get(user_rid, f"user_{user_rid}")
            # 找最后一个点击 item (action_type=1)
            last_click_rid = None
            for r in records:
                if r[1] is not None and r[4] == 1:  # item_id exists, action=click
                    last_click_rid = r[1]
            if last_click_rid is not None:
                ground_truth[user_str] = i_rev.get(last_click_rid, last_click_rid)
    return ground_truth


def evaluate_topk(top10s, user_list, ground_truth, k=10):
    """
    计算推理结果的 HR@k 和 NDCG@k。
    ground_truth: {user_id: target_creative_id}
    """
    hits = []
    ndcgs = []
    for user, top10 in zip(user_list, top10s):
        target = ground_truth.get(user)
        if target is None:
            continue
        if target in top10[:k]:
            rank = top10[:k].index(target) + 1
            hits.append(1.0)
            ndcgs.append(1.0 / np.log2(rank + 1))
        else:
            hits.append(0.0)
            ndcgs.append(0.0)

    hr = np.mean(hits) if hits else 0.0
    ndcg = np.mean(ndcgs) if ndcgs else 0.0
    return hr, ndcg, len(hits)


def infer():
    print("infer start")
    set_seed(42)
    args = get_args()
    data_path = os.environ.get('EVAL_DATA_PATH')
    test_dataset = MyTestDataset(data_path, args)
    collate_fn = make_collate_with_uid(test_dataset.collate_fn)  # 【8.18.2修改】包装以保证 user_id 可用
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=8,  # 【8.18.2修改】多进程数据加载
        collate_fn=collate_fn,
        pin_memory=True  # 【8.18.2修改】锁页内存，加速CPU->GPU传输
    )
    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    feat_statistics, feat_types = test_dataset.feat_statistics, test_dataset.feature_types
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    model.eval()
    # 【8.11.1修改】推理保持与训练一致的温度/损失设置（参数已在 get_args 中加入）
    # 【TimeDelta】确保 padding=0 行为 0 【8.17.2修改】
    if hasattr(model, 'time_delta_emb'):
        model.time_delta_emb.weight.data[0, :] = 0  # 【8.17.2修改】

    ckpt_path = get_ckpt_path()
        # --- 修改开始 ---
    # 1. 先把 state_dict 加载到一个临时的字典里
    state_dict = torch.load(ckpt_path, map_location=torch.device(args.device))

    # 2. 创建一个新的字典，用来存放修改过的 key
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    # 3. 遍历旧字典，去掉 '_orig_mod.' 前缀
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            name = k[10:] # 去掉 '_orig_mod.' (10个字符)
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v # 如果没有前缀，直接保留

    # 4. 加载修正后的 state_dict
    model.load_state_dict(new_state_dict)
    # --- 修改结束 ---
    # model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(args.device)))
    # all_embs = []
    # user_list = []
    # for step, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
    #     # 【8.18.2修改】改为字典批接口 + 新版推理入口
    #     logits = model.predict_batch(batch)  # 【8.18.2修改】
    #     # 【8.11.1修改】推理向量 L2 归一化（与 InfoNCE 训练保持一致）
    #     logits = torch.nn.functional.normalize(logits, p=2, dim=-1)
    #     for i in range(logits.shape[0]):
    #         emb = logits[i].unsqueeze(0).detach().cpu().numpy().astype(np.float32)
    #         all_embs.append(emb)
    #     user_list += batch['ids']['user_id']  # 【8.18.2修改】从批字典读取 user_id
    all_embs = []
    user_list = []
    for step, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="User embedding"):
        with torch.no_grad(): # 推理时最好都加上 no_grad()
            logits = model.predict_batch(batch)
            logits = torch.nn.functional.normalize(logits, p=2, dim=-1)
        
        # 直接对整个 batch 的 tensor 操作，效率高得多
        all_embs.append(logits.detach().cpu().numpy())
            
        user_list += batch['ids']['user_id']

    # 生成候选库的embedding 以及 id文件
    retrieve_id2creative_id = get_candidate_emb(
        test_dataset.indexer['i'],
        test_dataset.feature_types,
        test_dataset.feature_default_value,
        test_dataset.mm_emb_dict,
        model,
    )
    all_embs = np.concatenate(all_embs, axis=0)
    # 【8.11.1修改】候选库向量 L2 归一化
    all_embs = torch.nn.functional.normalize(torch.from_numpy(all_embs), p=2, dim=-1).numpy()
    # 保存query文件
    save_emb(all_embs, Path(os.environ.get('EVAL_RESULT_PATH'), 'query.fbin'))
    # 【8.18.2修改】PyTorch ANN Top‑K 检索（替换外部 faiss_demo 调用）
    dataset_vec_path = Path(os.environ.get("EVAL_RESULT_PATH"), "embedding.fbin")
    dataset_id_path = Path(os.environ.get("EVAL_RESULT_PATH"), "id.u64bin")
    query_vec_path = Path(os.environ.get("EVAL_RESULT_PATH"), "query.fbin")

    # 读取候选与查询向量/ID（按统一二进制格式）
    dataset_vecs = read_fbin(dataset_vec_path)
    dataset_ids = read_u64bin(dataset_id_path)
    query_vecs = read_fbin(query_vec_path)

    # 运行基于内积（与归一化后余弦等价）的 Top‑K 检索
    topk = 10
    top_indices = torch_ann_topk(
        dataset_vecs=dataset_vecs,
        query_vecs=query_vecs,
        topk=topk,
        device=args.device,
        chunk_size=200_000,        # 针对 N≈660k 的默认分块
        q_batch_size=args.batch_size,
    )

    # 映射为 creative_id 列表
    top10s_untrimmed = []
    for row in top_indices:
        for col in row.tolist():
            rid = int(dataset_ids[col])
            top10s_untrimmed.append(retrieve_id2creative_id.get(rid, 0))

    current_top10s = [top10s_untrimmed[i: i + topk] for i in range(0, len(top10s_untrimmed), topk)]
    current_user_list = user_list

    # --- 【修改】在此处添加结果的保存与融合逻辑 ---
    user_cache_path = os.environ.get('USER_CACHE_PATH')

    # 【保存模式】
    if args.ensemble_mode == 'save':
        if user_cache_path:
            cache_file_path = Path(user_cache_path, args.ensemble_cache_file)
            print(f"📦 [Ensemble SAVE] 正在保存当前模型结果到: {cache_file_path}")
            results_to_save = {'user_list': current_user_list, 'top10s': current_top10s}
            with open(cache_file_path, 'w') as f:
                json.dump(results_to_save, f)
            print("✅ 保存成功。")
        else:
            print("🔴 警告: USER_CACHE_PATH 未设置，无法保存结果。")
        return current_top10s, current_user_list

    # 【加载并直接合并模式】
    elif args.ensemble_mode == 'load_and_merge':
        if not user_cache_path:
            print("🔴 警告: USER_CACHE_PATH 未设置，无法加载用于融合的结果。将仅返回当前模型结果。")
            return current_top10s, current_user_list
        
        cache_file_path = Path(user_cache_path, args.ensemble_cache_file)
        if not cache_file_path.exists():
            print(f"🔴 警告: 缓存文件 {cache_file_path} 不存在，无法进行融合。将仅返回当前模型结果。")
            return current_top10s, current_user_list

        print(f"🚀 [Ensemble MERGE] 正在从 {cache_file_path} 加载结果并与当前结果融合...")
        with open(cache_file_path, 'r') as f:
            cached_data = json.load(f)
        
        cached_user_list = cached_data['user_list']
        cached_top10s = cached_data['top10s']

        # 使用字典进行高效匹配
        current_map = {user: top10 for user, top10 in zip(current_user_list, current_top10s)}
        cached_map = {user: top10 for user, top10 in zip(cached_user_list, cached_top10s)}
        
        # 以当前模型的 user_list 为准进行融合
        final_top10s = []
        for user in current_user_list:
            list_current = current_map.get(user, [])
            list_cached = cached_map.get(user, [])
            
            # 将两个列表合并，并去重，同时保持相对顺序
            merged_list = []
            seen = set()
            combined = list_current + list_cached
            for item in combined:
                if item not in seen:
                    seen.add(item)
                    merged_list.append(item)
            
            # 截取最终的Top10
            final_top10s.append(merged_list[:topk])
        
        print("✅ 结果融合完成。")
        return final_top10s, current_user_list

    # --- 【新增】RRF融合模式 ---
    elif args.ensemble_mode == 'load_and_rrf':
        if not user_cache_path:
            print("🔴 警告: USER_CACHE_PATH 未设置，无法加载用于RRF融合的结果。将仅返回当前模型结果。")
            return current_top10s, current_user_list
        
        cache_file_path = Path(user_cache_path, args.ensemble_cache_file)
        if not cache_file_path.exists():
            print(f"🔴 警告: 缓存文件 {cache_file_path} 不存在，无法进行RRF融合。将仅返回当前模型结果。")
            return current_top10s, current_user_list

        print(f"🚀 [Ensemble RRF] 正在从 {cache_file_path} 加载结果并与当前结果进行RRF融合...")
        with open(cache_file_path, 'r') as f:
            cached_data = json.load(f)
        
        cached_user_list = cached_data['user_list']
        cached_top10s = cached_data['top10s']

        # 使用字典进行高效匹配
        current_map = {user: top10 for user, top10 in zip(current_user_list, current_top10s)}
        cached_map = {user: top10 for user, top10 in zip(cached_user_list, cached_top10s)}
        
        final_top10s = []
        k = args.rrf_k # 从参数获取RRF的k值
        for user in current_user_list:
            list_current = current_map.get(user, [])
            list_cached = cached_map.get(user, [])
            
            # 计算RRF分数
            rrf_scores = {}
            # 处理当前模型的结果
            for rank, item in enumerate(list_current):
                if item not in rrf_scores:
                    rrf_scores[item] = 0
                rrf_scores[item] += 1 / (k + rank)
            
            # 处理缓存模型的结果
            for rank, item in enumerate(list_cached):
                if item not in rrf_scores:
                    rrf_scores[item] = 0
                rrf_scores[item] += 1 / (k + rank)
            
            # 根据RRF分数排序
            sorted_items = sorted(rrf_scores.keys(), key=lambda item: rrf_scores[item], reverse=True)
            
            # 截取最终的Top10
            final_top10s.append(sorted_items[:topk])
        
        print(f"✅ RRF结果融合完成 (k={k})。")
        return final_top10s, current_user_list
        
    # 【默认模式】
    else: # ensemble_mode == 'none'
        return current_top10s, current_user_list


def write_result(top10s, user_list, result_path):
    """将推理结果写入 result.jsonl"""
    result_path = Path(result_path)
    result_path.mkdir(parents=True, exist_ok=True)
    output_file = result_path / "result.jsonl"
    with open(output_file, 'w') as f:
        for user, top10 in zip(user_list, top10s):
            line = json.dumps({"user_id": user, "top10": top10}, ensure_ascii=False)
            f.write(line + '\n')
    print(f"✅ 推理结果已保存: {output_file} ({len(user_list)} 条)")


def print_summary(ckpt_dir, result_path, num_users, hr, ndcg, eval_count):
    """打印推理结果汇总 (含推理评测指标和最终 Score)"""
    score = ndcg * 0.85 + hr * 0.15
    # 训练验证指标 (仅作参考)
    val_ndcg, val_hr, val_score = extract_ckpt_metrics(ckpt_dir)

    print("")
    print("=" * 60)
    print("  📊 推理结果汇总")
    print("=" * 60)
    print(f"  Checkpoint : {os.path.basename(ckpt_dir)}")
    print(f"  推理用户数 : {num_users}")
    print(f"  评测用户数 : {eval_count} (有 ground truth 的用户)")
    print(f"  结果文件   : {result_path}/result.jsonl")
    print(f"  ────────────────────────────────")
    print(f"  [推理评测 - Top-K ANN 检索]")
    print(f"  NDCG@10    : {ndcg:.4f}")
    print(f"  HR@10      : {hr:.4f}")
    print(f"  Score      : {score:.4f}  (NDCG×0.85 + HR×0.15)")
    if val_ndcg is not None:
        print(f"  ────────────────────────────────")
        print(f"  [训练验证 - In-batch, 仅供参考]")
        print(f"  NDCG@10    : {val_ndcg:.4f}")
        print(f"  HR@10      : {val_hr:.4f}")
        print(f"  Score      : {val_score:.4f}")
    print("=" * 60)

    # 保存到文件
    summary_file = Path(result_path) / "metrics.json"
    summary = {
        "checkpoint": os.path.basename(ckpt_dir),
        "num_users": num_users,
        "eval_count": eval_count,
        "infer_NDCG@10": round(ndcg, 4),
        "infer_HR@10": round(hr, 4),
        "infer_Score": round(score, 4),
        "formula": "NDCG@10 * 0.85 + HR@10 * 0.15",
    }
    if val_ndcg is not None:
        summary["valid_NDCG@10"] = val_ndcg
        summary["valid_HR@10"] = val_hr
        summary["valid_Score"] = round(val_score, 4)
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  指标已保存: {summary_file}")


if __name__ == '__main__':
    top10s, user_list = infer()
    result_path = os.environ.get('EVAL_RESULT_PATH', './results')
    ckpt_dir = os.environ.get('MODEL_OUTPUT_PATH', '')
    data_path = os.environ.get('EVAL_DATA_PATH', './data')
    write_result(top10s, user_list, result_path)

    # 计算真实推理指标
    print("\n📊 加载 ground truth 并评测...")
    ground_truth = load_ground_truth(data_path)
    hr, ndcg, eval_count = evaluate_topk(top10s, user_list, ground_truth, k=10)
    print_summary(ckpt_dir, result_path, len(user_list), hr, ndcg, eval_count)
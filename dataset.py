# import json
import os, sys; os.system(f'"{sys.executable}" -m pip install --quiet orjson')
import orjson as json # <-- 换成这个
import pickle
import struct
from pathlib import Path
import time
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd  

class AliasMethod:
    """
    Alias Method for efficient sampling from a discrete probability distribution.
    """
    # 【9.5.1】: 在 AliasMethod 类中，添加一个新的类方法 from_precomputed
    @classmethod
    def from_precomputed(cls, filepath):
        """从预计算的 .npz 文件快速加载 prob 和 alias 表。"""
        tables = np.load(filepath)
        instance = cls(probs=[1]) # 用一个虚拟值实例化，避免重新计算
        instance.prob = tables['prob']
        instance.alias = tables['alias']
        instance.n = len(instance.prob)
        return instance

    def __init__(self, probs):
        """
        Initializes the AliasMethod by constructing the probability and alias tables.
        
        Args:
            probs (list or np.ndarray): A 1D array-like object of probabilities.
                                        Need not sum to 1, will be normalized.
        """
        self.n = len(probs)
        # 使用 deque 替代 list，因为 pop(0) 操作是 O(1)
        from collections import deque

        # 1. Normalize probabilities
        norm = sum(probs)
        if norm == 0:
            raise ValueError("Probabilities must not all be zero.")
        scaled_probs = [p * self.n / norm for p in probs]
        
        # 2. Initialize tables
        self.prob = np.zeros(self.n)
        self.alias = np.zeros(self.n, dtype=np.int32)
        
        # 3. Create worklists for small and large probabilities
        small = deque()
        large = deque()
        
        for i, p in enumerate(scaled_probs):
            if p < 1.0:
                small.append(i)
            else:
                large.append(i)
                
        # 4. Construct the tables
        while small and large:
            s = small.popleft()
            l = large.popleft()
            
            self.prob[s] = scaled_probs[s]
            self.alias[s] = l
            
            scaled_probs[l] = (scaled_probs[l] + scaled_probs[s]) - 1.0
            
            if scaled_probs[l] < 1.0:
                small.append(l)
            else:
                large.append(l)
                
        # Fill in the remaining probabilities (due to floating point inaccuracies)
        while large:
            self.prob[large.popleft()] = 1.0
        while small:
            self.prob[small.popleft()] = 1.0

    def draw(self, size=1):
        """
        Draw samples from the distribution in O(1) time.
        """
        i = np.random.randint(0, self.n, size=size)
        u = np.random.rand(size)
        
        res = np.where(u < self.prob[i], i, self.alias[i])

        if size == 1:
            return res[0]
        return res

class MyDataset(torch.utils.data.Dataset):
    """
    用户序列数据集

    Args:
        data_dir: 数据文件目录
        args: 全局参数

    Attributes:
        data_dir: 数据文件目录
        maxlen: 最大长度
        item_feat_dict: 物品特征字典
        mm_emb_ids: 激活的mm_emb特征ID
        mm_emb_dict: 多模态特征字典
        itemnum: 物品数量
        usernum: 用户数量
        indexer_i_rev: 物品索引字典 (reid -> item_id)
        indexer_u_rev: 用户索引字典 (reid -> user_id)
        indexer: 索引字典
        feature_default_value: 特征缺省值
        feature_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feat_statistics: 特征统计信息，包括user和item的特征数量
    """

    def __init__(self, data_dir, args):
        """
        初始化数据集
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self._load_data_and_offsets()
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id

        # self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
        with open(Path(data_dir, "item_feat_dict.json"), 'rb') as f:
            self.item_feat_dict = json.loads(f.read())
        # 【9.5.1】缓存 item_feat_dict 的键集合，用于后续O(1)快速检查
        self.item_feat_keys_set = set(self.item_feat_dict.keys())
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer
        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()
        # 【9.5.1】: 从缓存加载 Alias Table
        self.use_popularity_sampling = False
        self.alias_sampler = None
        self.alias_cache_path = None # 初始化缓存路径属性

        # 检查预计算的缓存文件是否存在
        try:
            cache_dir = os.environ.get('USER_CACHE_PATH')
            precomputed_file = Path(cache_dir) / 'alias_tables.npz'

            if precomputed_file.exists():
                print(f"检测到预计算的Alias Table缓存: {precomputed_file}")
                self.alias_cache_path = str(precomputed_file) # 保存路径
                self.use_popularity_sampling = True
                print("已启用基于【缓存】的Alias Method高效带权负采样。")
            else:
                # 如果缓存不存在，可以考虑保留原始的动态加载逻辑作为备用
                print("警告: 未找到预计算的Alias Table缓存，将尝试动态构建（若有概率文件）。")
                # (此处可以保留原来的逻辑，或者直接报错提示用户先运行预处理)

        except Exception as e:
            print(f"加载Alias Table缓存时发生错误: {e}，将使用均匀负采样。")

    def _load_data_and_offsets(self):
        """
        加载用户序列数据和每一行的文件偏移量(预处理好的), 用于快速随机访问数据并I/O
        """
        # 【8.18.2修改】不要在此处打开文件句柄，避免 DataLoader 多进程共享指针
        # 改为仅保存路径，并延迟到每个 worker 的首次读取时再打开
        self.data_path = self.data_dir / "seq.jsonl"
        self.data_file = None
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _load_user_data(self, uid):
        """
        从数据文件中加载单个用户的数据

        Args:
            uid: 用户ID(reid)

        Returns:
            data: 用户序列数据，格式为[(user_id, item_id, user_feat, item_feat, action_type, timestamp)]
        """
        # 【8.18.2修改】延迟在 worker 内打开独立文件句柄，避免多进程共享导致的竞态
        if self.data_file is None:
            self.data_file = open(self.data_path, 'rb')
        self.data_file.seek(self.seq_offsets[uid])
        line = self.data_file.readline()
        data = json.loads(line)
        return data

    def _random_neq(self, l, r, s):
        """
        生成一个不在序列s中的随机整数, 用于训练时的负采样

        Args:
            l: 随机整数的最小值
            r: 随机整数的最大值
            s: 序列

        Returns:
            t: 不在序列s中的随机整数
        """
        t = np.random.randint(l, r)
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t

    def _popularity_neq_final(self, s):
        """
        基于CTR感知的Alias Method负采样，避免选择已见物品
        
        Args:
            s: 已见物品集合
            
        Returns:
            t: 采样得到的负样本物品ID（不在s中且存在于item_feat_dict中）
        """
        if not self.use_popularity_sampling or self.alias_sampler is None:
            # 如果未启用流行性采样或采样器未初始化，回退到均匀采样
            return self._random_neq(1, self.itemnum + 1, s)
        
        # 延迟加载 Alias 采样器（仅在首次调用时加载）
        if self.alias_sampler is None and self.alias_cache_path is not None:
            try:
                self.alias_sampler = AliasMethod.from_precomputed(self.alias_cache_path)
                print(f"成功从缓存加载Alias采样器: {self.alias_cache_path}")
            except Exception as e:
                print(f"加载Alias采样器失败: {e}，回退到均匀采样")
                self.use_popularity_sampling = False
                return self._random_neq(1, self.itemnum + 1, s)
        
        # 使用Alias Method进行CTR感知的负采样
        max_attempts = 100  # 防止无限循环
        attempts = 0
        
        while attempts < max_attempts:
            # 使用Alias Method采样一个物品ID
            t = self.alias_sampler.draw()
            
            # 检查采样结果的有效性
            if (t > 0 and  # 确保不是padding ID
                t not in s and  # 确保不在已见物品中
                str(t) in self.item_feat_keys_set):  # 确保物品特征存在（使用O(1)检查）
                return t
                
            attempts += 1
        
        # 如果多次尝试都失败，回退到均匀采样
        print(f"警告: Alias采样{max_attempts}次均失败，回退到均匀采样")
        return self._random_neq(1, self.itemnum + 1, s)

    def _bucketize_time_delta(self, delta_seconds):  # 【8.17.2修改】
        """将时间差(秒)分桶，返回桶的索引ID。
        0: padding/无有效时间差；其余从1开始编号。
        桶边界: 60s, 10m, 1h, 1d, 1w, >1w -> 共 7 桶(含0)。
        """
        if delta_seconds is None:
            return 0
        try:
            ds = int(delta_seconds)
        except Exception:
            return 0
        if ds <= 0:
            return 1  # 0~60s 统一视作最短桶
        buckets = [60, 600, 3600, 3600 * 24, 3600 * 24 * 7]
        for i, edge in enumerate(buckets, start=1):
            if ds <= edge:
                return i
        return len(buckets) + 1

    def _add_temporal_features_to_sequence(self, user_sequence, tau=86400): # 【8.28.1修改】
        """
        使用矢量化操作为整个序列预先计算丰富的上下文特征。
        计算出的特征会被添加到每个记录的 user_feat 字典中。
        """
        if not user_sequence:
            return []

        ts_array = np.array([r[-1] for r in user_sequence], dtype=np.int64)

        # a. 对数时间差 (log_gap), 即 Δt_prev
        prev_ts_array = np.roll(ts_array, 1)
        prev_ts_array[0] = ts_array[0]
        time_gap = ts_array - prev_ts_array
        log_gap = np.log1p(time_gap)

        # b. 周期特征 (使用Pandas更清晰、更鲁棒)
        dt_series = pd.to_datetime(ts_array, unit='s', errors='coerce')
        hours = dt_series.hour.to_numpy(na_value=0, dtype=np.int32)
        weekdays = dt_series.weekday.to_numpy(na_value=0, dtype=np.int32)
        months = dt_series.month.to_numpy(na_value=0, dtype=np.int32)

        # c. 时间衰减 (time_decay), 即 recency
        last_ts = ts_array[-1]
        delta_t = last_ts - ts_array
        delta_scaled = delta_t / tau

        new_sequence = []
        for idx, record in enumerate(user_sequence):
            u, i, user_feat, item_feat, action_type, ts = record
            
            if user_feat is None:
                user_feat = {}
            
            # --- 修改点 START ---
            # 使用新的数字ID作为特征名
            user_feat['201'] = int(hours[idx])
            user_feat['202'] = int(weekdays[idx])
            user_feat['203'] = int(months[idx])
            user_feat['204'] = float(log_gap[idx])
            user_feat['205'] = float(delta_scaled[idx])
            # --- 修改点 END ---
            
            new_sequence.append((u, i, user_feat, item_feat, action_type, ts))
            
        return new_sequence

    def _transfer_context_features(self, user_feat, item_feat, cols_to_trans): # 【8.28.1修改】
        """
        将指定的上下文特征从user_feat拷贝到item_feat。
        """
        if not user_feat:
            return item_feat
        
        # 复制item_feat以避免修改原始字典
        item_feat = item_feat.copy() if item_feat else {}

        for col in cols_to_trans:
            if col in user_feat:
                item_feat[col] = user_feat[col]
                
        return item_feat

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户ID(reid)

        Returns:
            seq: 用户序列ID
            pos: 正样本ID（即下一个真实访问的item）
            neg: 负样本ID
            token_type: 用户序列类型，1表示item，2表示user
            next_token_type: 下一个token类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            pos_feat: 正样本特征，每个元素为字典，key为特征ID，value为特征值
            neg_feat: 负样本特征，每个元素为字典，key为特征ID，value为特征值
        """
        user_sequence_raw = self._load_user_data(uid)  # 动态加载用户数据
        user_sequence = self._add_temporal_features_to_sequence(user_sequence_raw) # 【8.28.1修改】

        # 【8.25.1修改】分离用户数据和物品序列，固定用户token在index=0
        user_profile_data = None
        item_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, timestamp = record_tuple
            if u and user_feat:
                user_profile_data = (u, user_feat, 2, action_type, timestamp)
            # if i and item_feat:
            #     item_sequence.append((i, item_feat, 1, action_type, timestamp))
            if i and item_feat:
                item_sequence.append((i, user_feat, item_feat, 1, action_type, timestamp)) # <--- 新增 user_feat

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        pos = np.zeros([self.maxlen + 1], dtype=np.int32)
        neg = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        # 新增：当前步 action_type（用于注意力动作门控，避免标签泄漏）
        action_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        # 新增：时间差分桶ID（仅对 item 位置有效；padding/user 位置=0）
        time_deltas = np.zeros([self.maxlen + 1], dtype=np.int32)  # 【8.17.2修改】

        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        pos_feat = np.empty([self.maxlen + 1], dtype=object)
        neg_feat = np.empty([self.maxlen + 1], dtype=object)

        # 【8.25.1修改】固定用户token在index=0位置
        if user_profile_data:
            u, user_feat, type_, act_type, timestamp = user_profile_data
            seq[0] = u
            token_type[0] = 2  # user type
            seq_feat[0] = self.fill_missing_feat(user_feat, u)
            # action_type[0], time_deltas[0] 等保持默认值0

        # 【8.25.1修改】截取最新的物品序列，为正样本预留一个位置
        max_items = self.maxlen  # 因为index=0被用户占用
        if len(item_sequence) > max_items + 1:
            item_sequence = item_sequence[-(max_items + 1):]

        # 【8.25.1修改】正样本是最后一个物品
        if item_sequence:
            nxt = item_sequence[-1]
            items_for_sequence = item_sequence[:-1]
        else:
            nxt = None
            items_for_sequence = []

        idx = self.maxlen
        # 从最后一个时间戳开始，用于计算当前步到下一步的时间差
        last_timestamp = nxt[-1] if nxt and nxt[-1] is not None else None  # 【8.17.2修改】

        ts = set()
        for record_tuple in item_sequence:  # 【8.25.1修改】只遍历物品序列
            if record_tuple[0]:
                ts.add(record_tuple[0])

        # 【8.25.1修改】left-padding, 从后往前遍历物品序列，填充到[1, maxlen]位置
        for record_tuple in reversed(items_for_sequence):
            i, user_feat, feat, type_, act_type, timestamp = record_tuple
            if nxt:
                next_i, _, next_feat, next_type, next_act_type, _ = nxt
            else:
                next_i, _, next_feat, next_type, next_act_type, _ = 0, {}, {}, 1, None, None
            feat = self.fill_missing_feat(feat, i)
            next_feat = self.fill_missing_feat(next_feat, next_i)
            
            # # 【8.27.4修改】对历史序列动态注入时间上下文特征，保证训推一致性
            # # 只有seq_feat注入时间特征，pos_feat/neg_feat保持纯净
            # if timestamp is not None:
            #     enhanced_feat = self._inject_temporal_features(feat, timestamp)
            # else:
            #     enhanced_feat = feat

            # 【8.28.1修改】将时间特征从 user_feat 移动到 item_feat
            # 定义需要从 user_feat 转移到 item_feat 的特征列表
            context_cols = [
                '201', 
                '202', 
                '203',
                '204',
                '205'
            ]
            # 调用新函数，完成从 user_feat -> item_feat 的注入
            enhanced_feat = self._transfer_context_features(user_feat, feat, context_cols)
            # 别忘了填充缺失值
            enhanced_feat = self.fill_missing_feat(enhanced_feat, i)
            
            # 同样，处理 next_feat 时也要填充
            next_feat = self.fill_missing_feat(next_feat, next_i)

            seq[idx] = i
            token_type[idx] = type_
            next_token_type[idx] = next_type
            if next_act_type is not None:
                next_action_type[idx] = next_act_type
            # 当前步的动作（仅对 item 位置生效；user 位置置 0）
            if act_type is not None and type_ == 1:
                action_type[idx] = act_type
            # 时间差分桶（仅对 item 位置计算） 【8.17.2修改】
            if type_ == 1:
                if last_timestamp is not None and timestamp is not None:
                    dt = int(max(0, last_timestamp - timestamp))  # 【8.17.2修改】
                else:
                    dt = None  # 【8.17.2修改】
                time_deltas[idx] = self._bucketize_time_delta(dt)  # 【8.17.2修改】
            seq_feat[idx] = enhanced_feat  # 【8.27.4修改】使用注入时间特征后的特征
            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                pos_feat[idx] = next_feat
                # neg_id = self._random_neq(1, self.itemnum + 1, ts)
                if self.use_popularity_sampling:
                    neg_id = self._popularity_neq_final(ts)
                else:
                    # 否则，回退到原始的均匀采样
                    neg_id = self._random_neq(1, self.itemnum + 1, ts)
                neg[idx] = neg_id
                neg_feat[idx] = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)
            nxt = record_tuple
            last_timestamp = timestamp
            idx -= 1
            if idx == 0:  # 【8.25.1修改】停止条件改为idx==0，因为index=0被用户占用
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        pos_feat = np.where(pos_feat == None, self.feature_default_value, pos_feat)
        neg_feat = np.where(neg_feat == None, self.feature_default_value, neg_feat)

        # 返回包含 time_deltas（位于末尾）的 11 元组  【8.17.2修改】
        return seq, pos, neg, token_type, action_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, time_deltas  # 【8.17.2修改】

    def __len__(self):
        """
        返回数据集长度，即用户数量

        Returns:
            usernum: 用户数量
        """
        return len(self.seq_offsets)

    def _init_feat_info(self):
        """
        初始化特征信息, 包括特征缺省值和特征类型

        Returns:
            feat_default_value: 特征缺省值，每个元素为字典，key为特征ID，value为特征缺省值
            feat_types: 特征类型，key为特征类型名称，value为包含的特征ID列表
        """
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        feat_types['user_sparse'] = ['103', '104', '105', '109']
        feat_types['item_sparse'] = [
            '100',
            '117',
            '111',
            '118',
            '101',
            '102',
            '119',
            '120',
            '114',
            '112',
            '121',
            '115',
            '122',
            '116',
            # 【8.27.4修改】新增时间上下文特征，动态注入到历史序列中
            '201',    # 小时特征 (0-23)  
            '202',   # 星期几特征 (0-6)
            '203',   # 月份特征 (1-12)
        ]
        feat_types['item_array'] = []
        feat_types['user_array'] = ['106', '107', '108', '110']
        feat_types['item_emb'] = self.mm_emb_ids
        feat_types['user_continual'] = []
        feat_types['item_continual'] = ['204', '205'] # 对数时间差， 时间衰减

        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            # 【8.27.4修改】为时间上下文特征定义词典大小，其他特征从indexer获取
            if feat_id == '203':
                feat_statistics[feat_id] = 13   # 1-12月 + padding(0)
            elif feat_id == '201':
                feat_statistics[feat_id] = 25   # 0-23小时 + padding(0)
            elif feat_id == '202':
                feat_statistics[feat_id] = 8    # 0-6星期 + padding(0)
            else:
                feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_continual']:
            feat_default_value[feat_id] = 0.0 # 使用浮点数0.0
        for feat_id in feat_types['item_emb']:
            feat_default_value[feat_id] = np.zeros(
                list(self.mm_emb_dict[feat_id].values())[0].shape[0], dtype=np.float32
            )

        return feat_default_value, feat_types, feat_statistics

    def fill_missing_feat(self, feat, item_id):
        """
        对于原始数据中缺失的特征进行填充缺省值

        Args:
            feat: 特征字典
            item_id: 物品ID

        Returns:
            filled_feat: 填充后的特征字典
        """
        if feat == None:
            feat = {}
        filled_feat = {}
        for k in feat.keys():
            filled_feat[k] = feat[k]

        all_feat_ids = []
        for feat_type in self.feature_types.values():
            all_feat_ids.extend(feat_type)
        missing_fields = set(all_feat_ids) - set(feat.keys())
        for feat_id in missing_fields:
            filled_feat[feat_id] = self.feature_default_value[feat_id]
        for feat_id in self.feature_types['item_emb']:
            if item_id != 0 and self.indexer_i_rev[item_id] in self.mm_emb_dict[feat_id]:
                if type(self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]) == np.ndarray:
                    filled_feat[feat_id] = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]

        return filled_feat

    def collate_fn(self, batch):  # 【8.18.2修改】统一批字典Schema，含数组特征动态padding
        """
        【8.18.2修改】将训练批次拼接为统一的批字典结构，包含 ids/masks/time_deltas/features/meta。
        - 所有离散ID/数组元素/时间桶/动作/类型均为 torch.int64。
        - 连续特征与多模态向量为 torch.float32。
        - 数组特征按 feat_id 在批内动态 padding 到 [B, L, A_max]。
        - mm_emb 特征按其维度 D 组装为 [B, L, D]。

        Args:
            batch: 多个 __getitem__ 返回的 11 元组。

        Returns:
            batch_dict: 统一的批字典。
        """
        # 解包并转 tensor（long）  【8.18.2修改】
        seq, pos, neg, token_type, action_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, time_deltas = zip(*batch)
        seq = torch.from_numpy(np.array(seq)).long()
        pos = torch.from_numpy(np.array(pos)).long()
        neg = torch.from_numpy(np.array(neg)).long()
        token_type = torch.from_numpy(np.array(token_type)).long()
        action_type = torch.from_numpy(np.array(action_type)).long()
        next_token_type = torch.from_numpy(np.array(next_token_type)).long()
        next_action_type = torch.from_numpy(np.array(next_action_type)).long()
        time_deltas = torch.from_numpy(np.array(time_deltas)).long()

        B = seq.shape[0]
        L = seq.shape[1]
        seq_feat_list = list(seq_feat)
        pos_feat_list = list(pos_feat)
        neg_feat_list = list(neg_feat)

        # 构建各类型特征的批量张量  【8.18.2修改】
        def build_sparse_from(feat_list, feat_id):
            arr = np.zeros((B, L), dtype=np.int64)
            for b in range(B):
                seq_b = feat_list[b]
                for t in range(L):
                    v = seq_b[t].get(feat_id, 0)
                    if isinstance(v, list):
                        v = v[0] if len(v) > 0 else 0
                    try:
                        arr[b, t] = int(v)
                    except Exception:
                        arr[b, t] = 0
            return torch.from_numpy(arr).long()

        def build_array_from(feat_list, feat_id):
            A_max = 0
            for b in range(B):
                seq_b = feat_list[b]
                for t in range(L):
                    v = seq_b[t].get(feat_id, [])
                    if isinstance(v, list):
                        A_max = max(A_max, len(v))
                    else:
                        A_max = max(A_max, 1)
            A_max = max(A_max, 1)
            arr = np.zeros((B, L, A_max), dtype=np.int64)
            for b in range(B):
                seq_b = feat_list[b]
                for t in range(L):
                    v = seq_b[t].get(feat_id, [])
                    if isinstance(v, list):
                        a = v[:A_max]
                    else:
                        a = [v]
                    if len(a) > 0:
                        arr[b, t, : len(a)] = np.asarray(a, dtype=np.int64)
            return torch.from_numpy(arr).long()

        def build_continual_from(feat_list, feat_id):
            arr = np.zeros((B, L), dtype=np.float32)
            for b in range(B):
                seq_b = feat_list[b]
                for t in range(L):
                    v = seq_b[t].get(feat_id, 0.0)
                    try:
                        arr[b, t] = float(v)
                    except Exception:
                        arr[b, t] = 0.0
            return torch.from_numpy(arr).float()

        def build_mm_from(feat_list, feat_id):
            default_vec = self.feature_default_value[feat_id]
            D = int(default_vec.shape[0]) if hasattr(default_vec, 'shape') else int(len(default_vec))
            arr = np.zeros((B, L, D), dtype=np.float32)
            for b in range(B):
                seq_b = feat_list[b]
                for t in range(L):
                    v = seq_b[t].get(feat_id, None)
                    if isinstance(v, np.ndarray):
                        if v.shape[0] != D:
                            dlen = min(v.shape[0], D)
                            arr[b, t, :dlen] = v[:dlen].astype(np.float32)
                        else:
                            arr[b, t] = v.astype(np.float32)
                    elif isinstance(v, list) and len(v) > 0:
                        vv = np.asarray(v, dtype=np.float32)
                        dlen = min(vv.shape[0], D)
                        arr[b, t, :dlen] = vv[:dlen]
            return torch.from_numpy(arr).float()

        # 初始化特征容器  【8.18.2修改】
        features = {
            'seq': {
                'user_sparse': {}, 'item_sparse': {},
                'user_array': {}, 'item_array': {},
                'user_continual': {}, 'item_continual': {},
                'mm_emb': {}
            },
            'pos': {
                'item_sparse': {}, 'item_array': {}, 'item_continual': {}, 'mm_emb': {}
            },
            'neg': {
                'item_sparse': {}, 'item_array': {}, 'item_continual': {}, 'mm_emb': {}
            }
        }

        # 序列侧（含 user/item）  【8.18.2修改】
        for fid in self.feature_types['user_sparse']:
            features['seq']['user_sparse'][fid] = build_sparse_from(seq_feat_list, fid)
        for fid in self.feature_types['item_sparse']:
            features['seq']['item_sparse'][fid] = build_sparse_from(seq_feat_list, fid)
        for fid in self.feature_types['user_array']:
            features['seq']['user_array'][fid] = build_array_from(seq_feat_list, fid)
        for fid in self.feature_types['item_array']:
            features['seq']['item_array'][fid] = build_array_from(seq_feat_list, fid)
        for fid in self.feature_types['user_continual']:
            features['seq']['user_continual'][fid] = build_continual_from(seq_feat_list, fid)
        for fid in self.feature_types['item_continual']:
            features['seq']['item_continual'][fid] = build_continual_from(seq_feat_list, fid)
        for fid in self.feature_types['item_emb']:
            features['seq']['mm_emb'][fid] = build_mm_from(seq_feat_list, fid)

        # 正/负样本侧（仅 item）  【8.18.2修改】
        for fid in self.feature_types['item_sparse']:
            features['pos']['item_sparse'][fid] = build_sparse_from(pos_feat_list, fid)
            features['neg']['item_sparse'][fid] = build_sparse_from(neg_feat_list, fid)
        for fid in self.feature_types['item_array']:
            features['pos']['item_array'][fid] = build_array_from(pos_feat_list, fid)
            features['neg']['item_array'][fid] = build_array_from(neg_feat_list, fid)
        for fid in self.feature_types['item_continual']:
            features['pos']['item_continual'][fid] = build_continual_from(pos_feat_list, fid)
            features['neg']['item_continual'][fid] = build_continual_from(neg_feat_list, fid)
        for fid in self.feature_types['item_emb']:
            features['pos']['mm_emb'][fid] = build_mm_from(pos_feat_list, fid)
            features['neg']['mm_emb'][fid] = build_mm_from(neg_feat_list, fid)

        batch_dict = {
            'ids': {
                'seq': seq, 'pos': pos, 'neg': neg,
                'action_type': action_type, 'next_action_type': next_action_type,
            },
            'masks': {
                'token_type': token_type, 'next_token_type': next_token_type,
            },
            'time_deltas': time_deltas,
            'features': features,
            'meta': {
                'batch_size': int(B), 'seq_len': int(L), 'maxlen': int(L),
            }
        }

        return batch_dict

class MyTestDataset(MyDataset):
    """
    测试数据集
    """

    def __init__(self, data_dir, args):
        super().__init__(data_dir, args)

    def _load_data_and_offsets(self):
        # 【8.18.2修改】测试集同样采用延迟打开策略
        self.data_path = self.data_dir / "predict_seq.jsonl"
        self.data_file = None
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _process_cold_start_feat(self, feat):
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

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户在self.data_file中储存的行号
        Returns:
            seq: 用户序列ID
            token_type: 用户序列类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            user_id: user_id eg. user_xxxxxx ,便于后面对照答案
        """
        user_sequence_raw = self._load_user_data(uid)  # 动态加载用户数据
        user_sequence = self._add_temporal_features_to_sequence(user_sequence_raw) # 【8.28.1修改】

        # 【8.25.1修改】分离用户数据和物品序列，固定用户token在index=0
        user_profile_data = None
        item_sequence = []
        for record_tuple in user_sequence:
            # 预测序列中也保留 action_type（若缺失则为 None）
            u, i, user_feat, item_feat, action_type_val, timestamp = record_tuple
            if u:
                if type(u) == str:  # 如果是字符串，说明是user_id
                    user_id = u
                else:  # 如果是int，说明是re_id
                    user_id = self.indexer_u_rev[u]
            if u and user_feat:
                if type(u) == str:
                    u = 0
                if user_feat:
                    user_feat = self._process_cold_start_feat(user_feat)
                user_profile_data = (u, user_feat, 2, action_type_val, timestamp)

            if i and item_feat:
                # 序列对于训练时没见过的item，不会直接赋0，而是保留creative_id，creative_id远大于训练时的itemnum
                if i > self.itemnum:
                    i = 0
                if item_feat:
                    item_feat = self._process_cold_start_feat(item_feat)
                item_sequence.append((i, user_feat, item_feat, 1, action_type_val, timestamp))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        # 新增：当前步 action_type（用于推理阶段的门控；若缺失则为 0）
        action_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        # 新增：时间差分桶ID（仅对 item 位置有效；padding/user 位置=0）
        time_deltas = np.zeros([self.maxlen + 1], dtype=np.int32)  # 【8.17.2修改】

        # 【8.25.1修改】固定用户token在index=0位置
        if user_profile_data:
            u, user_feat, type_, act_type, timestamp = user_profile_data
            seq[0] = u
            token_type[0] = 2  # user type
            seq_feat[0] = self.fill_missing_feat(user_feat, u)
            # action_type[0], time_deltas[0] 等保持默认值0

        idx = self.maxlen
        # 【8.21.1修改】推理阶段不再丢弃最后一个位置；为最后一个位置的 time_delta 置 0
        # 通过将初始 last_timestamp 置为 None，使首次迭代得到 time_delta 桶=0
        last_timestamp = None  # 【8.21.1修改】

        ts = set()
        for record_tuple in item_sequence:  # 【8.25.1修改】只遍历物品序列
            if record_tuple[0]:
                ts.add(record_tuple[0])

        # 【8.25.1修改】遍历物品序列，填充到[1, maxlen]位置
        for record_tuple in reversed(item_sequence):  # 【8.21.1修改】包含最后一个位置
            i, user_feat, feat, type_, act_type, timestamp = record_tuple
            feat = self.fill_missing_feat(feat, i)
            
            # 定义需要转移的特征列表  [8.28.1修改]
            context_cols = ['201', '202', '203', '204', '205']
            # 调用新函数进行注入
            enhanced_feat = self._transfer_context_features(user_feat, feat, context_cols)
            # 填充
            enhanced_feat = self.fill_missing_feat(enhanced_feat, i)

            seq[idx] = i
            token_type[idx] = type_
            if act_type is not None and type_ == 1:
                action_type[idx] = act_type
            # 时间差分桶（仅对 item 位置计算） 【8.17.2修改】
            if type_ == 1:
                if last_timestamp is not None and timestamp is not None:
                    dt = int(max(0, last_timestamp - timestamp))  # 【8.17.2修改】
                else:
                    dt = None  # 【8.17.2修改】
                time_deltas[idx] = self._bucketize_time_delta(dt)  # 【8.17.2修改】
            seq_feat[idx] = enhanced_feat  # 【8.27.4修改】使用注入时间特征后的特征
            last_timestamp = timestamp
            idx -= 1
            if idx == 0:  # 【8.25.1修改】停止条件改为idx==0，因为index=0被用户占用
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)

        return seq, token_type, action_type, seq_feat, time_deltas, user_id  # 【8.17.2修改】

    def __len__(self):
        """
        Returns:
            len(self.seq_offsets): 用户数量
        """
        return len(self.seq_offsets)

    def collate_fn(self, batch):  # 【8.18.2修改】推理批字典Schema
        """
        【8.18.2修改】将推理批次拼接为统一的批字典结构（仅包含 seq 分支）。
        Args:
            batch: 多个 __getitem__ 返回的 6 元组。
        Returns:
            batch_dict: 统一的批字典。
        """
        seq, token_type, action_type, seq_feat, time_deltas, user_id = zip(*batch)
        seq = torch.from_numpy(np.array(seq)).long()
        token_type = torch.from_numpy(np.array(token_type)).long()
        action_type = torch.from_numpy(np.array(action_type)).long()
        time_deltas = torch.from_numpy(np.array(time_deltas)).long()

        B = seq.shape[0]
        L = seq.shape[1]
        seq_feat_list = list(seq_feat)

        # 与训练侧一致的构建函数  【8.18.2修改】
        def build_sparse_from(feat_list, feat_id):
            arr = np.zeros((B, L), dtype=np.int64)
            for b in range(B):
                seq_b = feat_list[b]
                for t in range(L):
                    v = seq_b[t].get(feat_id, 0)
                    if isinstance(v, list):
                        v = v[0] if len(v) > 0 else 0
                    try:
                        arr[b, t] = int(v)
                    except Exception:
                        arr[b, t] = 0
            return torch.from_numpy(arr).long()

        def build_array_from(feat_list, feat_id):
            A_max = 0
            for b in range(B):
                seq_b = feat_list[b]
                for t in range(L):
                    v = seq_b[t].get(feat_id, [])
                    if isinstance(v, list):
                        A_max = max(A_max, len(v))
                    else:
                        A_max = max(A_max, 1)
            A_max = max(A_max, 1)
            arr = np.zeros((B, L, A_max), dtype=np.int64)
            for b in range(B):
                seq_b = feat_list[b]
                for t in range(L):
                    v = seq_b[t].get(feat_id, [])
                    if isinstance(v, list):
                        a = v[:A_max]
                    else:
                        a = [v]
                    if len(a) > 0:
                        arr[b, t, : len(a)] = np.asarray(a, dtype=np.int64)
            return torch.from_numpy(arr).long()

        def build_continual_from(feat_list, feat_id):
            arr = np.zeros((B, L), dtype=np.float32)
            for b in range(B):
                seq_b = feat_list[b]
                for t in range(L):
                    v = seq_b[t].get(feat_id, 0.0)
                    try:
                        arr[b, t] = float(v)
                    except Exception:
                        arr[b, t] = 0.0
            return torch.from_numpy(arr).float()

        def build_mm_from(feat_list, feat_id):
            default_vec = self.feature_default_value[feat_id]
            D = int(default_vec.shape[0]) if hasattr(default_vec, 'shape') else int(len(default_vec))
            arr = np.zeros((B, L, D), dtype=np.float32)
            for b in range(B):
                seq_b = feat_list[b]
                for t in range(L):
                    v = seq_b[t].get(feat_id, None)
                    if isinstance(v, np.ndarray):
                        if v.shape[0] != D:
                            dlen = min(v.shape[0], D)
                            arr[b, t, :dlen] = v[:dlen].astype(np.float32)
                        else:
                            arr[b, t] = v.astype(np.float32)
                    elif isinstance(v, list) and len(v) > 0:
                        vv = np.asarray(v, dtype=np.float32)
                        dlen = min(vv.shape[0], D)
                        arr[b, t, :dlen] = vv[:dlen]
            return torch.from_numpy(arr).float()

        features = {
            'seq': {
                'user_sparse': {}, 'item_sparse': {},
                'user_array': {}, 'item_array': {},
                'user_continual': {}, 'item_continual': {},
                'mm_emb': {}
            }
        }
        for fid in self.feature_types['user_sparse']:
            features['seq']['user_sparse'][fid] = build_sparse_from(seq_feat_list, fid)
        for fid in self.feature_types['item_sparse']:
            features['seq']['item_sparse'][fid] = build_sparse_from(seq_feat_list, fid)
        for fid in self.feature_types['user_array']:
            features['seq']['user_array'][fid] = build_array_from(seq_feat_list, fid)
        for fid in self.feature_types['item_array']:
            features['seq']['item_array'][fid] = build_array_from(seq_feat_list, fid)
        for fid in self.feature_types['user_continual']:
            features['seq']['user_continual'][fid] = build_continual_from(seq_feat_list, fid)
        for fid in self.feature_types['item_continual']:
            features['seq']['item_continual'][fid] = build_continual_from(seq_feat_list, fid)
        for fid in self.feature_types['item_emb']:
            features['seq']['mm_emb'][fid] = build_mm_from(seq_feat_list, fid)

        batch_dict = {
            'ids': {
                'seq': seq, 'action_type': action_type, 'user_id': list(user_id),
            },
            'masks': {
                'token_type': token_type,
            },
            'time_deltas': time_deltas,
            'features': features,
            'meta': {
                'batch_size': int(B), 'seq_len': int(L), 'maxlen': int(L),
            }
        }

        return batch_dict

def save_emb(emb, save_path):
    """
    将Embedding保存为二进制文件

    Args:
        emb: 要保存的Embedding，形状为 [num_points, num_dimensions]
        save_path: 保存路径
    """
    num_points = emb.shape[0]  # 数据点数量
    num_dimensions = emb.shape[1]  # 向量的维度
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)

def load_mm_emb(mm_path, feat_ids):
    """
    加载多模态特征Embedding

    Args:
        mm_path: 多模态特征Embedding路径
        feat_ids: 要加载的多模态特征ID列表

    Returns:
        mm_emb_dict: 多模态特征Embedding字典，key为特征ID，value为特征Embedding字典（key为item ID，value为Embedding）
    """
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mm_emb_dict = {}
    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
        shape = SHAPE_DICT[feat_id]
        emb_dict = {}
        if feat_id != '81':
            try:
                base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
                for json_file in base_path.glob('*.json'):
                    with open(json_file, 'r', encoding='utf-8') as file:
                        for line in file:
                            data_dict_origin = json.loads(line.strip())
                            insert_emb = data_dict_origin['emb']
                            if isinstance(insert_emb, list):
                                insert_emb = np.array(insert_emb, dtype=np.float32)
                            data_dict = {data_dict_origin['anonymous_cid']: insert_emb}
                            emb_dict.update(data_dict)
            except Exception as e:
                print(f"transfer error: {e}")
        if feat_id == '81':
            with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                emb_dict = pickle.load(f)
        mm_emb_dict[feat_id] = emb_dict
        print(f'Loaded #{feat_id} mm_emb')
    return mm_emb_dict

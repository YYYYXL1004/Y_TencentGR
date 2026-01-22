import os
import sys
os.system(f'"{sys.executable}" -m pip install --quiet orjson')
import orjson as json
from pathlib import Path
from collections import Counter
import numpy as np
from tqdm import tqdm
import pickle
# 确保你的 dataset.py 和这个脚本在同一个目录，或者在 Python Path 中
from dataset import AliasMethod 

def run_complete_preprocessing():
    """
    一个自包含、健壮的预处理脚本，完成从原始数据到Alias Table的全过程。
    1. 遍历 seq.jsonl，统计每个物品的曝光和点击。
    2. 计算CTR感知的采样权重。
    3. 构建并保存 Alias Table。
    """
    # ========================= 参数配置 =========================
    # 从环境变量获取平台路径，如果不存在则使用默认相对路径
    data_path = os.environ.get('TRAIN_DATA_PATH') 
    cache_path = os.environ.get('USER_CACHE_PATH')
    
    # --- 关键假设：Action Type 定义 ---
    # 0 代表曝光 (impression/exposure), 1 代表点击 (click)。
    # 如果你的数据定义不同，请务必修改这里！
    ACTION_TYPE_MAP = {'click': 1}
    
    # 平滑CTR时的常数K
    CTR_SMOOTHING_CONSTANT = 11
    # ==========================================================

    # 确保缓存目录存在
    Path(cache_path).mkdir(parents=True, exist_ok=True)
    
    # 动态读取总物品数
    try:
        with open(Path(data_path) / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            item_total_count = len(indexer['i']) + 1
            print(f"检测到总物品数为: {item_total_count - 1}")
    except FileNotFoundError:
        print(f"错误: indexer.pkl 未在 {data_path} 找到。请检查路径。")
        return

    # --- 步骤 1: 遍历原始数据，统计曝光与点击 (这是最耗时的部分) ---
    print("步骤 1/4: 开始遍历 seq.jsonl 统计曝光与点击...")
    impression_counts = Counter()
    click_counts = Counter()
    seq_file = Path(data_path) / 'seq.jsonl'
    
    if not seq_file.exists():
        print(f"错误: 原始序列文件 seq.jsonl 未在 {data_path} 找到。")
        return

    with open(seq_file, 'rb') as f:
        for line in tqdm(f, desc="处理用户记录", unit=" users"):
            user_sequence = json.loads(line)
            for record in user_sequence:
                item_id, action_type = record[1], record[4]
                
                if item_id is None or action_type is None:
                    continue
                
                impression_counts.update([item_id])
                if action_type == ACTION_TYPE_MAP['click']:
                    click_counts.update([item_id])

    print(f"统计完成！共发现 {len(impression_counts)} 个独立物品被曝光过。")
    # print(f"有{len(click_counts)}物品被点击")
    # print(f"ctr:{len(click_counts)/len(impression_counts)}")
    total_impressions = sum(impression_counts.values())
    total_clicks = sum(click_counts.values())

    if total_impressions > 0:
        global_ctr = total_clicks / total_impressions
    else:
        global_ctr = 0

    print(f"总曝光次数 (Total Impressions): {total_impressions}")
    print(f"总点击次数 (Total Clicks): {total_clicks}")
    print(f"真正的全局平均CTR (Global Average CTR): {global_ctr:.6f}")

    # --- 步骤 2: 计算CTR感知的采样权重 ---
    print("步骤 2/4: 计算CTR感知的采样权重...")
    all_impressions = np.zeros(item_total_count, dtype=np.float64)
    all_clicks = np.zeros(item_total_count, dtype=np.float64)

    for item_id, count in impression_counts.items():
        if item_id < item_total_count:
            all_impressions[item_id] = count
    for item_id, count in click_counts.items():
        if item_id < item_total_count:
            all_clicks[item_id] = count

    # 计算平滑CTR
    smoothed_ctr = (all_clicks + 1) / (all_impressions + CTR_SMOOTHING_CONSTANT)
    # 计算最终权重 W(i) = log(Imp + 1) * (1 - smoothed_CTR)
    final_weights = np.log1p(all_impressions) * np.maximum(0, 1 - smoothed_ctr)
    final_weights[0] = 0.0 # 确保 padding=0 不被采样

    # --- 步骤 3: 构建 Alias Table ---
    print("步骤 3/4: 构建 Alias Method 采样表...")
    sampler = AliasMethod(final_weights.tolist())

    # --- 步骤 4: 保存结果 ---
    save_file = Path(cache_path) / 'alias_tables.npz'
    print(f"步骤 4/4: 保存采样表到 {save_file}...")
    np.savez_compressed(save_file, prob=sampler.prob, alias=sampler.alias)
    
    print("\n预处理全部完成！")
    print("新的CTR感知负采样策略已生成，现在可以开始模型训练了。")


if __name__ == '__main__':
    run_complete_preprocessing()
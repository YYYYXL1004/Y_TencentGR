"""
将 HuggingFace 开源的 Parquet 格式数据转换为比赛平台所需的 JSONL/pkl/json 格式。

使用方法:
    python scripts/convert_hf_to_competition.py [--data_dir ./data] [--skip_mm_emb]
"""

import argparse
import os
import sys
import pickle
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import orjson
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ====================== 工具函数 ======================

def _format_size(size_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _clean_val(val):
    """将 numpy 类型转为 Python 原生类型，跳过 NaN/None。"""
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return [_clean_val(v) for v in val]
    if isinstance(val, list):
        return [_clean_val(v) for v in val]
    return val


def _get_parquet_files(directory: Path) -> list:
    """获取目录下所有 parquet 文件，排序返回。"""
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"未找到 parquet 文件: {directory}")
    return files


def _find_column(df: pd.DataFrame, candidates: list):
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ====================== Step 1: item_feat ======================

def convert_item_feat(data_dir: Path, output_dir: Path) -> dict:
    """item_feat parquet → item_feat_dict.json (流式逐文件处理)"""
    print("\n" + "=" * 50)
    print("📦 [1/6] 转换 item_feat → item_feat_dict.json")
    print("=" * 50)

    pq_files = _get_parquet_files(data_dir / "item_feat")
    print(f"  共 {len(pq_files)} 个 parquet 文件")

    item_feat_dict = {}
    total_rows = 0

    for fi, pf in enumerate(pq_files):
        df = pd.read_parquet(pf)
        feat_cols = [c for c in df.columns if c != "item_id"]
        n = len(df)
        total_rows += n

        item_ids = df["item_id"].values
        feat_arrays = {col: df[col].values for col in feat_cols}

        for i in tqdm(range(n), desc=f"  文件 {fi+1}/{len(pq_files)} ({pf.name})",
                      mininterval=1.0):
            iid = str(int(item_ids[i]))
            feat = {}
            for col in feat_cols:
                v = _clean_val(feat_arrays[col][i])
                if v is not None:
                    feat[col] = v
            item_feat_dict[iid] = feat

        del df  # 释放内存

    output_path = output_dir / "item_feat_dict.json"
    print(f"  💾 写入 {output_path.name} ({total_rows:,} 行)...")
    with open(output_path, 'w') as f:
        json.dump(item_feat_dict, f, ensure_ascii=False)
    size = output_path.stat().st_size
    print(f"  ✅ 完成: {len(item_feat_dict):,} 个物品, 文件大小 {_format_size(size)}")
    return item_feat_dict


# ====================== Step 2: user_feat ======================

def load_user_feat(data_dir: Path) -> dict:
    """user_feat parquet → {user_id(int): {feat_id: val}} (流式逐文件处理)"""
    print("\n" + "=" * 50)
    print("📦 [2/6] 加载 user_feat")
    print("=" * 50)

    pq_files = _get_parquet_files(data_dir / "user_feat")
    print(f"  共 {len(pq_files)} 个 parquet 文件")

    user_feat_dict = {}

    for fi, pf in enumerate(pq_files):
        df = pd.read_parquet(pf)
        feat_cols = [c for c in df.columns if c != "user_id"]
        n = len(df)

        user_ids = df["user_id"].values
        feat_arrays = {col: df[col].values for col in feat_cols}

        for i in tqdm(range(n), desc=f"  文件 {fi+1}/{len(pq_files)} ({pf.name})",
                      mininterval=1.0):
            uid = int(user_ids[i])
            feat = {}
            for col in feat_cols:
                v = _clean_val(feat_arrays[col][i])
                if v is not None:
                    feat[col] = v
            user_feat_dict[uid] = feat

        del df

    print(f"  ✅ 完成: {len(user_feat_dict):,} 个用户特征")
    return user_feat_dict


# ====================== Step 3: seq ======================

def convert_seq(data_dir: Path, output_dir: Path, item_feat_dict: dict,
                user_feat_dict: dict, eval_ratio: float = 0.01):
    """
    seq parquet → seq.jsonl + predict_seq.jsonl (流式逐文件处理)

    比赛格式 (每行一个用户的 JSON list):
      - 物品记录: [uid, iid, None, {item_feat}, action_type, timestamp]
      - 末尾用户记录: [uid, None, {user_feat}, None, None, last_ts]
    """
    print("\n" + "=" * 50)
    print(f"📦 [3/6] 转换 seq → seq.jsonl + predict_seq.jsonl")
    print("=" * 50)

    pq_files = _get_parquet_files(data_dir / "seq")
    print(f"  共 {len(pq_files)} 个 parquet 文件, eval_ratio={eval_ratio}")

    # 先统计总行数以确定 eval 抽样
    row_counts = []
    for pf in pq_files:
        pf_meta = pd.read_parquet(pf, columns=["user_id"])
        row_counts.append(len(pf_meta))
    n_total = sum(row_counts)
    n_eval = max(1, int(n_total * eval_ratio))
    np.random.seed(42)
    eval_indices = set(np.random.choice(n_total, size=n_eval, replace=False).tolist())
    print(f"  总用户数: {n_total:,}, 评测集: {n_eval:,}")

    seq_path = output_dir / "seq.jsonl"
    predict_seq_path = output_dir / "predict_seq.jsonl"
    train_count = 0
    eval_count = 0
    global_idx = 0

    with open(seq_path, 'wb') as f_train, open(predict_seq_path, 'wb') as f_eval:
        for fi, pf in enumerate(pq_files):
            df = pd.read_parquet(pf)
            n = len(df)
            user_ids = df["user_id"].values
            seqs = df["seq"].values

            pbar = tqdm(range(n),
                        desc=f"  文件 {fi+1}/{len(pq_files)} ({pf.name})",
                        mininterval=0.5)
            for i in pbar:
                try:
                    user_id = int(user_ids[i])
                    seq_data = seqs[i]

                    if seq_data is None or len(seq_data) == 0:
                        global_idx += 1
                        continue

                    u_feat = user_feat_dict.get(user_id, {})
                    seq_records = []
                    last_ts = None

                    for item_rec in seq_data:
                        iid = int(item_rec['item_id'])
                        at_raw = item_rec.get('action_type')
                        action_type = int(at_raw) if at_raw is not None else None
                        ts_raw = item_rec.get('timestamp')
                        timestamp = int(ts_raw) if ts_raw is not None else None
                        if timestamp is not None:
                            last_ts = timestamp
                        i_feat = item_feat_dict.get(str(iid))
                        seq_records.append([user_id, iid, None,
                                            i_feat if i_feat else None,
                                            action_type, timestamp])

                    # 末尾用户画像记录
                    seq_records.append([user_id, None,
                                        u_feat if u_feat else None,
                                        None, None, last_ts])

                    line = orjson.dumps(seq_records) + b'\n'
                    f_train.write(line)
                    train_count += 1

                    # 评测集
                    if global_idx in eval_indices:
                        last_click_idx = -1
                        for ri in range(len(seq_records) - 2, -1, -1):
                            if seq_records[ri][4] == 1:
                                last_click_idx = ri
                                break
                        if last_click_idx > 0:
                            predict_records = seq_records[:last_click_idx] + [seq_records[-1]]
                            f_eval.write(orjson.dumps(predict_records) + b'\n')
                            eval_count += 1

                    global_idx += 1

                except Exception as e:
                    print(f"\n  ❌ 第 {global_idx} 行转换失败: {e}")
                    traceback.print_exc()
                    sys.exit(1)

            pbar.close()
            del df

    print(f"  ✅ seq.jsonl: {train_count:,} 个用户")
    print(f"  ✅ predict_seq.jsonl: {eval_count:,} 个用户 (评测集)")


# ====================== Step 4: candidate ======================

def convert_candidate(data_dir: Path, output_dir: Path, item_feat_dict: dict, indexer: dict):
    """candidate parquet → predict_set.jsonl"""
    print("\n" + "=" * 50)
    print("📦 [4/6] 转换 candidate → predict_set.jsonl")
    print("=" * 50)

    candidate_dir = data_dir / "candidate"
    if not candidate_dir.exists() or not list(candidate_dir.glob("*.parquet")):
        print("  ⚠️ candidate 目录为空或不存在，跳过")
        return

    i_indexer = indexer.get('i', {})
    pq_files = _get_parquet_files(candidate_dir)
    output_path = output_dir / "predict_set.jsonl"
    count = 0

    with open(output_path, 'w') as f:
        for pf in pq_files:
            df = pd.read_parquet(pf)
            cols = df.columns.tolist()
            print(f"  候选集列名: {cols}")

            for i in tqdm(range(len(df)), desc=f"  生成 predict_set", mininterval=1.0):
                row = df.iloc[i]
                oid = int(row.get('item_id', 0)) if 'item_id' in cols else 0
                retrieval_id = int(row.get('retrieval_id', count)) if 'retrieval_id' in cols else count
                rid = i_indexer.get(oid, 0)
                features = item_feat_dict.get(str(rid), {})
                record = {"creative_id": oid, "retrieval_id": retrieval_id, "features": features}
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                count += 1
            del df

    print(f"  ✅ predict_set.jsonl: {count:,} 个候选物品")


# ====================== Step 5: offsets ======================

def generate_offsets(jsonl_path: Path, output_path: Path):
    """为 JSONL 文件生成随机访问偏移表。"""
    offsets = []
    with open(jsonl_path, 'rb') as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if line.strip():
                offsets.append(offset)
    with open(output_path, 'wb') as f:
        pickle.dump(offsets, f)
    print(f"  ✅ {output_path.name}: {len(offsets):,} 条偏移")


# ====================== Step 6: mm_emb ======================

def convert_mm_emb(data_dir: Path, output_dir: Path, indexer: dict):
    """mm_emb parquet → creative_emb/ (pkl/json)"""
    print("\n" + "=" * 50)
    print("📦 [6/6] 转换 mm_emb → creative_emb/")
    print("=" * 50)

    mm_emb_dir = data_dir / "mm_emb"
    creative_emb_dir = output_dir / "creative_emb"
    creative_emb_dir.mkdir(parents=True, exist_ok=True)

    if not mm_emb_dir.exists():
        print("  ⚠️ mm_emb 目录不存在，跳过")
        return

    for emb_dir in sorted(mm_emb_dir.iterdir()):
        if not emb_dir.is_dir() or not emb_dir.name.endswith("_parquet"):
            continue

        parts = emb_dir.name.replace("_parquet", "").split("_")
        if len(parts) < 3:
            continue
        feat_id, dim = parts[1], int(parts[2])

        parquet_files = sorted(emb_dir.glob("*.parquet"))
        if not parquet_files:
            print(f"  ⚠️ {emb_dir.name} 为空，跳过")
            continue

        print(f"  🔄 emb_{feat_id}_{dim} ({len(parquet_files)} 个文件)...")

        if feat_id == "81":
            emb_dict = {}
            for pf in tqdm(parquet_files, desc=f"  emb_{feat_id}"):
                df = pd.read_parquet(pf)
                id_col = _find_column(df, ['anonymous_cid', 'item_id', 'cid'])
                emb_col = _find_column(df, ['emb', 'embedding', 'vector'])
                if id_col is None or emb_col is None:
                    print(f"    ⚠️ 列名不匹配: {df.columns.tolist()}")
                    continue
                for idx in range(len(df)):
                    emb_dict[str(df[id_col].iloc[idx])] = np.array(df[emb_col].iloc[idx], dtype=np.float32)
            out_path = creative_emb_dir / f"emb_{feat_id}_{dim}.pkl"
            with open(out_path, 'wb') as f:
                pickle.dump(emb_dict, f)
            print(f"    ✅ {out_path.name}: {len(emb_dict):,} 条嵌入")
        else:
            out_subdir = creative_emb_dir / f"emb_{feat_id}_{dim}"
            out_subdir.mkdir(parents=True, exist_ok=True)
            total_count = 0
            for pi, pf in enumerate(tqdm(parquet_files, desc=f"  emb_{feat_id}")):
                df = pd.read_parquet(pf)
                id_col = _find_column(df, ['anonymous_cid', 'item_id', 'cid'])
                emb_col = _find_column(df, ['emb', 'embedding', 'vector'])
                if id_col is None or emb_col is None:
                    print(f"    ⚠️ 列名不匹配: {df.columns.tolist()}")
                    continue
                out_json = out_subdir / f"part_{pi:04d}.json"
                with open(out_json, 'w') as f:
                    for idx in range(len(df)):
                        oid = str(df[id_col].iloc[idx])
                        emb = df[emb_col].iloc[idx]
                        if isinstance(emb, np.ndarray):
                            emb = emb.tolist()
                        f.write(json.dumps({"anonymous_cid": oid, "emb": emb}, ensure_ascii=False) + '\n')
                        total_count += 1
            print(f"    ✅ emb_{feat_id}_{dim}/: {total_count:,} 条嵌入")


# ====================== main ======================

def main():
    parser = argparse.ArgumentParser(description="将 HF Parquet 数据转换为比赛格式")
    parser.add_argument("--data_dir", default=str(PROJECT_ROOT / "data"), help="HF 数据目录")
    parser.add_argument("--output_dir", default=None, help="输出目录 (默认与 data_dir 相同)")
    parser.add_argument("--eval_ratio", default=0.01, type=float, help="评测集比例")
    parser.add_argument("--skip_mm_emb", action="store_true", help="跳过多模态嵌入转换")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  HuggingFace Parquet → 比赛格式 转换工具")
    print("=" * 60)
    print(f"  数据目录: {data_dir}")
    print(f"  输出目录: {output_dir}")

    # 检查必要目录
    required_dirs = ["seq", "item_feat", "user_feat"]
    missing = [d for d in required_dirs
               if not (data_dir / d).exists() or not list((data_dir / d).glob("*.parquet"))]
    if missing:
        print(f"\n❌ 缺少必要数据目录: {missing}")
        print("   请先运行: python scripts/download_hf_data.py")
        sys.exit(1)

    indexer_path = data_dir / "indexer.pkl"
    if not indexer_path.exists():
        print(f"\n❌ 缺少 indexer.pkl")
        sys.exit(1)
    with open(indexer_path, 'rb') as f:
        indexer = pickle.load(f)
    print(f"\n📊 indexer: {len(indexer.get('i', {})):,} items, "
          f"{len(indexer.get('u', {})):,} users")

    try:
        item_feat_dict = convert_item_feat(data_dir, output_dir)
        user_feat_dict = load_user_feat(data_dir)
        convert_seq(data_dir, output_dir, item_feat_dict, user_feat_dict, args.eval_ratio)
        convert_candidate(data_dir, output_dir, item_feat_dict, indexer)

        print("\n" + "=" * 50)
        print("📦 [5/6] 生成偏移文件")
        print("=" * 50)
        for name in ["seq.jsonl", "predict_seq.jsonl"]:
            p = output_dir / name
            if p.exists():
                generate_offsets(p, output_dir / name.replace(".jsonl", "_offsets.pkl"))

        if args.skip_mm_emb:
            print("\n⏭️  [6/6] 跳过多模态嵌入转换 (--skip_mm_emb)")
        else:
            convert_mm_emb(data_dir, output_dir, indexer)

    except Exception as e:
        print(f"\n❌ 转换失败: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 完成
    print("\n" + "=" * 60)
    print("  ✅ 全部转换完成！")
    print("=" * 60)
    print(f"\n生成的文件:")
    for f in sorted(output_dir.iterdir()):
        if f.is_file() and not f.name.startswith('.') and f.suffix in ('.jsonl', '.json', '.pkl'):
            print(f"  📄 {f.name}  ({_format_size(f.stat().st_size)})")
        elif f.is_dir() and f.name == "creative_emb":
            cnt = sum(1 for _ in f.rglob('*') if _.is_file())
            print(f"  📁 {f.name}/  ({cnt} 个文件)")

    print(f"\n下一步:")
    print(f"  bash run_local.sh preprocess   # 生成 Alias 采样表")
    print(f"  bash run_local.sh train        # 训练")


if __name__ == "__main__":
    main()

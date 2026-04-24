"""
为 seq.jsonl 和 predict_seq.jsonl 生成随机访问偏移文件 (seq_offsets.pkl, predict_seq_offsets.pkl)。

代码中的 MyDataset 和 MyTestDataset 使用 file.seek(offset) 来按行随机读取 JSONL 文件，
因此需要预先计算每一行在文件中的字节偏移量。

使用方法:
    python scripts/generate_offsets.py [--data_dir ./data]
"""

import argparse
import os
import pickle
from pathlib import Path
from tqdm import tqdm


def generate_offsets(jsonl_path, output_path):
    """
    遍历 JSONL 文件，记录每一行的起始字节偏移量，并保存为 pickle 文件。
    
    Args:
        jsonl_path: JSONL 文件路径
        output_path: 输出的 offsets pickle 文件路径
    """
    offsets = []
    
    with open(jsonl_path, 'rb') as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            # 只记录非空行的偏移
            if line.strip():
                offsets.append(offset)
    
    with open(output_path, 'wb') as f:
        pickle.dump(offsets, f)
    
    print(f"  ✅ 生成 {len(offsets)} 条偏移量 -> {output_path}")
    return len(offsets)


def main():
    parser = argparse.ArgumentParser(description="生成 JSONL 文件的随机访问偏移表")
    parser.add_argument(
        "--data_dir",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
        help="数据目录 (默认: ./data)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # 生成训练序列偏移
    seq_file = data_dir / "seq.jsonl"
    if seq_file.exists():
        print(f"📄 处理训练序列: {seq_file}")
        generate_offsets(seq_file, data_dir / "seq_offsets.pkl")
    else:
        print(f"⚠️  未找到 {seq_file}，跳过")

    # 生成预测序列偏移
    predict_seq_file = data_dir / "predict_seq.jsonl"
    if predict_seq_file.exists():
        print(f"📄 处理预测序列: {predict_seq_file}")
        generate_offsets(predict_seq_file, data_dir / "predict_seq_offsets.pkl")
    else:
        print(f"⚠️  未找到 {predict_seq_file}，跳过")

    print("\n🎉 偏移文件生成完成！")


if __name__ == "__main__":
    main()

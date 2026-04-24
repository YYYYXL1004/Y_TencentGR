"""
从 HuggingFace 下载 TAAC2025 TencentGR-1M 数据集到本地 ./data 目录。

使用方法:
    pip install huggingface_hub
    python scripts/download_hf_data.py [--repo TAAC2025/TencentGR-1M] [--local_dir ./data]

说明:
    HuggingFace 数据集地址: https://huggingface.co/datasets/TAAC2025/TencentGR-1M
    
    内网用户: 默认使用 hf-mirror.com 镜像，也可通过 --mirror 指定其他镜像站。
    可选镜像:
      - https://hf-mirror.com          (推荐，国内速度快)
      - https://huggingface.sukaka.top
    
    下载后会包含 seq.jsonl, indexer.pkl, item_feat_dict.json, predict_seq.jsonl, 
    predict_set.jsonl, creative_emb/ 等文件。
    
    下载完成后，运行 scripts/generate_offsets.py 生成随机访问偏移文件。
"""

import argparse
import os
import sys
from pathlib import Path


# ========================= 镜像配置 =========================
MIRROR_URLS = {
    "hf-mirror":  "https://hf-mirror.com",
    "sukaka":     "https://huggingface.sukaka.top",
    "official":   "https://huggingface.co",
}
DEFAULT_MIRROR = "hf-mirror"


def setup_mirror(mirror_key_or_url: str):
    """设置 HuggingFace 镜像端点。"""
    if mirror_key_or_url in MIRROR_URLS:
        url = MIRROR_URLS[mirror_key_or_url]
    else:
        url = mirror_key_or_url  # 用户直接传入完整 URL

    os.environ["HF_ENDPOINT"] = url
    print(f"🌐 HuggingFace 镜像: {url}")


def main():
    parser = argparse.ArgumentParser(description="下载 TAAC2025 TencentGR-1M 数据集")
    parser.add_argument(
        "--repo", 
        default="TAAC2025/TencentGR-1M", 
        help="HuggingFace 数据集仓库名"
    )
    parser.add_argument(
        "--local_dir", 
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
        help="本地保存目录 (默认: ./data)"
    )
    parser.add_argument(
        "--mirror",
        default=DEFAULT_MIRROR,
        help=f"镜像源: {list(MIRROR_URLS.keys())} 或自定义URL (默认: {DEFAULT_MIRROR})"
    )
    parser.add_argument(
        "--emb_ids",
        nargs='+',
        default=['82'],
        help="要下载的多模态嵌入ID (默认: 82)。可选: 81 82 83 84 85 86，或 'all' 下载全部，'none' 跳过所有嵌入"
    )
    parser.add_argument(
        "--token", 
        default=None, 
        help="HuggingFace token (如果数据集需要认证)"
    )
    args = parser.parse_args()

    # 构建 allow_patterns: 始终下载核心文件，按需下载 mm_emb
    allow_patterns = [
        "seq/**",
        "user_feat/**",
        "item_feat/**",
        "candidate/**",
        "indexer.pkl",
        "README.md",
        ".gitattributes",
    ]
    if args.emb_ids == ['all']:
        allow_patterns.append("mm_emb/**")
        print(f"📥 多模态嵌入: 全部下载 (~128GB)")
    elif args.emb_ids == ['none']:
        print(f"📥 多模态嵌入: 跳过")
    else:
        EMB_DIM = {'81': 32, '82': 1024, '83': 3584, '84': 4096, '85': 3584, '86': 3584}
        EMB_SIZE = {'81': '~901MB', '82': '~9.4GB', '83': '~31GB', '84': '~30GB', '85': '~31GB', '86': '~26GB'}
        for eid in args.emb_ids:
            if eid in EMB_DIM:
                allow_patterns.append(f"mm_emb/emb_{eid}_{EMB_DIM[eid]}_parquet/**")
                print(f"📥 多模态嵌入 emb_{eid} (dim={EMB_DIM[eid]}, {EMB_SIZE[eid]})")
            else:
                print(f"⚠️ 未知嵌入ID: {eid}，跳过")

    # 配置镜像 (必须在 import huggingface_hub 之前设置环境变量)
    if args.mirror != "official":
        setup_mirror(args.mirror)

    try:
        from huggingface_hub import snapshot_download, list_repo_tree
    except ImportError:
        print("请先安装 huggingface_hub:")
        print("  pip install huggingface_hub")
        return

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 数据集: {args.repo}")
    print(f"📂 保存到: {local_dir.resolve()}")

    # 先列出仓库文件，展示即将下载的内容
    print("\n📋 正在获取文件列表...")
    try:
        from fnmatch import fnmatch
        files = list(list_repo_tree(args.repo, repo_type="dataset", token=args.token))
        total_size = 0
        file_count = 0
        for f in files:
            if hasattr(f, 'size') and f.size is not None:
                # 检查是否在 allow_patterns 中
                matched = any(fnmatch(f.rfilename, pat) for pat in allow_patterns)
                marker = "✅" if matched else "⏭️"
                size_str = _format_size(f.size)
                if matched:
                    total_size += f.size
                    file_count += 1
                print(f"  {marker} {f.rfilename}  ({size_str})")
        
        if total_size > 0:
            print(f"\n  将下载 {file_count} 个文件，总大小约 {_format_size(total_size)}")
    except Exception as e:
        print(f"  ⚠️ 无法获取文件列表: {e}，将直接开始下载")

    print("\n🚀 开始下载 (支持断点续传，Ctrl+C 暂停后重跑即可恢复)...")
    
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=str(local_dir),
        token=args.token,
        allow_patterns=allow_patterns,
    )

    print(f"\n✅ 数据集下载完成！文件保存在: {local_dir.resolve()}")
    
    # 展示下载结果
    print("\n� 下载的文件:")
    _list_files(local_dir, indent=2)
    
    print("\n�� 下一步: 运行 python scripts/generate_offsets.py 生成偏移文件")


def _format_size(size_bytes: int) -> str:
    """将字节数格式化为人类可读的大小。"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def _list_files(path: Path, indent: int = 0, max_depth: int = 2, current_depth: int = 0):
    """递归列出目录内容。"""
    if current_depth >= max_depth:
        return
    prefix = " " * indent
    try:
        items = sorted(path.iterdir())
        for item in items:
            if item.name.startswith('.'):
                continue
            if item.is_file():
                size_str = _format_size(item.stat().st_size)
                print(f"{prefix}📄 {item.name}  ({size_str})")
            elif item.is_dir():
                count = sum(1 for _ in item.rglob('*') if _.is_file())
                print(f"{prefix}📁 {item.name}/  ({count} 个文件)")
                _list_files(item, indent + 2, max_depth, current_depth + 1)
    except PermissionError:
        pass


if __name__ == "__main__":
    main()

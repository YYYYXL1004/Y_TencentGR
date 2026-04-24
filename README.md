# 腾讯广告算法大赛 2025 - 全模态生成式推荐

基于 Transformer 的序列推荐模型，在官方 Baseline 基础上进行了多项深度优化。初赛前 10%。

- **官方数据集**: [TAAC2025/TencentGR-1M](https://huggingface.co/datasets/TAAC2025/TencentGR-1M) (HuggingFace, ~137GB)
- **官方 Baseline**: [baseline_2025](https://github.com/TencentAdvertisingAlgorithmCompetition/baseline_2025)

## 核心改进点 (对比 Baseline)

### 模型架构
| 模块 | Baseline | Ours |
|------|----------|------|
| 归一化 | LayerNorm | **RMSNorm** (效率↑10-15%) |
| 位置编码 | 绝对位置 | **RoPE** 旋转位置编码 |
| 注意力 | 标准 Q/K/V | **ActionGate** + **TimeDelta-ATTN** |
| FFN | GELU | 可选 **SwiGLU** |
| 特征融合 | Linear + ReLU | **MLPBlock** (4H + 残差 + RMSNorm) |

### 损失函数
- **Baseline**: BCEWithLogitsLoss (独立正负样本)
- **Ours**: **InfoNCE 对比学习** (批内负采样, τ=0.03, 点击权重=2.5)

### 训练策略
| 项目 | Baseline | Ours |
|------|----------|------|
| 优化器 | Adam | **SparseAdam** + **AdamW** |
| 学习率 | 固定 | **Cosine Annealing** + Warmup |
| 精度 | FP32 | **BF16 混合精度** |
| 其他 | - | 梯度累积、梯度检查点 |

### 数据处理
- **负采样**: 均匀随机 → **CTR 感知 Alias Method**
- **时间特征**: 新增小时/星期/月份/时间差/衰减共 5 个特征
- **数据加载**: json → **orjson** (解析速度↑2-3x)

### 推理优化
- **ANN 检索**: 外部 faiss → **PyTorch 原生** Top-K
- **结果融合**: 支持 **RRF** (Reciprocal Rank Fusion)

---

## 项目结构

```
Y_TencentGR/
├── model.py          # 模型 (RMSNorm、ActionGate、RoPE、TimeDelta-ATTN)
├── dataset.py        # 数据加载 (AliasMethod 负采样、时间特征、批字典)
├── main.py           # 训练 (双优化器、混合精度、梯度累积)
├── infer.py          # 推理 (PyTorch ANN Top-K、RRF 融合)
├── run.sh            # 比赛平台运行脚本
├── run_local.sh      # 本地一键复现脚本 (download/convert/preprocess/train/infer)
├── requirements.txt  # Python 依赖
├── scripts/
│   ├── download_hf_data.py          # HuggingFace 数据下载 (镜像加速, 断点续传)
│   ├── convert_hf_to_competition.py # Parquet → 比赛 JSONL/pkl/json 格式转换
│   └── generate_offsets.py          # JSONL 随机访问偏移文件生成
├── util/
│   └── preprocess_alias.py          # CTR 感知 Alias Method 负采样表生成
├── data/             # 正式 1M 数据集 (HF 下载 + 转换生成, gitignored)
├── data_demo/        # 小规模 Demo 数据 (用于快速验证)
└── docs/
    └── 平台规范.md
```

---

## 快速开始

### 1. 环境准备

```bash
conda create -n TAAC python=3.10
conda activate TAAC
pip install -r requirements.txt
```

### 2. 本地复现 (HuggingFace 1M 数据集)

```bash
# 下载数据集 (默认 hf-mirror.com 镜像, 默认只下载 emb_82)
bash run_local.sh download
# 可选: --emb_ids 81 82  下载多个模态
# 可选: --emb_ids none    不下载嵌入

# Parquet → 比赛格式转换
bash run_local.sh convert --skip_mm_emb  # 跳过多模态嵌入, 快速跑通
# 或: bash run_local.sh convert          # 完整转换 (含 mm_emb)

# 预处理 (生成 Alias 采样表)
bash run_local.sh preprocess

# 训练
bash run_local.sh train

# 推理
bash run_local.sh infer
```

### 3. Demo 数据快速验证

`data_demo/` 包含小规模样例数据，适合验证代码能否跑通：

```bash
bash run_local.sh preprocess --demo
bash run_local.sh train --demo
```

### 4. 比赛平台运行

直接提交代码，平台自动设置环境变量 (`TRAIN_DATA_PATH`, `USER_CACHE_PATH`, `TRAIN_CKPT_PATH`, `EVAL_DATA_PATH` 等)：

```bash
bash run.sh
```

---

## 数据格式说明

HuggingFace 开源的是 **Parquet** 格式，比赛代码需要 **JSONL/pkl/json** 格式。`convert_hf_to_competition.py` 负责转换:

| HF Parquet | → | 比赛格式 | 大小 (1M) |
|---|---|---|---|
| `seq/*.parquet` | → | `seq.jsonl` + `predict_seq.jsonl` | ~16 GB |
| `item_feat/*.parquet` | → | `item_feat_dict.json` | ~831 MB |
| `user_feat/*.parquet` | → | (嵌入到 seq.jsonl 末尾记录) | - |
| `candidate/*.parquet` | → | `predict_set.jsonl` | ~126 MB |
| `mm_emb/emb_*_parquet/` | → | `creative_emb/` | 按需 |
| `indexer.pkl` | → | (直接使用) | 142 MB |

---

## 关键参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| hidden_units | 128 | 隐藏层维度 |
| num_blocks | 8 | Transformer 层数 |
| num_heads | 8 | 注意力头数 |
| batch_size | 128 | 批大小 |
| lr | 0.001 | 学习率 |
| tau | 0.03 | InfoNCE 温度参数 |
| click_weight | 2.5 | 点击行为损失权重 |
| mm_emb_id | 82 | 多模态嵌入 ID (dim=1024) |

---

## 依赖
- Python 3.8+
- PyTorch 2.0+
- orjson, numpy, pandas, pyarrow, tqdm, huggingface_hub

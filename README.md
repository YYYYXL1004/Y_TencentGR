# 腾讯算法大赛 

基于 Transformer 的序列推荐模型，在官方 Baseline 基础上进行了多项深度优化。

## 核心改进点 (对比 Baseline)

### 1. 模型架构
| 模块 | Baseline | SOTA |
|------|----------|------|
| 归一化 | LayerNorm | **RMSNorm** (效率↑10-15%) |
| 位置编码 | 绝对位置 | **RoPE** 旋转位置编码 |
| 注意力 | 标准 Q/K/V | **ActionGate** + **TimeDelta-ATTN** |
| FFN | GELU | 可选 **SwiGLU** |
| 特征融合 | Linear + ReLU | **MLPBlock** (4H + 残差 + RMSNorm) |

### 2. 损失函数
- **Baseline**: BCEWithLogitsLoss (独立正负样本)
- **SOTA**: **InfoNCE 对比学习**
  - 批内负采样 → 负样本数量从 1 扩展到 B×L
  - 温度参数 τ = 0.03
  - 点击行为权重 = 2.5

### 3. 训练策略
| 项目 | Baseline | SOTA |
|------|----------|------|
| 优化器 | Adam | **SparseAdam** + **AdamW** |
| 学习率 | 固定 | **Cosine Annealing** + Warmup |
| 精度 | FP32 | **BF16 混合精度** |
| 其他 | - | 梯度累积、梯度检查点 |

### 4. 数据处理
- **负采样**: 均匀随机 → **CTR感知 Alias Method**
- **时间特征**: 新增小时/星期/月份/时间差/衰减共 5 个特征
- **数据加载**: json → **orjson** (解析速度↑2-3x)

### 5. 推理优化
- **ANN 检索**: 外部 faiss_demo → **PyTorch 原生** Top-K
- **结果融合**: 支持 **RRF** (Reciprocal Rank Fusion)

---

## 项目结构
```
Y_TencentGR/
├── dataset.py      # 数据处理 (AliasMethod、时间特征、批字典)
├── model.py        # 模型 (RMSNorm、ActionGate、TimeDelta-ATTN)
├── main.py         # 训练 (双优化器、混合精度、梯度累积)
├── infer.py        # 推理 (PyTorch ANN、RRF 融合)
├── run.sh          # 运行脚本
├── data/           # Demo 数据集
│   ├── README.md          # 数据格式说明
│   ├── seq.jsonl          # 用户行为序列
│   ├── indexer.pkl        # ID 映射表
│   ├── item_feat_dict.json # 物品特征
│   ├── seq_offsets.pkl    # 随机访问偏移
│   ├── predict_seq.jsonl  # 预测序列
│   ├── predict_set.jsonl  # 预测集
│   └── creative_emb/      # 多模态特征
├── util/           # 预处理工具
│   └── preprocess_alias.py  # CTR感知负采样表生成
└── docs/平台规范.md   # 官方平台规范
```

---

## 运行指南

### 平台运行 (腾讯比赛环境)
直接提交代码，平台自动设置环境变量：
- `TRAIN_DATA_PATH`: 训练数据路径
- `USER_CACHE_PATH`: 缓存路径 (存放 alias_tables.npz)
- `TRAIN_CKPT_PATH`: 模型保存路径
- `EVAL_DATA_PATH`: 评测数据路径

```bash
bash run.sh
```

### 本地运行
需手动设置环境变量：
```bash
export TRAIN_DATA_PATH=./data
export USER_CACHE_PATH=./cache
export TRAIN_CKPT_PATH=./ckpt
export TRAIN_LOG_PATH=./log
export TRAIN_TF_EVENTS_PATH=./log/tf_events

# 1. 预处理：生成负采样表
cd util && python preprocess_alias.py

# 2. 训练
python main.py --amp_bf16 --use_gradient_checkpointing

# 3. 推理
python infer.py
```

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

---

## 依赖
- Python 3.8+
- PyTorch 2.0+
- orjson, numpy, pandas, tqdm

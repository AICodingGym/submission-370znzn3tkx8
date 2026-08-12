# AES 2.0 MLE-Bench Strategy

## 1. Original Kaggle Top Solutions — Public Recipe

| Rank | Team | Core Approach |
|------|------|--------------|
| 1st | ferdinandlimburg | 4× DeBERTa V3 Large ensemble, two-stage training (+0.015), pseudo labeling (+0.005), per-model threshold optimization |
| 2nd | syhens | 5× DeBERTa V3 Large, two-stage training, ordinal regression, contextual positional encoding, hard voting |
| 3rd | dsohonosom | 7× DeBERTa V3 Base + 4× Large, two-stage training, Nelder-Mead weight optimization |
| 4th | tascj0 | Data source classification head, DeBERTa large + v3-large + Qwen2-1.5B-Instruct |
| 5th | GPU From onethingai | Two-stage training, DeBERTa V3 Small/Base/Large blend |

**公共配方：**
- Backbone: DeBERTa V3 (Large preferred, Base也能用)
- Head: 线性层 -> 6分类或回归，attention/mean pooling
- Loss: CrossEntropy 或 MSE + QWK style loss
- CV: Group K-Fold by prompt
- Special tokens: 添加 `\n\n` 和双空格 token
- Threshold optimization: OOF 上优化 5 个阈值最大化 QWK
- Two-stage training: Persuade 2.0 pretrain -> Kaggle Only finetune

## 2. MLE-Bench Split Analysis — CRITICAL

**MLE-Bench prepare.py 切分逻辑：**
```python
new_train, answers = train_test_split(old_train, test_size=0.1, random_state=0)
```
- 从原始 train.csv (17307行) 随机抽取 10% (1731行) 作为 test
- `random_state=0`, 纯随机，不分组，不分层
- **Test 和 Train 是同分布的**（都是 Persuade 2.0 + Kaggle Only 混合）

**与原始 Kaggle 比赛的关键差异：**
| 维度 | 原比赛 | MLE-Bench |
|------|--------|-----------|
| Test 来源 | 纯 Kaggle Only 新数据 (~8k) | 原 train.csv 随机 10% |
| 分布偏移 | **有** (Persuade 2.0 vs Kaggle Only 评分标准不同) | **无** (完全同分布) |
| 外部 Persuade 2.0 数据 | 可以安全使用 | **不能用**（会和 test 集重叠） |

**冠军技术取舍论证：**

| 技术 | 原比赛价值 | MLE-Bench 价值 | 判断 |
|------|-----------|---------------|------|
| Two-stage training | +0.015 QWK | 有害（限制模型学习 Persuade 2.0 模式，而这些恰好在 test 中） | **丢弃** |
| Pseudo labeling | +0.005 QWK | 无用（没有评分标准差异需要对齐） | **丢弃** |
| Source classification head | 有用 | 无用 | **丢弃** |
| DeBERTa V3 backbone | 核心 | 核心 | **保留** |
| Special tokens (\n\n, double-space) | 有用 | 有用 | **保留** |
| Attention/mean pooling | 有用 | 有用 | **保留** |
| QWK threshold optimization | 核心 | 核心 | **保留** |
| Stratified K-Fold CV | 有用 | 有用 | **保留** |
| Ordinal regression/classification | 有用 | 有用 | **保留** |
| Multi-seed ensemble | 有用 | 有用 | **保留** |

## 3. Data Leakage — Forbidden Zones

| 禁区 | 原因 | 判定 |
|------|------|------|
| 外部 Persuade 2.0 公开数据集 | MLE-Bench test 包含原 train.csv 中的 Persuade 2.0 文章，外部下载会有重叠 | **禁止** |
| 比赛期间公开的预训练权重 | 这些权重可能在完整 train.csv 上训练过（含 test 集文章） | **禁止** |
| Kaggle 公开 notebook 的 OOF 预测 | 可能用了包含 test 的完整数据 | **禁止** |
| GPT-4 等生成的数据增强 | 如用原始文章做 prompt 生成，属于泄露 | **禁止** |
| HuggingFace 上针对此比赛的 finetuned 模型 | 训练数据不透明，可能包含 test 文章 | **禁止** |
| 官方预训练模型 (DeBERTa V3 等) | 通用预训练语料，不含此数据集 | **安全** |
| 自己提取的 NLP 特征 (TF-IDF, 可读性分数等) | 从训练文本中提取，无外部泄露 | **安全** |
| 本脚本的 OOF 预测 | 正确 5-fold CV，每个 test fold 只用了对应 train fold | **安全** |

## 4. Our Strategy for MLE-Bench

### Architecture
- Backbone: `microsoft/deberta-v3-base` (主力) / `microsoft/deberta-v3-large` (可选增强)
- Pooling: Mean pooling over last hidden states
- Head: Linear(hidden_size, 6) classification
- Special tokens: `[NEWLINE]` token for `\n\n`

### Training
- Loss: CrossEntropyLoss (6-class classification, 天然处理序数结构)
- Optimizer: AdamW, lr=2e-5, cosine schedule with 10% warmup
- Max length: 1024 tokens
- Batch size: 8 (base) with gradient accumulation
- Epochs: 4
- CV: 5-fold StratifiedKFold (stratify by score)
- Mixed precision (AMP)

### Post-processing
- Softmax probabilities -> expected score: E[score] = Σ(p_i × i)
- OOF predictions -> optimize 5 thresholds to maximize QWK
- Threshold optimization via scipy.optimize.minimize (Nelder-Mead)
- Test predictions -> apply same thresholds

### Ensemble
- 3-5 models with different seeds
- Average probabilities before thresholding
- Final threshold optimization on averaged OOF

### Smoke Test
- DeBERTa V3 Base, 500 samples, 2 folds, 1 epoch
- Quick verification of pipeline end-to-end

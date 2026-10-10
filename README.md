# README

首次运行需联网下载词表。

```bash
uv sync

# TinyStories
uv run hf download \
  roneneldan/TinyStories \
  --repo-type dataset \
  --include "data/*.parquet" \
  --local-dir data/raw/tinystories

# OWT
uv run hf download \
  Skylion007/openwebtext \
  --repo-type dataset \
  --include "plain_text/*.parquet" \
  --local-dir data/raw/openwebtext
```

## 1. 预处理

---
预处理 TinyStories

```bash
uv run preprocessing.py \
  --raw-dir data/raw/tinystories/data \
  --output-dir data/processed/tinystories-gpt2
```

---
预处理 OWT

```bash
uv run preprocessing.py \
  --dataset owt \
  --raw-dir data/raw/openwebtext/plain_text \
  --train-pattern "train-*.parquet" \
  --output-dir data/processed/owt-gpt2 \
  --tokenizer gpt2 \
  --batch-size 1024 \
  --flush-tokens 1000000 \
  --valid-ratio 0.005 \
  --split-seed 42
```

## 2. 预训练

```
uv run train.py --help
```

### 预训练实验

```
# 运行实验
uv run python -m experiments.exp_scaling --scale small

# 分析实验结果
uv run analyze.py --input results --output output
```

实验记录详见 [[experiments/README.md]]。

## 3. 推理


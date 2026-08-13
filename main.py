#!/usr/bin/env python3
"""
Automated Essay Scoring 2.0 — MLE-Bench Pipeline
=================================================
Strategy: DeBERTa V3 ordinal classification + QWK threshold optimization + ensemble.

Key design decisions for MLE-Bench (vs. original Kaggle):
- NO two-stage training (test is random split from train, same distribution)
- NO pseudo labeling (no grading-criteria shift to correct)
- NO source classification head (no Persuade 2.0 vs Kaggle-Only shift)
- YES: DeBERTa V3 backbone, ordinal classification, QWK-optimized thresholds, ensemble

Usage:
  python main.py                  # full training
  python main.py --smoke          # smoke test (small model, few samples)
  python main.py --model large    # use deberta-v3-large
  python main.py --data /path/to/data
"""

import argparse
import gc
import os
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from torch.amp import GradScaler, autocast
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

warnings.filterwarnings("ignore")

# ╔═══════════════════════════════════════════════════════════════╗
# ║  ★ 编辑区 — Kaggle 每次跑之前改这里就行，不用碰命令行         ║
# ╚═══════════════════════════════════════════════════════════════╝
# 预设模式：
#   "smoke" : 冒烟测试，固定 500样本/2fold/1epoch/2seed，~10分钟验证流程
#   "fast"  : base + 关checkpointing，配下面 1seed/3epoch，2xT4 约 40-60min，~0.80
#   "base"  : DeBERTa-v3-base，3 seeds，2xT4 约 2-3h，预期 0.82-0.83
#   "large" : DeBERTa-v3-large，最佳效果，3 seeds，2xT4 约 8-9h，预期 0.84-0.85
EDIT_PRESET = "large"          # ← 改这里切换模式

# 集成 seeds（仅 base/large 生效；smoke 固定 2 个）
# 每个 seed 跑完都会存一份可交的 submission.csv，设 3 个只赚不亏。
EDIT_SEEDS = [42, 3407, 2024]

# 训练超参（仅 base/large 生效；smoke 固定 1 epoch / 2 fold）
EDIT_FOLDS = 5
EDIT_EPOCHS = 4
EDIT_MAX_LENGTH = 1024         # 多数作文<500 token；想加速可改 512
EDIT_LR = 2e-5
EDIT_BATCH_SIZE = None         # None = 预设默认(base=4, large=2)；OOM 时调小
EDIT_NO_AMP = False            # True = 关混合精度（仅 AMP 反复报错时用）
EDIT_HF_OFFLINE = True         # True = 只读本地缓存不联网（模型已下载后设True，避免SSL抖动）
# ╔═══════════════════════════════════════════════════════════════╗
# ║  ★ 编辑区结束，下面一般不用动                                  ║
# ╚═══════════════════════════════════════════════════════════════╝

# ============================================================
# Environment detection — set HF cache BEFORE any model load
# ============================================================

_KAGGLE = os.path.exists("/kaggle/working") and os.path.exists("/kaggle/input")
# All paths below are relative to the current working directory. On Kaggle we
# override to the writable /kaggle/working dir; elsewhere (e.g. a 4090 box at
# /root/wzt) relative paths resolve against the project folder.
_LOG_DIR = "/kaggle/working" if _KAGGLE else "."
_LOG_FILE = os.path.join(_LOG_DIR, "training.log")

# Reduce CUDA memory fragmentation — the correct env var (note: _CUDA_) helps avoid
# OOM with long sequences. Must be set before any CUDA context is created.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# HF cache: relative on local boxes, /kaggle/working on Kaggle.
_HF_CACHE = "/kaggle/working/.cache/huggingface" if _KAGGLE else "./.cache/huggingface"
os.environ["HF_HOME"] = _HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = _HF_CACHE
# If model is already downloaded, force offline so from_pretrained never pings HF
# (avoids flaky SSL on the cache-validity HEAD check). Toggle via EDIT_HF_OFFLINE.
if EDIT_HF_OFFLINE:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print("[config] HF_HUB_OFFLINE=1 (using local cache only, no network)")
# Longer timeout for large model downloads over flaky networks (default 10s is too short).
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
# If huggingface.co is unreachable (e.g. CN networks), point at a mirror:
#   set HF_ENDPOINT=https://hf-mirror.com   (or EDIT_HF_ENDPOINT below)
if os.environ.get("HF_ENDPOINT"):
    print(f"[config] HF_ENDPOINT = {os.environ['HF_ENDPOINT']}")
os.makedirs(_HF_CACHE, exist_ok=True)


# ============================================================
# Logging: tee stdout/stderr to both console and file
# ============================================================

class _TeeWriter:
    """Write to both a file and the original stream. Shares log file handle.
    Delegates all non-overridden attributes to the original stream."""
    _log_fh = None  # class-level shared file handle

    def __init__(self, orig_stream):
        self._orig = orig_stream

    def __getattr__(self, name):
        # Delegate to _orig stream; provide safe defaults for common attrs
        if name == "_orig":
            raise AttributeError(name)
        if hasattr(self._orig, name):
            return getattr(self._orig, name)
        # Safe defaults for attrs that third-party libs call on stdout
        _DEFAULTS = {
            "isatty": lambda: False,
            "encoding": "utf-8",
            "fileno": lambda: -1,
        }
        if name in _DEFAULTS:
            return _DEFAULTS[name]
        raise AttributeError(f"'_TeeWriter' object has no attribute '{name}'")

    def write(self, message):
        self._orig.write(message)
        if _TeeWriter._log_fh is not None:
            _TeeWriter._log_fh.write(message)
        if "\n" in message:
            self._orig.flush()
            if _TeeWriter._log_fh is not None:
                _TeeWriter._log_fh.flush()

    def flush(self):
        self._orig.flush()
        if _TeeWriter._log_fh is not None:
            _TeeWriter._log_fh.flush()


def _setup_logging(log_path):
    # Unwrap any previous TeeWriter (duck-type: has _orig attr)
    # Jupyter re-imports change class identity so isinstance fails
    while hasattr(sys.stdout, "_orig"):
        sys.stdout = sys.stdout._orig
    _TeeWriter._log_fh = open(log_path, "w", buffering=1)
    sys.stdout = _TeeWriter(sys.stdout)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Logging to {log_path}")
    if _KAGGLE:
        print(f"HF_HOME = {os.environ.get('HF_HOME', 'NOT SET')}")
        print(f"TRANSFORMERS_CACHE = {os.environ.get('TRANSFORMERS_CACHE', 'NOT SET')}")


# Global hook for uncaught exceptions
def _log_exception(exc_type, exc_value, exc_tb):
    msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print(f"\n{'='*60}\nFATAL ERROR:\n{msg}{'='*60}\n", file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()

sys.excepthook = _log_exception

# ============================================================
# Configuration
# ============================================================


class Config:
    """Central configuration — override via CLI flags."""

    # --- Paths (relative to CWD; overridden to /kaggle/working on Kaggle) ---
    data_dir: str = "./data"
    output_dir: str = "./output"
    submission_file: str = "./output/submission.csv"
    cache_dir: str = "./.cache/huggingface"

    # --- Model ---
    model_name: str = "microsoft/deberta-v3-base"
    num_classes: int = 6
    max_length: int = 1024

    # --- Training ---
    # batch_size=4 + grad_accum=2 = effective batch 8. Fits T4 (15GB) with
    # gradient checkpointing even on long (1024-token) batches.
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 4
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True

    # --- Multi-GPU (auto) ---
    world_size: int = 1  # set in main(); >1 enables nn.DataParallel

    # --- CV ---
    n_folds: int = 5
    seed: int = 42

    # --- Hardware ---
    use_amp: bool = True  # automatic mixed precision
    # num_workers=0 is safest on Kaggle/Jupyter: num_workers>0 frequently deadlocks
    # on the first batch (DataLoader worker processes hang, esp. with DataParallel).
    # Cost is negligible here because data is pre-tokenized (getitem is a dict lookup).
    num_workers: int = 0

    # --- Smoke test overrides ---
    smoke: bool = False
    smoke_samples: int = 500
    smoke_epochs: int = 1
    smoke_folds: int = 2

    # --- Ensemble ---
    # 3 seeds keeps total runtime (~6h on T4 with dynamic padding) under Kaggle's 9h GPU limit.
    # Each seed produces a submittable CSV, so more seeds = pure upside if time allows.
    ensemble_seeds: tuple = (42, 3407, 2024)

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# QWK Metric (NumPy, threshold-optimization friendly)
# ============================================================


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute QWK between integer labels (1–6)."""
    min_rating = 1
    max_rating = 6
    y_true = y_true.astype(int)
    y_pred = np.round(y_pred).astype(int)

    # Build confusion matrix O
    O = np.zeros((max_rating, max_rating))
    for t, p in zip(y_true, y_pred):
        t_idx = int(t) - min_rating
        p_idx = int(p) - min_rating
        O[t_idx, p_idx] += 1

    # Weight matrix: quadratic
    N = max_rating
    w = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            w[i, j] = ((i - j) ** 2) / ((N - 1) ** 2)

    # Expected matrix E (chance-level agreement)
    hist_true = O.sum(axis=1)
    hist_pred = O.sum(axis=0)
    E = np.outer(hist_true, hist_pred) / O.sum()

    # Kappa
    num = (w * O).sum()
    den = (w * E).sum()
    if den == 0:
        return 0.0
    return 1.0 - num / den


def qwk_loss_torch(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Differentiable soft QWK approximation.
    Converts logits->softmax->expected_value, and penalizes MSE in label space.
    This is a proxy; the actual QWK is optimized via threshold tuning on OOF.
    """
    probs = torch.softmax(logits, dim=-1)
    # Expected score: sum(p_i * (i+1))  for classes 0..5 → scores 1..6
    weights = torch.arange(1, 7, device=logits.device, dtype=logits.dtype)
    expected = (probs * weights).sum(dim=-1)  # [batch]
    targets_f = targets.float()
    return nn.functional.mse_loss(expected, targets_f)


class CombinedLoss(nn.Module):
    """CrossEntropy + Soft QWK proxy."""

    def __init__(self, ce_weight: float = 0.7, qwk_weight: float = 0.3):
        super().__init__()
        self.ce_weight = ce_weight
        self.qwk_weight = qwk_weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, targets)
        qwk = qwk_loss_torch(logits, targets)
        return self.ce_weight * ce + self.qwk_weight * qwk


# ============================================================
# Threshold Optimization
# ============================================================


def apply_thresholds(continuous: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Map continuous scores to integer 1-6 via thresholds.
    thresholds: 5 values defining boundaries between classes.
    E.g., thresholds = [1.5, 2.5, 3.5, 4.5, 5.5] ≈ rounding.
    We predict:
      score = 1 if x <= thresh[0]
              i+1 if thresh[i-1] < x <= thresh[i]
              6 if x > thresh[-1]
    """
    scores = np.full_like(continuous, 6, dtype=int)
    for i in range(5):
        mask = continuous <= thresholds[i]
        scores[mask] = i + 1
        continuous = np.where(mask, np.inf, continuous)  # deactivate assigned
    return scores


def threshold_objective(thresholds: np.ndarray, y_true: np.ndarray, y_cont: np.ndarray) -> float:
    """Objective to MINIMIZE: negative QWK."""
    if np.any(np.diff(thresholds) <= 0):
        return 1e9  # penalty for non-monotonic
    y_pred = apply_thresholds(y_cont, thresholds)
    return -quadratic_weighted_kappa(y_true, y_pred)


def optimize_thresholds(y_true: np.ndarray, y_cont: np.ndarray, n_restarts: int = 10) -> np.ndarray:
    """
    Find optimal thresholds maximizing QWK using Nelder-Mead with random restarts.
    """
    best_thresh = None
    best_score = -np.inf

    # Default: midpoint thresholds
    default = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

    for _ in range(n_restarts):
        init = default + np.random.uniform(-0.3, 0.3, size=5)
        init = np.sort(np.clip(init, 1.1, 5.9))
        # Ensure minimum gap
        for i in range(1, 5):
            if init[i] - init[i - 1] < 0.05:
                init[i] = init[i - 1] + 0.05

        res = minimize(
            threshold_objective,
            init,
            args=(y_true, y_cont),
            method="Nelder-Mead",
            options={"maxiter": 1000, "xatol": 1e-6},
        )
        score = -res.fun
        if score > best_score:
            best_score = score
            best_thresh = res.x

    # Fall back to sorted
    best_thresh = np.sort(best_thresh)
    return best_thresh


# ============================================================
# Safe Model Loading (with retry + clear error messages)
# ============================================================


def _load_with_retry(load_fn, *args, label: str = "resource", retries: int = 8,
                     delay: float = 8.0, **kwargs):
    """Generic retry wrapper for HF downloads — SSL/network flakes are common and
    usually succeed on retry. Once anything loads, the files are cached locally."""
    import time
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            obj = load_fn(*args, **kwargs)
            # First successful download → switch to offline so later folds/epochs
            # never re-hit the network (avoids flaky SSL on cache-validity checks).
            os.environ["HF_HUB_OFFLINE"] = "1"
            return obj
        except Exception as e:
            last_err = e
            # Re-enable online for the next retry attempt
            os.environ.pop("HF_HUB_OFFLINE", None)
            print(f"  [retry {attempt}/{retries}] {label} download failed: {type(e).__name__}: {e}")
            if attempt < retries:
                time.sleep(delay)
    raise RuntimeError(f"{label} failed to load after {retries} attempts") from last_err


def safe_load_tokenizer(model_name: str, retries: int = 3):
    """Load tokenizer with retry on network failures."""
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    action = "Loading tokenizer from cache" if offline else "Downloading tokenizer"
    cache_info = f"HF_HOME={os.environ.get('HF_HOME', 'default')}"
    for attempt in range(1, retries + 1):
        try:
            print(f"  {action} for {model_name} ... [{cache_info}]")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.add_special_tokens({"additional_special_tokens": ["[NEWLINE]"]})
            print(f"  Tokenizer loaded OK (vocab={len(tokenizer)})")
            return tokenizer
        except Exception as e:
            print(f"  [ERROR] Tokenizer download failed (attempt {attempt}/{retries}): {e}")
            if attempt == retries:
                traceback.print_exc()
                raise RuntimeError(
                    f"\n{'='*60}\n"
                    f"TOKENIZER DOWNLOAD FAILED after {retries} attempts.\n"
                    f"Model: {model_name}\n"
                    f"Cache: {cache_info}\n"
                    f"Last error: {e}\n"
                    f"{'='*60}"
                ) from e
            print(f"  Retrying in 5s ...")
            import time
            time.sleep(5)


def safe_load_model(model_name: str, num_classes: int = 6, retries: int = 3):
    """Load model with retry on network failures."""
    cache_info = f"HF_HOME={os.environ.get('HF_HOME', 'default')}"
    for attempt in range(1, retries + 1):
        try:
            print(f"  Downloading model weights for {model_name} ... [{cache_info}]")
            model = EssayScoringModel(model_name, num_classes)
            n_params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"  Model loaded OK ({n_params:.1f}M params)")
            return model
        except Exception as e:
            print(f"  [ERROR] Model download failed (attempt {attempt}/{retries}): {e}")
            if attempt == retries:
                traceback.print_exc()
                raise RuntimeError(
                    f"\n{'='*60}\n"
                    f"MODEL DOWNLOAD FAILED after {retries} attempts.\n"
                    f"Model: {model_name}\n"
                    f"Cache: {cache_info}\n"
                    f"Last error: {e}\n"
                    f"{'='*60}"
                ) from e
            print(f"  Retrying in 5s ...")
            import time
            time.sleep(5)


# ============================================================
# Dataset & Preprocessing
# ============================================================


class EssayDataset(Dataset):
    """Holds PRE-TOKENIZED data (plain ints) — safe for num_workers multiprocessing."""

    def __init__(self, input_ids: list, attention_masks: list, scores=None):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.scores = scores  # None for test

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
        }
        if self.scores is not None:
            item["label"] = self.scores[idx]
        return item


def pretokenize(texts, tokenizer, max_length):
    """Tokenize all texts ONCE (no padding). Returns lists of python ints — picklable."""
    input_ids, attention_masks = [], []
    for text in tqdm(texts, desc="Tokenizing", leave=False):
        enc = tokenizer(text, max_length=max_length, truncation=True, add_special_tokens=True)
        input_ids.append(enc["input_ids"])
        attention_masks.append(enc["attention_mask"])
    return input_ids, attention_masks


def make_collate_fn(pad_token_id: int):
    """Dynamic padding: pad each batch to its longest sequence (not global max_length)."""
    def collate_fn(batch):
        input_ids = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
        attention_mask = [torch.tensor(b["attention_mask"], dtype=torch.long) for b in batch]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        out = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "label" in batch[0]:
            out["label"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        return out
    return collate_fn


def preprocess_text(text: str) -> str:
    """Lightweight text preprocessing."""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Add special token for paragraph breaks — DeBERTa treats this as meaningful
    text = text.replace("\n\n", " [NEWLINE] ")
    return text


def load_data(data_dir: str) -> tuple:
    """Load train and test CSVs."""
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Preprocess texts
    train_df["full_text"] = train_df["full_text"].apply(preprocess_text)
    test_df["full_text"] = test_df["full_text"].apply(preprocess_text)

    return train_df, test_df


# ============================================================
# Model
# ============================================================


class EssayScoringModel(nn.Module):
    """DeBERTa backbone + mean pooling + classification head."""

    def __init__(self, model_name: str, num_classes: int = 6, dropout: float = 0.1,
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.config = _load_with_retry(AutoConfig.from_pretrained, model_name,
                                       label=f"config({model_name})")
        self.config.hidden_dropout_prob = dropout
        self.config.attention_probs_dropout_prob = dropout

        self.backbone = _load_with_retry(
            AutoModel.from_pretrained, model_name,
            label=f"weights({model_name})", config=self.config, torch_dtype=torch.float32,
        )
        # Hard guarantee: force ALL params/buffers to FP32. microsoft/deberta-v3-* may
        # load as FP16 from the hub; under AMP that yields FP16 grads and crashes
        # GradScaler.unscale_ ("Attempting to unscale FP16 gradients").
        self.backbone = self.backbone.float()
        if gradient_checkpointing:
            # use_reentrant=False is REQUIRED under AMP — otherwise backward
            # recompute mishandles the autocast context and produces FP16 grads.
            try:
                self.backbone.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.backbone.gradient_checkpointing_enable()
            self.backbone.config.use_cache = False
        hidden_size = self.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Initialize classifier
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling over non-padded tokens
        last_hidden = outputs.last_hidden_state  # [B, L, H]
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = (last_hidden * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_embeddings / sum_mask  # [B, H]

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # [B, 6]
        return logits


# ============================================================
# Training
# ============================================================


def set_seed(seed: int):
    """Reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def wrap_data_parallel(model, config):
    """Wrap model in nn.DataParallel if multiple GPUs are visible."""
    if config.world_size > 1 and torch.cuda.device_count() > 1:
        device_ids = list(range(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=device_ids)
        print(f"  Using DataParallel on {len(device_ids)} GPUs: {device_ids}")
    return model


def unwrap_model(model):
    """Get the underlying model whether or not it's wrapped in DataParallel."""
    return model.module if isinstance(model, nn.DataParallel) else model


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, scaler, config):
    """Single training epoch. Robust to AMP fp16-grad drift: self-heals and never aborts."""
    model.train()
    # Hard reset all params to FP32 at epoch start. Some param/buffer can drift to FP16
    # across epochs (e.g. via GradScaler state residue / state_dict round-trips), which
    # makes GradScaler.unscale_ raise "Attempting to unscale FP16 gradients". This is cheap.
    model.float()
    total_loss = 0.0
    run_loss = 0.0  # running loss for periodic logging
    optimizer.zero_grad(set_to_none=True)
    n_steps = len(dataloader)
    log_interval = max(1, n_steps // 10)  # print ~10 loss lines per epoch

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        labels = batch["label"].to(config.device)

        if config.use_amp and scaler is not None:
            with autocast("cuda"):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss = loss / config.gradient_accumulation_steps
            scaler.scale(loss).backward()
        else:
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss = loss / config.gradient_accumulation_steps
            loss.backward()

        step_loss = loss.item() * config.gradient_accumulation_steps
        total_loss += step_loss
        run_loss += step_loss

        # Live loss in the progress bar + periodic line to stdout (→ train.log)
        pbar.set_postfix(loss=f"{step_loss:.4f}", avg=f"{total_loss/(step+1):.4f}")
        if (step + 1) % log_interval == 0:
            recent = run_loss / log_interval
            print(f"    step {step+1}/{n_steps} | "
                  f"recent_loss={recent:.4f} | avg_loss={total_loss/(step+1):.4f}")
            run_loss = 0.0

        if (step + 1) % config.gradient_accumulation_steps == 0:
            try:
                if config.use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
            except ValueError as e:
                if "FP16" in str(e) or "unscale" in str(e).lower():
                    # Auto-heal: the grads accumulated under scaler.scale() are SCALED,
                    # so a plain optimizer.step() here would be wrong. Safest is to drop
                    # this one optimizer step, zero grads, force FP32, and switch to a
                    # clean FP32 path for the rest of the run. One skipped step out of
                    # thousands is negligible; correctness is not.
                    print(f"\n[WARN] AMP fp16-grad issue at step {step}: "
                          f"dropping this step, switching to FP32 for the rest of the run.")
                    model.float()
                    optimizer.zero_grad(set_to_none=True)
                    config.use_amp = False
                    scheduler.step()
                    continue
                raise
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

    return total_loss / len(dataloader)


@torch.no_grad()
def predict(model, dataloader, config) -> np.ndarray:
    """Return predicted probabilities of shape [N, 6]."""
    model.eval()
    all_probs = []
    for batch in tqdm(dataloader, desc="Predicting", leave=False):
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)

        if config.use_amp and config.device.type == "cuda":
            with autocast("cuda"):
                logits = model(input_ids, attention_mask)
        else:
            logits = model(input_ids, attention_mask)

        probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)


def probabilities_to_scores(probs: np.ndarray) -> np.ndarray:
    """Convert class probabilities [N, 6] to continuous expected scores."""
    weights = np.arange(1, 7, dtype=np.float32)  # scores 1..6
    return (probs * weights).sum(axis=-1)


# ============================================================
# Single Model Training (with OOF)
# ============================================================


def train_single_model(train_input_ids, train_attention_masks, scores_list,
                       test_input_ids, test_attention_masks, tokenizer, config, fold_seeds=None):
    """
    Train DeBERTa with 5-fold CV on PRE-TOKENIZED data.
    Returns OOF probs, TEST probs (averaged across the N fold models — no separate
    full-data retrain needed), optimized thresholds, and OOF QWK.
    """
    score_ints = np.array(scores_list, dtype=int)
    n = len(train_input_ids)
    collate = make_collate_fn(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)

    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    oof_probs = np.zeros((n, config.num_classes), dtype=np.float32)
    # Accumulate test predictions from each fold model (ensemble of the N fold models).
    test_probs_sum = np.zeros((len(test_input_ids), config.num_classes), dtype=np.float32)

    eff_bs = config.batch_size * config.world_size
    test_dataset = EssayDataset(test_input_ids, test_attention_masks)
    test_loader = DataLoader(
        test_dataset, batch_size=eff_bs * 2, shuffle=False,
        num_workers=config.num_workers, pin_memory=True, collate_fn=collate,
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(range(n), score_ints)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{config.n_folds}")
        print(f"{'='*50}")

        fold_seed = config.seed if fold_seeds is None else fold_seeds[fold]
        set_seed(fold_seed)

        # Data: slice pre-tokenized encodings by index
        tr_ids = [train_input_ids[i] for i in train_idx]
        tr_mask = [train_attention_masks[i] for i in train_idx]
        tr_scores = [scores_list[i] - 1 for i in train_idx]  # 1-6 → 0-5
        va_ids = [train_input_ids[i] for i in val_idx]
        va_mask = [train_attention_masks[i] for i in val_idx]
        va_scores = [scores_list[i] for i in val_idx]  # keep 1-6 for QWK

        train_dataset = EssayDataset(tr_ids, tr_mask, tr_scores)
        val_dataset = EssayDataset(va_ids, va_mask)

        # Scale batch by world_size so each GPU gets config.batch_size samples.
        eff_bs = config.batch_size * config.world_size
        train_loader = DataLoader(
            train_dataset, batch_size=eff_bs, shuffle=True,
            num_workers=config.num_workers, pin_memory=True, collate_fn=collate,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=eff_bs * 2, shuffle=False,
            num_workers=config.num_workers, pin_memory=True, collate_fn=collate,
        )

        # Model
        model = EssayScoringModel(config.model_name, config.num_classes,
                                  gradient_checkpointing=config.gradient_checkpointing)
        model.backbone.resize_token_embeddings(len(tokenizer))
        model = model.float()  # ensure new embedding rows are FP32 (AMP-safe)
        model.to(config.device)
        model = wrap_data_parallel(model, config)
        if config.device.type == "cuda":
            torch.cuda.synchronize()

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        total_steps = len(train_loader) * config.epochs // config.gradient_accumulation_steps
        warmup_steps = int(total_steps * config.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        criterion = CombinedLoss(ce_weight=0.7, qwk_weight=0.3)
        scaler = GradScaler("cuda") if (config.use_amp and config.device.type == "cuda") else None

        # Training loop
        best_val_qwk = -1.0
        best_state = None
        for epoch in range(config.epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, config)
            probs = predict(model, val_loader, config)
            val_cont = probabilities_to_scores(probs)
            val_preds = np.round(val_cont).clip(1, 6).astype(int)
            val_qwk = quadratic_weighted_kappa(np.array(va_scores), val_preds)

            print(f"  Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_qwk(rounded)={val_qwk:.4f}")

            if val_qwk > best_val_qwk:
                best_val_qwk = val_qwk
                best_state = {k: v.cpu().clone() for k, v in unwrap_model(model).state_dict().items()}

        unwrap_model(model).load_state_dict(best_state)
        fold_probs = predict(model, val_loader, config)
        oof_probs[val_idx] = fold_probs

        # Predict test with this fold's best model; accumulate (ensemble across folds).
        test_probs_sum += predict(model, test_loader, config)

        fold_cont = probabilities_to_scores(fold_probs)
        fold_preds = np.round(fold_cont).clip(1, 6).astype(int)
        fold_qwk = quadratic_weighted_kappa(np.array(va_scores), fold_preds)
        print(f"  Fold {fold + 1} best QWK (rounded): {fold_qwk:.4f}")

        del model, optimizer, scheduler, scaler, best_state
        gc.collect()
        torch.cuda.empty_cache()

    test_probs = test_probs_sum / config.n_folds  # average of the N fold models

    # Optimize thresholds on full OOF
    oof_cont = probabilities_to_scores(oof_probs)
    thresholds = optimize_thresholds(score_ints, oof_cont)
    oof_preds = apply_thresholds(oof_cont, thresholds)
    oof_qwk = quadratic_weighted_kappa(score_ints, oof_preds)

    print(f"\n{'='*50}")
    print(f"OOF QWK (threshold-optimized): {oof_qwk:.4f}")
    print(f"Thresholds: {np.round(thresholds, 4)}")
    print(f"{'='*50}")

    return oof_probs, test_probs, thresholds, oof_qwk


# ============================================================
# Ensemble Pipeline
# ============================================================


def run_ensemble(train_df, test_df, config):
    """
    Train ensemble of models with different seeds.
    Tokenizes ONCE (shared across seeds), then per seed: N-fold CV produces both OOF
    (for thresholds/score estimate) AND test predictions (averaged across the N fold
    models — no separate full-data retrain). Saves a submittable CSV after EVERY seed.
    """
    print(f"\n{'#'*60}")
    print(f"ENSEMBLE TRAINING — {len(config.ensemble_seeds)} seeds × {config.n_folds} folds")
    print(f"Model: {config.model_name}")
    print(f"{'#'*60}")

    # Load tokenizer + pre-tokenize ALL data once (big speedup vs per-epoch tokenization)
    tokenizer = safe_load_tokenizer(config.model_name)
    train_texts = train_df["full_text"].tolist()
    scores_list = train_df["score"].tolist()
    test_texts = test_df["full_text"].tolist()

    print("Pre-tokenizing training data...")
    train_ids, train_masks = pretokenize(train_texts, tokenizer, config.max_length)
    print("Pre-tokenizing test data...")
    test_ids, test_masks = pretokenize(test_texts, tokenizer, config.max_length)
    print(f"Done. Train seq lengths: mean={np.mean([len(x) for x in train_ids]):.0f}, "
          f"max={max(len(x) for x in train_ids)} (cap={config.max_length})")

    score_ints = np.array(scores_list, dtype=int)
    all_oof_probs = []
    all_test_probs = []
    oof_qwks = []

    for i, seed in enumerate(config.ensemble_seeds):
        print(f"\n{'*'*50}")
        print(f"Ensemble Member {i + 1}/{len(config.ensemble_seeds)} — seed={seed}")
        print(f"{'*'*50}")

        config.seed = seed
        oof_probs, test_probs, _, oof_qwk = train_single_model(
            train_ids, train_masks, scores_list, test_ids, test_masks, tokenizer, config
        )
        all_oof_probs.append(oof_probs)
        all_test_probs.append(test_probs)
        oof_qwks.append(oof_qwk)

        # ---- Incremental checkpoint: save raw probs + a submittable CSV with seeds so far ----
        ckpt_dir = os.path.join(config.output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        np.save(os.path.join(ckpt_dir, f"oof_seed{seed}.npy"), oof_probs)
        np.save(os.path.join(ckpt_dir, f"test_seed{seed}.npy"), test_probs)

        # Build a partial submission from seeds completed so far
        avg_oof = np.mean(all_oof_probs, axis=0)
        avg_test = np.mean(all_test_probs, axis=0)
        partial_thresholds = optimize_thresholds(score_ints, probabilities_to_scores(avg_oof))
        partial_oof_qwk = quadratic_weighted_kappa(
            score_ints, apply_thresholds(probabilities_to_scores(avg_oof), partial_thresholds)
        )
        partial_test_preds = apply_thresholds(probabilities_to_scores(avg_test), partial_thresholds)
        pd.DataFrame({"essay_id": test_df["essay_id"], "score": partial_test_preds}).to_csv(
            config.submission_file, index=False
        )
        print(f"  >> Checkpoint saved after seed {seed} "
              f"({i + 1}/{len(config.ensemble_seeds)} done). "
              f"Partial OOF QWK={partial_oof_qwk:.4f}. submission.csv is submittable now.")

    # ---- Final ensemble (all seeds) ----
    avg_oof_probs = np.mean(all_oof_probs, axis=0)
    avg_test_probs = np.mean(all_test_probs, axis=0)

    oof_cont = probabilities_to_scores(avg_oof_probs)
    final_thresholds = optimize_thresholds(score_ints, oof_cont)
    oof_preds = apply_thresholds(oof_cont, final_thresholds)
    final_oof_qwk = quadratic_weighted_kappa(score_ints, oof_preds)

    test_cont = probabilities_to_scores(avg_test_probs)
    test_preds = apply_thresholds(test_cont, final_thresholds)

    print(f"\n{'#'*60}")
    print(f"ENSEMBLE RESULTS")
    print(f"Individual OOF QWKs: {[round(q, 4) for q in oof_qwks]}")
    print(f"Ensemble OOF QWK:   {final_oof_qwk:.4f}  (note: threshold fit on OOF → slightly optimistic, expect ~0.002-0.005 lower on LB)")
    print(f"Thresholds:          {np.round(final_thresholds, 4)}")
    print(f"{'#'*60}")

    return test_preds, final_oof_qwk, final_thresholds


def run_smoke_test(config):
    """Quick pipeline validation with small data and model."""
    print("\n*** SMOKE TEST MODE ***")
    train_df, test_df = load_data(config.data_dir)

    # Subsample
    train_df = train_df.sample(n=min(config.smoke_samples, len(train_df)), random_state=42).reset_index(drop=True)
    config.epochs = config.smoke_epochs
    config.n_folds = config.smoke_folds
    config.ensemble_seeds = (42, 99)  # only 2 members

    print(f"Data: {len(train_df)} train, {len(test_df)} test")
    print(f"Epochs: {config.epochs}, Folds: {config.n_folds}")
    print(f"Ensemble seeds: {config.ensemble_seeds}")

    test_preds, oof_qwk, thresholds = run_ensemble(train_df, test_df, config)

    # Generate submission
    submission = pd.DataFrame({
        "essay_id": test_df["essay_id"],
        "score": test_preds,
    })
    submission.to_csv(config.submission_file, index=False)
    # Show score distribution
    unique, counts = np.unique(test_preds, return_counts=True)
    dist = dict(zip(unique.astype(int), counts))
    print(f"\nSmoke test submission saved to {config.submission_file}")
    print(f"Score distribution: {dist}")
    print(f"Estimated OOF QWK: {oof_qwk:.4f} (THIS IS AN OVER-OPTIMISTIC ESTIMATE — use only for pipeline verification)")
    return test_preds, oof_qwk


def run_full(config):
    """Full training pipeline."""
    print(f"\n*** FULL TRAINING MODE ***")
    print(f"Model: {config.model_name}")
    print(f"Device: {config.device}")

    train_df, test_df = load_data(config.data_dir)
    print(f"Data: {len(train_df)} train, {len(test_df)} test")

    test_preds, oof_qwk, thresholds = run_ensemble(train_df, test_df, config)

    # Generate submission
    submission = pd.DataFrame({
        "essay_id": test_df["essay_id"],
        "score": test_preds,
    })
    submission.to_csv(config.submission_file, index=False)
    unique, counts = np.unique(test_preds, return_counts=True)
    dist = dict(zip(unique.astype(int), counts))
    print(f"\nSubmission saved to {config.submission_file}")
    print(f"Score distribution: {dist}")
    print(f"OOF QWK estimate: {oof_qwk:.4f}")
    return test_preds, oof_qwk, thresholds


# ============================================================
# Main
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(description="AES 2.0 MLE-Bench Pipeline")
    parser.add_argument("--smoke", action="store_true", help="Smoke test mode (small data, fast)")
    parser.add_argument("--model", type=str, default="base", choices=["base", "large"],
                        help="Model size: base or large")
    parser.add_argument("--data", type=str, default=None, help="Override data directory")
    parser.add_argument("--output", type=str, default="submission.csv", help="Submission file path")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--folds", type=int, default=None, help="Override number of CV folds")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Ensemble seeds")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--max_length", type=int, default=None, help="Max token length (default 1024)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable CUDA_LAUNCH_BLOCKING for detailed CUDA error stacks (slow)")
    return parser.parse_known_args()[0]  # ignore Jupyter/kernel extra args like -f


def main():
    args = parse_args()
    config = Config()

    # --debug sets CUDA_LAUNCH_BLOCKING for detailed CUDA stacks (must be before any CUDA use).
    # Checked here (not in __main__) so it also works via run.py / cell import.
    if args.debug or "--debug" in sys.argv:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        print("[debug] CUDA_LAUNCH_BLOCKING=1 (synchronous CUDA, slower)")

    # ---- Setup logging ASAP ----
    log_path = "/kaggle/working/training.log" if _KAGGLE else os.path.join(config.output_dir, "training.log")
    _setup_logging(log_path)

    # ===== Apply EDIT block (top of file) — the default source of truth =====
    if EDIT_PRESET == "smoke":
        config.smoke = True
        config.gradient_checkpointing = False  # tiny data, not needed
    elif EDIT_PRESET in ("fast", "base"):
        # DeBERTa-v3-base fits on T4 WITHOUT gradient checkpointing now that we use
        # dynamic padding (most batches are short) → ~1.5x faster than checkpointing.
        config.gradient_checkpointing = False
    elif EDIT_PRESET == "large":
        config.model_name = "microsoft/deberta-v3-large"
        config.batch_size = 2  # per-GPU micro-batch (memory-safe on T4)
        config.gradient_checkpointing = True  # large needs checkpointing on T4
    else:
        raise ValueError(f'EDIT_PRESET 必须是 "smoke"/"fast"/"base"/"large"，当前: {EDIT_PRESET}')

    if not config.smoke:
        config.ensemble_seeds = tuple(EDIT_SEEDS)
        config.n_folds = EDIT_FOLDS
        config.epochs = EDIT_EPOCHS
    config.max_length = EDIT_MAX_LENGTH
    config.learning_rate = EDIT_LR
    if EDIT_BATCH_SIZE is not None:
        config.batch_size = EDIT_BATCH_SIZE
    if EDIT_NO_AMP:
        config.use_amp = False

    # ===== CLI overrides (only if explicitly passed; default None → skip) =====
    if args.data:
        config.data_dir = args.data
    if args.output:
        config.submission_file = args.output
    if args.model == "large":
        config.model_name = "microsoft/deberta-v3-large"
        config.batch_size = 2
        config.gradient_checkpointing = True
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.epochs = args.epochs
    if args.folds:
        config.n_folds = args.folds
    if args.seeds:
        config.ensemble_seeds = tuple(args.seeds)
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.no_amp:
        config.use_amp = False
    if args.max_length:
        config.max_length = args.max_length
    if args.smoke:
        config.smoke = True

    # Environment-specific path adjustments
    if _KAGGLE:
        # Kaggle: force writable /kaggle/working paths (input data is read-only under /kaggle/input)
        config.data_dir = "/kaggle/input/datasets/arkria/lalaes2"
        config.output_dir = "/kaggle/working"
        config.submission_file = "/kaggle/working/submission.csv"
        if not os.path.exists(config.data_dir):
            print(f"WARNING: Kaggle data path not found: {config.data_dir}")
    else:
        # Local/4090 box: keep relative paths; just warn if data missing.
        if not os.path.exists(config.data_dir):
            print(f"WARNING: Data directory not found: {config.data_dir} "
                  f"(expected ./data with train.csv/test.csv). Use --data to override.")

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Print config
    print("=" * 60)
    print("AES 2.0 MLE-Bench Pipeline")
    print("=" * 60)
    print(f"Data:       {config.data_dir}")
    print(f"Model:      {config.model_name}")
    print(f"Device:     {config.device}")
    print(f"Batch:      {config.batch_size}")
    print(f"Epochs:     {config.epochs}")
    print(f"Folds:      {config.n_folds}")
    print(f"Seeds:      {config.ensemble_seeds}")
    print(f"AMP:        {config.use_amp}")
    print(f"Smoke:      {config.smoke}")
    print("=" * 60)

    # --- CUDA self-check + multi-GPU detection ---
    if config.device.type == "cuda":
        torch.cuda.empty_cache()
        n_gpus = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        config.world_size = n_gpus
        print(f"CUDA device:    {gpu_name}  (x{n_gpus})")
        try:
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        except AttributeError:
            mem = torch.cuda.get_device_properties(0).total_mem / 1e9  # older torch
        print(f"CUDA memory:    {mem:.1f} GB per GPU")

        # --- Memory-aware tuning for large model (only if user didn't force EDIT_BATCH_SIZE) ---
        is_large = "large" in config.model_name
        if is_large and EDIT_BATCH_SIZE is None:
            if mem >= 22:  # 4090 (24GB), A100, etc.
                # batch 4 needs checkpointing ON even on 24GB — long (1024-token)
                # batches spike activation memory. Checkpointing keeps it safe and
                # batch 4 still utilizes the GPU well.
                config.batch_size = 8
                config.gradient_checkpointing = True
                print(f"Large model on {mem:.0f}GB GPU → batch=4, checkpointing ON")
            else:  # T4 (16GB)
                config.batch_size = 2
                config.gradient_checkpointing = True
                print(f"Large model on {mem:.0f}GB GPU → batch=2, checkpointing ON (memory-safe)")
        try:
            t = torch.zeros(1, device=config.device)
            torch.cuda.synchronize()
            del t
            print("CUDA tensor test: OK")
        except Exception as e:
            print(f"CUDA tensor test: FAILED — {e}")
            print("FALLING BACK TO CPU. This will be slow but will work.")
            config.use_amp = False
            config.world_size = 1
            torch.cuda.is_available = lambda: False  # force cpu
        else:
            # Keep effective batch ~8 regardless of GPU count:
            # effective = batch_size * world_size * grad_accum
            target_eff = 8
            per_step = config.batch_size * config.world_size
            config.gradient_accumulation_steps = max(1, round(target_eff / per_step))
            if n_gpus > 1:
                print(f"Multi-GPU:     DataParallel across {n_gpus} GPUs "
                      f"(per-GPU batch={config.batch_size}, total/step={per_step}, "
                      f"grad_accum={config.gradient_accumulation_steps})")
        # P100 has known CUDA kernel issues — disable AMP
        if "P100" in gpu_name:
            print("P100 detected — disabling AMP (CUDA kernel compat)")
            config.use_amp = False
        else:
            print(f"GPU AMP:       {'enabled' if config.use_amp else 'disabled'}")
    print("=" * 60)

    try:
        if config.smoke:
            run_smoke_test(config)
        else:
            run_full(config)
    except Exception:
        # Always capture traceback to a dedicated error file
        tb = traceback.format_exc()
        print(tb)
        err_path = os.path.join(config.output_dir, "error.log")
        with open(err_path, "w") as f:
            f.write(tb)
        print(f"FATAL ERROR — stack trace saved to {err_path}")


if __name__ == "__main__":
    main()

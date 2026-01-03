# FinAI — Financial News Sentiment & Next-Day Direction Prediction (FinBERT + XGBoost)

FinAI is an end-to-end pipeline that aggregates daily financial news, extracts FinBERT-based semantic and sentiment features, enriches them with technical/commodity signals, and trains an XGBoost classifier to predict **next-day market direction** (Up/Down) on a **daily panel**.

It also includes a **Streamlit dashboard** for:
- price charting + a lightweight LSTM demo forecast, and
- live news sentiment inspection + XGBoost directional inference.

> Disclaimer: This project is for research/educational use and does not constitute financial advice.

---

## Key Ideas

### 1) Row-level FinBERT → Daily Aggregation
Instead of treating each headline as an independent training sample, the project:
1. computes **FinBERT embeddings (768-d)** and sentiment probabilities **per headline**, then
2. aggregates them into **one row per (Market_Index, Date)**:
   - mean-pooled embedding (768)
   - embedding disagreement proxy (`emb_std_mean`)
   - sentiment aggregates (mean/max/min/std)
   - news volume (`News_Count`)
   - keyword flags (bullish/bearish/mentions)

This reduces noise and aligns the target to “next day” movement.

### 2) Exogenous + Technical + Return Dynamics
For each daily row we add:
- **technical indicators** (volatility, RSI, cumulative return)
- **return dynamics** (lags + shifted rolling mean/std)
- **commodity context** (Gold and Oil, from Yahoo Finance)

### 3) Adaptive Labeling (Neutral Drop)
Rather than using a fixed threshold for Up/Down labels, the pipeline uses:
- a **volatility-adaptive threshold**: `thr_vec = max(MIN_THR, K_VOL * roll_std_10)`
- then labels:
  - Up if `return_1d > thr_vec`
  - Down if `return_1d < -thr_vec`
  - otherwise **Neutral** → dropped from training

This focuses learning on higher-signal days.

---

## Repository Structure

```

baturyusuf-financial-news-sentiment-analysis/
├─ app.py                      # Streamlit dashboard (price + news + inference)
├─ main.py                     # Main training pipeline (FinBERT row-cache → daily panel → XGB)
├─ inference_xgb.py            # Programmatic inference helper (requires history panel)
├─ ablation_sanity.py          # Logistic regression ablations + shuffle-label sanity checks
├─ analyze_correlations.py     # Small sample correlation analysis
├─ dataset_audit.py            # Duplicate/day-level audit
├─ diagnose_data.py            # Basic data diagnostics
├─ sweep_label_params.py       # Grid sweep over labeling params (K_VOL, MIN_THR)
├─ sweep_xgb_params.py         # Random sweep over XGB hyperparameters (walk-forward CV)
├─ src/
│  ├─ data/
│  │  ├─ make_dataset.py       # loading, preprocessing, chronological split utilities
│  │  └─ market_data.py        # yfinance downloader + CSV cache for commodity series
│  ├─ features/
│  │  ├─ build_features.py     # FinBERT extractor + feature engineering
│  │  └─ text_preprocessing.py # classic NLP cleaning (optional)
│  └─ models/
│     ├─ train_model.py        # XGB training from ENV hyperparameters
│     └─ train_projection.py   # optional NN projection head (experimental)
├─ data/
│  ├─ external/                # cached Yahoo series (Gold/Oil)
│  └─ processed/               # cached FinBERT embeddings/sentiment
├─ models/                     # trained artifacts (XGB, scalers, encoders, threshold, config)
└─ results/                    # sweep results CSVs

````

---

## Setup

### 1) Create an environment

**Option A — venv (Windows PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
````

**Option B — venv (macOS/Linux)**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Place the dataset

The training script expects:

```
data/raw/financial_news_market_events_2025.csv
```

If your dataset filename or path differs, update it in `main.py` and related scripts.

---

## Training Pipeline (main.py)

### What `main.py` does

1. Loads and preprocesses the raw dataset
2. Computes (or loads) **row-level FinBERT cache**
3. Aggregates to a **daily panel**
4. Adds commodity + technical + return-dynamics features
5. Builds **next-day** labels with volatility-adaptive thresholding
6. Fits preprocessing artifacts on train only:

   * `StandardScaler` for numeric features
   * `OneHotEncoder` for categoricals
   * `TruncatedSVD` to compress embeddings (default 64)
7. Trains `XGBClassifier` with early stopping
8. Searches an optimal decision threshold on validation (`MCC` or `Balanced Accuracy`)
9. Saves artifacts to `models/`

### Minimal run

```bash
python main.py
```

### Configuration via environment variables

**Labeling**

* `K_VOL` (float): volatility multiplier for adaptive labeling
* `MIN_THR` (float): minimum labeling threshold floor

**Threshold selection**

* `THR_OBJECTIVE`: `mcc` (default) or `bacc`

**Train/val split**

* `CUTOFF_FRAC` (float): time cutoff fraction for single split (default `0.80`)
* `SVD_DIM` (int): embedding compression (default `64`)

**CV mode**

* `CV_MODE`: `1` enables walk-forward evaluation, no model saving
* `CV_POINTS`: e.g. `0.60,0.70,0.80,0.90`

**Model saving**

* `SAVE_MODELS`: `1` saves artifacts, `0` skips

**XGBoost hyperparameters (via ENV)**

* `XGB_MAX_DEPTH`
* `XGB_MIN_CHILD_WEIGHT`
* `XGB_SUBSAMPLE`
* `XGB_COLSAMPLE_BYTREE`
* `XGB_REG_LAMBDA`
* `XGB_REG_ALPHA`
* `XGB_GAMMA`
* `XGB_LEARNING_RATE`
* `XGB_N_ESTIMATORS`
* `XGB_EARLY_STOPPING`
* `XGB_TREE_METHOD`
* `XGB_RANDOM_STATE`
* `XGB_N_JOBS`
* Optional: `XGB_PARAMS_JSON` to override many at once

**Example (PowerShell)**

```powershell
$env:K_VOL="1.1"
$env:MIN_THR="0.2"
$env:THR_OBJECTIVE="mcc"
$env:SVD_DIM="64"
$env:CUTOFF_FRAC="0.90"
$env:SAVE_MODELS="1"

$env:XGB_MAX_DEPTH="4"
$env:XGB_MIN_CHILD_WEIGHT="5"
$env:XGB_SUBSAMPLE="0.6"
$env:XGB_COLSAMPLE_BYTREE="0.7"
$env:XGB_REG_LAMBDA="3.0"
$env:XGB_LEARNING_RATE="0.02"
$env:XGB_GAMMA="0.5"
$env:XGB_N_ESTIMATORS="6000"
$env:XGB_EARLY_STOPPING="150"
$env:XGB_TREE_METHOD="hist"
$env:XGB_RANDOM_STATE="123"

python main.py
```

### Saved artifacts (models/)

Typical outputs:

* `xgb_model.pkl`
* `scaler.pkl`
* `ohe.pkl`
* `svd.pkl`
* `threshold.pkl`
* `numeric_cols.pkl`
* `cat_cols.pkl`
* `run_config.json`

If the Streamlit app says “model files missing”, run `python main.py` first to generate them.

---

## Streamlit App (app.py)

The dashboard has two tabs:

### Tab 1 — Technical + LSTM (demo)

* Candlestick chart from Yahoo Finance (`yfinance`)
* On-the-fly small LSTM training to forecast **t+1** price (demo baseline)

Run:

```bash
streamlit run app.py
```

### Tab 2 — News + FinBERT + XGBoost inference

* Pulls ticker news:

  * first from `yfinance` (`ticker.news`)
  * fallback to Google News RSS
* Computes FinBERT sentiment per headline (cached in `st.session_state`)
* Uses the latest headline for a direction inference demo

Notes:

* The training pipeline is **daily-panel based**, while live inference in the app is a simplified approximation (it builds features from the current ticker price series plus one latest headline).
* For more faithful inference using the same feature engineering as training, use `inference_xgb.py` with a proper `history_panel`.

---

## Programmatic Inference (inference_xgb.py)

`predict_from_panel_row(...)` is designed to match training behavior more closely:

* you must provide a `history_panel` (past days), because rolling/technical features cannot be computed from a single row.

Typical workflow:

1. Load artifacts via `load_artifacts()`
2. Build today’s row with `build_single_day_row(...)` (includes news aggregation)
3. Concatenate with `history_panel` and call `predict_from_panel_row(...)`

---

## Caching

### FinBERT row-level cache (training)

`main.py` stores:

* `data/processed/finbert_row_emb_768.npy`
* `data/processed/finbert_row_sent.npz`
* `data/processed/finbert_row_cache_sig.txt`

A lightweight signature is computed from dataset headline order/length. If signature or length mismatches, the cache is recomputed automatically.

### Commodity data cache

`src/data/market_data.py` caches Yahoo Finance daily closes to:

* `data/external/yahoo_GC_F_1d.csv` (Gold futures)
* `data/external/yahoo_CL_F_1d.csv` (Oil futures)

---

## Experiments and Diagnostics

### 1) Label sweep (adaptive labeling)

Grid search over `K_VOL` and `MIN_THR`:

```bash
python sweep_label_params.py
```

Outputs:

* `results/label_sweep.csv`

### 2) XGB hyperparameter sweep (walk-forward CV)

Random trials in CV mode:

```bash
# optionally set:
# export N_TRIALS=20
python sweep_xgb_params.py
```

Outputs:

* `results/xgb_sweep.csv`

### 3) Ablations + sanity checks

Runs Logistic Regression baselines (numeric-only, embedding-only, combined) and a **shuffle-label sanity check**:

```bash
python ablation_sanity.py
```

### 4) Correlation analysis (small sample)

```bash
python analyze_correlations.py
```

### 5) Dataset audits

```bash
python dataset_audit.py
python diagnose_data.py
```

---

## Metrics You Will See

* **PR-AUC (Average Precision)**: useful under class imbalance / when positive precision matters
* **ROC-AUC**: ranking quality across thresholds
* **Balanced Accuracy**: mean of recall per class
* **MCC (Matthews Correlation Coefficient)**: robust for imbalanced binary classification
* **Best threshold**: selected on validation set to maximize `THR_OBJECTIVE` (`mcc` or `bacc`)

---

## Troubleshooting

### “Model files missing” in Streamlit

Run training first:

```bash
python main.py
```

Confirm `models/` contains `xgb_model.pkl`, `svd.pkl`, `scaler.pkl`, `ohe.pkl`, etc.

### FinBERT is very slow

* First run will be slow due to model downloads and embedding computation.
* Subsequent runs are faster thanks to row-level caching.
* If you changed the dataset significantly and want a clean rebuild, delete:

  * `data/processed/finbert_row_emb_768.npy`
  * `data/processed/finbert_row_sent.npz`
  * `data/processed/finbert_row_cache_sig.txt`

### GPU usage

FinBERT will use CUDA if available (`torch.cuda.is_available()`).

### Embedding shape confusion

`build_features.FinbertFeatureExtractor.get_embedding()` returns a **768-d mean pooled** embedding (despite older docstrings mentioning 1536). The training pipeline assumes 768 and compresses it with SVD.

---

## Limitations (Current)

* Live Streamlit inference is a practical demo and not a perfect mirror of the daily-panel training distribution.
* Data quality and label design dominate performance; news-to-next-day mapping is inherently noisy.
* The dataset appears to contain multiple rows per day/index; the pipeline aggregates them, but the underlying event alignment still matters.

---

## Roadmap / Possible Improvements

* True “next trading day” calendar handling in the app (exchange calendars).
* Multi-task learning: predict both **direction** and **volatility**.
* Better headline grouping: deduplication, clustering, and publisher weighting.
* Add proper backtesting and trading-cost-aware evaluation.

---


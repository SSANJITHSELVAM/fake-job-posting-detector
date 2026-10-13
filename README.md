# Fake Job Posting Detector

An explainable fraud-detection pipeline: RoBERTa (feature extraction) →
XGBoost (classification) → SHAP (explanation), wrapped in a Streamlit app
with a login page and a detection page.

## Architecture

```
Job posting text
      │
      ▼
data_preprocessing.py   -- clean & combine fields into one text document
      │
      ▼
feature_extraction.py   -- RoBERTa tokenizer + frozen RoBERTa encoder
      │                     (mean-pooled 768-dim embedding)
      ▼
train.py                -- XGBoost trained on RoBERTa embeddings
      │
      ▼
inference.py             -- predict() for label + confidence
                            explain() for SHAP word-level attributions,
                            computed on the whole text→probability pipeline
                            (not on the opaque embedding dims — see the
                            docstring in inference.py for why)
      │
      ▼
app.py (Streamlit)       -- login page + detection page
```

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

A GPU is not required (RoBERTa runs on CPU fine for inference/feature
extraction at this scale) but will speed up training if available.

## 2. Get the dataset

This project uses the Kaggle **"Real or Fake? Fake Job Posting
Prediction"** dataset:
https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

Download `fake_job_postings.csv` and place it at:

```
data/fake_job_postings.csv
```

> If the file isn't present, `data_preprocessing.py` automatically falls
> back to a tiny synthetic sample so you can smoke-test the whole pipeline
> end-to-end — but you'll want the real dataset for a model that's
> actually useful. This fallback logs a clear warning when it kicks in.

## 3. Train the model

```bash
python -m src.train
```

This will:
- load and clean the dataset
- split into train/val/test
- extract RoBERTa embeddings for each split
- train an XGBoost classifier (with early stopping and class-imbalance
  weighting, since fake postings are a small minority of the dataset)
- print a classification report and save:
  - `models/xgb_classifier.json` — the trained classifier
  - `models/metrics.json` — evaluation metrics
  - `models/shap_background.json` — a small reference text sample

Training time is dominated by RoBERTa feature extraction over the dataset
(a few minutes on CPU for ~18k postings, faster on GPU).

## 4. Run the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (typically http://localhost:8501).

### Login

Demo accounts are created automatically on first run in `users.json`
(bcrypt-hashed, not plaintext):

| username | password |
|----------|----------|
| `demo`   | `demo1234` |
| `admin`  | `admin123` |

Change these (`src/auth.py` → `register_user`) or delete `users.json`
before any real deployment.

### Detection page

Paste a job posting (or click one of the example buttons) and click
**Analyze posting** to see:
- **Prediction** — Fake or Real
- **Confidence score** — from XGBoost's predicted probability
- **SHAP explanation** — a bar chart of the words that most pushed the
  prediction toward "Fake" or "Real", plus a highlighted-text view

The "SHAP thoroughness" slider in the sidebar trades explanation quality
for speed: each unit of thoroughness re-runs the full RoBERTa+XGBoost
pipeline on a perturbed version of the text, so lower it if explanations
feel slow.

## Notes on design choices

- **RoBERTa is frozen**, used only as a feature extractor (not
  fine-tuned). This keeps training fast and reproducible without a GPU
  cluster, while still giving XGBoost contextual language features
  instead of raw word counts. Fine-tuning RoBERTa end-to-end would likely
  improve accuracy further but is a heavier, separate project.
- **SHAP explains the full pipeline, not the embeddings.** SHAP is run as
  a model-agnostic text explainer over the function
  `text → clean → RoBERTa → XGBoost → probability`, using word-masking
  perturbations. This is what makes the explanations human-readable
  ("these words pushed it toward Fake") instead of meaningless embedding
  dimension indices.
- **Auth is intentionally minimal** (local bcrypt-hashed JSON file) to
  keep the app self-contained and easy to run. Swap in a real identity
  provider for production.

## File overview

| File | Purpose |
|---|---|
| `src/config.py` | Paths, hyperparameters, constants |
| `src/data_preprocessing.py` | Load, clean, split the dataset |
| `src/feature_extraction.py` | RoBERTa tokenization + embedding |
| `src/train.py` | Train & evaluate the XGBoost classifier |
| `src/inference.py` | Prediction + SHAP explanation at request time |
| `src/auth.py` | Login/session authentication |
| `app.py` | Streamlit app (login page + detection page) |

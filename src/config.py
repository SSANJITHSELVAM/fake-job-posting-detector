"""
config.py
---------
Central place for every path, hyperparameter, and constant used across the
pipeline. Keeping these in one file means train.py, inference.py, and app.py
all agree on where models live, how text is processed, and how caches are used.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Auto-detect dataset name (supports standard name or DataSet.csv from Kaggle)
if os.path.exists(os.path.join(DATA_DIR, "fake_job_postings.csv")):
    RAW_DATA_PATH = os.path.join(DATA_DIR, "fake_job_postings.csv")
elif os.path.exists(os.path.join(DATA_DIR, "DataSet.csv")):
    RAW_DATA_PATH = os.path.join(DATA_DIR, "DataSet.csv")
else:
    RAW_DATA_PATH = os.path.join(DATA_DIR, "fake_job_postings.csv")

PROCESSED_DATA_PATH = os.path.join(CACHE_DIR, "processed_jobs.parquet")
TRAIN_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "train_embeddings.npy")
VAL_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "val_embeddings.npy")
TEST_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "test_embeddings.npy")
SPLITS_METADATA_PATH = os.path.join(CACHE_DIR, "splits_metadata.json")

XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_classifier.json")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
BACKGROUND_SAMPLE_PATH = os.path.join(MODEL_DIR, "shap_background.json")
FEATURE_STATS_PATH = os.path.join(MODEL_DIR, "feature_stats.json")

USERS_FILE = os.path.join(BASE_DIR, "users.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Transformer feature extraction
# Using sentence-transformers/all-MiniLM-L6-v2 (cached locally, 384-dim semantic embeddings)
# ---------------------------------------------------------------------------
TRANSFORMER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 192          # truncate/pad job text to this many tokens
EMBEDDING_BATCH_SIZE = 64
POOLING_STRATEGY = "mean"

# ---------------------------------------------------------------------------
# XGBoost classifier
# ---------------------------------------------------------------------------
XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.04,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.5,
    min_child_weight=2,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
)
EARLY_STOPPING_ROUNDS = 25

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2
VAL_SIZE = 0.1  # carved out of the training portion, used for early stopping
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Explainability (SHAP & Fast Keyword Saliency)
# ---------------------------------------------------------------------------
SHAP_MAX_EVALS = 100          # default model calls per SHAP explanation
SHAP_TOP_K = 12               # how many top words/phrases to show in UI
FAST_SALIENCY_TOP_K = 15      # words evaluated in sub-second quick scan

# ---------------------------------------------------------------------------
# Dataset Columns
# ---------------------------------------------------------------------------
TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]

METADATA_COLUMNS = [
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "salary_range",
]

LABEL_COLUMN = "fraudulent"

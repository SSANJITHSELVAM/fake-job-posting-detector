"""
train.py
--------
Step 4 of the pipeline: Classification Model Training.

End-to-end training run:
  1. Load + preprocess dataset (data_preprocessing.py)
  2. Extract or load cached RoBERTa features for train/val/test splits
  3. Train an XGBoost classifier with class-imbalance weighting & early stopping
  4. Evaluate thoroughly on held-out test set
  5. Save model artifacts, metrics, and SHAP reference background sample

Run with:
    python -m src.train
"""

import json
import logging
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from . import config
from .data_preprocessing import load_and_prepare
from .feature_extraction import RobertaFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_and_evaluate(force_recompute_embeddings: bool = False):
    # ---------------------------------------------------------------
    # 1. Load Data Splits
    # ---------------------------------------------------------------
    train_df, val_df, test_df = load_and_prepare()

    # ---------------------------------------------------------------
    # 2. Tokenization + Feature Extraction (with Disk Cache)
    # ---------------------------------------------------------------
    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    cache_valid = (
        not force_recompute_embeddings
        and os.path.exists(config.TRAIN_EMBEDDINGS_PATH)
        and os.path.exists(config.VAL_EMBEDDINGS_PATH)
        and os.path.exists(config.TEST_EMBEDDINGS_PATH)
    )

    if cache_valid:
        logger.info("Loading precomputed RoBERTa embeddings from cache: %s", config.CACHE_DIR)
        X_train = np.load(config.TRAIN_EMBEDDINGS_PATH)
        X_val = np.load(config.VAL_EMBEDDINGS_PATH)
        X_test = np.load(config.TEST_EMBEDDINGS_PATH)

        if len(X_train) != len(y_train) or len(X_val) != len(y_val) or len(X_test) != len(y_test):
            logger.warning("Cache size mismatch! Re-computing RoBERTa embeddings...")
            cache_valid = False

    if not cache_valid:
        logger.info("Initializing RoBERTa Feature Extractor...")
        extractor = RobertaFeatureExtractor()

        logger.info("Extracting RoBERTa features for training split (%d samples)...", len(train_df))
        X_train = extractor.extract_features(train_df["text"].tolist())
        np.save(config.TRAIN_EMBEDDINGS_PATH, X_train)

        logger.info("Extracting RoBERTa features for validation split (%d samples)...", len(val_df))
        X_val = extractor.extract_features(val_df["text"].tolist())
        np.save(config.VAL_EMBEDDINGS_PATH, X_val)

        logger.info("Extracting RoBERTa features for test split (%d samples)...", len(test_df))
        X_test = extractor.extract_features(test_df["text"].tolist())
        np.save(config.TEST_EMBEDDINGS_PATH, X_test)
        logger.info("Saved all split embeddings to cache.")

    # ---------------------------------------------------------------
    # 3. Model Training (XGBoost)
    # ---------------------------------------------------------------
    logger.info("Training XGBoost classifier on %d samples (dim=%d)...", X_train.shape[0], X_train.shape[1])

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = float(n_neg / max(n_pos, 1))
    logger.info("Class distribution -> Real: %d, Fake: %d | scale_pos_weight: %.2f", n_neg, n_pos, scale_pos_weight)

    model = XGBClassifier(
        **config.XGB_PARAMS,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # ---------------------------------------------------------------
    # 4. Comprehensive Evaluation
    # ---------------------------------------------------------------
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate optimal threshold on validation set
    val_proba = model.predict_proba(X_val)[:, 1]
    best_thresh = 0.5
    best_f1 = 0.0
    for thresh in np.arange(0.2, 0.8, 0.02):
        score = f1_score(y_val, (val_proba >= thresh).astype(int), zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(thresh)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if len(set(y_test)) > 1 else None,
        "pr_auc": float(average_precision_score(y_test, y_proba)) if len(set(y_test)) > 1 else None,
        "optimal_threshold": round(best_thresh, 3),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_train_samples": len(y_train),
        "n_val_samples": len(y_val),
        "n_test_samples": len(y_test),
        "best_iteration": int(getattr(model, "best_iteration", 0)),
    }

    logger.info("=== Evaluation Results ===")
    logger.info("Accuracy:  %.4f", metrics["accuracy"])
    logger.info("Precision: %.4f", metrics["precision"])
    logger.info("Recall:    %.4f", metrics["recall"])
    logger.info("F1 Score:  %.4f", metrics["f1"])
    if metrics["roc_auc"]:
        logger.info("ROC-AUC:   %.4f", metrics["roc_auc"])

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["Legitimate (Real)", "Fraudulent (Fake)"], zero_division=0))

    # ---------------------------------------------------------------
    # 5. Persist Model & Metadata Artifacts
    # ---------------------------------------------------------------
    model.save_model(config.XGB_MODEL_PATH)
    logger.info("Saved XGBoost model to %s", config.XGB_MODEL_PATH)

    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", config.METRICS_PATH)

    # Save balanced background reference text for SHAP
    background_texts = []
    for label in (0, 1):
        subset = train_df[train_df["label"] == label]["text"]
        sample_size = min(15, len(subset))
        background_texts.extend(subset.sample(sample_size, random_state=config.RANDOM_STATE).tolist())

    with open(config.BACKGROUND_SAMPLE_PATH, "w") as f:
        json.dump(background_texts, f, indent=2)
    logger.info("Saved SHAP background reference sample (%d texts) to %s", len(background_texts), config.BACKGROUND_SAMPLE_PATH)

    return model, metrics


if __name__ == "__main__":
    train_and_evaluate()

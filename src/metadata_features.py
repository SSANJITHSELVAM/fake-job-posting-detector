"""
metadata_features.py
---------------------
Encodes the dataset's structured metadata columns (separate from the free
text fields) into a fixed-length numeric vector that gets concatenated onto
the RoBERTa text embedding before XGBoost -- turning this into a hybrid
text + structured-metadata classifier.

Why this helps: on the Kaggle "Fake Job Postings" dataset, a handful of
non-text signals are strongly correlated with fraud even before you read a
single word of the posting -- e.g. genuine postings overwhelmingly have a
company logo and screening questions attached, while a large share of scam
postings have neither. Folding those signals in directly, instead of asking
RoBERTa to infer them indirectly from phrasing, measurably improves
separability.

Categorical columns are encoded against a *fixed* vocabulary (config.py)
rather than a sklearn encoder fitted at training time, so the exact same
mapping is guaranteed at inference time with no extra artifact to load.
Unknown/blank values fall into a dedicated "unknown" bucket rather than
being dropped, so a posting missing this metadata is still handled
gracefully instead of erroring out.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from . import config


def _one_hot(value, vocab: List[str]) -> List[float]:
    value = (str(value).strip() if value is not None else "")
    vec = [0.0] * (len(vocab) + 1)  # last slot = "unknown / other"
    if value in vocab:
        vec[vocab.index(value)] = 1.0
    else:
        vec[-1] = 1.0
    return vec


def encode_row(row: Dict) -> np.ndarray:
    """Encode one posting's metadata (any dict-like: a plain dict or a
    pandas Series) into a fixed-length float vector. Missing fields default
    to 0 / "unknown" rather than raising, so this works equally well for a
    full dataset row or a partial dict typed in by a user in the UI.
    """
    getter = row.get
    features: List[float] = []
    for col in config.METADATA_BINARY_COLUMNS:
        try:
            features.append(float(getter(col, 0) or 0))
        except (TypeError, ValueError):
            features.append(0.0)
    features.extend(_one_hot(getter("employment_type", ""), config.EMPLOYMENT_TYPE_VOCAB))
    features.extend(_one_hot(getter("required_experience", ""), config.EXPERIENCE_VOCAB))
    features.extend(_one_hot(getter("required_education", ""), config.EDUCATION_VOCAB))
    return np.array(features, dtype=np.float32)


VECTOR_SIZE = (
    len(config.METADATA_BINARY_COLUMNS)
    + len(config.EMPLOYMENT_TYPE_VOCAB) + 1
    + len(config.EXPERIENCE_VOCAB) + 1
    + len(config.EDUCATION_VOCAB) + 1
)


def encode_dataframe(df: pd.DataFrame) -> np.ndarray:
    """Vectorized convenience wrapper: encodes every row of a DataFrame.
    Returns an (N, VECTOR_SIZE) array, even when N == 0.
    """
    if len(df) == 0:
        return np.zeros((0, VECTOR_SIZE), dtype=np.float32)
    return np.vstack([encode_row(row) for _, row in df.iterrows()])

"""
data_preprocessing.py
----------------------
Step 1 of the pipeline: Data Collection & Preprocessing.

Loads the Kaggle "Fake Job Postings" dataset, cleans free text, encodes structured
context, drops uninformative rows, and produces stratified train/val/test splits.
"""

import html
import logging
import os
import re
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text cleaning patterns
# ---------------------------------------------------------------------------
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_MULTI_SPACE_RE = re.compile(r"\s+")
_NON_TEXT_RE = re.compile(r"[^a-zA-Z0-9\.\,\!\?\'\-\$\@\%\:\/\#\+\s]")


def clean_text(text: str) -> str:
    """Unescape HTML entities, strip tags/URLs, and normalize whitespace.

    Preserves critical fraud indicators: currency symbols ($), exclamation
    marks (!), question marks (?), percentages (%), and contact symbols (@).
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" [URL] ", text)
    text = _EMAIL_RE.sub(" [EMAIL] ", text)
    text = _NON_TEXT_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def _safe_bool(val) -> Optional[bool]:
    """Robustly parse mixed boolean representations (0, 1, 't', 'f', 'True', 'False')."""
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    if s in ("1", "1.0", "t", "true", "yes", "y"):
        return True
    if s in ("0", "0.0", "f", "false", "no", "n"):
        return False
    return None


def _parse_label(val) -> int:
    """Parse label column containing 't'/'f', 1/0, True/False, or 'fake'/'real'."""
    if pd.isna(val):
        return 0
    s = str(val).strip().lower()
    if s in ("1", "1.0", "t", "true", "yes", "y", "fake", "fraud", "fraudulent"):
        return 1
    return 0


def combine_text_fields(row: pd.Series) -> str:
    """Concatenate relevant free-text and metadata fields into one unified document."""
    parts = []

    # Free text sections
    for col in config.TEXT_COLUMNS:
        val = row.get(col, "")
        if isinstance(val, str) and val.strip():
            cleaned = clean_text(val)
            if cleaned:
                parts.append(f"{col.upper()}: {cleaned}")

    # Metadata signals
    meta_parts = []
    if "employment_type" in row and pd.notna(row["employment_type"]):
        val_str = str(row["employment_type"]).strip()
        if val_str and val_str.lower() != "nan":
            meta_parts.append(f"Type: {val_str}")

    tele_b = _safe_bool(row.get("telecommuting"))
    if tele_b is not None:
        meta_parts.append("Location: Remote" if tele_b else "Location: On-site")

    if "salary_range" in row and pd.notna(row["salary_range"]):
        val_str = str(row["salary_range"]).strip()
        if val_str and val_str.lower() != "nan":
            meta_parts.append(f"Salary: {val_str}")

    logo_b = _safe_bool(row.get("has_company_logo"))
    if logo_b is not None:
        meta_parts.append("Logo: Verified" if logo_b else "Logo: Missing")

    if meta_parts:
        parts.append(f"METADATA: {' | '.join(meta_parts)}")

    return " \n ".join(parts)


# ---------------------------------------------------------------------------
# Loading & Synthetic Fallback
# ---------------------------------------------------------------------------
def load_raw_dataset(path: str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV dataset from disk."""
    if os.path.exists(path):
        logger.info("Loading dataset from %s", path)
        df = pd.read_csv(path, low_memory=False)
        return df

    # Search for alternative names in data directory
    if os.path.exists(config.DATA_DIR):
        for fname in os.listdir(config.DATA_DIR):
            if fname.endswith(".csv"):
                alt_path = os.path.join(config.DATA_DIR, fname)
                logger.info("Found alternative dataset at %s", alt_path)
                return pd.read_csv(alt_path, low_memory=False)

    logger.warning(
        "Real dataset not found at %s. Falling back to synthetic sample.",
        path,
    )
    return _generate_synthetic_sample()


def _generate_synthetic_sample(n_per_class: int = 150) -> pd.DataFrame:
    """Synthetic dataset used ONLY when no CSV is present."""
    import random

    random.seed(config.RANDOM_STATE)

    real_templates = [
        "Senior Software Engineer at {company}. We are seeking an experienced "
        "distributed systems engineer to join our infrastructure platform. "
        "Requirements: 4+ years of Python or Go experience, familiarity with AWS/GCP, "
        "Kubernetes, and relational databases. Bachelor's degree in Computer Science or equivalent. "
        "We offer competitive base salary, equity, 401k match, full medical/dental/vision.",
        "Financial Analyst at {company} Group. Join our global financial planning "
        "team. Responsibilities include quarterly forecasting, P&L modeling, and BI reports. "
        "Requirements: 2+ years corporate finance experience, advanced Excel & SQL. "
        "Competitive compensation, hybrid schedule, comprehensive healthcare benefits.",
        "Staff Registered Nurse at {company} Medical Center. Immediate opening in our "
        "intensive care unit. Requirements: Active state RN license, BLS and ACLS certification, "
        "minimum 2 years acute care experience. Night and day shifts available with shift differential.",
        "Data Scientist - Machine Learning at {company} Labs. Build predictive models "
        "and recommendation engines. Requirements: MS/PhD in Quantitative field, strong PyTorch/Scikit-learn, "
        "proven track record deploying ML pipelines. Great company culture and flexible remote options.",
    ]

    fake_templates = [
        "URGENT WORK FROM HOME OPPORTUNITY!!! Make $4000-$8000 per week guaranteed with NO EXPERIENCE! "
        "Acme Global Inc is looking for immediate online data entry workers. All you need is a phone. "
        "To receive your starter kit and company check, send your bank account details and wire a $75 "
        "refundable registration fee via Western Union or CashApp. Contact HR manager immediately on WhatsApp: +1-800-FAKE-JOB",
        "Mystery Shopper / Payment Processing Clerk Needed IMMEDIATELY! Earn $500 daily evaluating retail stores. "
        "You will receive cashier checks to deposit into your personal bank account and wire back excess funds. "
        "No interview required. 100% guaranteed hire! Send your SSN and credit card copy to job-desk-online@gmail.com",
        "Remote Package Inspector / Reshipment Specialist! Earn thousands weekly working only 1 hour a day. "
        "Receive luxury goods at your home address, inspect them, and forward them overseas. "
        "Requires paying a small security deposit via crypto or prepaid Visa card. Guaranteed fast payout!",
        "Executive Assistant (Home Based). High pay $95/hour with zero qualifications! "
        "Must be willing to purchase office supplies using your own account with promised reimbursement. "
        "Apply via Telegram @instant_hire_manager. Limited positions remaining, ACT NOW!!!",
    ]

    companies = ["Acme Corp", "Globex Global", "Initech Systems", "Umbrella Life Sciences", "Stark Logistics", "Apex Cloud"]
    rows = []
    for _ in range(n_per_class):
        comp = random.choice(companies)
        rows.append({
            "title": "Legitimate Career Opening",
            "company_profile": f"{comp} is a leading enterprise technology company founded in 2012.",
            "description": random.choice(real_templates).format(company=comp),
            "requirements": "Strong communication, relevant experience, university degree or equivalent.",
            "benefits": "Full healthcare, 401(k) matching, paid time off, commuter benefits.",
            "telecommuting": random.choice([0, 1]),
            "has_company_logo": 1,
            "has_questions": 1,
            "employment_type": "Full-time",
            "required_experience": "Mid-Senior level",
            "required_education": "Bachelor's Degree",
            "industry": "Information Technology",
            "function": "Engineering",
            "salary_range": "$90,000 - $140,000",
            "fraudulent": 0,
        })
        rows.append({
            "title": "URGENT HIRING - EASY WORK $$$",
            "company_profile": "",
            "description": random.choice(fake_templates),
            "requirements": "No requirements, anyone can do this!",
            "benefits": "Guaranteed income, make $5000/week",
            "telecommuting": 1,
            "has_company_logo": 0,
            "has_questions": 0,
            "employment_type": "Part-time",
            "required_experience": "Not Applicable",
            "required_education": "Unspecified",
            "industry": "",
            "function": "",
            "salary_range": "$150,000+",
            "fraudulent": 1,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Preprocessing Pipeline
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Combine text fields, clean text, drop duplicates, and prepare dataset."""
    df = df.copy()

    label_col = config.LABEL_COLUMN
    if label_col not in df.columns:
        for cand in ["fraudulent", "label", "is_fake", "fake", "target"]:
            if cand in df.columns:
                label_col = cand
                break
        else:
            raise ValueError(
                f"Expected label column '{config.LABEL_COLUMN}' not found in dataset. "
                f"Found columns: {list(df.columns)}"
            )

    df["label"] = df[label_col].apply(_parse_label).astype(int)
    df["text"] = df.apply(combine_text_fields, axis=1)

    # Filter out empty texts and exact duplicate texts
    df = df[df["text"].str.len() > 30]
    df = df.drop_duplicates(subset=["text"])

    df = df[["text", "label"]].reset_index(drop=True)

    logger.info(
        "Preprocessed dataset: %d rows (%.2f%% fraudulent, %d fake, %d real)",
        len(df),
        100.0 * df["label"].mean(),
        int(df["label"].sum()),
        len(df) - int(df["label"].sum()),
    )
    return df


def split_dataset(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split into train (70%), validation (10%), and test (20%)."""
    train_val, test = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        stratify=df["label"],
        random_state=config.RANDOM_STATE,
    )
    val_fraction = config.VAL_SIZE / (1 - config.TEST_SIZE)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        stratify=train_val["label"],
        random_state=config.RANDOM_STATE,
    )
    logger.info(
        "Split sizes -> train: %d (fake=%d), val: %d (fake=%d), test: %d (fake=%d)",
        len(train),
        int(train["label"].sum()),
        len(val),
        int(val["label"].sum()),
        len(test),
        int(test["label"].sum()),
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def load_and_prepare() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load, preprocess, and split the dataset."""
    raw = load_raw_dataset()
    clean = preprocess(raw)
    return split_dataset(clean)


if __name__ == "__main__":
    train_df, val_df, test_df = load_and_prepare()
    print("Train label distribution:\n", train_df["label"].value_counts())

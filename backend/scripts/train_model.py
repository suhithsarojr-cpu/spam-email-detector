"""Train the TF-IDF + Logistic Regression spam classifier and save the artifact.

Expected CSV columns: subject (optional), body, label  (label: spam/ham or 1/0)

Usage (from backend/):
    python -m scripts.train_model
    python -m scripts.train_model --data data/my_dataset.csv --dataset-name "My dataset" --dataset-source "URL / license"
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app import config
from app.ml_service import build_text

RANDOM_STATE = 42
TEST_SIZE = 0.2
LABEL_MAP = {"spam": "spam", "1": "spam", "ham": "ham", "0": "ham", "not spam": "ham"}
SAMPLE_DESCRIPTION = ("Synthetic, template-generated demo emails (scripts/generate_sample_data.py). "
                      "No real email data. Metrics on this data are NOT indicative of real-world performance.")


def load_and_clean(path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, on_bad_lines="skip", encoding="utf-8")
    original = len(df)
    for col in ("subject", "body"):
        if col not in df.columns:
            df[col] = ""
    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")
    df["subject"] = df["subject"].astype(str).str.strip()
    df["body"] = df["body"].astype(str).str.strip().str.slice(0, config.MAX_BODY_CHARS)
    df["label"] = df["label"].astype(str).str.strip().str.lower().map(LABEL_MAP)
    df = df.dropna(subset=["label"])                       # unknown / missing labels
    df = df[(df["subject"] != "") | (df["body"] != "")]    # rows with no text
    df = df.drop_duplicates(subset=["subject", "body"])    # avoid duplicates leaking across split
    df = df.reset_index(drop=True)
    if df["label"].nunique() < 2 or df["label"].value_counts().min() < 5:
        raise ValueError("Need at least 5 examples of both 'spam' and 'ham' after cleaning.")
    return df, {"rows_in_file": original, "rows_used": len(df), "rows_dropped": original - len(df)}


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2),
                                  min_df=2, sublinear_tf=True, max_features=20000)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])


def train(data_path: Path, model_dir: Path, dataset_name: str = None,
          dataset_source: str = None, dataset_description: str = None) -> dict:
    data_path, model_dir = Path(data_path), Path(model_dir)
    df, cleaning = load_and_clean(data_path)
    X = [build_text(s, b) for s, b in zip(df["subject"], df["body"])]
    y = df["label"].tolist()

    # Split BEFORE fitting anything, so the test set never influences the vectorizer or classifier.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    pipe = build_pipeline().fit(X_train, y_train)
    pred = pipe.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, pos_label="spam", zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, pos_label="spam", zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, pos_label="spam", zero_division=0)), 4),
        "positive_class": "spam",
        "confusion_matrix": {"labels": ["ham", "spam"],
                             "matrix": confusion_matrix(y_test, pred, labels=["ham", "spam"]).tolist()},
        "train_size": len(X_train), "test_size": len(X_test),
    }
    is_sample = data_path.name == "sample_emails.csv"
    metadata = {
        "model_type": "TF-IDF (1-2 grams) + Logistic Regression (scikit-learn)",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_state": RANDOM_STATE, "test_fraction": TEST_SIZE, "sklearn_version": sklearn.__version__,
        "dataset": {
            "name": dataset_name or ("Synthetic sample emails" if is_sample else data_path.name),
            "source": dataset_source or ("Generated locally" if is_sample else "Not specified"),
            "description": dataset_description or (SAMPLE_DESCRIPTION if is_sample else "User-supplied dataset"),
            "class_counts": {k: int(v) for k, v in df["label"].value_counts().items()},
            **cleaning,
        },
        "metrics": metrics,
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_dir / config.MODEL_FILENAME)
    (model_dir / config.METADATA_FILENAME).write_text(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=config.DATA_DIR / "sample_emails.csv")
    p.add_argument("--model-dir", type=Path, default=config.MODEL_DIR)
    p.add_argument("--dataset-name")
    p.add_argument("--dataset-source")
    p.add_argument("--dataset-description")
    a = p.parse_args()
    if not a.data.exists():
        sys.exit(f"Dataset not found: {a.data}\nCreate the demo data with: python -m scripts.generate_sample_data")
    meta = train(a.data, a.model_dir, a.dataset_name, a.dataset_source, a.dataset_description)
    m = meta["metrics"]
    print(f"Trained on {m['train_size']} rows, evaluated on {m['test_size']} held-out rows.")
    print(f"Accuracy {m['accuracy']} | Precision {m['precision']} | Recall {m['recall']} | F1 {m['f1']}")
    print(f"Saved to {a.model_dir}")


if __name__ == "__main__":
    main()

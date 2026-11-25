"""Loads the saved scikit-learn pipeline and runs inference."""
import json
import logging
from pathlib import Path
from typing import Optional

import joblib

from . import config

logger = logging.getLogger("spam_detector")

DISCLAIMER = (
    "These terms show which words in this message pushed the model's score the most. "
    "They are a rough, model-specific hint, not a definitive reason for the prediction, "
    "and the model can be wrong."
)


class ModelNotAvailable(RuntimeError):
    pass


def build_text(subject: str, body: str) -> str:
    """Single place that defines how subject+body become model input (also used in training)."""
    return f"{subject.strip()}\n{body.strip()}".strip()


class SpamModel:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.pipeline = None
        self.metadata: dict = {}

    @property
    def available(self) -> bool:
        return self.pipeline is not None

    def load(self) -> bool:
        path = self.model_dir / config.MODEL_FILENAME
        meta_path = self.model_dir / config.METADATA_FILENAME
        if not path.exists():
            logger.warning("Model file not found at %s. Run: python -m scripts.train_model", path)
            self.pipeline, self.metadata = None, {}
            return False
        self.pipeline = joblib.load(path)
        self.metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return True

    def predict(self, subject: str, body: str) -> dict:
        if not self.available:
            raise ModelNotAvailable("Model artifact not found. Train it with: python -m scripts.train_model")
        text = build_text(subject, body)
        classes = list(self.pipeline.classes_)
        proba = self.pipeline.predict_proba([text])[0]
        idx = int(proba.argmax())
        label = str(classes[idx])
        return {
            "label": label,
            "confidence": float(proba[idx]),
            "probabilities": {str(c): float(p) for c, p in zip(classes, proba)},
            "explanation": self._explain(text, label),
        }

    def _explain(self, text: str, label: str, top_n: int = 5) -> dict:
        """Top TF-IDF terms (weight x coefficient) supporting the predicted class.
        Only available for linear classifiers; otherwise returns a generic summary."""
        terms: list[dict] = []
        try:
            vec = self.pipeline.named_steps["tfidf"]
            clf = self.pipeline.named_steps["clf"]
            X = vec.transform([text]).tocoo()
            names = vec.get_feature_names_out()
            sign = 1.0 if label == "spam" else -1.0  # coef_ > 0 points to classes_[1] == "spam"
            contribs = [(names[j], sign * float(v) * float(clf.coef_[0][j]))
                        for j, v in zip(X.col, X.data)]
            contribs = [c for c in contribs if c[1] > 0]
            contribs.sort(key=lambda c: c[1], reverse=True)
            terms = [{"term": t, "weight": round(w, 4)} for t, w in contribs[:top_n]]
        except Exception:  # explanation is best-effort and must never break prediction
            terms = []
        if label == "spam":
            summary = "The message was classified as spam because its wording resembles spam examples seen in training."
        else:
            summary = "The message was classified as not spam because its wording resembles legitimate emails seen in training."
        if not terms:
            summary += " No individual terms from the message stood out."
        return {"summary": summary, "top_terms": terms, "disclaimer": DISCLAIMER}


model_service = SpamModel(config.MODEL_DIR)


def get_model_service() -> SpamModel:
    return model_service

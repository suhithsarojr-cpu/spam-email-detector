from app import config
from app.ml_service import SpamModel, build_text


def test_metrics_are_computed_not_hardcoded(metadata):
    m = metadata["metrics"]
    for key in ("accuracy", "precision", "recall", "f1"):
        assert 0.0 <= m[key] <= 1.0
    assert m["train_size"] + m["test_size"] == metadata["dataset"]["rows_used"]


def test_predict_returns_valid_structure():
    model = SpamModel(config.MODEL_DIR)
    assert model.load()
    out = model.predict("You have won a free gift card!!!", "Click http://example.com now to claim your prize.")
    assert out["label"] in ("spam", "ham")
    assert 0.5 <= out["confidence"] <= 1.0
    assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-6
    assert "disclaimer" in out["explanation"]


def test_obvious_examples():
    model = SpamModel(config.MODEL_DIR)
    model.load()
    assert model.predict("Congratulations! Claim your prize", "Free gift card, click now, limited time offer")["label"] == "spam"
    assert model.predict("Meeting moved to Tuesday", "Hi Sam, are you free to talk about the project agenda?")["label"] == "ham"


def test_build_text():
    assert build_text(" a ", " b ") == "a\nb"

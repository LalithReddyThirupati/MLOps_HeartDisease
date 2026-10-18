import pytest

from src.predict import load_metadata, load_model, predict_one


def test_predict_one_output_contract(fitted_model, sample_record):
    out = predict_one(fitted_model, sample_record)
    assert set(out) == {"prediction", "label", "probability_disease", "confidence"}
    assert out["prediction"] in (0, 1)
    assert 0.5 <= out["confidence"] <= 1.0
    assert out["confidence"] == pytest.approx(
        max(out["probability_disease"], 1 - out["probability_disease"]), abs=1e-4)


def test_predict_one_accepts_missing_optional_fields(fitted_model, sample_record):
    record = {**sample_record, "ca": None, "thal": None}
    assert predict_one(fitted_model, record)["prediction"] in (0, 1)


def test_load_model_uses_env_path(model_file, sample_record):
    model = load_model()
    assert load_metadata()["model_name"] == "test_lr"
    assert predict_one(model, sample_record)["label"] in ("heart disease", "no heart disease")

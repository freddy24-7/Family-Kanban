import numpy as np
import pytest

from ml.evaluate import (
    bootstrap_ci,
    expected_calibration_error,
    group_cv_predict,
    macro_f1,
    ordinal_mae,
    threshold_table,
)
from ml.models import (
    KeywordBaseline,
    OrdinalClassifier,
    TextModelConfig,
    build_text_classifier,
    mask_names,
)

TEXTS = [
    "gras maaien",
    "dakgoot schoonmaken",
    "melk halen",
    "brood en kaas",
    "zwemles betalen",
    "belasting doen",
] * 4
CATS = ["home_maintenance", "home_maintenance", "groceries", "groceries", "finance", "finance"] * 4
GROUPS = np.repeat([1, 2, 3, 4], 6)


@pytest.mark.parametrize("features", ["word", "char", "word+char"])
def test_text_classifier_learns_simple_data(features):
    model = build_text_classifier(TextModelConfig(features=features, C=10)).fit(TEXTS, CATS)
    assert list(model.predict(["gras maaien", "melk halen"])) == ["home_maintenance", "groceries"]
    assert np.allclose(model.predict_proba(["x"]).sum(axis=1), 1)


def test_ordinal_classifier_probabilities_are_valid():
    efforts = ["S", "M", "L", "S", "M", "L"] * 4
    model = build_text_classifier(TextModelConfig(features="word", ordinal=True, C=10)).fit(
        TEXTS, efforts
    )
    proba = model.predict_proba(TEXTS)
    assert proba.shape == (len(TEXTS), 3)
    assert np.allclose(proba.sum(axis=1), 1) and (proba >= 0).all()
    assert list(model.classes_) == ["S", "M", "L"]
    assert isinstance(model.named_steps["classifier"], OrdinalClassifier)


def test_keyword_baseline_rules_and_default():
    model = KeywordBaseline().fit(TEXTS, CATS)
    assert list(model.predict(["Belasting betalen", "iets vaags"])) == ["finance", "finance"]


def test_mask_names():
    assert mask_names(["Lieke naar zwemles", "Max uitlaten"]) == [
        " naamx  naar zwemles",
        " naamx  uitlaten",
    ]


def test_group_cv_never_sees_own_family():
    preds = group_cv_predict(
        build_text_classifier(TextModelConfig(C=10)), TEXTS, CATS, GROUPS, n_splits=4
    )
    assert len(preds) == len(TEXTS)
    assert macro_f1(CATS, preds) > 0.5


def test_metrics():
    assert ordinal_mae(["S", "M", "L"], ["L", "M", "S"]) == pytest.approx(4 / 3)
    low, high = bootstrap_ci(CATS, CATS)
    assert low == high == 1.0
    conf = np.array([0.9] * 10)
    assert expected_calibration_error(conf, [1] * 9 + [0]) == pytest.approx(0.0)
    assert expected_calibration_error(conf, [1] * 5 + [0] * 5) == pytest.approx(0.4)
    table = threshold_table([0.2, 0.95], [0, 1], thresholds=(0.5,))
    assert table.iloc[0].to_dict() == {
        "threshold": 0.5,
        "auto_accepted": 0.5,
        "accuracy_of_accepted": 1.0,
        "flagged_for_review": 0.5,
    }


def test_group_cv_proba_keeps_ordinal_column_order():
    efforts = ["S", "M", "L", "S", "M", "L"] * 4
    model = build_text_classifier(TextModelConfig(features="word", ordinal=True, C=10))
    proba = group_cv_predict(
        model, TEXTS, efforts, GROUPS, n_splits=4, proba=True, classes=["S", "M", "L"]
    )
    assert proba.shape == (len(TEXTS), 3) and np.allclose(proba.sum(axis=1), 1)

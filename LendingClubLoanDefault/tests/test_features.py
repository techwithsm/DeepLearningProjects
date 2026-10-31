import numpy as np
import pytest

from loan_default.features import FeaturePipeline, find_correlated_features, learn_categories, one_hot_encode

from .conftest import PURPOSES, TARGET


def test_one_hot_drops_reference_level(loans):
    encoded = one_hot_encode(loans, learn_categories(loans, ["purpose"]))
    dummies = [c for c in encoded.columns if c.startswith("purpose_")]
    assert "purpose" not in encoded.columns
    assert sorted(dummies) == [f"purpose_{p}" for p in PURPOSES[1:]]
    assert encoded[dummies].sum(axis=1).max() == 1


def test_one_hot_unseen_level_becomes_all_zeros(loans, caplog):
    categories = learn_categories(loans, ["purpose"])
    new = loans.head(2).copy()
    new["purpose"] = "wedding"
    encoded = one_hot_encode(new, categories)
    assert encoded.filter(like="purpose_").to_numpy().sum() == 0
    assert "unseen levels" in caplog.text


def test_find_correlated_drops_feature_weaker_with_target(loans):
    encoded = one_hot_encode(loans, learn_categories(loans, ["purpose"]))
    pairs, to_drop = find_correlated_features(encoded, TARGET, threshold=0.7)
    assert {pairs[0][0], pairs[0][1]} == {"int.rate", "fico"}
    corr = encoded.corr()[TARGET].abs()
    expected = "fico" if corr["fico"] < corr["int.rate"] else "int.rate"
    assert to_drop == [expected]


def test_pipeline_transform_is_consistent(loans):
    pipe = FeaturePipeline(TARGET).fit(loans)
    X = pipe.transform(loans)
    assert X.shape == (len(loans), len(pipe.feature_names_))
    assert np.allclose(X.mean(axis=0), 0, atol=1e-8)
    assert "log.revol.bal" in pipe.feature_names_ and "revol.bal" not in pipe.feature_names_
    assert not set(pipe.dropped_features_) & set(pipe.feature_names_)


def test_pipeline_handles_rows_without_target_in_any_column_order(loans):
    pipe = FeaturePipeline(TARGET).fit(loans)
    shuffled = loans.drop(columns=TARGET)[loans.columns[::-1].drop(TARGET)]
    assert np.allclose(pipe.transform(shuffled), pipe.transform(loans))


def test_pipeline_imputes_missing_values(loans):
    pipe = FeaturePipeline(TARGET).fit(loans)
    row = loans.head(1).copy()
    row["dti"] = np.nan
    assert np.isfinite(pipe.transform(row)).all()


def test_pipeline_requires_fit(loans):
    with pytest.raises(RuntimeError, match="not fitted"):
        FeaturePipeline(TARGET).transform(loans)

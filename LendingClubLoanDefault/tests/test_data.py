import pytest

from loan_default.data import load_data, split_data, validate_data

from .conftest import TARGET


def test_validate_rejects_missing_columns(loans):
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_data(loans.drop(columns="fico"), TARGET)


def test_validate_rejects_non_binary_target(loans):
    loans.loc[0, TARGET] = 2
    with pytest.raises(ValueError, match="binary"):
        validate_data(loans, TARGET)


def test_validate_without_target_accepts_feature_only_frame(loans):
    validate_data(loans.drop(columns=TARGET))


def test_validate_warns_on_missing_values(loans, caplog):
    loans.loc[0, "dti"] = None
    validate_data(loans, TARGET)
    assert "Missing values found" in caplog.text


def test_split_is_stratified_and_disjoint(loans):
    splits = split_data(loans, TARGET, val_size=0.15, test_size=0.15, seed=1)
    assert len(splits.train) + len(splits.val) + len(splits.test) == len(loans)
    assert not set(splits.train.index) & set(splits.test.index)
    rate = loans[TARGET].mean()
    for _, part in splits.items():
        assert part[TARGET].mean() == pytest.approx(rate, abs=0.03)


def test_load_data_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "nope.csv")

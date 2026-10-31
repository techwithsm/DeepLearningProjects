import numpy as np
import pandas as pd
import pytest

TARGET = "not.fully.paid"
PURPOSES = ["all_other", "credit_card", "debt_consolidation", "educational",
            "home_improvement", "major_purchase", "small_business"]


@pytest.fixture
def loans() -> pd.DataFrame:
    """Small synthetic dataset with the real schema; `int.rate` is built to correlate strongly with `fico`."""
    rng = np.random.default_rng(0)
    n = 400
    fico = rng.integers(620, 820, n)
    int_rate = 0.30 - 0.00025 * fico + rng.normal(0, 0.005, n)
    default = (rng.random(n) < 0.1 + 2 * (int_rate - int_rate.mean())).astype(int)
    return pd.DataFrame({
        "credit.policy": rng.integers(0, 2, n),
        "purpose": rng.choice(PURPOSES, n),
        "int.rate": int_rate,
        "installment": rng.uniform(20, 900, n),
        "log.annual.inc": rng.normal(10.9, 0.6, n),
        "dti": rng.uniform(0, 30, n),
        "fico": fico,
        "days.with.cr.line": rng.uniform(200, 15000, n),
        "revol.bal": rng.exponential(15000, n).round(),
        "revol.util": rng.uniform(0, 100, n),
        "inq.last.6mths": rng.poisson(1.5, n),
        "delinq.2yrs": rng.poisson(0.2, n),
        "pub.rec": rng.poisson(0.05, n),
        TARGET: default,
    })

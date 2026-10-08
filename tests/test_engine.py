from __future__ import annotations

import numpy as np
import pandas as pd

from macropulse.crisis import next_year_labels
from macropulse.data import load_dataset
from macropulse.features import engineer_features
from macropulse.portfolio import illustrative_weights


def tiny_panel() -> pd.DataFrame:
    return pd.DataFrame({
        "countryname": ["A", "A", "A", "B", "B"],
        "ISO3": ["AAA", "AAA", "AAA", "BBB", "BBB"],
        "year": [2000, 2001, 2003, 2000, 2001],
        "rGDP": [100.0, 110.0, 120.0, 100.0, 102.0],
        "infl": [2.0, 3.0, 4.0, 2.0, 2.5],
        "CurrencyCrisis": [0.0, 1.0, 0.0, 0.0, 0.0],
        "SovDebtCrisis": [0.0, 0.0, 0.0, 0.0, 0.0],
        "BankingCrisis": [0.0, 0.0, 0.0, 0.0, 0.0],
    })


def test_loader_filters_years_and_deduplicates(tmp_path):
    source = pd.DataFrame({
        "countryname": ["A", "A", "A"], "ISO3": ["aaa", "AAA", "AAA"],
        "year": [1959, 1960, 1960], "rGDP": [1, 2, 3], "infl": [1, 2, 3],
    })
    path = tmp_path / "sample.csv"
    source.to_csv(path, index=False)
    loaded, audit = load_dataset(path)
    assert loaded[["ISO3", "year"]].values.tolist() == [["AAA", 1960]]
    assert audit["dropped_outside_window"] == 1
    assert audit["dropped_duplicate_country_year"] == 1


def test_loader_accepts_uploaded_csv_bytes():
    source = pd.DataFrame({
        "countryname": ["A"], "ISO3": ["AAA"], "year": [2020],
        "rGDP": [100.0], "infl": [2.0],
    })
    loaded, audit = load_dataset(source.to_csv(index=False).encode("utf-8"))
    assert len(loaded) == 1
    assert audit["source_name"] == "GMD.csv (uploaded)"
    assert audit["source_bytes"] > 0


def test_growth_does_not_bridge_missing_years():
    result = engineer_features(tiny_panel())
    a = result.loc[result.ISO3.eq("AAA")].set_index("year")
    assert np.isclose(a.loc[2001, "gdp_growth"], 100 * np.log(1.1))
    assert np.isnan(a.loc[2003, "gdp_growth"])


def test_next_year_crisis_labels_align_only_consecutive_years():
    panel = tiny_panel()
    labels = next_year_labels(panel, "CurrencyCrisis")
    row_2000_a = panel.index[(panel.ISO3 == "AAA") & (panel.year == 2000)][0]
    row_2001_a = panel.index[(panel.ISO3 == "AAA") & (panel.year == 2001)][0]
    assert labels.loc[row_2000_a] == 1
    assert pd.isna(labels.loc[row_2001_a])


def test_illustrative_weights_are_long_only_and_sum_to_one():
    scores = pd.DataFrame({
        "countryname": ["A", "B", "C"], "ISO3": ["AAA", "BBB", "CCC"],
        "year": [2024] * 3, "macro_score": [2.0, 0.0, -1.0],
    })
    result = illustrative_weights(scores, top_n=2, cash_weight=0.1)
    assert (result.weight >= 0).all()
    assert np.isclose(result.weight.sum(), 1.0)
    assert result.loc[result.ISO3.eq("CASH"), "weight"].iloc[0] == 0.1

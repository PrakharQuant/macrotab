"""Transparent country-year feature engineering; no market returns are inferred."""
from __future__ import annotations

import numpy as np
import pandas as pd

MODEL_FEATURES = (
    "gdp_growth",
    "infl",
    "govdebt_GDP",
    "govdef_GDP",
    "CA_GDP",
    "unemp",
    "strate",
    "ltrate",
    "REER_change",
)


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add within-country annual growth and exchange-rate changes without crossing gaps."""
    out = frame.sort_values(["ISO3", "year"]).copy()
    if "REER" not in out:
        out["REER"] = np.nan
    prior_year = out.groupby("ISO3", sort=False)["year"].shift(1)
    prior_gdp = out.groupby("ISO3", sort=False)["rGDP"].shift(1)
    valid_pair = prior_year.eq(out["year"] - 1)
    good_gdp = valid_pair & out["rGDP"].gt(0) & prior_gdp.gt(0)
    out["gdp_growth"] = np.where(good_gdp, 100.0 * np.log(out["rGDP"] / prior_gdp), np.nan)

    prior_reer = out.groupby("ISO3", sort=False)["REER"].shift(1)
    good_reer = valid_pair & out["REER"].gt(0) & prior_reer.gt(0)
    out["REER_change"] = np.where(good_reer, 100.0 * np.log(out["REER"] / prior_reer), np.nan)
    out["infl"] = pd.to_numeric(out["infl"], errors="coerce")
    out["is_projection_period"] = out["year"].ge(2025)
    return out.replace([np.inf, -np.inf], np.nan)

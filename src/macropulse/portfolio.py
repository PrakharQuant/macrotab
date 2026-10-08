"""Exploratory regime discovery and macro tilts—not a return backtest or advice."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from .features import MODEL_FEATURES


def fit_regimes(frame: pd.DataFrame, training_end_year: int = 2024, n_clusters: int = 4) -> pd.DataFrame:
    """Fit K-means on historical rows only, then map all years to those clusters.

    Names are heuristic descriptions of standardized growth/inflation centroids;
    they are not economically validated states or investment forecasts.
    """
    out = frame.copy()
    feature_columns = list(MODEL_FEATURES)
    usable = out[feature_columns].notna().sum(axis=1).ge(3)
    training_mask = usable & out["year"].le(training_end_year)
    if int(training_mask.sum()) < max(40, n_clusters * 5):
        out["regime"] = "Insufficient history"
        out["regime_cluster"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
        return out
    k = min(n_clusters, int(training_mask.sum()))
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    train_raw = out.loc[training_mask, feature_columns]
    lower = train_raw.quantile(0.01)
    upper = train_raw.quantile(0.99)
    train_clipped = train_raw.clip(lower=lower, upper=upper, axis=1)
    train_x = scaler.fit_transform(imputer.fit_transform(train_clipped))
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    model.fit(train_x)

    centers = model.cluster_centers_
    g_idx = MODEL_FEATURES.index("gdp_growth")
    i_idx = MODEL_FEATURES.index("infl")
    names: dict[int, str] = {}
    descriptors = [
        ("Goldilocks-like", lambda center: center[g_idx] - center[i_idx]),
        ("Expansion / inflation pressure", lambda center: center[g_idx] + center[i_idx]),
        ("Stagflation-like", lambda center: center[i_idx] - center[g_idx]),
        ("Weak growth / disinflation", lambda center: -center[g_idx] - center[i_idx]),
    ]
    remaining = set(range(k))
    for label, score in descriptors:
        if not remaining:
            break
        cluster_id = max(remaining, key=lambda idx: score(centers[idx]))
        names[cluster_id] = label
        remaining.remove(cluster_id)
    out["regime_cluster"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["regime"] = "Insufficient features"
    scoring_mask = usable
    score_raw = out.loc[scoring_mask, feature_columns].clip(lower=lower, upper=upper, axis=1)
    score_x = scaler.transform(imputer.transform(score_raw))
    predicted = model.predict(score_x)
    out.loc[scoring_mask, "regime_cluster"] = predicted
    out.loc[scoring_mask, "regime"] = pd.Series(predicted, index=out.index[scoring_mask]).map(names).to_numpy()
    return out


def _cross_section_zscore(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = series.astype(float).copy()
    if values.notna().sum() == 0:
        return values
    low, high = values.quantile([0.02, 0.98])
    values = values.clip(lower=low, upper=high)
    median = values.median()
    mad = (values - median).abs().median()
    scale = 1.4826 * mad if pd.notna(mad) and mad > 0 else values.std(ddof=0)
    if pd.isna(scale) or scale == 0:
        return values.where(values.isna(), 0.0)
    z = (values - median) / scale
    return z if higher_is_better else -z


def country_macro_scores(frame: pd.DataFrame, as_of_year: int = 2024) -> pd.DataFrame:
    """Build a disclosed equal-weight heuristic score from latest available country rows."""
    sample = frame.loc[frame["year"].le(as_of_year)].sort_values(["ISO3", "year"])
    if sample.empty:
        return pd.DataFrame(columns=["countryname", "ISO3", "year", "macro_score"])
    sample = sample.groupby("ISO3", as_index=False, sort=False).tail(1).copy()
    if "gdp_growth" not in sample:
        sample = sample.sort_values(["ISO3", "year"])
        prior_year = sample.groupby("ISO3")["year"].shift()
        prior_gdp = sample.groupby("ISO3")["rGDP"].shift()
        sample["gdp_growth"] = np.where(
            prior_year.eq(sample["year"] - 1) & sample["rGDP"].gt(0) & prior_gdp.gt(0),
            100.0 * np.log(sample["rGDP"] / prior_gdp), np.nan,
        )
    components = {
        "growth_signal": ("gdp_growth", True),
        "inflation_signal": ("absolute_inflation", False),
        "debt_signal": ("govdebt_GDP", False),
        "external_balance_signal": ("CA_GDP", True),
        "employment_signal": ("unemp", False),
    }
    sample["absolute_inflation"] = sample["infl"].abs()
    for name, (column, _) in components.items():
        sample[name] = _cross_section_zscore(sample[column], higher_is_better=components[name][1])
    component_names = list(components)
    sample["macro_score"] = sample[component_names].mean(axis=1, skipna=True).where(sample[component_names].notna().any(axis=1))
    coverage = sample[component_names].notna().sum(axis=1)
    sample.loc[coverage.lt(2), "macro_score"] = np.nan
    return sample.sort_values("macro_score", ascending=False).reset_index(drop=True)


def illustrative_weights(scores: pd.DataFrame, top_n: int = 10, cash_weight: float = 0.10, temperature: float = 0.8) -> pd.DataFrame:
    """Convert the top macro scores to illustrative, non-investable weights plus cash."""
    if not 0 <= cash_weight < 1:
        raise ValueError("cash_weight must be in [0, 1)")
    chosen = scores.dropna(subset=["macro_score"]).nlargest(max(1, int(top_n)), "macro_score").copy()
    if chosen.empty:
        return pd.DataFrame(columns=["countryname", "ISO3", "macro_score", "weight"])
    temp = max(float(temperature), 1e-6)
    logits = (chosen["macro_score"] - chosen["macro_score"].max()) / temp
    raw = np.exp(logits)
    chosen["weight"] = (1 - cash_weight) * raw / raw.sum()
    cash = pd.DataFrame([{"countryname": "Cash reserve (illustrative)", "ISO3": "CASH", "year": chosen["year"].max(), "macro_score": np.nan, "weight": cash_weight}])
    columns = ["countryname", "ISO3", "year", "macro_score", "weight"]
    return pd.concat([chosen[columns], cash], ignore_index=True).sort_values("weight", ascending=False).reset_index(drop=True)

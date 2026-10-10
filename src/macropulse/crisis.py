"""Leakage-aware next-year crisis classification and out-of-time diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from .data import CRISIS_TARGETS
from .features import MODEL_FEATURES


def next_year_labels(frame: pd.DataFrame, target: str) -> pd.Series:
    """Align target at t+1 to predictor row t; do not bridge missing calendar years."""
    if target not in frame:
        return pd.Series(np.nan, index=frame.index, name=f"{target}_next_year")
    ordered = frame.sort_values(["ISO3", "year"])
    next_year = ordered.groupby("ISO3", sort=False)["year"].shift(-1)
    next_value = ordered.groupby("ISO3", sort=False)[target].shift(-1)
    label = next_value.where(next_year.eq(ordered["year"] + 1))
    label = label.where(label.isin([0, 1]))
    return label.reindex(frame.index).rename(f"{target}_next_year")


def _pipeline() -> object:
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        RobustScaler(quantile_range=(10.0, 90.0)),
        LogisticRegression(max_iter=1200, random_state=42),
    )


def build_crisis_radar(
    frame: pd.DataFrame,
    as_of_year: int = 2024,
    train_end_year: int = 2010,
    start_year: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return latest-country model scores and temporal holdout metrics for each crisis.

    The training outcome year is predictor year + 1. Forecast/estimation rows from
    2025 onward are never used as labeled training or evaluation outcomes.
    """
    sample = frame if start_year is None else frame.loc[frame["year"].ge(start_year)]
    output: pd.DataFrame | None = None
    metrics: list[dict[str, object]] = []
    latest = sample.loc[sample["year"].le(as_of_year)].sort_values(["ISO3", "year"]).groupby("ISO3", as_index=False, sort=False).tail(1).copy()

    for target in CRISIS_TARGETS:
        column = f"{target}_next_year"
        work = sample.copy()
        work[column] = next_year_labels(work, target)
        labels = work.loc[work[column].notna() & work["year"].lt(as_of_year)].copy()
        if not labels.empty:
            labels[column] = labels[column].astype(int)
        train = labels.loc[labels["year"].le(train_end_year)]
        test = labels.loc[labels["year"].gt(train_end_year) & labels["year"].lt(as_of_year)]
        entry: dict[str, object] = {"crisis": target.replace("Crisis", " crisis"), "train_rows": len(train), "train_events": int(train[column].sum()) if len(train) else 0, "test_rows": len(test), "test_events": int(test[column].sum()) if len(test) else 0, "ROC-AUC": np.nan, "PR-AUC": np.nan, "Brier": np.nan}

        if len(train) >= 50 and train[column].nunique() == 2 and len(test) >= 10:
            evaluator = _pipeline()
            evaluator.fit(train[list(MODEL_FEATURES)], train[column])
            if test[column].nunique() == 2:
                p_test = evaluator.predict_proba(test[list(MODEL_FEATURES)])[:, 1]
                entry["ROC-AUC"] = float(roc_auc_score(test[column], p_test))
                entry["PR-AUC"] = float(average_precision_score(test[column], p_test))
                entry["Brier"] = float(brier_score_loss(test[column], p_test))

        full_train = labels.loc[labels["year"].le(as_of_year - 1)]
        enough = len(full_train) >= 50 and full_train[column].nunique() == 2 and int(full_train[column].sum()) >= 8
        if enough:
            model = _pipeline()
            model.fit(full_train[list(MODEL_FEATURES)], full_train[column])
            valid_latest = latest[list(MODEL_FEATURES)].notna().sum(axis=1).ge(3)
            scores = pd.Series(np.nan, index=latest.index, dtype=float)
            if valid_latest.any():
                scores.loc[valid_latest] = model.predict_proba(latest.loc[valid_latest, list(MODEL_FEATURES)])[:, 1]
            if output is None:
                output = latest[["countryname", "ISO3", "year"]].copy()
            output[target] = scores.to_numpy()
            entry["model_status"] = "fit"
        else:
            entry["model_status"] = "insufficient labeled history"
            if output is None:
                output = latest[["countryname", "ISO3", "year"]].copy()
            output[target] = np.nan
        metrics.append(entry)

    if output is None:
        output = pd.DataFrame(columns=["countryname", "ISO3", "year", *CRISIS_TARGETS])
    return output.sort_values("ISO3").reset_index(drop=True), pd.DataFrame(metrics)

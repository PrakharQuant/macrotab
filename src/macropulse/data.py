"""Loading, validation, and audit helpers for the supplied Global Macro Dataset."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FIRST_YEAR = 1960
LAST_YEAR = 2031
HISTORICAL_END_YEAR = 2024
FORECAST_START_YEAR = 2025
CRISIS_TARGETS = ("SovDebtCrisis", "CurrencyCrisis", "BankingCrisis")
RAW_FEATURES = ("rGDP", "infl", "govdebt_GDP", "govdef_GDP", "CA_GDP", "unemp", "strate", "ltrate", "REER")
REQUIRED_COLUMNS = ("countryname", "ISO3", "year", "rGDP", "infl")


class DatasetError(ValueError):
    """Raised when the input file cannot be used safely."""


def load_dataset(path: str | Path | bytes, first_year: int = FIRST_YEAR, last_year: int = LAST_YEAR) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the CSV, validate its schema, and retain only the documented analysis window.

    The source contains observations outside the stated analysis window. Those rows are
    counted in the audit and excluded, rather than silently entering model fitting.
    """
    if isinstance(path, bytes):
        source_name = "GMD.csv (uploaded)"
        source_bytes = len(path)
        input_source = io.BytesIO(path)
    else:
        source = Path(path)
        if not source.is_file():
            raise DatasetError(f"Dataset not found: {source}. Place the supplied GMD.csv in data/ or upload it in the app.")
        source_name = source.name
        source_bytes = source.stat().st_size
        input_source = source
    try:
        frame = pd.read_csv(input_source, low_memory=False)
    except Exception as exc:
        raise DatasetError(f"Could not read {source_name}: {exc}") from exc
    missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise DatasetError(f"Missing required columns: {', '.join(missing)}")

    input_rows = len(frame)
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    invalid_year = frame["year"].isna()
    frame = frame.loc[~invalid_year].copy()
    frame["year"] = frame["year"].astype(int)
    outside_window = ~frame["year"].between(first_year, last_year)
    dropped_outside_window = int(outside_window.sum())
    frame = frame.loc[~outside_window].copy()

    frame["ISO3"] = frame["ISO3"].astype("string").str.strip().str.upper()
    frame["countryname"] = frame["countryname"].astype("string").str.strip()
    for column in (*RAW_FEATURES, *CRISIS_TARGETS):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame[column] = frame[column].replace([np.inf, -np.inf], np.nan)
    for column in RAW_FEATURES:
        if column not in frame:
            frame[column] = np.nan
    for column in CRISIS_TARGETS:
        if column not in frame:
            frame[column] = np.nan

    duplicate_rows = int(frame.duplicated(["ISO3", "year"], keep="last").sum())
    frame = frame.drop_duplicates(["ISO3", "year"], keep="last")
    frame = frame.dropna(subset=["ISO3", "countryname"])
    frame = frame.sort_values(["ISO3", "year"]).reset_index(drop=True)

    audit: dict[str, Any] = {
        "source_name": source_name,
        "source_bytes": source_bytes,
        "input_rows": input_rows,
        "analysis_rows": len(frame),
        "dropped_invalid_year": int(invalid_year.sum()),
        "dropped_outside_window": dropped_outside_window,
        "dropped_duplicate_country_year": duplicate_rows,
        "first_year": int(frame["year"].min()) if not frame.empty else None,
        "last_year": int(frame["year"].max()) if not frame.empty else None,
        "countries": int(frame["ISO3"].nunique()),
        "columns": int(frame.shape[1]),
        "missing_share": frame[list(RAW_FEATURES)].isna().mean().to_dict(),
    }
    if frame.empty:
        raise DatasetError(f"No observations remain inside {first_year}–{last_year}.")
    return frame, audit


def data_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize annual panel coverage and observed crisis labels."""
    aggregations: dict[str, tuple[str, str]] = {"country_years": ("ISO3", "nunique")}
    for col in (*RAW_FEATURES, *CRISIS_TARGETS):
        if col in frame:
            aggregations[f"{col}_observed"] = (col, "count")
    return frame.groupby("year", as_index=False).agg(**aggregations).sort_values("year")

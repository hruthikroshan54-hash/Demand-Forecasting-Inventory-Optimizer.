"""Data loading, validation and a synthetic sales generator.

The generator lets the project run end to end with zero downloads. To use real
data (e.g. Kaggle Walmart / Rossmann), reshape it to the columns
``date, sku, units`` (optional: ``promo``) and upload it in the app.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_sales.csv"

SKU_PROFILES = {
    "SKU-101 Instant Noodles": {"base": 120, "trend": 0.00025, "yearly": 0.10, "phase": 0.5, "weekend": 1.25},
    "SKU-102 Cold Drink 1L": {"base": 90, "trend": 0.00030, "yearly": 0.45, "phase": -1.2, "weekend": 1.35},
    "SKU-103 Basmati Rice 5kg": {"base": 45, "trend": 0.00010, "yearly": 0.15, "phase": 2.0, "weekend": 1.10},
    "SKU-104 Detergent 2kg": {"base": 60, "trend": 0.00015, "yearly": 0.08, "phase": 1.0, "weekend": 1.05},
    "SKU-105 Cooking Oil 1L": {"base": 75, "trend": 0.00020, "yearly": 0.12, "phase": 0.0, "weekend": 1.20},
}

REQUIRED_COLUMNS = ["date", "sku", "units"]


def generate_sample_data(start: str = "2023-01-01", end: str = "2025-06-30", seed: int = 42) -> pd.DataFrame:
    """Daily unit sales with trend, weekly + yearly seasonality, promotions and noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    t = np.arange(n)
    doy = dates.dayofyear.to_numpy()
    dow = dates.dayofweek.to_numpy()

    frames = []
    for sku, p in SKU_PROFILES.items():
        weekday = np.array([0.95, 0.95, 0.97, 1.0, 1.1, p["weekend"], p["weekend"] * 0.95])[dow]
        seasonal = 1 + p["yearly"] * np.sin(2 * np.pi * doy / 365.25 + p["phase"])
        trend = 1 + p["trend"] * t

        promo = np.zeros(n)
        for s in rng.choice(n - 7, size=n // 45, replace=False):
            promo[s : s + 5] = 1

        mean = p["base"] * trend * seasonal * weekday * (1 + 0.35 * promo)
        units = np.maximum(0, rng.normal(mean, mean * 0.12)).round()
        frames.append(pd.DataFrame({"date": dates, "sku": sku, "units": units.astype(int), "promo": promo.astype(int)}))

    return pd.concat(frames, ignore_index=True)


def validate_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """Check columns/types and return a clean copy. Raises ValueError with a readable message."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}. Expected: date, sku, units (optional: promo).")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["units"] = pd.to_numeric(df["units"], errors="coerce")
    bad = df["date"].isna() | df["units"].isna()
    if bad.all():
        raise ValueError("No valid rows: check the date format and that 'units' is numeric.")
    df = df[~bad].copy()
    df["units"] = df["units"].clip(lower=0)
    df["sku"] = df["sku"].astype(str)
    if "promo" in df.columns:
        df["promo"] = pd.to_numeric(df["promo"], errors="coerce").fillna(0).clip(0, 1).astype(int)
    return df.sort_values(["sku", "date"]).reset_index(drop=True)


def load_sample_data() -> pd.DataFrame:
    if SAMPLE_PATH.exists():
        return validate_sales_data(pd.read_csv(SAMPLE_PATH))
    return validate_sales_data(generate_sample_data())


if __name__ == "__main__":
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    generate_sample_data().to_csv(SAMPLE_PATH, index=False)
    print(f"Wrote {SAMPLE_PATH}")

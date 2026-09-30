"""Demand forecasting: gradient boosting with lag features vs simple baselines."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

FEATURES = ["dow", "month", "dayofyear", "weekofyear", "promo",
            "lag_7", "lag_14", "lag_28", "roll_7", "roll_28"]
MIN_TRAIN_DAYS = 120


def prepare_sku_frame(df: pd.DataFrame, sku: str) -> pd.DataFrame:
    """One SKU -> continuous daily frame (date, units, promo); missing days become 0 sales."""
    d = df[df["sku"] == sku].copy()
    if "promo" not in d.columns:
        d["promo"] = 0
    d = d.groupby("date", as_index=False).agg(units=("units", "sum"), promo=("promo", "max"))
    full = pd.date_range(d["date"].min(), d["date"].max(), freq="D")
    d = d.set_index("date").reindex(full).rename_axis("date")
    d[["units", "promo"]] = d[["units", "promo"]].fillna(0)
    return d.reset_index()


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.reset_index(drop=True)
    dates = pd.DatetimeIndex(frame["date"])
    u = frame["units"].astype(float)
    f = pd.DataFrame({
        "dow": dates.dayofweek,
        "month": dates.month,
        "dayofyear": dates.dayofyear,
        "weekofyear": dates.isocalendar().week.astype(int).to_numpy(),
        "promo": frame["promo"].to_numpy(),
        "lag_7": u.shift(7),
        "lag_14": u.shift(14),
        "lag_28": u.shift(28),
        "roll_7": u.shift(1).rolling(7).mean(),
        "roll_28": u.shift(1).rolling(28).mean(),
        "y": u,
    })
    return f.dropna()


def fit_model(frame: pd.DataFrame) -> HistGradientBoostingRegressor:
    feats = build_features(frame)
    model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, max_depth=5, random_state=0)
    model.fit(feats[FEATURES], feats["y"])
    return model


def recursive_forecast(model, frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Forecast day by day, feeding each prediction back in as history. Future promos assumed off."""
    hist = list(frame["units"].astype(float))
    dates = pd.date_range(frame["date"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    preds = []
    for d in dates:
        row = {
            "dow": d.dayofweek, "month": d.month, "dayofyear": d.dayofyear,
            "weekofyear": int(d.isocalendar().week), "promo": 0.0,
            "lag_7": hist[-7], "lag_14": hist[-14], "lag_28": hist[-28],
            "roll_7": float(np.mean(hist[-7:])), "roll_28": float(np.mean(hist[-28:])),
        }
        p = max(0.0, float(model.predict(pd.DataFrame([row])[FEATURES])[0]))
        preds.append(p)
        hist.append(p)
    return pd.DataFrame({"date": dates, "forecast": preds})


def seasonal_naive(frame: pd.DataFrame, horizon: int) -> np.ndarray:
    last7 = frame["units"].astype(float).to_numpy()[-7:]
    return np.tile(last7, int(np.ceil(horizon / 7)))[:horizon]


def moving_average(frame: pd.DataFrame, horizon: int, window: int = 28) -> np.ndarray:
    return np.full(horizon, frame["units"].astype(float).to_numpy()[-window:].mean())


def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    err = actual - pred
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "WAPE %": float(np.abs(err).sum() / max(actual.sum(), 1e-9) * 100),
    }


def backtest(frame: pd.DataFrame, horizon: int = 28, max_folds: int = 3) -> dict:
    """Rolling-origin backtest. Returns metrics per model, predictions, and forecast-error sigma."""
    frame = frame.reset_index(drop=True)
    folds = min(max_folds, (len(frame) - MIN_TRAIN_DAYS) // horizon)
    if folds < 1:
        raise ValueError(
            f"Not enough history: need at least {MIN_TRAIN_DAYS + horizon} days for a {horizon}-day horizon, "
            f"found {len(frame)}. Reduce the horizon or upload more data."
        )

    actuals, preds = {"gbm": [], "seasonal_naive": [], "moving_avg": []}, []
    for k in range(folds, 0, -1):
        end = len(frame) - (k - 1) * horizon
        start = end - horizon
        train, test = frame.iloc[:start], frame.iloc[start:end]
        model = fit_model(train)
        fc = recursive_forecast(model, train, horizon)["forecast"].to_numpy()
        sn, ma = seasonal_naive(train, horizon), moving_average(train, horizon)
        a = test["units"].astype(float).to_numpy()
        for name, p in (("gbm", fc), ("seasonal_naive", sn), ("moving_avg", ma)):
            actuals[name].append((a, p))
        preds.append(pd.DataFrame({"date": test["date"].to_numpy(), "actual": a, "gbm": fc,
                                   "seasonal_naive": sn, "moving_avg": ma, "fold": folds - k + 1}))

    labels = {"gbm": "Gradient Boosting (ML)", "seasonal_naive": "Seasonal Naive", "moving_avg": "28-day Moving Avg"}
    rows = []
    for name, pairs in actuals.items():
        a = np.concatenate([x for x, _ in pairs])
        p = np.concatenate([y for _, y in pairs])
        rows.append({"Model": labels[name], **_metrics(a, p)})

    metrics = pd.DataFrame(rows).sort_values("WAPE %").reset_index(drop=True)
    gbm_row = metrics[metrics["Model"] == labels["gbm"]].iloc[0]
    return {"metrics": metrics, "predictions": pd.concat(preds, ignore_index=True),
            "sigma": float(gbm_row["RMSE"]), "folds": folds}


def forecast_future(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Fit on all available history and forecast the next `horizon` days."""
    frame = frame.reset_index(drop=True)
    if len(frame) < MIN_TRAIN_DAYS:
        raise ValueError(f"Need at least {MIN_TRAIN_DAYS} days of history, found {len(frame)}.")
    return recursive_forecast(fit_model(frame), frame, horizon)

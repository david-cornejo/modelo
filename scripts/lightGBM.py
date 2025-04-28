#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

# 1) Load & aggregate to monthly
def load_series(path, date_col="Fecha", value_col="Cargos"):
    df = (
        pd.read_csv(path, parse_dates=[date_col], dayfirst=False)
          .loc[lambda d: d[value_col] > 0, [date_col, value_col]]
    )
    df.set_index(date_col, inplace=True)
    # resample to month-end
    monthly = df[value_col].resample("M").sum()
    return monthly

# 2) Smooth outliers via IQR + rolling median
def smooth_outliers(s, q_low=15, q_high=85, factor=1.5, window=5):
    q1, q3 = np.percentile(s, [q_low, q_high])
    iqr = q3 - q1
    low, up = q1 - factor * iqr, q3 + factor * iqr
    med = s.rolling(window, center=True, min_periods=1).median()
    out = s.copy()
    mask = (s < low) | (s > up)
    out[mask] = med[mask]
    return out

# 3) Feature engineering: lags, rolling, month, Fourier
def make_features(s):
    df = pd.DataFrame({"y": s})
    df["month"] = df.index.month
    # lag features
    for lag in range(1, 13):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    # rolling statistics
    for w in (3, 6, 12):
        df[f"roll_mean_{w}"] = df["y"].shift(1).rolling(window=w).mean()
        df[f"roll_std_{w}"] = df["y"].shift(1).rolling(window=w).std()
    # annual seasonality via Fourier
    period = 12
    idx = np.arange(len(df))
    for k in range(1, 4):
        df[f"sin_{k}"] = np.sin(2 * np.pi * k * idx / period)
        df[f"cos_{k}"] = np.cos(2 * np.pi * k * idx / period)
    df = df.dropna()
    return df

# 4) Split into train/test
def train_test_split(df, train_size=0.85):
    n = int(len(df) * train_size)
    train = df.iloc[:n]
    test  = df.iloc[n:]
    return (
        train.drop(columns="y"), train["y"],
        test.drop(columns="y"),  test["y"]
    )

# 5) Optuna objective for LGBM
def objective(trial, X, y):
    params = {
        "objective": "regression",
        "metric": "mape",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
    }
    tscv = TimeSeriesSplit(n_splits=3)
    mape_scores = []
    for tr_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model = lgb.LGBMRegressor(**params, n_estimators=1000)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)
            ]
        )
        y_pred = model.predict(X_val, num_iteration=model.best_iteration_)
        mape_scores.append(mean_absolute_percentage_error(y_val, y_pred))
    return np.mean(mape_scores)

def main():
    DATA_PATH = "../data/processed/merged/ventas_2015-2024.csv"

    # load & smooth
    series = load_series(DATA_PATH)
    series_smooth = smooth_outliers(series)

    # build features + target
    df = make_features(series_smooth)

    # split
    X_tr, y_tr, X_te, y_te = train_test_split(df, train_size=0.85)

    # scale numeric features
    scaler = StandardScaler()
    X_tr = pd.DataFrame(
        scaler.fit_transform(X_tr),
        index=X_tr.index, columns=X_tr.columns
    )
    X_te = pd.DataFrame(
        scaler.transform(X_te),
        index=X_te.index, columns=X_te.columns
    )

    # hyperparameter tuning
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_tr, y_tr),
                   n_trials=50, show_progress_bar=True)
    print("Best params:", study.best_params)

    # final train with best params
    best_params = study.best_params.copy()
    best_params.update({
        "objective": "regression",
        "metric": "mape",
        "verbosity": -1,
        "boosting_type": "gbdt"
    })
    model = lgb.LGBMRegressor(**best_params, n_estimators=1000)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0)
        ]
    )

    # predict on test
    y_pred = model.predict(X_te, num_iteration=model.best_iteration_)
    mape = mean_absolute_percentage_error(y_te, y_pred) * 100
    print(f"\nTest MAPE: {mape:.2f}%")

    # plot results
    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series.values, color="0.7", label="Serie completa")
    plt.plot(y_te.index, y_te, color="black", label="Test real")
    plt.plot(y_te.index, y_pred, "--", color="green", label="LGBM predicho")
    plt.xlabel("Fecha")
    plt.ylabel("Ventas")
    plt.title("Pronóstico Mensual con LightGBM (Optuna)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
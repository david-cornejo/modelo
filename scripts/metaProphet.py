#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
from tqdm import tqdm

# ───────────────────────── Parámetros ─────────────────────────
DATA_PATH     = "../data/processed/merged/ventas_2015-2024.csv"
TEST_HORIZON  = 18     # últimos meses para test
N_ITER        = 100    # cuántos modelos probar
RANDOM_SEED   = 42
# ──────────────────────────────────────────────────────────────

def load_and_resample(path):
    df = (
        pd.read_csv(path, parse_dates=["Fecha"], index_col="Fecha")
          .assign(Cargos=lambda d: pd.to_numeric(d["Cargos"], errors="coerce"))
          .dropna(subset=["Cargos"])
    )
    return df["Cargos"].resample("ME").sum()

def smooth_outliers_mean(s, window=5):
    q1, q3 = np.percentile(s, [15, 85])
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr

    # media móvil sobre toda la serie (min_periods=1 para evitar NaN)
    mean_rolling = s.rolling(window=window, center=True, min_periods=1).mean()

    s_smooth = s.copy()
    is_outlier = (s < lower) | (s > upper)
    s_smooth.loc[is_outlier] = mean_rolling.loc[is_outlier]
    return s_smooth

# 1) Cargo la serie y suavizo outliers, igual que en Holt-Winters
serie = load_and_resample(DATA_PATH)
serie_s = smooth_outliers_mean(serie)

# 2) Preparo DataFrame para Prophet
df_prop = (
    serie_s.reset_index()
           .rename(columns={"Fecha":"ds", "Cargos":"y"})
)

# 3) Split train/test
train_df = df_prop.iloc[:-TEST_HORIZON].copy()
test_df  = df_prop.iloc[-TEST_HORIZON:].copy()
y_true   = test_df["y"].values

# 4) Defino distribuciones de hiper-parámetros
def sample_params():
    return {
        # escala de ruptura de tendencia: 0.001 → 1
        "changepoint_prior_scale": 10**np.random.uniform(-3, 0),
        # escala de estacionalidad: 0.1 → 20
        "seasonality_prior_scale":  np.random.uniform(0.1, 20),
        # aditivo vs multiplicativo
        "seasonality_mode":         random.choice(["additive", "multiplicative"]),
        # Fourier orders para la estacionalidad mensual: 3 → 20
        "fourier_order":            random.randint(3, 20),
        # incluir o no estacionalidad semanal
        "weekly_seasonality":       random.choice([True, False]),
    }

best_mape   = np.inf
best_params = None

# 5) Búsqueda aleatoria
for i in tqdm(range(N_ITER), desc="Random Search"):
    params = sample_params()
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=params["weekly_seasonality"],
        daily_seasonality=False,
        changepoint_prior_scale=params["changepoint_prior_scale"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
        seasonality_mode=params["seasonality_mode"],
        interval_width=0.9,
    )
    # añado estacionalidad mensual con orden variable
    m.add_seasonality(
        name="monthly",
        period=30.5,
        fourier_order=params["fourier_order"]
    )
    # entrena
    m.fit(train_df)
    # forecast
    future = m.make_future_dataframe(periods=TEST_HORIZON, freq="M")
    fcst   = m.predict(future)
    y_pred = fcst.set_index("ds")["yhat"].loc[test_df["ds"]].values
    mape   = mean_absolute_percentage_error(y_true, y_pred)*100
    mae = np.mean(np.abs(y_true - y_pred))

    if mape < best_mape:
        best_mape, best_params = mape, params

print(f"\n>>> Mejor configuración tras {N_ITER} iteraciones:")
print(f"    MAPE = {best_mape:.2f}%")
print(f"    MAE  = {mae:.2f}")
print("    Parámetros:", best_params)

# 6) Entreno el modelo final con la mejor configuración
m_final = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=best_params["weekly_seasonality"],
    daily_seasonality=False,
    changepoint_prior_scale=best_params["changepoint_prior_scale"],
    seasonality_prior_scale=best_params["seasonality_prior_scale"],
    seasonality_mode=best_params["seasonality_mode"],
    interval_width=0.9,
)
m_final.add_seasonality(
    name="monthly",
    period=30.5,
    fourier_order=best_params["fourier_order"]
)
m_final.fit(train_df)

# 7) Forecast final
future = m_final.make_future_dataframe(periods=TEST_HORIZON, freq="M")
fcst   = m_final.predict(future)
y_pred = fcst.set_index("ds")["yhat"].loc[test_df["ds"]].values
final_mape = mean_absolute_percentage_error(y_true, y_pred)*100

# 8) Gráfica
plt.figure(figsize=(12,5))
plt.plot(df_prop["ds"], df_prop["y"],      color="gray", alpha=0.3, label="Real (suavizado)")
plt.plot(train_df["ds"], train_df["y"],    color="C0", label="Train")
plt.plot(test_df["ds"],  test_df["y"],     color="black", label="Test")
plt.plot(test_df["ds"],  y_pred, "--",     color="C2", label="Prophet pred")
plt.title(f"Prophet RandomSearch — MAPE={final_mape:.1f}% | MAE={mae:.1f}")
plt.xlabel("Fecha"); plt.ylabel("Ventas mensuales")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()
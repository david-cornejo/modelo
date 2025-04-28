#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm
from sklearn.metrics import mean_absolute_percentage_error

DATA_PATH    = "../data/processed/merged/ventas_2015-2024.csv"
TEST_HORIZON = 18
SEASONAL_M   = 12

def load_series(path):
    df = (
        pd.read_csv(path, parse_dates=["Fecha"], index_col="Fecha")
          .assign(Cargos=lambda d: pd.to_numeric(d["Cargos"], errors="coerce"))
          .dropna(subset=["Cargos"])
    )
    # Agrega mensual
    return df["Cargos"].resample("M").sum()

def smooth_outliers(s, q_low=15, q_high=85, factor=1.5, window=5):
    q1, q3 = np.percentile(s, [q_low, q_high])
    iqr = q3 - q1
    low, up = q1 - factor*iqr, q3 + factor*iqr
    med = s.rolling(window, center=True, min_periods=1).median()
    out = s.copy()
    mask = (s<low)|(s>up)
    out[mask] = med[mask]
    return out

# 1) Cargo y suavizo outliers
serie = load_series(DATA_PATH)
serie_s = smooth_outliers(serie)

# 2) split train/test
train = serie_s.iloc[:-TEST_HORIZON]
test  = serie_s.iloc[-TEST_HORIZON:]

# 3) Ajuste auto_arima FORZANDO AR (start_p/P>=1)
print("🔍 Ajustando SARIMA con búsqueda de órdenes que incluyan AR...")
sarima = pm.auto_arima(
    train,
    seasonal=True, m=SEASONAL_M,
    start_p=1, max_p=5,
    start_P=1, max_P=2,
    d=None, D=None,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    trace=True
)
print("\n▶ Modelo seleccionado:", sarima.summary())

# 4) Rolling one-step forecasts
history = train.copy().tolist()
preds = []
idxs  = test.index

for t in range(TEST_HORIZON):
    # Cada vez ajusto con todo lo observado hasta ahora
    sar_mod = pm.ARIMA(order=sarima.order,
                       seasonal_order=sarima.seasonal_order
                      ).fit(history, suppress_warnings=True)
    # pronostico 1 paso
    yhat = sar_mod.predict(n_periods=1)[0]
    preds.append(yhat)
    # meto el valor real para el siguiente loop
    history.append(test.iloc[t])

# 5) Métrica
mape = mean_absolute_percentage_error(test, preds)*100
print(f"\n📊 MAPE rolling SARIMA = {mape:.2f}%")

# 6) Gráfica
plt.figure(figsize=(12,5))
plt.plot(serie_s,           color="gray", alpha=0.3, label="Real (suavizado)")
plt.plot(train.index, train, color="C0",       label="Train")
plt.plot(test.index,  test,  color="black",    label="Test")
plt.plot(idxs, preds, "--", color="C2",        label="SARIMA rolling pred")
plt.title(f"Rolling SARIMA{sarima.order}x{sarima.seasonal_order} — MAPE={mape:.1f}%")
plt.xlabel("Fecha"); plt.ylabel("Ventas mensuales")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()
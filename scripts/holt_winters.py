import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# 1) Carga de la serie diaria y agregación semanal
serie_diaria = (
    pd.read_csv("../data/processed/merged/ventas_2015-2024.csv",
                parse_dates=["Fecha"], index_col="Fecha")
      ["Cargos"]
)

# Agregar los datos a nivel semanal (suma de cargos por semana)
serie = serie_diaria.resample("M").sum()
# 2) Suavizado de todos los outliers usando IQR + mediana móvil
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

serie_smooth = smooth_outliers_mean(serie, window=5)

# 3) Split train/test (80% train)
n = int(len(serie_smooth) * 0.85)
train, test = serie_smooth.iloc[:n], serie_smooth.iloc[n:]

# 4) Ajuste Holt–Winters sobre serie suavizada
modelo = ExponentialSmoothing(
    train,
    trend="add",
    seasonal="add",
    seasonal_periods=32  # 52 semanas en un año
).fit()

pred = modelo.forecast(len(test))

# 5) Métricas
mae  = mean_absolute_error(test, pred)
mape = mean_absolute_percentage_error(test, pred) * 100
print(f"Holt-Winters -> MAE: {mae:.2f}, MAPE: {mape:.2f}%")

# 6) Comparar reales vs predichos
print("\nValores reales vs predichos:")
for fecha, (real, p) in zip(test.index, zip(test, pred)):
    print(f"{fecha.date()} -> Real: {real:.2f}, Predicho: {p:.2f}")

# 7) Gráfica
plt.figure(figsize=(12,6))
plt.plot(train, label="Train (suavizado)")
plt.plot(serie.loc[train.index[-1]:], label="Original Test", alpha=0.3)
plt.plot(test.index, test, label="Test (suavizado)", color="black")
plt.plot(test.index, pred, label="HW Forecast", linestyle="--")
plt.legend()
plt.title("Pronóstico Holt–Winters sobre Serie Suavizada (Agregación Semanal)")
plt.grid(True)
plt.show()
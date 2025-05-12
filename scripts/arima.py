import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# -------------------------------
# 1. Cargar y preparar datos (agregación mensual igual a SARIMA)
# -------------------------------
def load_monthly_data(path):
    df = (
        pd.read_csv(path, parse_dates=["Fecha"], index_col="Fecha")
          .assign(Cargos=lambda d: pd.to_numeric(d["Cargos"], errors="coerce"))
          .dropna(subset=["Cargos"])
    )
    monthly = df["Cargos"].resample("M").sum()
    return monthly

# Función para suavizar outliers (similar a la de SARIMA)
def smooth_outliers(s, window=5):
    q1, q3 = np.percentile(s, [15, 85])
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr

    # media móvil sobre toda la serie (min_periods=1 para evitar NaN)
    mean_rolling = s.rolling(window=window, center=True, min_periods=1).mean()

    s_smooth = s.copy()
    is_outlier = (s < lower) | (s > upper)
    s_smooth.loc[is_outlier] = mean_rolling.loc[is_outlier]
    return s_smooth

# -------------------------------
# 2. División de datos
# -------------------------------
def train_test_split_series(series, test_weeks=12):
    train = series[:-test_weeks]
    test = series[-test_weeks:]
    return train, test

# -------------------------------
# 3. Ajustar ARIMA
# -------------------------------
def fit_arima_model(train, order=(1,1,1)):
    model = ARIMA(train, order=order)
    model_fit = model.fit()
    return model_fit

# -------------------------------
# 4. Predicción y evaluación
# -------------------------------
def evaluate_predictions(test, forecast):
    mae = mean_absolute_error(test, forecast)
    mape = mean_absolute_percentage_error(test, forecast) * 100
    print(f"MAE: {mae:.2f}, MAPE: {mape:.2f}%")
    return mae, mape

# -------------------------------
# 5. Visualización
# -------------------------------
def plot_forecast(train, test, forecast):
    plt.figure(figsize=(12, 6))
    plt.plot(train, label="Train")
    plt.plot(test, label="Test")
    plt.plot(test.index, forecast, label="ARIMA Forecast", linestyle="--", color="green")
    plt.title("Pronóstico ARIMA")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -------------------------------
# 6. Main
# -------------------------------
def main():
    path = "../data/processed/merged/ventas_2015-2024.csv"
    # Uso de la agregación mensual similar a SARIMA
    series = load_monthly_data(path)

    # Aplicar suavizado de outliers
    series = smooth_outliers(series)
    
    train, test = train_test_split_series(series, test_weeks=18)
    
    print(f"Datos totales: {len(series)}, Train: {len(train)}, Test: {len(test)}")
    
    model_fit = fit_arima_model(train, order=(1,1,1))
    forecast = model_fit.forecast(steps=len(test))
    
    evaluate_predictions(test, forecast)
    plot_forecast(train, test, forecast)

if __name__ == "__main__":
    main()
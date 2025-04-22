import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# -------------------------------
# 1. Cargar y preparar datos
# -------------------------------
def load_weekly_data(path):
    df = pd.read_csv(path)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df.set_index("Fecha", inplace=True)
    weekly = df["Cargos"].resample("W").sum().fillna(0)
    return weekly

# -------------------------------
# 2. División de datos
# -------------------------------
def train_test_split_series(series, test_weeks=52):
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

# -------------------------------∫
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
    plt.figure(figsize=(12,6))
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
    path = "../data/processed/ventas_2015-2020.csv"
    series = load_weekly_data(path)
    train, test = train_test_split_series(series, test_weeks=52)
    
    print(f"Datos totales: {len(series)}, Train: {len(train)}, Test: {len(test)}")
    
    model_fit = fit_arima_model(train, order=(1,1,1))
    forecast = model_fit.forecast(steps=len(test))
    
    evaluate_predictions(test, forecast)
    plot_forecast(train, test, forecast)

if __name__ == "__main__":
    main()
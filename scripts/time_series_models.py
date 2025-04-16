import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_absolute_percentage_error

from pmdarima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings("ignore")

# ------------------------------
# 1. Carga y limpieza de datos
# ------------------------------
def load_clean_data(path):
    df = pd.read_csv(path)
    df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")
    df["Cargos"] = pd.to_numeric(df["Cargos"], errors="coerce").fillna(0)
    df = df[df["Cargos"] > 0]  # Eliminar valores cero o negativos
    return df

def remove_outliers(series, q=0.99):
    upper = series.quantile(q)
    return series.clip(upper=upper)

# ------------------------------
# 2. Agregación semanal
# ------------------------------
def aggregate_weekly(df):
    df = df.set_index("Fecha").sort_index()
    weekly_cargos = df["Cargos"].resample("W").sum()
    weekly_cargos.index.freq = "W"
    return weekly_cargos

# ------------------------------
# 3. Transformaciones log
# ------------------------------
def apply_log(series):
    return np.log1p(series)

def inverse_log(series):
    return np.expm1(series)

# ------------------------------
# 4. División Train/Test
# ------------------------------
def train_test_split_time_series(series, split_date):
    train = series.loc[:split_date].copy()
    test = series.loc[split_date:].copy()
    return train, test

# ------------------------------
# 5. Análisis de estacionalidad
# ------------------------------
def analyze_seasonality(series, lags=60):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(series, lags=lags, ax=axes[0])
    plot_pacf(series, lags=lags, ax=axes[1])
    axes[0].set_title("ACF")
    axes[1].set_title("PACF")
    plt.tight_layout()
    plt.show()

# ------------------------------
# 6. Modelos ARIMA y SARIMA
# ------------------------------
def run_arima_model(train, test):
    model_arima = auto_arima(
        train,
        seasonal=False,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,
        max_p=3,
        max_d=2,
        max_q=3
    )
    print("Auto ARIMA (sin estacionalidad) - Order:", model_arima.order)

    forecast_log = model_arima.predict(n_periods=len(test))
    forecast = inverse_log(forecast_log)
    test_orig = inverse_log(test)

    forecast = pd.Series(forecast, index=test.index).dropna()
    test_aligned = test_orig.loc[forecast.index].dropna()

    mae_val = np.mean(np.abs(test_aligned - forecast))
    mape_val = mean_absolute_percentage_error(test_aligned, forecast) * 100
    print(f"ARIMA => MAE: {mae_val:.2f}, MAPE: {mape_val:.2f}%")

    return model_arima, forecast

def run_sarima_model(train, test, m=26):
    model_sarima = auto_arima(
        train,
        seasonal=True,
        m=m,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,
        max_p=3,
        max_d=2,
        max_q=3,
        max_P=2,
        max_D=2,
        max_Q=2
    )
    print(f"Auto SARIMA con m={m} - Order: {model_sarima.order}, Seasonal: {model_sarima.seasonal_order}")

    forecast_log = model_sarima.predict(n_periods=len(test))
    forecast = inverse_log(forecast_log)
    test_orig = inverse_log(test)

    forecast = pd.Series(forecast, index=test.index).dropna()
    test_aligned = test_orig.loc[forecast.index].dropna()

    mae_val = np.mean(np.abs(test_aligned - forecast))
    mape_val = mean_absolute_percentage_error(test_aligned, forecast) * 100
    print(f"SARIMA (m={m}) => MAE: {mae_val:.2f}, MAPE: {mape_val:.2f}%")

    return model_sarima, forecast

def run_sarima_multiple_m(train, test, m_values=[26, 52]):
    results = {}
    test_orig = inverse_log(test)
    for m in m_values:
        print(f"\n--- Ajuste SARIMA con m={m} ---")
        model, forecast_series = run_sarima_model(train, test, m=m)

        forecast_series = forecast_series.dropna()
        test_aligned = test_orig.loc[forecast_series.index].dropna()

        mae_val = np.mean(np.abs(test_aligned - forecast_series))
        mape_val = mean_absolute_percentage_error(test_aligned, forecast_series) * 100
        results[m] = {
            "model": model,
            "forecast": forecast_series,
            "mae": mae_val,
            "mape": mape_val
        }
    return results

# ------------------------------
# 7. main()
# ------------------------------
def main():
    data_path = "../data/processed/cargos_2015-2017.csv"
    df = load_clean_data(data_path)
    df["Cargos"] = remove_outliers(df["Cargos"], q=0.99)

    weekly_cargos = aggregate_weekly(df)

    plt.figure(figsize=(12, 4))
    plt.plot(weekly_cargos.index, weekly_cargos, label="Cargos semanales (original)")
    plt.title("Serie original antes de log")
    plt.grid(True)
    plt.legend()
    plt.show()

    weekly_cargos_log = apply_log(weekly_cargos)
    analyze_seasonality(weekly_cargos_log, lags=60)

    split_date = "2017-01-01"
    train, test = train_test_split_time_series(weekly_cargos_log, split_date)
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)

    print("\n--- Ajuste ARIMA (sin estacionalidad) ---")
    arima_model, arima_forecast = run_arima_model(train, test)

    m_candidates = [26, 52]
    sarima_results = run_sarima_multiple_m(train, test, m_values=m_candidates)

    test_original = inverse_log(test)

    plt.figure(figsize=(12, 6))
    plt.plot(train.index, inverse_log(train), label="Train Cargos", color="blue")
    plt.plot(test.index, test_original, label="Test Cargos", color="black")
    plt.plot(arima_forecast.index, arima_forecast, label="ARIMA Forecast", color="red", linewidth=2)

    for m, res in sarima_results.items():
        plt.plot(res["forecast"].index, res["forecast"], label=f"SARIMA (m={m})", linewidth=2)

    plt.title("Comparación de ARIMA y SARIMA con distintos m")
    plt.xlabel("Fecha (semanas)")
    plt.ylabel("Cargos Semanales")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nResumen de métricas SARIMA:")
    for m, vals in sarima_results.items():
        print(f" m={m}: MAE={vals['mae']:.2f}, MAPE={vals['mape']:.2f}%")

    print("\n--- Fin del script ---")

if __name__ == "__main__":
    main()

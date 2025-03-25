"""
time_series_models.py

Incluye:
1) Carga y limpieza de datos
2) Agregación de datos a nivel semanal
3) Transformación logarítmica (log1p) y su inversa (expm1)
4) División en train/test
5) Función para analizar la estacionalidad (ACF y PACF)
6) Modelos ARIMA y SARIMA (auto_arima)
7) Comparación con distintos valores de m para SARIMA
8) Visualización con propiedades para distinguir líneas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# Para auto_arima y ACF/PACF
from pmdarima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings("ignore")

# ------------------------------
# 1. Carga y limpieza de datos
# ------------------------------
def load_clean_data(path):
    df = pd.read_csv(path)
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)
    return df

def remove_outliers(series, q=0.99):
    upper = series.quantile(q)
    return series.clip(upper=upper)

# ------------------------------
# 2. Agregación semanal
# ------------------------------
def aggregate_weekly(df):
    df = df.set_index("Order Date").sort_index()
    weekly_sales = df["Sales"].resample("W").sum()
    # Forzamos la frecuencia semanal explícitamente
    weekly_sales.index.freq = "W"
    return weekly_sales

# ------------------------------
# 3. Transformaciones log
# ------------------------------
def apply_log(series):
    # Usamos log1p para evitar problemas con ceros
    return np.log1p(series)

def inverse_log(series):
    # Inversa de log1p
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
    """
    Genera los gráficos de ACF y PACF para 'lags' retardos.
    Útil para detectar picos y periodos estacionales.
    """
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
    """
    Ajusta un modelo ARIMA (sin estacionalidad) usando auto_arima.
    """
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
    
    # Pronosticamos
    forecast_log = model_arima.predict(n_periods=len(test))
    forecast = inverse_log(forecast_log)
    test_orig = inverse_log(test)
    
    # Métricas
    mae_val = np.mean(np.abs(test_orig - forecast))
    mape_val = mean_absolute_percentage_error(test_orig, forecast) * 100
    print(f"ARIMA => MAE: {mae_val:.2f}, MAPE: {mape_val:.2f}%")
    
    return model_arima, pd.Series(forecast, index=test.index)

def run_sarima_model(train, test, m=26):
    """
    Ajusta un modelo SARIMA (con estacionalidad) usando auto_arima,
    para un m específico.
    """
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
    
    # Pronosticamos
    forecast_log = model_sarima.predict(n_periods=len(test))
    forecast = inverse_log(forecast_log)
    test_orig = inverse_log(test)
    
    # Métricas
    mae_val = np.mean(np.abs(test_orig - forecast))
    mape_val = mean_absolute_percentage_error(test_orig, forecast) * 100
    print(f"SARIMA (m={m}) => MAE: {mae_val:.2f}, MAPE: {mape_val:.2f}%")
    
    return model_sarima, pd.Series(forecast, index=test.index)

def run_sarima_multiple_m(train, test, m_values=[26, 52]):
    """
    Prueba varios valores de m para detectar si hay alguna estacionalidad
    clara que mejore el modelo. Devuelve un dict con pronósticos y métricas.
    """
    results = {}
    test_orig = inverse_log(test)
    for m in m_values:
        print(f"\n--- Ajuste SARIMA con m={m} ---")
        model, forecast_series = run_sarima_model(train, test, m=m)
        
        mae_val = np.mean(np.abs(test_orig - forecast_series))
        mape_val = mean_absolute_percentage_error(test_orig, forecast_series) * 100
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
    # Ajusta la ruta de tu CSV
    data_path = "../data/processed/walmart_cleaned.csv"
    df = load_clean_data(data_path)
    df["Sales"] = remove_outliers(df["Sales"], q=0.99)
    
    # Agregación semanal
    weekly_sales = aggregate_weekly(df)
    
    # Visualiza la serie original (opcional)
    plt.figure(figsize=(12,4))
    plt.plot(weekly_sales.index, weekly_sales, label="Ventas semanales (original)")
    plt.title("Serie original antes de log")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Análisis de estacionalidad con ACF/PACF
    print("Analizando estacionalidad en la serie original (o transformada).")
    # Se recomienda analizar en log si hay mucha varianza:
    weekly_sales_log = apply_log(weekly_sales)
    analyze_seasonality(weekly_sales_log, lags=60)  # Ajusta lags según necesites
    
    # División en train/test
    split_date = "2015-01-01"
    train, test = train_test_split_time_series(weekly_sales_log, split_date)
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    
    # Modelo ARIMA (sin estacionalidad)
    print("\n--- Ajuste ARIMA (sin estacionalidad) ---")
    arima_model, arima_forecast = run_arima_model(train, test)
    
    # Prueba varios m para SARIMA
    m_candidates = [26, 52]  # Ajusta los valores según sospechas de estacionalidad
    sarima_results = run_sarima_multiple_m(train, test, m_values=m_candidates)
    
    # Visualización comparativa
    test_original = inverse_log(test)
    
    plt.figure(figsize=(12,6))
    plt.plot(train.index, inverse_log(train), label="Train Sales", color="blue")
    plt.plot(test.index, test_original, label="Test Sales", color="black")
    
    # ARIMA sin estacionalidad
    plt.plot(arima_forecast.index, arima_forecast, label="ARIMA Forecast", 
             color="red", alpha=0.7, zorder=5, linewidth=2)
    
    # Dibuja cada SARIMA con distinto m
    color_map = {26: "green", 52: "orange", 104: "purple"}  # Ejemplo de colores
    for m, res in sarima_results.items():
        forecast_series = res["forecast"]
        plt.plot(
            forecast_series.index, forecast_series, 
            label=f"SARIMA (m={m})", 
            color=color_map.get(m, "gray"), 
            alpha=0.9, zorder=10, linewidth=2
        )
    
    plt.title("Comparación de ARIMA y SARIMA con distintos m")
    plt.xlabel("Fecha (semanas)")
    plt.ylabel("Ventas Semanales")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Muestra un resumen de las métricas para cada m
    print("\nResumen de métricas SARIMA:")
    for m, vals in sarima_results.items():
        print(f" m={m}: MAE={vals['mae']:.2f}, MAPE={vals['mape']:.2f}%")
    
    print("\n--- Fin del script ---")
    print("Si ambos modelos devuelven una línea horizontal, significa que la "
          "serie no presenta estacionalidad o tendencia significativa para "
          "mejorar el criterio AIC/BIC frente a un modelo constante.")


if __name__ == "__main__":
    main()

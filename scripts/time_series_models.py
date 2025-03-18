"""
time_series_models.py

Script para:
1) Cargar el dataset limpio (data/processed/walmart_retail_data_limpio.csv)
2) Agregar las ventas a nivel diario
3) Crear sets de entrenamiento y prueba
4) Implementar varios modelos de pronóstico (Naive, Moving Average, ARIMA)
5) Comparar su desempeño (MAE, MAPE)

Requisitos:
  pip install pandas numpy matplotlib statsmodels pmdarima
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Para ARIMA
import statsmodels.api as sm
# (Opcional) Auto ARIMA
try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False


def load_clean_data(path):
    """
    Carga el CSV limpio con columnas:
    - 'Order Date' (parseable a datetime)
    - 'Sales'
    Retorna un DataFrame de pandas.
    """
    df = pd.read_csv(path, parse_dates=["Order Date"])
    return df


def aggregate_daily(df):
    """
    Agrupa las ventas a nivel diario, asegurándose de que
    no existan huecos de fecha.
    Retorna un DataFrame con índice fecha y columna 'Sales'.
    """
    # Sumar ventas por día
    daily = df.groupby(df["Order Date"].dt.date)["Sales"].sum().reset_index()
    daily["Order Date"] = pd.to_datetime(daily["Order Date"])

    # Crear rango de fechas completo
    start_date = daily["Order Date"].min()
    end_date = daily["Order Date"].max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    df_full = pd.DataFrame({"Order Date": date_range})

    # Merge para rellenar con 0 días sin venta
    daily_sales = df_full.merge(daily, on="Order Date", how="left")
    daily_sales["Sales"] = daily_sales["Sales"].fillna(0)

    # Ordenar y poner índice
    daily_sales.sort_values("Order Date", inplace=True)
    daily_sales.set_index("Order Date", inplace=True)
    return daily_sales


def train_test_split_time_series(df, split_date):
    """
    Separa el DataFrame (indexado por fecha) en train y test,
    usando 'split_date' como corte (string 'YYYY-MM-DD').
    Retorna (train, test).
    """
    train = df.loc[:split_date].copy()  # incluye hasta split_date
    test = df.loc[split_date:].copy()   # a partir de split_date
    return train, test


def mae_mape(y_true, y_pred):
    """
    Calcula MAE y MAPE entre series de pandas.
    Retorna (mae, mape).
    """
    mae_val = (y_true - y_pred).abs().mean()
    # Evitar división cero si hay valores 0
    # Reemplazar 0 con un valor muy pequeño (o ignorar)
    mape_val = ((y_true - y_pred).abs() / (y_true.replace(0, np.finfo(float).eps)).abs()).mean() * 100
    return mae_val, mape_val


def naive_forecast(train, test):
    """
    Modelo Naive: Forecast(t) = Actual(t-1).
    Maneja el 1er día de test usando la última venta de train.
    Retorna pd.Series con las predicciones para test.
    """
    # Crear copia de test
    predictions = test.copy()

    # Desplazamos 1 día las ventas en test
    predictions["Pred_naive"] = test["Sales"].shift(1)

    # Tomar la última venta de train para el primer día de test
    last_train_day = train.index.max()
    last_train_sales = train.loc[last_train_day, "Sales"]
    first_test_day = test.index.min()
    predictions.loc[first_test_day, "Pred_naive"] = last_train_sales

    return predictions["Pred_naive"]


def moving_average_forecast(train, test, window=7):
    """
    Modelo de Media Móvil (ej. 7 días).
    Toma la venta promedio de los últimos 'window' días
    para predecir el día actual.
    Retorna pd.Series con las predicciones para test.
    """
    # Unimos train + test para calcular rolling
    combined = pd.concat([train["Sales"], test["Sales"]], axis=0)
    rolling_mean = combined.rolling(window=window).mean()

    # Extraemos sólo para fechas de test
    predictions = test.copy()
    predictions["Pred_ma"] = rolling_mean.loc[test.index]
    return predictions["Pred_ma"]


def arima_forecast(train, test, order=(1,1,1)):
    """
    Modelo ARIMA simple con statsmodels, usando order=(p,d,q).
    Retorna pd.Series con predicciones en el rango de test.
    """
    # Ajustar ARIMA en la serie de entrenamiento
    train_series = train["Sales"]
    model = sm.tsa.ARIMA(train_series, order=order)
    model_fit = model.fit()

    # Predicción para el rango de test
    start_date = test.index[0]
    end_date = test.index[-1]

    forecast = model_fit.predict(start=start_date, end=end_date, typ="levels")
    return forecast


def main():
    # 1. Ruta del dataset limpio
    data_path = os.path.join("data", "processed", "walmart_retail_data_limpio.csv")
    df = load_clean_data(data_path)

    # 2. Agregar ventas diarias
    daily_sales = aggregate_daily(df)
    print("Datos diarios:", daily_sales.shape, "Fechas de", daily_sales.index.min(), "a", daily_sales.index.max())

    # 3. Separar Train/Test
    # Ejemplo: Usamos todo hasta '2014-12-31' como Train, y '2015-01-01' en adelante como Test
    split_date = "2015-01-01"
    train, test = train_test_split_time_series(daily_sales, split_date)
    print("Entrenamiento:", train.shape, "Prueba:", test.shape)

    # Asegurar que no haya nulos en Sales
    train["Sales"] = train["Sales"].fillna(0)
    test["Sales"] = test["Sales"].fillna(0)

    # 4. Modelo Naive
    pred_naive = naive_forecast(train, test)
    mae_n, mape_n = mae_mape(test["Sales"], pred_naive)
    print(f"Naive Model -> MAE: {mae_n:.2f}, MAPE: {mape_n:.2f}%")

    # 5. Modelo de Media Móvil (7 días)
    pred_ma7 = moving_average_forecast(train, test, window=7)
    mae_ma, mape_ma = mae_mape(test["Sales"], pred_ma7)
    print(f"Moving Average(7) -> MAE: {mae_ma:.2f}, MAPE: {mape_ma:.2f}%")

    # 6. Modelo ARIMA(1,1,1) (ejemplo)
    pred_arima = arima_forecast(train, test, order=(1,1,1))
    # Alinear con test (por si faltan índices)
    pred_arima = pred_arima.reindex(test.index, method="ffill")
    mae_a, mape_a = mae_mape(test["Sales"], pred_arima)
    print(f"ARIMA(1,1,1) -> MAE: {mae_a:.2f}, MAPE: {mape_a:.2f}%")

    # 7. (Opcional) Auto-ARIMA
    if PMDARIMA_AVAILABLE:
        print("\nProbando auto_arima (pmdarima)...")
        from pmdarima import auto_arima
        stepwise_fit = auto_arima(train["Sales"], 
                                  start_p=1, start_q=1,
                                  max_p=5, max_q=5,
                                  seasonal=False,  # Cambiar a True si consideras estacionalidad (semanal, etc.)
                                  trace=True,
                                  error_action='ignore',
                                  suppress_warnings=True,
                                  stepwise=True)
        print(stepwise_fit.summary())

        # Predicción con auto_arima
        n_periods = len(test)
        forecast_auto = stepwise_fit.predict(n_periods=n_periods)
        pred_auto_arima = pd.Series(forecast_auto, index=test.index)

        mae_auto, mape_auto = mae_mape(test["Sales"], pred_auto_arima)
        print(f"Auto-ARIMA -> MAE: {mae_auto:.2f}, MAPE: {mape_auto:.2f}%")
    else:
        print("\nPMDARIMA no está instalado, omitiendo auto_arima test.")

    # 8. Visualización comparativa (opcional)
    plt.figure(figsize=(12,6))
    plt.plot(train.index, train["Sales"], label="Train Sales", color="blue")
    plt.plot(test.index, test["Sales"], label="Test Sales", color="black")
    plt.plot(test.index, pred_naive, label="Naive Forecast", color="red", alpha=0.6)
    plt.plot(test.index, pred_ma7, label="MA(7) Forecast", color="green", alpha=0.6)
    plt.plot(test.index, pred_arima, label="ARIMA(1,1,1)", color="orange", alpha=0.6)
    plt.title("Comparación de Modelos Clásicos vs. Ventas Reales")
    plt.xlabel("Fecha")
    plt.ylabel("Ventas Diarias")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

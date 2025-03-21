"""
time_series_models.py

1) Carga datos limpios
2) Aplica transformación logarítmica (log1p) para estabilizar la varianza
3) Divide en train/test
4) Modelos ARIMA y SARIMA (rolling)
5) Usa auto_arima para optimizar parámetros

Requisitos:
    pip install pandas numpy matplotlib statsmodels pmdarima scipy
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from pmdarima import auto_arima

warnings.filterwarnings("ignore")


# Carga de datos
def load_clean_data(path):
    df = pd.read_csv(path)
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)
    return df


# Agregación diaria y limpieza
def aggregate_daily(df):
    daily = df.groupby(df["Order Date"].dt.date)["Sales"].sum().reset_index()
    daily["Order Date"] = pd.to_datetime(daily["Order Date"])
    
    date_range = pd.date_range(start=daily["Order Date"].min(), end=daily["Order Date"].max(), freq='D')
    df_full = pd.DataFrame({"Order Date": date_range})
    
    daily_sales = df_full.merge(daily, on="Order Date", how="left")
    daily_sales["Sales"] = daily_sales["Sales"].fillna(0)
    
    daily_sales.sort_values("Order Date", inplace=True)
    daily_sales.set_index("Order Date", inplace=True)
    daily_sales.index.freq = 'D'
    
    return daily_sales


# Transformación logarítmica (log1p) y su inversa
def apply_log(series):
    return np.log1p(series)

def inverse_log(series):
    return np.expm1(series)


# División de datos
def train_test_split_time_series(df, split_date):
    train = df.loc[:split_date].copy()
    test = df.loc[split_date:].copy()
    
    train.index.freq = 'D'
    test.index.freq = 'D'
    
    return train, test


# ARIMA rolling forecast usando datos log-transformados
def arima_rolling_forecast(train, test, order=(1,1,1), use_actuals=False):
    # Se espera que los datos ya estén en log1p
    df_rolling = train.copy().asfreq('D', method='ffill')
    preds = []
    
    for current_date in test.index:
        model = sm.tsa.ARIMA(df_rolling["Sales"], order=order)
        model_fit = model.fit()
        forecast_series = model_fit.forecast(steps=1)
        forecast_val = forecast_series.iloc[0] if isinstance(forecast_series, pd.Series) else forecast_series[0]
        
        new_sale = test.loc[current_date, "Sales"] if use_actuals else forecast_val
        preds.append(float(forecast_val))
        
        new_row = pd.DataFrame({"Sales": [float(new_sale)]}, index=[current_date])
        df_rolling = pd.concat([df_rolling, new_row])
        df_rolling = df_rolling[~df_rolling.index.duplicated(keep='last')]
        df_rolling = df_rolling.asfreq('D', method='ffill')
    
    preds = np.array(preds)
    # Aplicamos la inversa de log para regresar a la escala original
    preds = inverse_log(preds)
    return pd.Series(preds, index=test.index, name="ARIMA_Rolling")


# SARIMA rolling forecast usando datos log-transformados
def sarima_rolling_forecast(train, test, order=(1,1,1), seasonal_order=(1,1,1,7), use_actuals=False):
    # Se espera que los datos ya estén en log1p
    df_rolling = train.copy().asfreq('D', method='ffill')
    preds = []
    
    for current_date in test.index:
        model = sm.tsa.statespace.SARIMAX(df_rolling["Sales"],
                                          order=order,
                                          seasonal_order=seasonal_order,
                                          enforce_stationarity=False,
                                          enforce_invertibility=False)
        model_fit = model.fit(method='powell', disp=False)
        forecast_series = model_fit.forecast(steps=1)
        forecast_val = forecast_series.iloc[0] if isinstance(forecast_series, pd.Series) else forecast_series[0]
        
        new_sale = test.loc[current_date, "Sales"] if use_actuals else forecast_val
        preds.append(float(forecast_val))
        
        new_row = pd.DataFrame({"Sales": [float(new_sale)]}, index=[current_date])
        df_rolling = pd.concat([df_rolling, new_row])
        df_rolling = df_rolling[~df_rolling.index.duplicated(keep='last')]
        df_rolling = df_rolling.asfreq('D', method='ffill')
    
    preds = np.array(preds)
    preds = inverse_log(preds)
    return pd.Series(preds, index=test.index, name="SARIMA_Rolling")


# Evaluación de errores
def mae_mape(y_true, y_pred):
    common_idx = y_true.index.intersection(y_pred.index)
    y_true, y_pred = y_true.loc[common_idx], y_pred.loc[common_idx]
    mask = y_true != 0
    mae_val = (y_true[mask] - y_pred[mask]).abs().mean()
    mape_val = ((y_true[mask] - y_pred[mask]).abs() / y_true[mask]).mean() * 100
    return mae_val, mape_val


def main():
    data_path = "../data/processed/walmart_cleaned.csv"
    df = load_clean_data(data_path)
    daily_sales = aggregate_daily(df)
    
    print("Daily sales shape:", daily_sales.shape)
    print("Rango de fechas:", daily_sales.index.min(), "->", daily_sales.index.max())
    
    # Aplicar transformación logarítmica
    daily_sales["Sales"] = apply_log(daily_sales["Sales"])
    
    split_date = "2015-01-01"
    train, test = train_test_split_time_series(daily_sales, split_date)
    
    print(f"Duplicados en train: {train.index.duplicated().sum()}")
    print(f"Duplicados en test: {test.index.duplicated().sum()}")
    
    # Auto ARIMA para optimizar parámetros (forzamos d=1 para asegurar estacionariedad)
    best_arima = auto_arima(train["Sales"], seasonal=False, stepwise=True, trace=True, d=1)
    print(best_arima.summary())
    
    # Para SARIMA, se usa m=30 como ciclo aproximado mensual
    best_sarima = auto_arima(train["Sales"], seasonal=True, m=30, stepwise=True, trace=True)
    print(best_sarima.summary())
    
    order_arima = best_arima.order
    order_sarima = best_sarima.order
    seasonal_order_sarima = best_sarima.seasonal_order
    
    # ARIMA Rolling Forecast
    pred_arima = arima_rolling_forecast(train, test, order=order_arima, use_actuals=False)
    mae_a, mape_a = mae_mape(inverse_log(test["Sales"]), pred_arima)
    print(f"ARIMA {order_arima} Rolling -> MAE: {mae_a:.2f}, MAPE: {mape_a:.2f}%")
    
    # SARIMA Rolling Forecast
    pred_sarima = sarima_rolling_forecast(train, test, order=order_sarima, seasonal_order=seasonal_order_sarima, use_actuals=False)
    mae_s, mape_s = mae_mape(inverse_log(test["Sales"]), pred_sarima)
    print(f"SARIMA {order_sarima}x{seasonal_order_sarima} Rolling -> MAE: {mae_s:.2f}, MAPE: {mape_s:.2f}%")
    
    # Gráfico de resultados en la escala original
    plt.figure(figsize=(12,6))
    plt.plot(train.index, inverse_log(train["Sales"]), label="Train Sales", color="blue")
    plt.plot(test.index, inverse_log(test["Sales"]), label="Test Sales", color="black")
    plt.plot(pred_arima.index, pred_arima, label="ARIMA Rolling", color="red", alpha=0.6)
    plt.plot(pred_sarima.index, pred_sarima, label="SARIMA Rolling", color="green", alpha=0.6)
    
    plt.title("Comparación ARIMA vs. SARIMA (Rolling)")
    plt.xlabel("Fecha")
    plt.ylabel("Ventas Diarias")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

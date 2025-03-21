"""
Ejemplo de:
1) Cargar datos limpios (con 'Order Date' y 'Sales')
2) Agregar ventas diarias
3) Dividir train/test
4) Modelos ARIMA vs. SARIMA (rolling)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

warnings.filterwarnings("ignore")


def load_clean_data(path):
    df = pd.read_csv(path)
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)
    return df


def aggregate_daily(df):
    daily = df.groupby(df["Order Date"].dt.date)["Sales"].sum().reset_index()
    daily["Order Date"] = pd.to_datetime(daily["Order Date"])
    start_date = daily["Order Date"].min()
    end_date = daily["Order Date"].max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    df_full = pd.DataFrame({"Order Date": date_range})
    daily_sales = df_full.merge(daily, on="Order Date", how="left")
    daily_sales["Sales"] = daily_sales["Sales"].fillna(0)
    daily_sales.sort_values("Order Date", inplace=True)
    daily_sales.set_index("Order Date", inplace=True)
    daily_sales.index.freq = 'D'
    return daily_sales


def train_test_split_time_series(df, split_date):
    train = df.loc[:split_date].copy()
    test = df.loc[split_date:].copy()
    train.index.freq = 'D'
    test.index.freq = 'D'
    return train, test


def check_stationarity(series):
    result = adfuller(series)
    print(f"ADF Test p-value: {result[1]:.5f}")
    return result[1] < 0.05


def mae_mape(y_true, y_pred):
    common_idx = y_true.index.intersection(y_pred.index)
    y_true = y_true.loc[common_idx]
    y_pred = y_pred.loc[common_idx]
    mask = (y_true != 0)
    if mask.sum() == 0:
        return 0.0, 0.0
    diff = (y_true[mask] - y_pred[mask])
    mae_val = diff.abs().mean()
    mape_val = (diff.abs() / y_true[mask].abs()).mean() * 100
    return mae_val, mape_val


def arima_rolling_forecast(train, test, order=(1,0,1), use_actuals=False):
    df_rolling = train.copy()
    df_rolling = df_rolling.asfreq('D', method='ffill')
    preds = []

    for current_date in test.index:
        print(f"Fecha actual: {current_date}, Frecuencia inferida: {pd.infer_freq(df_rolling.index)}")

        model = sm.tsa.ARIMA(df_rolling["Sales"], order=order)
        model_fit = model.fit()
        forecast_val = model_fit.forecast(steps=1)[0]
        new_sale = test.loc[current_date, "Sales"] if use_actuals else forecast_val
        preds.append(float(forecast_val))

        new_row = pd.DataFrame({"Sales": [float(new_sale)]}, index=[current_date])
        df_rolling = pd.concat([df_rolling, new_row])
        df_rolling = df_rolling[~df_rolling.index.duplicated(keep='last')]
        df_rolling = df_rolling.asfreq('D', method='ffill')

    return pd.Series(preds, index=test.index, name="ARIMA_Rolling")


def sarima_rolling_forecast(train, test, order=(1,1,1), seasonal_order=(1,1,1,7), use_actuals=False):
    df_rolling = train.copy()
    df_rolling = df_rolling.asfreq('D', method='ffill')
    preds = []

    for current_date in test.index:
        print(f"Fecha actual: {current_date}, Frecuencia inferida: {pd.infer_freq(df_rolling.index)}")

        model = sm.tsa.statespace.SARIMAX(
            df_rolling["Sales"],
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_fit = model.fit(method='powell', disp=False)
        forecast_val = model_fit.forecast(steps=1)[0]
        new_sale = test.loc[current_date, "Sales"] if use_actuals else forecast_val
        preds.append(float(forecast_val))

        new_row = pd.DataFrame({"Sales": [float(new_sale)]}, index=[current_date])
        df_rolling = pd.concat([df_rolling, new_row])
        df_rolling = df_rolling[~df_rolling.index.duplicated(keep='last')]
        df_rolling = df_rolling.asfreq('D', method='ffill')

    return pd.Series(preds, index=test.index, name="SARIMA_Rolling")


def main():
    data_path = "../data/processed/walmart_cleaned.csv"
    df = load_clean_data(data_path)
    daily_sales = aggregate_daily(df)

    print("Daily sales shape:", daily_sales.shape)
    print("Rango de fechas:", daily_sales.index.min(), "->", daily_sales.index.max())

    split_date = "2015-01-01"
    train, test = train_test_split_time_series(daily_sales, split_date)

    print(f"Duplicados en train: {train.index.duplicated().sum()}")
    print(f"Duplicados en test: {test.index.duplicated().sum()}")

    pred_arima = arima_rolling_forecast(train, test, order=(1,0,1), use_actuals=False)
    mae_a, mape_a = mae_mape(test["Sales"], pred_arima)
    print(f"ARIMA(1,0,1) Rolling -> MAE: {mae_a:.2f}, MAPE: {mape_a:.2f}%")

    pred_sarima = sarima_rolling_forecast(train, test, order=(1,1,1), seasonal_order=(1,1,1,7))
    mae_s, mape_s = mae_mape(test["Sales"], pred_sarima)
    print(f"SARIMA(1,1,1)x(1,1,1,7) Rolling -> MAE: {mae_s:.2f}, MAPE: {mape_s:.2f}%")

    plt.figure(figsize=(12,6))
    plt.plot(train.index, train["Sales"], label="Train Sales", color="blue")
    plt.plot(test.index, test["Sales"], label="Test Sales", color="black")
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

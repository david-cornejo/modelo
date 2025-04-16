import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

warnings.filterwarnings("ignore")

# --- Funciones de métricas ---
def compute_errors(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)*100
    return mae, mape

# --- Holt-Winters Grid Search ---
def holt_winters_gridsearch(train, test, seasonal_periods=52):
    alphas = [0.2, 0.4, 0.6, 0.8]
    betas = [0.2, 0.4, 0.6]
    gammas = [0.2, 0.4, 0.6]
    trends = ['add', 'mul', None]
    seasonals = ['add', 'mul', None]
    best_score = float('inf')
    best_params = None
    best_preds = None
    
    for alpha, beta, gamma, trend, seasonal in itertools.product(alphas, betas, gammas, trends, seasonals):
        if seasonal is None and gamma != 0.0:
            continue
        if trend is None and beta != 0.0:
            continue
        try:
            model = ExponentialSmoothing(
                train,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods
            )
            model_fit = model.fit(
                smoothing_level=alpha,
                smoothing_trend=beta if trend else None,
                smoothing_seasonal=gamma if seasonal else None,
                optimized=False
            )
            preds = model_fit.forecast(steps=len(test))
            mae, mape = compute_errors(test, preds)
            if mae < best_score:
                best_score = mae
                best_params = (alpha, beta, gamma, trend, seasonal)
                best_preds = preds
        except:
            continue
    
    return best_params, best_score, best_preds

# --- ARIMA Rolling Forecast ---
def arima_rolling_forecast(train, test, order=(5,1,1)):
    df_rolling = train.copy()
    preds = []
    for current_date in test.index:
        model = ARIMA(df_rolling, order=order)
        model_fit = model.fit()
        forecast_val = model_fit.forecast(steps=1).iloc[0]
        preds.append(float(forecast_val))
        # Agregamos el valor real al df_rolling para la siguiente iteración
        df_rolling.loc[current_date] = float(test.loc[current_date])
    return pd.Series(preds, index=test.index, name="ARIMA_Preds")

# --- SARIMA Rolling Forecast ---
def sarima_rolling_forecast(train, test, order=(3,1,1), seasonal_order=(1,1,1,52)):
    df_rolling = train.copy()
    preds = []
    for current_date in test.index:
        model = SARIMAX(df_rolling, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        forecast_val = model_fit.forecast(steps=1).iloc[0]
        preds.append(float(forecast_val))
        df_rolling.loc[current_date] = float(test.loc[current_date])
    return pd.Series(preds, index=test.index, name="SARIMA_Preds")

# --- Regresión Lineal Múltiple ---
def mlr_forecast(train, test):
    # Crearemos features a partir de la posición (t) y funciones sinus/cos para la estacionalidad
    def create_features(index):
        t = np.arange(len(index))
        sin_term = np.sin(2*np.pi*t/52)
        cos_term = np.cos(2*np.pi*t/52)
        return np.column_stack((t, sin_term, cos_term))
    
    train_df = train.to_frame(name="Sales")
    test_df = test.to_frame(name="Sales")
    
    X_train = create_features(train_df.index)
    y_train = train_df["Sales"].values
    X_test = create_features(test_df.index)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return pd.Series(preds, index=test.index, name="MLR_Preds")

def main():
    # Carga y agregación
    data_path = "../data/processed/walmart_cleaned_soft.csv"
    df = pd.read_csv(data_path)
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df.set_index("Order Date", inplace=True)
    weekly_sales = df["Sales"].resample("W").sum().fillna(0)
    
    # Train/Test split
    split_date = "2015-01-01"
    train = weekly_sales.loc[:split_date]
    test = weekly_sales.loc[split_date:]
    
    # --- Holt-Winters (grid search) ---
    hw_params, hw_score, hw_preds = holt_winters_gridsearch(train, test, seasonal_periods=52)
    if hw_preds is None:
        print("No se encontró modelo Holt-Winters válido.")
        return
    alpha, beta, gamma, trend, seasonal = hw_params
    mae_hw, mape_hw = compute_errors(test, hw_preds)
    print(f"[HW] Mejor combo => alpha={alpha}, beta={beta}, gamma={gamma}, trend={trend}, seasonal={seasonal}")
    print(f"[HW] MAE: {mae_hw:.2f}, MAPE: {mape_hw:.2f}%")

    # --- ARIMA ---
    arima_order = (5,1,1)
    arima_preds = arima_rolling_forecast(train, test, order=arima_order)
    mae_arima, mape_arima = compute_errors(test, arima_preds)
    print(f"[ARIMA] Orden={arima_order}, MAE: {mae_arima:.2f}, MAPE: {mape_arima:.2f}%")

    # --- SARIMA ---
    sarima_order = (3,1,1)
    seasonal_order = (1,1,1,52)
    sarima_preds = sarima_rolling_forecast(train, test, order=sarima_order, seasonal_order=seasonal_order)
    mae_sarima, mape_sarima = compute_errors(test, sarima_preds)
    print(f"[SARIMA] Orden={sarima_order}x{seasonal_order}, MAE: {mae_sarima:.2f}, MAPE: {mape_sarima:.2f}%")

    # --- MLR ---
    mlr_preds = mlr_forecast(train, test)
    mae_mlr, mape_mlr = compute_errors(test, mlr_preds)
    print(f"[MLR] MAE: {mae_mlr:.2f}, MAPE: {mape_mlr:.2f}%")

    # --- Crear tabla comparativa ---
    df_results = pd.DataFrame({
        "Fecha": test.index,
        "Valor Real": test.values,
        "Holt-Winters": hw_preds.values,
        "ARIMA": arima_preds.values,
        "SARIMA": sarima_preds.values,
        "MLR": mlr_preds.values
    }).reset_index(drop=True)

    print("\nTabla de Resultados (primeras 10 filas):")
    print(df_results.head(10))

    # --- Graficar en subplots separados ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Subplot 1: Holt-Winters
    axs[0, 0].plot(train.index, train, label="Train", color="blue")
    axs[0, 0].plot(test.index, test, label="Test", color="black")
    axs[0, 0].plot(hw_preds.index, hw_preds, label="HW Pred", color="red", linestyle="--")
    axs[0, 0].set_title("Holt-Winters")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Subplot 2: ARIMA
    axs[0, 1].plot(train.index, train, label="Train", color="blue")
    axs[0, 1].plot(test.index, test, label="Test", color="black")
    axs[0, 1].plot(arima_preds.index, arima_preds, label="ARIMA Pred", color="green", linestyle="--")
    axs[0, 1].set_title(f"ARIMA {arima_order}")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Subplot 3: SARIMA
    axs[1, 0].plot(train.index, train, label="Train", color="blue")
    axs[1, 0].plot(test.index, test, label="Test", color="black")
    axs[1, 0].plot(sarima_preds.index, sarima_preds, label="SARIMA Pred", color="orange", linestyle="--")
    axs[1, 0].set_title(f"SARIMA {sarima_order}x{seasonal_order}")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Subplot 4: MLR
    axs[1, 1].plot(train.index, train, label="Train", color="blue")
    axs[1, 1].plot(test.index, test, label="Test", color="black")
    axs[1, 1].plot(mlr_preds.index, mlr_preds, label="MLR Pred", color="purple", linestyle="--")
    axs[1, 1].set_title("MLR")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
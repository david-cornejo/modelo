import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import warnings

warnings.filterwarnings("ignore")

def compute_errors(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)*100
    return mae, mape

def holt_winters_gridsearch(train, test, seasonal_periods=52):
    """
    Realiza una búsqueda en un conjunto de parámetros (alpha, beta, gamma, trend, seasonal)
    y retorna el que minimice el MAE en test.
    """
    alphas = [0.2, 0.4, 0.6, 0.8]
    betas = [0.2, 0.4, 0.6]
    gammas = [0.2, 0.4, 0.6]
    trends = ['add', 'mul', None]       # None => sin tendencia
    seasonals = ['add', 'mul', None]    # None => sin estacionalidad

    best_score = float('inf')
    best_params = None
    best_preds = None

    # Generamos todas las combinaciones posibles
    for alpha, beta, gamma, trend, seasonal in itertools.product(alphas, betas, gammas, trends, seasonals):
        # Si no usamos estacionalidad, gamma no aplica; y si no usamos tendencia, beta no aplica
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
                smoothing_trend=beta if trend else None,       # Cambiar smoothing_slope -> smoothing_trend
                smoothing_seasonal=gamma if seasonal else None,
                optimized=False  # Desactivamos la optimización interna para respetar nuestros parámetros
            )
            preds = model_fit.forecast(steps=len(test))
            mae, mape = compute_errors(test, preds)

            # Verificamos si mejoró
            if mae < best_score:
                best_score = mae
                best_params = (alpha, beta, gamma, trend, seasonal)
                best_preds = preds
        except Exception as e:
            # Saltamos combos que causan error (ej: 'mul' con ceros)
            continue

    return best_params, best_score, best_preds

def main():
    # 1. Carga de datos
    data_path = "../data/processed/walmart_cleaned.csv"  # Ajusta la ruta a tu archivo
    df = pd.read_csv(data_path)
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df.set_index("Order Date", inplace=True)

    # 2. Agregar a nivel semanal
    weekly_sales = df["Sales"].resample("W").sum().fillna(0)

    # 3. Train/Test split
    split_date = "2015-01-01"
    train = weekly_sales.loc[:split_date]
    test = weekly_sales.loc[split_date:]

    # 4. Grid search Holt-Winters
    best_params, best_score, best_preds = holt_winters_gridsearch(train, test, seasonal_periods=52)
    if best_preds is None:
        print("No se encontró ningún modelo válido en la búsqueda.")
        return

    alpha, beta, gamma, trend, seasonal = best_params
    mae, mape = compute_errors(test, best_preds)

    print(f"Mejor combo => alpha={alpha}, beta={beta}, gamma={gamma}, trend={trend}, seasonal={seasonal}")
    print(f"MAE: {mae:.2f}, MAPE: {mape:.2f}%")

    # 5. Crear DataFrame con Fecha, Valor Real, Valor Predicho
    df_result = pd.DataFrame({
        'Fecha': test.index,
        'Valor Real': test.values,
        'Valor Predicho': best_preds.values
    }).reset_index(drop=True)

    # Mostrar las primeras filas
    print("\nTabla de Resultados:")
    print(df_result.head(10))

    # 6. (Opcional) Guardar en CSV
    # df_result.to_csv("tabla_resultados_hw.csv", index=False)

    # 7. Gráfica
    plt.figure(figsize=(10,6))
    plt.plot(train.index, train, label="Train")
    plt.plot(test.index, test, label="Test")
    plt.plot(best_preds.index, best_preds, label="Best Holt-Winters")
    plt.legend()
    plt.title("Búsqueda de parámetros Holt-Winters")
    plt.show()

if __name__ == "__main__":
    main()

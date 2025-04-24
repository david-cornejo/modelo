import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from pmdarima import auto_arima
import itertools

warnings.filterwarnings('ignore')


# --- Suavizado de outliers en toda la serie ---
def smooth_outliers(s: pd.Series, window: int = 5) -> pd.Series:
    """
    Detecta outliers en la serie usando IQR y los reemplaza
    por la mediana móvil de la ventana especificada.
    """
    q1, q3 = np.percentile(s, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    med_rolling = s.rolling(window=window, center=True).median()
    s_smooth = s.copy()
    outliers = (s < lower) | (s > upper)
    s_smooth[outliers] = med_rolling[outliers]
    return s_smooth


def load_and_aggregate_weekly(path: str,
                              date_col: str = 'Fecha',
                              value_col: str = 'Cargos') -> pd.Series:
    """
    Carga un CSV con fechas y un valor, y lo agrega a frecuencia semanal.
    """
    df = pd.read_csv(path, parse_dates=[date_col], dayfirst=True)
    df.set_index(date_col, inplace=True)
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce').fillna(0)
    weekly = df[value_col].resample('W').sum().fillna(0)
    return weekly


def test_stationarity(series: pd.Series, s: int = 52):
    """
    Ejecuta Augmented Dickey-Fuller en la serie y sus diferencias.
    """
    print("=== ADF Test ===")
    for diff, name in [
        (series, 'Original'),
        (series.diff().dropna(), 'd=1'),
        (series.diff(s).dropna(), f'D={s}')
    ]:
        pval = adfuller(diff)[1]
        print(f"{name}: p-value={pval:.4f}")


def plot_acf_pacf(series: pd.Series, lags: int = 30):
    """
    Grafica la ACF y PACF de la serie y de su primera diferencia.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_acf(series, lags=lags, ax=axes[0, 0]); axes[0, 0].set_title('ACF Original')
    plot_pacf(series, lags=lags, ax=axes[0, 1]); axes[0, 1].set_title('PACF Original')
    d1 = series.diff().dropna()
    plot_acf(d1, lags=lags, ax=axes[1, 0]); axes[1, 0].set_title('ACF d=1')
    plot_pacf(d1, lags=lags, ax=axes[1, 1]); axes[1, 1].set_title('PACF d=1')
    plt.tight_layout()
    plt.show()


def sarima_grid_search(series: pd.Series,
                       p_vals, d_vals, q_vals,
                       P_vals, D_vals, Q_vals,
                       s: int = 52):
    """
    Grid-search de SARIMA(p,d,q)x(P,D,Q,s) optimizando AIC.
    """
    best_aic = np.inf
    best_cfg = None
    for p, d, q in itertools.product(p_vals, d_vals, q_vals):
        for P, D, Q in itertools.product(P_vals, D_vals, Q_vals):
            try:
                m = SARIMAX(series,
                            order=(p, d, q),
                            seasonal_order=(P, D, Q, s),
                            enforce_stationarity=False,
                            enforce_invertibility=False)
                res = m.fit(disp=False)
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_cfg = ((p, d, q), (P, D, Q, s))
            except Exception:
                continue
    return best_cfg, best_aic


def rolling_forecast(series: pd.Series, cfg, train_size: int) -> pd.Series:
    """
    Realiza rolling-forecast: ajusta el modelo hasta cada punto de test.
    """
    (p, d, q), (P, D, Q, s) = cfg
    train, test = series[:train_size], series[train_size:]
    history = train.copy()
    preds = []
    for t in test.index:
        res = SARIMAX(history,
                      order=(p, d, q),
                      seasonal_order=(P, D, Q, s),
                      enforce_stationarity=False,
                      enforce_invertibility=False).fit(disp=False)
        f = res.forecast(1)[0]
        preds.append(f)
        history.loc[t] = test.loc[t]
    return pd.Series(preds, index=test.index)


def main():
    path = '../data/processed/merged/ventas_2015-2024.csv'

    # 1) Carga y agregación semanal
    weekly = load_and_aggregate_weekly(path, value_col='Cargos')

    # 2) Suavizado global de outliers
    weekly = smooth_outliers(weekly, window=5)

    # 3) Visualizar serie suavizada
    plt.figure(figsize=(10, 4))
    plt.plot(weekly, label='Ventas Semanales (Suavizada)')
    plt.title('Serie Semanal Suavizada')
    plt.grid(True)
    plt.show()

    # 4) Pruebas de estacionariedad y ACF/PACF
    test_stationarity(weekly)
    plot_acf_pacf(weekly)

    # 5) División train/test
    split_date = '2018-12-31'
    train = weekly[:split_date]
    test = weekly[split_date:]
    if not test.empty and test.index[0] == train.index[-1]:
        test = test.iloc[1:]
    train_size = len(train)

    # 6) Grid‑search SARIMA manual
    p_vals, d_vals, q_vals = range(0, 3), [0, 1], range(0, 3)
    P_vals, D_vals, Q_vals = range(0, 2), [0, 1], range(0, 2)
    best_cfg, best_aic = sarima_grid_search(train,
                                            p_vals, d_vals, q_vals,
                                            P_vals, D_vals, Q_vals)
    print(f"Mejor SARIMA: order={best_cfg[0]}, seasonal={best_cfg[1]}, AIC={best_aic:.2f}")

    # 7) Ajuste y pronósticos
    model = SARIMAX(train,
                    order=best_cfg[0],
                    seasonal_order=best_cfg[1],
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    res = model.fit(disp=False)
    preds_man = res.forecast(steps=len(test))

    mask = test.notna() & preds_man.notna()
    mae_man = mean_absolute_error(test[mask], preds_man[mask])
    mape_man = mean_absolute_percentage_error(test, preds_man) * 100
    print(f"[Manual SARIMA] MAE={mae_man:.2f}, MAPE={mape_man:.2f}%")

    # 8) Benchmark auto_arima
    auto = auto_arima(train, seasonal=True, m=52,
                      start_p=0, max_p=3, start_q=0, max_q=3,
                      d=None, start_P=0, max_P=2,
                      start_Q=0, max_Q=2, D=None,
                      stepwise=True, suppress_warnings=True)
    preds_auto = pd.Series(auto.predict(n_periods=len(test)), index=test.index)
    mae_auto = mean_absolute_error(test, preds_auto)
    mape_auto = mean_absolute_percentage_error(test, preds_auto) * 100
    print(f"[Auto ARIMA] MAE={mae_auto:.2f}, MAPE={mape_auto:.2f}%")

    # 9) Rolling‑forecast
    preds_roll = rolling_forecast(weekly, best_cfg, train_size)
    mae_roll = mean_absolute_error(test, preds_roll)
    print(f"[Rolling SARIMA] MAE={mae_roll:.2f}")

    # 10) Gráfica comparativa
    plt.figure(figsize=(10, 5))
    plt.plot(train.index, train, label='Train')
    plt.plot(test.index, test, label='Test', color='black')
    plt.plot(preds_man.index, preds_man, '--', label='SARIMA Manual')
    plt.plot(preds_auto.index, preds_auto, '--', label='Auto ARIMA')
    plt.plot(preds_roll.index, preds_roll, '--', label='SARIMA Rolling')
    plt.legend()
    plt.title('Comparación de Modelos SARIMA vs AutoARIMA')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
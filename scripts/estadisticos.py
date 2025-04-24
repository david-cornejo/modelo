import warnings
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima      as pm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet      import Prophet
from tbats        import TBATS
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

warnings.filterwarnings("ignore")

def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)*100

if __name__=='__main__':
    # 1) carga + mensual
    df = (
        pd.read_csv('../data/processed/merged/ventas_2016-2024.csv',
                    parse_dates=['Fecha'], index_col='Fecha')['Cargos']
          .resample('M').sum()
          .asfreq('M')
    )

    # 2) split
    train, test = df[:-12], df[-12:]

    # 3) SARIMA con boxcox log
    sarima = pm.auto_arima(train, seasonal=True, m=12,
                           error_action='ignore', suppress_warnings=True,
                           boxcox='log', trace=False)
    sarima_fc = pd.Series(sarima.predict(12), index=test.index)

    # 4) ETS + log-transform manual (aditivo + damped)
    train_log = np.log(train)
    ets_log = ExponentialSmoothing(
        train_log,
        trend='add', damped_trend=True,
        seasonal='add', seasonal_periods=12,
        initialization_method='estimated'
    ).fit(optimized=True)
    ets_fc = np.exp( ets_log.forecast(12) )

    # 5) Prophet + log
    prophet_df = train.reset_index().rename(columns={'Fecha':'ds','Cargos':'y'})
    prophet_df['y'] = np.log(prophet_df['y'])
    m = Prophet(yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False)
    m.add_seasonality(name='monthly', period=12, fourier_order=5)
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=12, freq='M')
    fc = m.predict(future).set_index('ds')['yhat']
    prophet_fc = np.exp(fc[-12:]);  prophet_fc.index = test.index

    # 6) TBATS con box-cox built-in
    tbats_est = TBATS(seasonal_periods=[12], use_arma_errors=True, use_box_cox=True)
    tbats_mod = tbats_est.fit(train)
    tbats_fc = pd.Series(tbats_mod.forecast(12), index=test.index)

    # 7) ensemble simple
    ensemble_fc = pd.concat([sarima_fc, ets_fc, prophet_fc, tbats_fc], axis=1).mean(axis=1)

    # 8) métricas
    models = {
      'SARIMA-log'         : sarima_fc,
      'ETS-damped-log'     : ets_fc,
      'Prophet-log'        : prophet_fc,
      'TBATS-boxcox-auto'  : tbats_fc,
      'Ensemble'           : ensemble_fc
    }

    print("MAPE (%) por modelo:")
    for name, fc in models.items():
        print(f"  {name:15s}: {mape(test, fc):6.2f}%")

    # 9) gráfica
    plt.figure(figsize=(12,6))
    plt.plot(train.index, train, color='gray', label='Train')
    plt.plot(test.index,  test,  color='black', linewidth=2, label='Test')
    for name, fc in models.items():
        plt.plot(fc.index, fc, '--', label=name)
    plt.title('Comparativa Mensual: SARIMA vs ETS vs Prophet vs TBATS vs Ensemble')
    plt.xlabel('Fecha')
    plt.ylabel('Ventas Mensuales')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

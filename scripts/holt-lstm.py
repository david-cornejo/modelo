import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.callbacks import EarlyStopping

# ──────────────────────────────────────────────────────────────────────────────
# CPU only + silenciar logs
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
warnings.filterwarnings("ignore")
tf.config.optimizer.set_jit(False)
tf.config.set_visible_devices([], 'GPU')
# ──────────────────────────────────────────────────────────────────────────────

DATA_PATH = "../data/processed/merged/ventas_2016-2024.csv"
TEST_HORIZON = 18    # <-- número de meses en test (12 ó 18)

def smooth_outliers_iqr_moving_mean(s, window=5):
    q1, q3 = np.percentile(s, [15, 85])
    iqr = q3 - q1
    low, up = q1 - 1.5*iqr, q3 + 1.5*iqr
    roll = s.rolling(window, center=True, min_periods=1).mean()
    out = s.copy()
    mask = (s<low)|(s>up)
    out.loc[mask] = roll.loc[mask]
    return out

def load_monthly(path, smooth=False):
    df = pd.read_csv(path, parse_dates=["Fecha"], index_col="Fecha")
    df["Cargos"] = pd.to_numeric(df["Cargos"], errors="coerce")
    df = df[df["Cargos"]>0]
    m = df["Cargos"].resample("M").sum().fillna(0)
    return smooth_outliers_iqr_moving_mean(m) if smooth else m

def fit_hw(train):
    return ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=36).fit()

def build_lstm(input_shape, drop=0.01):
    m = Sequential([
        Input(shape=input_shape),
        LSTM(50, return_sequences=True),
        Dropout(drop),
        LSTM(50),
        Dropout(drop),
        Dense(1)
    ])
    m.compile("adam", "mse")
    return m

def main():
    # 1) cargamos series
    series_raw    = load_monthly(DATA_PATH, smooth=False)
    series_smooth = load_monthly(DATA_PATH, smooth=True)

    # 2) preparamos Holt–Winters con test fijo
    train_hw = series_smooth.iloc[:-TEST_HORIZON]
    test_hw  = series_smooth.iloc[-TEST_HORIZON:]

    hw = fit_hw(train_hw)
    pred_hw = hw.forecast(TEST_HORIZON)
    mape_hw = mean_absolute_percentage_error(test_hw, pred_hw)*100
    print(f"Holt–Winters MAPE = {mape_hw:.2f}%")

    # 3) preparamos datos para LSTM (log+MinMax, luego fixed horizon)
    log_raw = np.log1p(series_raw)
    # escalamos TODO junto para mantener alignment
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(log_raw.values.reshape(-1,1)).flatten()

    # creamos secuencias de ventana=4
    W = 4
    X, y = [], []
    for i in range(len(scaled)-W):
        X.append(scaled[i:i+W])
        y.append(scaled[i+W])
    X = np.array(X)[:, :, None]
    y = np.array(y)

    # división en train/test, usando TEST_HORIZON
    cut = len(scaled) - TEST_HORIZON
    Xtr, ytr = X[:cut-W], y[:cut-W]
    Xte, yte = X[cut-W:], y[cut-W:]

    # 4) entrenamos LSTM
    lstm = build_lstm((W,1))
    es = EarlyStopping("val_loss", patience=5, restore_best_weights=True)
    lstm.fit(Xtr, ytr, epochs=50, batch_size=8,
             validation_split=0.1, verbose=0, callbacks=[es])

    # 5) predecimos
    ps = lstm.predict(Xte).flatten()
    # desescalado y despeje de log
    ps = np.expm1(scaler.inverse_transform(ps.reshape(-1,1)).flatten())
    ts = np.expm1(scaler.inverse_transform(yte.reshape(-1,1)).flatten())

    # índices para el test LSTM
    idx_ls = series_raw.index[-TEST_HORIZON:]
    pred_ls = pd.Series(ps, index=idx_ls)
    true_ls = pd.Series(ts, index=idx_ls)

    mape_ls = mean_absolute_percentage_error(true_ls, pred_ls)*100
    print(f"LSTM         MAPE = {mape_ls:.2f}%")

    # 6) stacking ensemble con Ridge
    # usamos los mismos idx_ls == test_hw.index
    idx_all = test_hw.index.intersection(pred_ls.index)
    Xall = np.vstack([
        pred_hw.loc[idx_all].values,
        pred_ls.loc[idx_all].values
    ]).T
    yall = test_hw.loc[idx_all].values

    # mitad validación / mitad test
    mid = len(yall)//2
    Xval, Xf = Xall[:mid], Xall[mid:]
    yval, yf = yall[:mid], yall[mid:]
    idx_val, idx_f = idx_all[:mid], idx_all[mid:]

    meta = Ridge().fit(Xval, yval)
    ens = meta.predict(Xf)
    mape_en = mean_absolute_percentage_error(yf, ens)*100
    print(f"Ensemble     MAPE = {mape_en:.2f}%")

    # series para graficar
    s_hw = pd.Series(pred_hw.values, index=test_hw.index)
    s_ls = pred_ls
    s_en = pd.Series(ens, index=idx_f)

    # 7) gráfica
    plt.figure(figsize=(12,5))
    plt.plot(series_smooth, color="gray", alpha=0.3, label="Real (suav)")
    plt.plot(train_hw, label="Train HW", color="C0")
    plt.plot(test_hw,  label="Test HW",  color="black")
    plt.plot(s_hw, "--", label="HW pred",   color="C2")
    plt.plot(s_ls, "--", label="LSTM pred", color="C3")
    plt.plot(s_en, "--", label="Ensemble",  color="C1", linewidth=2)
    plt.title(f"Ensemble sobre {TEST_HORIZON} meses de test\n"
              f"HW={mape_hw:.1f}%  LSTM={mape_ls:.1f}%  ENS={mape_en:.1f}%")
    plt.xlabel("Fecha"); plt.ylabel("Ventas mensuales")
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()

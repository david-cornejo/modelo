import os
# ──────────────────────────────────────────────────────────────────────────────
# Forzar CPU y desactivar XLA/Metal para evitar errores de plataforma
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"        # suprime logs innecesarios
#os.environ["CUDA_VISIBLE_DEVICES"] = ""         # deshabilita GPUs/Metal
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
# ──────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")
tf.config.set_visible_devices([], 'GPU')

# ———  NUEVO: función de suavizado tal cual la tenías  ——————————————
def smooth_outliers_mean(s, window=5):
    q1, q3 = np.percentile(s, [15, 85])
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    mean_rolling = s.rolling(window=window, center=True, min_periods=1).mean()
    s_smooth = s.copy()
    is_outlier = (s < lower) | (s > upper)
    s_smooth.loc[is_outlier] = mean_rolling.loc[is_outlier]
    return s_smooth
# ————————————————————————————————————————————————————————————————

def load_and_aggregate(data_path, use_log_transform=False):
    df = pd.read_csv(data_path)
    df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")
    df["Cargos"] = pd.to_numeric(df["Cargos"], errors="coerce")
    df = df[df["Cargos"] > 0]
    df.set_index("Fecha", inplace=True)
    # agregación mensual
    weekly = df["Cargos"].resample("M").sum().fillna(0)
    return weekly

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, dropout_rate):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(50, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(50),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def main():
    data_path = "../data/processed/merged/ventas_2015-2024.csv"
    use_log = True

    # 1) Carga y agregación
    weekly_orig = load_and_aggregate(data_path)                      # ← guardamos la original
    # 2) Suavizado de outliers
    weekly_smooth = smooth_outliers_mean(weekly_orig, window=5)     # ← aplicamos suavizado

    # 3) Log transform si toca
    if use_log:
        weekly_orig_log   = np.log1p(weekly_orig)
        weekly_smooth_log = np.log1p(weekly_smooth)
    else:
        weekly_orig_log   = weekly_orig
        weekly_smooth_log = weekly_smooth

    # 4) División train/test sobre la serie suavizada
    split_idx = int(len(weekly_smooth_log) * 0.85)
    train_series = weekly_smooth_log.iloc[:split_idx]
    test_series  = weekly_smooth_log.iloc[split_idx:]

    # 5) Escalado
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_series.values.reshape(-1, 1))
    test_scaled  = scaler.transform(test_series.values.reshape(-1, 1))

    # 6) Parámetros manuales
    window_size  = 3
    dropout_rate = 0.1
    batch_size   = 32  
    epochs       = 50

    # 7) Secuencias
    Xtr, ytr = create_sequences(train_scaled, window_size)
    Xte, yte = create_sequences(test_scaled, window_size)
    Xtr = Xtr.reshape(*Xtr.shape, 1)
    Xte = Xte.reshape(*Xte.shape, 1)

    # 8) Entrenamiento
    model = build_lstm_model((window_size, 1), dropout_rate)
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(Xtr, ytr, validation_split=0.01,
              epochs=epochs, batch_size=batch_size,
              verbose=1, callbacks=[es])

    # 9) Predicción y desescalado
    pred_scaled = model.predict(Xte)
    pred = scaler.inverse_transform(pred_scaled)
    true = scaler.inverse_transform(yte.reshape(-1,1))
    if use_log:
        pred = np.expm1(pred)
        true = np.expm1(true)

    # Calcular métricas
    mae = mean_absolute_error(true, pred)
    mape = mean_absolute_percentage_error(true, pred) * 100
    print(f"LSTM Forecast -> MAE: {mae:.2f}, MAPE: {mape:.2f}%")

    # Fechas alineadas
    dates = test_series.index[window_size:]
    n = min(len(dates), len(pred))
    df_res = pd.DataFrame({
        "Fecha": dates[:n],
        "Pred":  pred.flatten()[:n]
    })

    # ———  PLOTTING “estilo HW” ——————————————————————————————————
    plt.figure(figsize=(12,6))

    # Train suavizado (en original scale)
    plt.plot(weekly_smooth.index[:split_idx],
            np.expm1(train_series) if use_log else train_series,
            label="Train (suavizado)")

    # Original Test (sin suavizar) en semitransparencia
    plt.plot(weekly_orig.index[split_idx:],
            np.expm1(test_series) if use_log else test_series,
            label="Original Test", alpha=0.3)

    # Test suavizado
    plt.plot(weekly_smooth.index[split_idx:],
            np.expm1(test_series) if use_log else test_series,
            label="Test (suavizado)", color="black")

    # Forecast LSTM
    plt.plot(df_res["Fecha"], df_res["Pred"],
            label="LSTM Forecast", linestyle="--")

    # Agregar métricas en la gráfica
    plt.figtext(0.15, 0.85, f"MAE: {mae:.2f}\nMAPE: {mape:.2f}%", fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5))

    plt.title("LSTM Forecast (configuración manual, con suavizado)")
    plt.xlabel("Fecha")
    plt.ylabel("Ventas Semanales")
    plt.legend()
    plt.grid(True)
    plt.show()
    # ————————————————————————————————————————————————————————————————

if __name__ == "__main__":
    main()
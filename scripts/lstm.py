import os
# ──────────────────────────────────────────────────────────────────────────────
# Forzar CPU y desactivar XLA/Metal para evitar errores de plataforma
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"        # suprime logs innecesarios
os.environ["CUDA_VISIBLE_DEVICES"] = ""         # deshabilita GPUs/Metal
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
# ──────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")

# Asegurarnos de que TF no intente JIT/XLA
tf.config.optimizer.set_jit(False)
# Confirmar que no hay GPUs visibles
tf.config.set_visible_devices([], 'GPU')

def load_and_aggregate(data_path, use_log_transform=False):
    df = pd.read_csv(data_path)
    df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")
    df["Cargos"] = pd.to_numeric(df["Cargos"], errors="coerce")
    df = df[df["Cargos"] > 0]
    df.set_index("Fecha", inplace=True)
    weekly = df["Cargos"].resample("M").sum().fillna(0)
    if use_log_transform:
        weekly = np.log1p(weekly)
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
    weekly = load_and_aggregate(data_path, use_log_transform=use_log)

    # 2) División train/test
    split_idx = int(len(weekly) * 0.85)
    train_series = weekly.iloc[:split_idx]
    test_series = weekly.iloc[split_idx:]

    # 3) Escalado
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_series.values.reshape(-1, 1))
    test_scaled  = scaler.transform(test_series.values.reshape(-1, 1))

    # 4) Parámetros definidos manualmente (valores ajustados)
    window_size  = 4
    dropout_rate = 0.01
    batch_size   = 16
    epochs       = 50

    print(f"Usando configuración: window={window_size}, dropout={dropout_rate}, batch={batch_size}, epochs={epochs}")

    # 5) Creación de secuencias
    Xtr, ytr = create_sequences(train_scaled, window_size)
    Xte, yte = create_sequences(test_scaled, window_size)
    Xtr = Xtr.reshape(*Xtr.shape, 1)
    Xte = Xte.reshape(*Xte.shape, 1)

    # 6) Construcción y entrenamiento del modelo
    model = build_lstm_model((window_size, 1), dropout_rate)
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(Xtr, ytr, validation_split=0.01, epochs=epochs, batch_size=batch_size,
              verbose=1, callbacks=[es])

    # 7) Predicción y desescalado
    pred_scaled = model.predict(Xte)
    pred = scaler.inverse_transform(pred_scaled)
    true = scaler.inverse_transform(yte.reshape(-1, 1))
    if use_log:
        pred = np.expm1(pred)
        true = np.expm1(true)

    mape_val = mean_absolute_percentage_error(true, pred) * 100
    print(f"\nMAPE: {mape_val:.2f}%")

    # Ajustar las fechas para alinear los datos (se pierde 'window_size' muestras iniciales)
    dates = test_series.index[window_size:]
    n = min(len(dates), len(pred))
    results_df = pd.DataFrame({
        "Fecha": dates[:n],
        "Real":  true.flatten()[:n],
        "Pred":  pred.flatten()[:n]
    })
    print(results_df.head())

    # 8) Gráfica final
    plt.figure(figsize=(12,6))
    train_vals = np.expm1(train_series) if use_log else train_series
    test_vals  = np.expm1(test_series)  if use_log else test_series
    plt.plot(train_series.index, train_vals, label="Train")
    plt.plot(test_series.index, test_vals, label="Test", color="black")
    plt.plot(results_df["Fecha"], results_df["Pred"], "--", label="LSTM Predicho", color="red")
    plt.xlabel("Fecha")
    plt.ylabel("Ventas Semanales")
    plt.title("LSTM Forecast (configuración manual)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
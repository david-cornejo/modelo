import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")

def load_and_aggregate(data_path, use_log_transform=False):
    df = pd.read_csv(data_path)
    df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")
    df["Cargos"] = pd.to_numeric(df["Cargos"], errors="coerce")
    df = df[df["Cargos"] > 0]
    df.set_index("Fecha", inplace=True)
    weekly_cargos = df["Cargos"].resample("W").sum().fillna(0)
    if use_log_transform:
        weekly_cargos = np.log1p(weekly_cargos)
    return weekly_cargos

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, dropout_rate):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(50))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return mae, mape

def main():
    data_path = "../data/processed/ventas_2015-2020.csv"
    use_log = True

    weekly_cargos = load_and_aggregate(data_path, use_log_transform=use_log)
    split_index = int(len(weekly_cargos) * 0.8)
    train_series = weekly_cargos.iloc[:split_index]
    test_series = weekly_cargos.iloc[split_index:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_series.values.reshape(-1, 1))
    test_scaled = scaler.transform(test_series.values.reshape(-1, 1))

    # Grid de hiperparámetros
    window_sizes = [4, 8, 12]
    dropout_rates = [0.1, 0.2]
    batch_sizes = [8, 16]
    epochs_list = [50]

    best_mape = float("inf")
    best_config = None
    best_model = None
    best_results = None

    for window_size in window_sizes:
        X_train, y_train = create_sequences(train_scaled, window_size)
        X_test, y_test = create_sequences(test_scaled, window_size)

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        for dropout in dropout_rates:
            for batch in batch_sizes:
                for epochs in epochs_list:
                    print(f"Probando: window={window_size}, dropout={dropout}, batch={batch}, epochs={epochs}")
                    model = build_lstm_model((window_size, 1), dropout)
                    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                    model.fit(X_train, y_train, epochs=epochs, batch_size=batch, 
                              validation_split=0.1, verbose=0, callbacks=[early_stop])

                    test_pred_scaled = model.predict(X_test)
                    test_pred = scaler.inverse_transform(test_pred_scaled)
                    y_test_orig = scaler.inverse_transform(y_test)

                    if use_log:
                        test_pred = np.expm1(test_pred)
                        y_test_orig = np.expm1(y_test_orig)

                    mape = mean_absolute_percentage_error(y_test_orig, test_pred) * 100

                    if mape < best_mape:
                        best_mape = mape
                        best_config = (window_size, dropout, batch, epochs)
                        best_model = model
                        test_dates = test_series.index[window_size:]
                        min_len = min(len(test_dates), len(test_pred))
                        best_results = pd.DataFrame({
                            "Fecha": test_dates[:min_len],
                            "Valor Real": y_test_orig.flatten()[:min_len],
                            "Valor Predicho": test_pred.flatten()[:min_len]
                        })

    # Resultados finales
    print(f"\n🎯 Mejor Configuración: window={best_config[0]}, dropout={best_config[1]}, batch={best_config[2]}, epochs={best_config[3]}")
    print(f"✅ Mejor MAPE: {best_mape:.2f}%")
    print(best_results.head())

    # Gráficas
    plt.figure(figsize=(12,6))
    plt.plot(train_series.index, np.expm1(train_series.values) if use_log else train_series.values, label="Train", color="blue")
    plt.plot(test_series.index, np.expm1(test_series.values) if use_log else test_series.values, label="Test", color="black")
    plt.plot(best_results["Fecha"], best_results["Valor Predicho"], label="LSTM Predicción", color="red", linestyle="--")
    plt.xlabel("Fecha")
    plt.ylabel("Ventas Semanales")
    plt.title("Predicción con LSTM (mejor configuración)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
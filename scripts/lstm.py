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

def build_lstm_model(input_shape, dropout_rate=0.2):
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
    data_path = "../data/processed/cargos_2015-2017.csv"
    use_log = True
    window_size = 4
    epochs = 50
    batch_size = 16

    weekly_cargos = load_and_aggregate(data_path, use_log_transform=use_log)
    split_index = int(len(weekly_cargos) * 0.8)
    train_series = weekly_cargos.iloc[:split_index]
    test_series = weekly_cargos.iloc[split_index:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_series.values.reshape(-1, 1))
    test_scaled = scaler.transform(test_series.values.reshape(-1, 1))

    X_train, y_train = create_sequences(train_scaled, window_size)
    X_test, y_test = create_sequences(test_scaled, window_size)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                        validation_split=0.1, verbose=1, callbacks=[early_stop])

    train_pred_scaled = model.predict(X_train)
    test_pred_scaled = model.predict(X_test)

    train_pred = scaler.inverse_transform(train_pred_scaled)
    y_train_orig = scaler.inverse_transform(y_train)
    test_pred = scaler.inverse_transform(test_pred_scaled)
    y_test_orig = scaler.inverse_transform(y_test)

    if use_log:
        train_pred = np.expm1(train_pred)
        y_train_orig = np.expm1(y_train_orig)
        test_pred = np.expm1(test_pred)
        y_test_orig = np.expm1(y_test_orig)
        train_series = np.expm1(train_series)
        test_series = np.expm1(test_series)

    mae_test, mape_test = evaluate_model(y_test_orig, test_pred)
    print(f"Test MAE: {mae_test:.2f}, Test MAPE: {mape_test:.2f}%")

    # Recalcular fechas válidas para y_test_orig (última fecha de cada secuencia)
    test_dates = test_series.index[window_size:]

    # En caso de que haya un desfase de 1 por el split, ajustamos la longitud
    min_len = min(len(test_dates), len(y_test_orig), len(test_pred))

    df_results = pd.DataFrame({
        "Fecha": test_dates[:min_len],
        "Valor Real": y_test_orig.flatten()[:min_len],
        "Valor Predicho": test_pred.flatten()[:min_len]
    })
    print("\nTabla de Resultados (primeras 10 filas):")
    print(df_results.head(10))

    plt.figure(figsize=(12,6))
    plt.plot(train_series.index, train_series.values, label="Train", color="blue")
    plt.plot(test_series.index, test_series.values, label="Test", color="black")
    plt.plot(df_results["Fecha"], df_results["Valor Predicho"], label="LSTM Predicción", color="red", linestyle="--")
    plt.xlabel("Fecha")
    plt.ylabel("Cargos Semanales")
    plt.title("Predicción con LSTM")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12,6))
    plt.plot(df_results["Fecha"], np.abs(df_results["Valor Real"] - df_results["Valor Predicho"]), label="Error Absoluto", color="magenta")
    plt.xlabel("Fecha")
    plt.ylabel("Error Absoluto")
    plt.title("Error Absoluto en Test")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
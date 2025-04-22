import pandas as pd

def cargar_datos(path, frecuencia='D'):
    df = pd.read_csv(path)
    df["Fecha"] = pd.to_datetime(df["Fecha"], format="%d/%m/%y", errors="coerce")
    df = df.dropna(subset=["Fecha"])

    # Agrupar por fecha para eliminar duplicados y sumar
    df_grouped = df.groupby("Fecha")["Cargos"].sum()

    # Reindexado a frecuencia deseada
    serie = df_grouped.resample(frecuencia).sum().fillna(0)
    return serie

if __name__ == "__main__":
    path = "../data/processed/ventas_2015-2020.csv"  # Ajusta según tu estructura
    diaria = cargar_datos(path, "D")
    semanal = cargar_datos(path, "W")
    mensual = cargar_datos(path, "M")

    diaria.to_csv("../data/processed/serie_diaria.csv")
    semanal.to_csv("../data/processed/serie_semanal.csv")
    mensual.to_csv("../data/processed/serie_mensual.csv")

    print("Series guardadas correctamente.")
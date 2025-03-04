"""
plot_time_series.py

Script que:
1) Carga un dataset limpio con columnas 'Order Date' y 'Sales'
2) Agrupa las ventas diarias
3) Grafica la serie de tiempo resultante
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # 1. Ruta de tu archivo (CSV) con datos limpios
    #    Ajusta el nombre y la ubicación según tu estructura
    data_path = "../data/processed/walmart_cleaned.csv"
    
    # 2. Carga el dataset
    df = pd.read_csv(data_path, parse_dates=["Order Date"])
    
    # 3. Agrupar las ventas diarias
    #    Asume que cada registro tiene 'Sales' y 'Order Date' (datetime)
    daily_sales = df.groupby(df["Order Date"].dt.date)["Sales"].sum()
    
    # Convierto el índice a tipo datetime de nuevo, si deseas
    daily_sales.index = pd.to_datetime(daily_sales.index)
    
    # 4. Graficar la serie de tiempo
    plt.figure(figsize=(10, 5))
    plt.plot(daily_sales.index, daily_sales.values, linewidth=1)

    # 5. Personalizar la gráfica
    plt.title("Ventas Diarias a lo Largo del Tiempo")
    plt.xlabel("Fecha")
    plt.ylabel("Ventas Totales (Daily Sales)")
    plt.grid(True)

    # 6. Mostrar o guardar la gráfica
    plt.show()
    # Si quisieras guardar, podrías:
    plt.savefig("daily_sales_timeseries.png", dpi=300)

if __name__ == "__main__":
    main()

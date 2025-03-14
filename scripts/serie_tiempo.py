"""
plot_time_series.py

Script que:
1) Carga un dataset limpio con columnas 'Order Date' y 'Sales'
2) Agrupa las ventas diarias
3) Grafica la serie de tiempo resultante
"""

import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import plotly.express as px

def interactive_plot(daily_sales):
    fig = px.line(daily_sales, x=daily_sales.index, y='Sales', title='Ventas Diarias a lo Largo del Tiempo')
    fig.update_xaxes(title_text='Fecha')
    fig.update_yaxes(title_text='Ventas Totales (Daily Sales)')
    fig.show()

def main(plot_type="superpuesta"):
    # 1. Ruta de tu archivo (CSV) con datos limpios
    data_path = "../data/processed/walmart_cleaned_w_out.csv"
    
    # 2. Carga el dataset
    df = pd.read_csv(data_path, parse_dates=["Order Date"])
    
    if plot_type == "superpuesta":
        # 3. Crear una columna para el día del año
        df["Day_of_Year"] = df["Order Date"].dt.dayofyear
        df["Year"] = df["Order Date"].dt.year
        
        # 4. Agrupar las ventas diarias por año y día del año
        daily_sales = df.groupby(["Year", "Day_of_Year"])["Sales"].sum().unstack(level=0)
        
        # 5. Graficar la serie de tiempo para cada año
        plt.figure(figsize=(10, 5))
        for year in daily_sales.columns:
            plt.plot(daily_sales.index, daily_sales[year], label=str(year), linewidth=1)

        # 6. Personalizar la gráfica
        plt.title("Ventas Diarias a lo Largo del Tiempo (Superpuestas por Año)")
        plt.xlabel("Día del Año")
        plt.ylabel("Ventas Totales (Daily Sales)")
        plt.legend(title="Año")
        plt.grid(True)
        # Formatear el eje X para mostrar todos los días del mes
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.gcf().autofmt_xdate()  # Rotar las etiquetas de fecha

    else:
        # 3. Agrupar las ventas diarias
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
        
        # Formatear el eje X para mostrar fechas completas
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gcf().autofmt_xdate()  # Rotar las etiquetas de fecha

    # 7. Mostrar la gráfica
    plt.show()

if __name__ == "__main__":
    # Cambia el valor de plot_type a "normal" para ver la gráfica de toda la serie
    main(plot_type="superpuesta")

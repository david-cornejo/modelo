"""
plot_time_series.py

Script que:
1) Carga un dataset limpio con columnas 'Order Date' y 'Sales'
2) Agrupa las ventas semanales
3) Grafica la serie de tiempo resultante
"""

import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import plotly.express as px

def interactive_plot(weekly_sales):
    fig = px.line(weekly_sales, x=weekly_sales.index, y='Sales', title='Ventas Semanales a lo Largo del Tiempo')
    fig.update_xaxes(title_text='Fecha')
    fig.update_yaxes(title_text='Ventas Totales (Weekly Sales)')
    fig.show()

def main(plot_type="superpuesta"):
    # 1. Ruta de tu archivo (CSV) con datos limpios
    data_path = "../data/processed/walmart_cleaned_w_out.csv"
    
    # 2. Carga el dataset
    df = pd.read_csv(data_path, parse_dates=["Order Date"])
    
    if plot_type == "superpuesta":
        # 3. Crear una columna para la semana del año
        df["Week_of_Year"] = df["Order Date"].dt.isocalendar().week
        df["Year"] = df["Order Date"].dt.year
        
        # 4. Agrupar las ventas semanales por año y semana del año
        weekly_sales = df.groupby(["Year", "Week_of_Year"])["Sales"].sum().unstack(level=0)
        
        # 5. Graficar la serie de tiempo para cada año
        plt.figure(figsize=(10, 5))
        for year in weekly_sales.columns:
            plt.plot(weekly_sales.index, weekly_sales[year], label=str(year), linewidth=1)

        # 6. Personalizar la gráfica
        plt.title("Ventas Semanales a lo Largo del Tiempo (Superpuestas por Año)")
        plt.xlabel("Semana del Año")
        plt.ylabel("Ventas Totales (Weekly Sales)")
        plt.legend(title="Año")
        plt.grid(True)
        # Formatear el eje X para mostrar todas las semanas
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%W'))
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator())
        plt.gcf().autofmt_xdate()  # Rotar las etiquetas de fecha

    else:
        # 3. Agrupar las ventas semanales
        df.set_index("Order Date", inplace=True)
        weekly_sales = df.resample('W')['Sales'].sum()
        
        # 4. Graficar la serie de tiempo
        plt.figure(figsize=(10, 5))
        plt.plot(weekly_sales.index, weekly_sales.values, linewidth=1)

        # 5. Personalizar la gráfica
        plt.title("Ventas Semanales a lo Largo del Tiempo")
        plt.xlabel("Fecha")
        plt.ylabel("Ventas Totales (Weekly Sales)")
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

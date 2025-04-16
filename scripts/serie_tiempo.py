"""
serie_tiempo.py

Script que:
1) Carga un dataset limpio con columnas 'Order Date' y 'Sales'
2) Agrupa las ventas semanales
3) Grafica la serie de tiempo resultante
4) Agrega análisis de estacionalidad (descomposición, ACF, heatmap)
"""

import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
import os
import plotly.express as px

def interactive_plot(weekly_sales):
    """
    Genera una gráfica interactiva de las ventas semanales.
    """
    fig = px.line(weekly_sales, x=weekly_sales.index, y='Sales', title='Ventas Semanales a lo Largo del Tiempo')
    fig.update_xaxes(title_text='Fecha')
    fig.update_yaxes(title_text='Ventas Totales (Weekly Sales)')
    fig.show()

def decompose_series(weekly_sales):
    """
    Descompone la serie de tiempo en tendencia, estacionalidad y residuales.
    """
    decomposition = seasonal_decompose(weekly_sales, model='additive', period=52)  # Periodo semanal
    decomposition.plot()
    plt.show()

def plot_acf_analysis(weekly_sales):
    """
    Grafica la función de autocorrelación (ACF) para identificar estacionalidad.
    """
    plt.figure(figsize=(10, 5))
    plot_acf(weekly_sales, lags=104)  # Hasta 2 años (104 semanas)
    plt.title("ACF - Autocorrelación")
    plt.show()

def plot_heatmap(df):
    """
    Crea un heatmap para visualizar patrones estacionales por semana del año y año.
    """
    df["Week_of_Year"] = df["Order Date"].dt.isocalendar().week
    df["Year"] = df["Order Date"].dt.year
    heatmap_data = df.groupby(["Week_of_Year", "Year"])["Sales"].sum().unstack()
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap="YlGnBu", annot=False)
    plt.title("Heatmap de Ventas Semanales por Año")
    plt.xlabel("Año")
    plt.ylabel("Semana del Año")
    plt.show()

def main(plot_type="superpuesta", analyze_seasonality=False):
    # 1. Ruta de tu archivo (CSV) con datos limpios
    data_path = "../data/processed/walmart_cleaned_soft.csv"
    
    # 2. Carga el dataset
    df = pd.read_csv(data_path, parse_dates=["Order Date"])
    
    if plot_type == "normal":
        # 3. Crear una columna para la semana del año
        df["Week_of_Year"] = df["Order Date"].dt.isocalendar().week
        df["Year"] = df["Order Date"].dt.year
        
        # 4. Agrupar las ventas semanales por año y semana del año
        weekly_sales = df.groupby(["Week_of_Year", "Year"])["Sales"].sum().unstack(level=1)
        
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

        # Formatear el eje X para mostrar todas las semanas del año
        plt.xticks(ticks=range(1, 53), labels=range(1, 53))  # Mostrar semanas del 1 al 52

    else:
        # 3. Agrupar las ventas semanales
        df.set_index("Order Date", inplace=True)
        weekly_sales = df.resample('M')['Sales'].sum()
        
        # 4. Graficar la serie de tiempo
        plt.figure(figsize=(10, 5))
        plt.plot(weekly_sales.index, weekly_sales.values, linewidth=1)

        # 5. Personalizar la gráfica
        plt.title("Ventas Semanales a lo Largo del Tiempo")
        plt.xlabel("Fecha")
        plt.ylabel("Ventas Totales (Weekly Sales)")
        plt.grid(True)
        
        # Formatear el eje X para mostrar fechas completas
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
        plt.gcf().autofmt_xdate()  # Rotar las etiquetas de fecha

        # Guardar la gráfica como PNG
        plt.savefig("serie_tiempo_soft.png", dpi=300, bbox_inches="tight")

    # 7. Mostrar la gráfica
    plt.show()

    # 8. Análisis de estacionalidad (opcional)
    if analyze_seasonality:
        print("\n--- Análisis de Estacionalidad ---")
        # Agrupar las ventas semanales
        weekly_sales = df.set_index("Order Date").resample('W')['Sales'].sum()
        decompose_series(weekly_sales)
        plot_acf_analysis(weekly_sales)
        plot_heatmap(df)

if __name__ == "__main__":
    # Cambia el valor de plot_type a "normal" para ver la gráfica de toda la serie
    # Cambia analyze_seasonality a True para realizar el análisis de estacionalidad
    main(plot_type="superpuesta", analyze_seasonality=True)
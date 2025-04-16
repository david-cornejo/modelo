import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import plotly.express as px

def cargar_datos(path):
    df = pd.read_csv(path)
    df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
    df['Cargos'] = df['Cargos'].astype(str).str.replace(',', '').str.strip()
    df = df[df['Cargos'].str.match(r'^\d+(\.\d+)?$')]
    df['Cargos'] = df['Cargos'].astype(float)
    df = df[df['Fecha'].notna()]
    return df

def graficar_serie(df):
    df.set_index("Fecha", inplace=True)
    cargos_semanales = df.resample('W')['Cargos'].sum()

    # Gráfico estático
    plt.figure(figsize=(12, 6))
    plt.plot(cargos_semanales.index, cargos_semanales.values, label='Cargos semanales', linewidth=1.5)
    plt.title("Serie de Tiempo de Cargos (Semanal)")
    plt.xlabel("Fecha")
    plt.ylabel("Monto de Cargos")
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.tight_layout()
    plt.savefig("cargos_semanales.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Gráfico interactivo con Plotly
    fig = px.line(x=cargos_semanales.index, y=cargos_semanales.values,
                  labels={'x': 'Fecha', 'y': 'Cargos semanales'},
                  title='Cargos Semanales - Interactivo')
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    fig.show()

def main():
    ruta = "../data/processed/cargos_2015-2017.csv"  # Cambia si tu archivo está en otra ruta
    df = cargar_datos(ruta)
    graficar_serie(df)

if __name__ == "__main__":
    main()
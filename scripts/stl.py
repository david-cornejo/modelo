import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# Leer datos desde un archivo CSV
file_path = '../data/processed/merged/ventas_2015-2024.csv'
df = pd.read_csv(file_path)

# Asegurarse de que la columna de fecha esté en formato datetime
df['Fecha'] = pd.to_datetime(df['Fecha'])

# Agrupar datos limpios por mes
df['Mes'] = df['Fecha'].dt.to_period('M').dt.to_timestamp()
monthly_data = df.groupby('Mes')['Cargos'].sum()

# Aplicar STL con periodicidad mensual (12)
stl = STL(monthly_data, period=12, seasonal=13)
res = stl.fit()

# Visualizar resultados
res.plot()
plt.show()  # Agregar esta línea para mostrar la gráfica
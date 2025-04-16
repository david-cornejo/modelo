import pandas as pd
import glob

# Ruta donde están los archivos
ruta_archivos = "../data/processed/ventas_20*.csv"

# Leer todos los archivos que coincidan con el patrón (por ejemplo 2015, 2016, 2017)
archivos = glob.glob(ruta_archivos)

# Cargar y unir los datasets
df_union = pd.concat([pd.read_csv(f) for f in archivos], ignore_index=True)

# # Asegurar que la fecha sea datetime y ordenar
# df_union['Fecha'] = pd.to_datetime(df_union['Fecha'], format='%d/%m/%Y', errors='coerce')
# df_union = df_union.sort_values(by='Fecha')

# Guardar archivo combinado
df_union.to_csv("../data/processed/abonos_2015-2017.csv", index=False)

print("Unión completada. Total de registros:", len(df_union))
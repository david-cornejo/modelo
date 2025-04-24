import pandas as pd
import glob

# Ruta donde están los archivos
ruta_archivos = "../data/processed/separated/ventas_20*.csv"
archivos = glob.glob(ruta_archivos)

# Cargar y unir los datasets
df_union = pd.concat([pd.read_csv(f) for f in archivos], ignore_index=True)

# Parsear fechas tratando múltiples formatos y sin perder registros
df_union['Fecha'] = pd.to_datetime(
    df_union['Fecha'],
    dayfirst=True,
    infer_datetime_format=True,
    errors='coerce'
)

# Ordenar todo el DF por Fecha (NaT al final) y reindexar
df_union = df_union.sort_values(
    by='Fecha',
    ascending=True,
    na_position='last'
).reset_index(drop=True)

# Guardar archivo combinado
df_union.to_csv("../data/processed/merged/abonos_2015-2024.csv", index=False)

print("Unión completada. Total de registros:", len(df_union))
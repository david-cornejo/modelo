import pandas as pd
import re

# Cargar el archivo original
df = pd.read_csv("../data/raw/2020.csv")

# Saltar las primeras filas irrelevantes (encabezados duplicados, información general)
df = df[6:]

# Renombrar columnas con encabezados correctos
df.columns = ['Fecha', 'Serie', 'Folio', 'Concepto', 'Cargos', 'Abonos', 
              'Saldo Documento', 'Vence', 'Tipo de Cambio', 
              'Estado del Documento', 'Referencia']

# Eliminar columnas innecesarias
df = df.drop(columns=['Serie', 'Folio', 'Abonos', 'Saldo Documento', 
                      'Vence', 'Tipo de Cambio', 'Estado del Documento', 
                      'Referencia'])

# Eliminar filas vacías o con 'Cargos' inválidos (texto como ' ', resúmenes, etc.)
df['Cargos'] = df['Cargos'].astype(str).str.replace(',', '').str.strip()

# Filtrar solo filas donde 'Cargos' es un número positivo (ignora abonos y totales)
df = df[df['Cargos'].str.match(r'^\d+(\.\d+)?$')]
df['Cargos'] = df['Cargos'].astype(float)

# Eliminar filas que no tengan fecha o concepto
df = df[df['Fecha'].notna() & df['Concepto'].notna()]

# Eliminar posibles resúmenes (como "Saldo Inicial", "(+)", "(=)", etc.) que se repiten
resumen_keywords = ['Saldo Inicial', '(+)', '(-)', '(=)', 'Abonos', 'Cargos', 'Saldo Final']

escaped_keywords = [re.escape(keyword) for keyword in resumen_keywords]
df = df[~df['Concepto'].astype(str).str.contains('|'.join(escaped_keywords), case=False)]
# Reiniciar índice
df = df.reset_index(drop=True)

df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
df = df.sort_values(by='Fecha')
df['Fecha'] = df['Fecha'].dt.strftime('%d/%m/%Y')

# Guardar el archivo limpio si lo deseas
df.to_csv("../data/processed/ventas_2020_limpias.csv", index=False)

# Mostrar preview (opcional si estás trabajando en Jupyter)
print(df.head())
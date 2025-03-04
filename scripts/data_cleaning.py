"""
data_cleaning.py

Script para:
1) Cargar el dataset "walmart Retail Data.xlsx" de la carpeta data/raw/
2) Realizar limpieza básica (eliminar columnas irrelevantes, manejar nulos, revisar outliers, etc.)
3) Guardar el dataset limpio en la carpeta data/processed/

Requerimientos:
- pandas
- numpy
- scipy
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

def load_data(input_path: str) -> pd.DataFrame:
    return pd.read_excel(input_path)

def reorder_columns(df: pd.DataFrame, column_order: list) -> pd.DataFrame:
    return df[column_order]

def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    # 1. Eliminar columnas que no aportan
    cols_to_drop = ["Customer Age", "Customer Name", "Discount", "Number of Records", "Order ID", "Order Priority",
                    "Order Quantity", "Product Base Margin", "Product Container", "Product Name", "Product Sub-Category",
                     "Profit", "Row ID", "Ship Date", "Ship Mode", "Shipping Cost", "Unit Price", "Zip Code"]
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # 2. Eliminar duplicados exactos
    df.drop_duplicates(inplace=True)

    # 4. Revisar outliers en 'Sales' (ejemplo con z-score)
    if "Sales" in df.columns:
        df["z_sales"] = stats.zscore(df["Sales"])
        # Muestra algunos outliers para tu revisión
        outliers_sales = df[abs(df["z_sales"]) > 3]
        print("Registros con outliers en 'Sales' (|z-score| > 3):")
        print(outliers_sales[["Sales", "z_sales"]].head(10))

        # Decide si excluyes outliers
        # df = df[abs(df["z_sales"]) <= 3]

        # Limpia la columna auxiliar
        df.drop(columns="z_sales", inplace=True)

    # 5. Crear columnas derivadas de fecha
    if "Order Date" in df.columns:
        df["Order_Year"] = df["Order Date"].dt.year
        df["Order_Month"] = df["Order Date"].dt.month
        df["Order_Day"] = df["Order Date"].dt.day

    # 6. Convertir columnas categóricas (opcional)
    categorical_cols = ["State", "Customer Segment", "Product Category", "Region"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df

def save_data(df: pd.DataFrame, output_path: str):
    """
    Guarda el DataFrame en formato CSV (puedes usar Excel también si deseas).
    """
    df.to_csv(output_path, index=False)
         
if __name__ == "__main__":

    input_file = "../data/raw/walmart Retail Data.xlsx"
    output_file = "../data/processed/walmart_cleaned.csv"

    # 1. Carga
    df_raw = load_data(input_file)
    print("Dimensiones originales:", df_raw.shape)

    # 2. Limpieza
    df_clean = clean_data(df_raw)
    print("Dimensiones después de limpieza:", df_clean.shape)

    # 3. Reordenar columnas
    column_order = ["Region", "State", "Order Date", "Order_Year", "Order_Month", "Order_Day",  "Customer Segment", "Product Category", "Sales"]
    df_clean = reorder_columns(df_clean, column_order)
    print("Columnas reordenadas:", df_clean.columns)

    # 4. Guardado
    save_data(df_clean,  output_file)
    print(f"Dataset limpio guardado en: {output_file}")

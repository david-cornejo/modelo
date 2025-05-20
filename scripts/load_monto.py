#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.dialects.postgresql import insert as pg_insert
from datetime import datetime

# 1) Lee el CSV y agrega ventas a nivel mensual (sumas de mes‐fin)
df = (
    pd.read_csv("../data/processed/merged/ventas_2015-2024.csv",
                parse_dates=["Fecha"], index_col="Fecha")
      ["Cargos"]
      .resample("M")         # Month-end
      .sum()
)

# 2) Conecta a la base de datos
DATABASE_URL = "postgresql://admin_timbrella:Timbrella1029.@api.timbrela.com:15432/modelo_test"
engine = create_engine(DATABASE_URL)
metadata = MetaData(bind=engine)
pred_table = Table("Predicciones", metadata, autoload_with=engine)

# 3) Upsert de monto_real mes a mes
with engine.begin() as conn:
    for fecha, monto_real in df.items():
        anio = fecha.year
        mes  = fecha.month
        now  = datetime.now()

        stmt = pg_insert(pred_table).values(
            anio=anio,
            mes=mes,
            id_empresa=1,
            monto_real=float(monto_real),
            fecha_actualizacion=now,
            updatedAt=now,
            createdAt=now
        ).on_conflict_do_update(
            index_elements=["anio", "mes", "id_empresa"],
            set_={
                "monto_real": monto_real,
                "fecha_actualizacion": now,
                "updatedAt": now
            }
        )
        conn.execute(stmt)

print("✅ monto_real cargado en Predicciones según la misma agregación mensual del CSV")
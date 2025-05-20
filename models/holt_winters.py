#!/usr/bin/env python3
import numpy as np
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.dialects.postgresql import insert as pg_insert
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# 1) Conexión a la DB (ajusta URL si cambia)
DATABASE_URL = "postgresql://admin_timbrella:Timbrella1029.@api.timbrela.com:15432/modelo_test"
engine = create_engine(DATABASE_URL)

# 2) Reflejar tabla Predicciones para el UPSERT
metadata   = MetaData()
pred_table = Table("Predicciones", metadata, autoload_with=engine)

# 3) Cargar datos reales mensuales
sql_hist = """
SELECT anio, mes, monto_real
FROM "Predicciones"
WHERE id_empresa = 1
  AND monto_real IS NOT NULL
ORDER BY anio, mes
"""
df = pd.read_sql(sql_hist, engine)

if df.empty:
    raise RuntimeError(
        "No se encontraron registros con `monto_real` en Predicciones. "
        "Asegúrate de haber cargado allí tus datos históricos."
    )

# 4) Construir serie con índice datetime y frecuencia mensual (Month Start)
df["fecha"] = pd.to_datetime(
    df["anio"].astype(str) + "-"
    + df["mes"].astype(str).str.zfill(2)
    + "-01"
)
df.set_index("fecha", inplace=True)
serie = df["monto_real"].resample("MS").sum()

# 5) Suavizado de outliers (IQR + media móvil)
def smooth_outliers_mean(s: pd.Series, window: int = 5) -> pd.Series:
    arr = s.dropna()
    if arr.empty:
        return s
    q1, q3 = np.percentile(arr, [15, 85])
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    roll = s.rolling(window=window, center=True, min_periods=1).mean()
    s_smooth = s.copy()
    mask = (s < lower) | (s > upper)
    s_smooth[mask] = roll[mask]
    return s_smooth

serie_smooth = smooth_outliers_mean(serie, window=5)

# 6) Split train/test (85% train, 15% test)
n = int(len(serie_smooth) * 0.85)
train, test = serie_smooth.iloc[:n], serie_smooth.iloc[n:]
if test.empty:
    raise RuntimeError(
        f"Tras el split 85/15 quedan 0 puntos para test (serie length={len(serie_smooth)})."
    )

# 7) Ajuste Holt–Winters
hw_model = ExponentialSmoothing(
    train,
    trend="add",
    seasonal="add",
    seasonal_periods=32
).fit()
pred = hw_model.forecast(len(test))

# 8) Métricas
mae  = mean_absolute_error(test, pred)
mape = mean_absolute_percentage_error(test, pred) * 100
print(f"\nHolt–Winters -> MAE: {mae:.2f}, MAPE: {mape:.2f}%")
print(f"Error absoluto medio: {mae:.2f}\n")

# 9) Comparativa mes a mes
print("Valores reales vs predichos:")
for fecha, real, p in zip(test.index, test.values, pred.values):
    print(f"  {fecha.strftime('%Y-%m')} → Real: {real:.2f}, Predicho: {p:.2f}")

# 10) UPSERT de predicciones y desviaciones en la base
with engine.begin() as conn:
    for fecha, real, p in zip(test.index, test.values, pred.values):
        y, m = fecha.year, fecha.month
        deviation = abs(real - p) / real * 100 if real != 0 else None

        record = {
            "anio": y,
            "mes": m,
            "id_empresa": 1,
            "monto_predicho": float(p),
            "desviacion": deviation,
            "fecha_actualizacion": datetime.now(),
            "updatedAt": datetime.now()
        }

        stmt = pg_insert(pred_table).values(**record).on_conflict_do_update(
            index_elements=["anio", "mes", "id_empresa"],
            set_={
                "monto_predicho": record["monto_predicho"],
                "desviacion":      record["desviacion"],
                "fecha_actualizacion": record["fecha_actualizacion"],
                "updatedAt":       record["updatedAt"]
            }
        )
        conn.execute(stmt)

print("\n✅ Predicciones y desviaciones guardadas en Predicciones.")
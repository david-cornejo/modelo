import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
from scipy.stats import randint, uniform

# ───────────────────────── Parámetros ─────────────────────────
DATA_PATH    = "../data/processed/merged/ventas_2015-2024.csv"
TEST_HORIZON = 18      # últimos meses para test
LAGS         = 12      # cuántos retardos básicos
RANDOM_STATE = 68
# ──────────────────────────────────────────────────────────────

# 1) Cargo y agrego a nivel mensual
df = (
    pd.read_csv(DATA_PATH, parse_dates=["Fecha"], index_col="Fecha")
      .assign(Cargos=lambda d: pd.to_numeric(d["Cargos"], errors="coerce"))
      .dropna(subset=["Cargos"])
)
serie = df["Cargos"].resample("M").sum()

# 2) Suavizado robusto de outliers
def smooth_outliers(s, q_low=15, q_high=85, factor=1.5, window=5):
    q1, q3 = np.percentile(s, [q_low, q_high])
    iqr = q3 - q1
    low, up = q1 - factor*iqr, q3 + factor*iqr
    med = s.rolling(window, center=True, min_periods=1).median()
    out = s.copy()
    mask = (s < low) | (s > up)
    out[mask] = med[mask]
    return out

serie_s = smooth_outliers(serie)

# 3) Construyo DataFrame de features
df_feat = pd.DataFrame({"y": serie_s})
# a) retardos
for lag in range(1, LAGS+1):
    df_feat[f"lag_{lag}"] = df_feat["y"].shift(lag)
# b) media y desviación móvil a 3,6,12 meses
for w in (3,6,12):
    df_feat[f"roll_mean_{w}"] = df_feat["y"].rolling(w).mean()
    df_feat[f"roll_std_{w}"]  = df_feat["y"].rolling(w).std()
# c) variable estacional: y - y_{t-12}
df_feat["seasonal_diff_12"] = df_feat["y"] - df_feat["y"].shift(12)
# d) calendario: mes cíclico
df_feat["month"]       = df_feat.index.month
df_feat["month_sin"]   = np.sin(2*np.pi*df_feat["month"]/12)
df_feat["month_cos"]   = np.cos(2*np.pi*df_feat["month"]/12)

df_feat.dropna(inplace=True)

# 4) split train/test
train = df_feat.iloc[:-TEST_HORIZON]
test  = df_feat.iloc[-TEST_HORIZON:]

X_train, y_train = train.drop("y",1), train["y"]
X_test,  y_test  = test .drop("y",1),  test["y"]

# 5) Hiperparámetros a explorar
param_distribs = {
    "max_iter":       randint(100, 500),
    "learning_rate":  uniform(0.01, 0.3),
    "max_depth":      randint(3, 10),
    "min_samples_leaf": randint(10, 100),
}

tscv = TimeSeriesSplit(n_splits=3)
base_model = HistGradientBoostingRegressor(random_state=RANDOM_STATE)

rs = RandomizedSearchCV(
    base_model,
    param_distributions=param_distribs,
    n_iter=30,
    cv=tscv,
    scoring="neg_mean_absolute_percentage_error",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1
)
rs.fit(X_train, y_train)

print("Mejores params:", rs.best_params_)
best_model = rs.best_estimator_

# 6) predicción y métrica
y_pred = best_model.predict(X_test)
mape_bt = mean_absolute_percentage_error(y_test, y_pred)*100
print(f"\nBoosted Trees mejorado — MAPE: {mape_bt:.2f}%")

# 7) gráfica
plt.figure(figsize=(12,5))
plt.plot(serie_s,         color="gray", alpha=0.25, label="Real (suavizado)")
plt.plot(serie_s.index[:-TEST_HORIZON], serie_s.iloc[:-TEST_HORIZON],
         label="Train", color="C0")
plt.plot(serie_s.index[-TEST_HORIZON:], serie_s.iloc[-TEST_HORIZON:],
         label="Test",  color="black")
plt.plot(y_test.index, y_pred, "--", label="Boosted pred", color="C2")
plt.title(f"Boosted Trees Tuned — MAPE={mape_bt:.1f}%")
plt.xlabel("Fecha"); plt.ylabel("Ventas mensuales")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()

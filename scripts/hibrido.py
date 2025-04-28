import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import randint, uniform

# ───────────────────────── Parámetros ─────────────────────────
DATA_PATH    = "../data/processed/merged/ventas_2015-2024.csv"
TEST_FRAC    = 0.15         # % finales para test
LAGS         = 12           # retardos para ML
RANDOM_STATE = 68
# ──────────────────────────────────────────────────────────────

# 1) Carga y agregación mensual
df = (
    pd.read_csv(DATA_PATH, parse_dates=["Fecha"], index_col="Fecha")
      .assign(Cargos=lambda d: pd.to_numeric(d["Cargos"], errors="coerce"))
      .dropna(subset=["Cargos"])
)
serie = df["Cargos"].resample("M").sum()

# 2) Suavizado robusto de outliers (IQR + media móvil)
def smooth_outliers(s, q_low=15, q_high=85, factor=1.5, window=5):
    q1, q3 = np.percentile(s, [q_low, q_high])
    iqr = q3 - q1
    low, high = q1 - factor*iqr, q3 + factor*iqr
    med = s.rolling(window, center=True, min_periods=1).median()
    out = s.copy()
    mask = (s < low) | (s > high)
    out[mask] = med[mask]
    return out

serie_s = smooth_outliers(serie)

# 3) Split train / test
n_test = int(len(serie_s) * TEST_FRAC)
train_s, test_s = serie_s.iloc[:-n_test], serie_s.iloc[-n_test:]
dates_test = test_s.index

# 4) Holt–Winters sobre suavizada
hw = ExponentialSmoothing(
    train_s,
    trend="add",
    seasonal="add",
    seasonal_periods=12
).fit()
hw_fore = hw.forecast(n_test)

# 5) Residuales en train
resid_train = train_s - hw.fittedvalues

# 6) Construir dataset de features para residuos
def make_features(s: pd.Series, lags):
    df = pd.DataFrame({"resid": s})
    for lag in range(1, lags+1):
        df[f"lag_{lag}"] = df["resid"].shift(lag)
    for w in (3,6,12):
        df[f"roll_mean_{w}"] = df["resid"].rolling(w).mean()
        df[f"roll_std_{w}"]  = df["resid"].rolling(w).std()
    df["seasonal_diff_12"] = df["resid"] - df["resid"].shift(12)
    df["month"]     = df.index.month
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
    return df.dropna()

feat_resid = make_features(resid_train, LAGS)
Xr_train = feat_resid.drop("resid", axis=1)
yr_train = feat_resid["resid"]

# 7) Hiper-búsqueda y entrenamiento de HistGradientBoosting sobre residuos
param_dist = {
    "max_iter":         randint(100, 500),
    "learning_rate":    uniform(0.01, 0.3),
    "max_depth":        randint(3, 10),
    "min_samples_leaf": randint(10, 100),
}
tscv = TimeSeriesSplit(n_splits=3)
base = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
rs = RandomizedSearchCV(
    base,
    param_distributions=param_dist,
    n_iter=30,
    cv=tscv,
    scoring="neg_mean_absolute_percentage_error",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1
)
rs.fit(Xr_train, yr_train)
best_model = rs.best_estimator_
print("Mejores params ML:", rs.best_params_)

# 8) Pronóstico recursivo de residuos
res_hist = resid_train.copy()
res_preds = []
for date in dates_test:
    # construir vector de features para 'date'
    feat = {}
    for lag in range(1, LAGS+1):
        feat[f"lag_{lag}"] = res_hist.shift(lag).iloc[-1]
    for w in (3,6,12):
        feat[f"roll_mean_{w}"] = res_hist.rolling(w).mean().iloc[-1]
        feat[f"roll_std_{w}"]  = res_hist.rolling(w).std().iloc[-1]
    feat["seasonal_diff_12"] = res_hist.iloc[-1] - res_hist.shift(12).iloc[-1]
    m = date.month
    feat["month"]     = m
    feat["month_sin"] = np.sin(2*np.pi*m/12)
    feat["month_cos"] = np.cos(2*np.pi*m/12)
    
    X0 = pd.DataFrame([feat], index=[date])
    r = best_model.predict(X0)[0]
    res_preds.append(r)
    res_hist = pd.concat([res_hist, pd.Series(r, index=[date])])

# 9) Construir pronóstico híbrido
hybrid_for = hw_fore.values + np.array(res_preds)
hybrid_for = pd.Series(hybrid_for, index=dates_test)

# 10) Métricas
mae_hw  = mean_absolute_error(test_s, hw_fore)
mape_hw = mean_absolute_percentage_error(test_s, hw_fore)*100
mae_hy  = mean_absolute_error(test_s, hybrid_for)
mape_hy = mean_absolute_percentage_error(test_s, hybrid_for)*100

print(f"Holt–Winters   → MAE: {mae_hw:.2f}, MAPE: {mape_hw:.2f}%")
print(f"Híbrido (HW+ML) → MAE: {mae_hy:.2f}, MAPE: {mape_hy:.2f}%")

# 11) Gráfica comparativa
plt.figure(figsize=(12,6))
plt.plot(train_s,                       label="Train (suavizado)", color="C0")
plt.plot(serie_s.loc[train_s.index[-1]:], label="Original Test",   color="gray", alpha=0.3)
plt.plot(test_s,                        label="Test (suavizado)",  color="C1")
plt.plot(hw_fore,    "--",  label="HW forecast",   color="C2")
plt.plot(hybrid_for,"-.",  label="Híbrido forecast",color="C3")
plt.axvline(train_s.index[-1], color="black", linestyle=":")
plt.legend()
plt.title("Holt–Winters vs Modelo Híbrido")
plt.xlabel("Fecha")
plt.ylabel("Cargos (Ventas)")
plt.grid(True)
plt.show()
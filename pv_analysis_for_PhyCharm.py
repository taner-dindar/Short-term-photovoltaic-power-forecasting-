import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense

import shap


# =========================
# SETTINGS
# =========================
file_name = "data_v1.csv"
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)


# =========================
# HELPER FUNCTIONS
# =========================
def save_plot(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()


def calc_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def create_sequences(X_data, y_data, time_steps=24):
    Xs, ys = [], []
    for i in range(time_steps, len(X_data)):
        Xs.append(X_data[i - time_steps:i])
        ys.append(y_data[i])
    return np.array(Xs), np.array(ys)


# =========================
# 1. DATA LOAD
# =========================
print("===== DATA LOAD STARTED =====")

df = pd.read_csv(file_name)

print("İlk veri boyutu:", df.shape)
print("\nSütunlar:")
print(df.columns.tolist())

df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
df = df.dropna(subset=["Time"]).copy()
df = df.sort_values("Time").reset_index(drop=True)

print("\nTime dönüşümü tamamlandı.")


# =========================
# 2. DROP UNUSED COLUMNS
# =========================
drop_cols = [
    "DailyAP",
    "DailyIrrediation",
    "Year",
    "Month",
    "month",
    "Month_Name",
    "Minute",
    "Season",
    "date",
    "snowfall_height",
    "lightning_potential"
]

df = df.drop(columns=drop_cols, errors="ignore")

print("\nDrop sonrası veri boyutu:", df.shape)
print("\nVeri tipleri:")
print(df.dtypes)

object_cols = df.select_dtypes(include=["object"]).columns.tolist()
print("\nObject/String sütunlar:")
print(object_cols)

for col in object_cols:
    if col != "Time":
        df[col] = pd.to_numeric(df[col], errors="coerce")


# =========================
# 3. CREATE LAG FEATURES
# =========================
print("\n===== LAG FEATURES =====")

df["lag_1"] = df["AP"].shift(1)
df["lag_4"] = df["AP"].shift(4)
df["lag_8"] = df["AP"].shift(8)
df["lag_96"] = df["AP"].shift(96)

df = df.dropna().reset_index(drop=True)

print("Lag sonrası veri boyutu:", df.shape)


# =========================
# 4. FEATURE / TARGET
# =========================
print("\n===== FEATURE / TARGET =====")

y = df["AP"]
X = df.drop(columns=["AP", "Time"], errors="ignore")
X = X.select_dtypes(include=[np.number])

print("\nKullanılan featurelar:")
print(X.columns.tolist())

print("\nFeature veri tipleri:")
print(X.dtypes)

print("\nFinal X shape:", X.shape)
print("Final y shape:", y.shape)

split = int(len(df) * 0.8)

X_train = X.iloc[:split]
y_train = y.iloc[:split]

X_test = X.iloc[split:]
y_test = y.iloc[split:]

print("\nTrain shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)


# =========================
# 5. RANDOM FOREST
# =========================
print("\n===== RANDOM FOREST =====")

rf_model = RandomForestRegressor(
    n_estimators=150,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = calc_rmse(y_test, y_pred_rf)

print("RF MAE:", mae_rf)
print("RF RMSE:", rmse_rf)

plt.figure(figsize=(15, 6))
plt.plot(y_test.values[:500], label="Actual Power")
plt.plot(y_pred_rf[:500], label="Predicted Power")
plt.legend()
plt.title("Photovoltaic Power Prediction (First 500 Test Samples) - Random Forest")
plt.xlabel("Time Step")
plt.ylabel("Power Output")
save_plot("rf_prediction.png")

importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print("\nTop 15 Feature Importances:")
print(importance_df.head(15))

importance_df.head(15).to_csv(
    os.path.join(output_dir, "rf_top15_feature_importance.csv"),
    index=False
)

top_features = importance_df.head(15)

plt.figure(figsize=(10, 6))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.gca().invert_yaxis()
plt.title("Feature Importance (Random Forest Model)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
save_plot("rf_feature_importance.png")


# =========================
# 6. XGBOOST
# =========================
print("\n===== XGBOOST =====")

xgb_model = XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = calc_rmse(y_test, y_pred_xgb)

print("XGBoost MAE:", mae_xgb)
print("XGBoost RMSE:", rmse_xgb)


# =========================
# 7. LSTM
# =========================
print("\n===== LSTM =====")

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

print("Scaled data shapes:")
print("X_train_scaled:", X_train_scaled.shape)
print("X_test_scaled:", X_test_scaled.shape)
print("y_train_scaled:", y_train_scaled.shape)
print("y_test_scaled:", y_test_scaled.shape)

time_steps = 24
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, time_steps)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, time_steps)

print("\nSequence shapes:")
print("X_train_seq:", X_train_seq.shape)
print("y_train_seq:", y_train_seq.shape)
print("X_test_seq:", X_test_seq.shape)
print("y_test_seq:", y_test_seq.shape)

lstm_model = Sequential([
    Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1)
])

lstm_model.compile(optimizer="adam", loss="mse")

history = lstm_model.fit(
    X_train_seq,
    y_train_seq,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

y_pred_lstm_scaled = lstm_model.predict(X_test_seq, verbose=0)

y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm_scaled)
y_test_actual = scaler_y.inverse_transform(y_test_seq)

mae_lstm = mean_absolute_error(y_test_actual, y_pred_lstm)
rmse_lstm = calc_rmse(y_test_actual, y_pred_lstm)

print("LSTM MAE:", mae_lstm)
print("LSTM RMSE:", rmse_lstm)

plt.figure(figsize=(12, 6))
plt.plot(y_test_actual[:500], label="Actual Power")
plt.plot(y_pred_lstm[:500], label="Predicted Power (LSTM)")
plt.title("Photovoltaic Power Prediction using LSTM")
plt.xlabel("Time Step")
plt.ylabel("Power Output")
plt.legend()
save_plot("lstm_prediction.png")


# =========================
# 8. MODEL COMPARISON
# =========================
print("\n===== MODEL COMPARISON =====")

models = ["Random Forest", "XGBoost", "LSTM"]
mae_values = [mae_rf, mae_xgb, mae_lstm]
rmse_values = [rmse_rf, rmse_xgb, rmse_lstm]

comparison_df = pd.DataFrame({
    "Model": models,
    "MAE": mae_values,
    "RMSE": rmse_values
})

print(comparison_df)
comparison_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)

x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width / 2, mae_values, width, label="MAE")
plt.bar(x + width / 2, rmse_values, width, label="RMSE")
plt.xticks(x, models)
plt.ylabel("Error Value")
plt.title("Model Performance Comparison")
plt.legend()
save_plot("model_comparison.png")


# =========================
# 9. SHAP SUMMARY
# =========================
print("\n===== SHAP SUMMARY STARTED =====")

try:
    X_sample = X_test.iloc[:200]

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_sample, check_additivity=False)

    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    save_plot("shap_summary.png")

    print("SHAP summary kaydedildi.")
except Exception as e:
    print("SHAP summary kısmında hata oluştu:")
    print(e)


# =========================
# 10. SHAP INTERACTION
# =========================
print("\n===== SHAP INTERACTION STARTED =====")

try:
    X_inter = X_train.sample(20, random_state=42).copy()

    explainer = shap.TreeExplainer(rf_model)
    shap_inter = explainer.shap_interaction_values(X_inter)

    # writable array yap
    interaction_matrix = np.array(np.abs(shap_inter).mean(axis=0), copy=True)

    # diagonal sıfırla
    np.fill_diagonal(interaction_matrix, 0)

    interaction_df = pd.DataFrame(
        interaction_matrix,
        index=X_inter.columns,
        columns=X_inter.columns
    )

    top = (
        interaction_df.where(np.triu(np.ones(interaction_df.shape), k=1).astype(bool))
        .stack()
        .sort_values(ascending=False)
    )

    print("\nTop 10 Interactions:")
    print(top.head(10))

    top.head(10).to_csv(os.path.join(output_dir, "top10_shap_interactions.csv"))

    plt.figure(figsize=(10, 6))
    top.head(10).sort_values().plot(kind="barh")
    plt.title("Top Feature Interactions")
    plt.xlabel("Mean |SHAP interaction value|")
    save_plot("top_shap_interactions.png")

    print("SHAP interaction kaydedildi.")

except Exception as e:
    print("SHAP interaction kısmında hata oluştu:")
    print(e)

print("\n===== SCRIPT COMPLETED SUCCESSFULLY =====")
print(f"Çıktılar '{output_dir}' klasörüne kaydedildi.")

print("\n===== SHAP INTERACTION ONLY =====")


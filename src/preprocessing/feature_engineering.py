import pandas as pd
import numpy as np
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")

PROCESSED_PATH = os.path.join(DATA_DIR, "cart_abandonment_preprocessed.csv")
FEATURED_PATH = os.path.join(DATA_DIR, "cart_abandonment_featured.csv")
ENCODERS_PATH = os.path.join(DATA_DIR, "label_encoders.json")

df = pd.read_csv(PROCESSED_PATH)

with open(ENCODERS_PATH, "r") as f:
    label_encoders = json.load(f)

df["time_per_item"] = df.apply(
    lambda row: row["session_duration"] / row["num_items_carted"] if row["num_items_carted"] > 0 else 0, axis=1
)

weekend_labels = [
    label_encoders["day_of_week"]["Saturday"],
    label_encoders["day_of_week"]["Sunday"]
]

df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x in weekend_labels else 0)

df["pages_per_minute"] = df.apply(
    lambda row: row["num_pages_viewed"] / (row["session_duration"]/60) if row["session_duration"] > 0 else 0, axis=1
)

df["scroll_ratio"] = df.apply(
    lambda row: row["scroll_depth"] / row["num_pages_viewed"] if row["num_pages_viewed"] > 0 else 0, axis=1
)

df["avg_item_price"] = df.apply(
    lambda row: row["cart_value"] / row["num_items_carted"] if row["num_items_carted"] > 0 else 0, axis=1
)

df["shipping_to_cart_ratio"] = df.apply(
    lambda row: row["shipping_fee"] / (row["cart_value"] + 1), axis=1
)

df["discount_to_cart_ratio"] = df.apply(
    lambda row: row["discount_applied"] / (row["cart_value"] + 1), axis=1
)

df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"].astype(float) / 7)
df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"].astype(float) / 7)

df["time_sin"] = np.sin(2 * np.pi * df["time_of_day"].astype(float) / 4)
df["time_cos"] = np.cos(2 * np.pi * df["time_of_day"].astype(float) / 4)

X = df.select_dtypes(include=["float64", "int64"]).to_numpy()
cov_matrix = np.cov(X, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
idx = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, idx]
X_pca = np.dot(X, eigenvectors[:, :2])
df["pca1"] = X_pca[:, 0]
df["pca2"] = X_pca[:, 1]

if "abandoned" in df.columns:
    corr = df.corr(numeric_only=True)["abandoned"].sort_values(ascending=False)
    print("\n🔹 Feature correlations with target:")
    print(corr)

df["log_cart_value"] = np.log1p(df["cart_value"])
df["cart_value_bin"] = pd.qcut(df["cart_value"], q=3, labels=["Low", "Medium", "High"])

df.to_csv(FEATURED_PATH, index=False)
print(f"✅ Feature engineering complete. Saved as {FEATURED_PATH}")

print(df[["day_of_week", "day_sin", "day_cos", "time_of_day", "time_sin", "time_cos"]].head())
print(df[["day_of_week", "time_of_day"]].dtypes)

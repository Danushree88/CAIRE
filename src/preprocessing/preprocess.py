import pandas as pd
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")

RAW_PATH = os.path.join(DATA_DIR, "cart_abandonment_dataset.csv")
PROCESSED_PATH = os.path.join(DATA_DIR, "cart_abandonment_preprocessed.csv")
ENCODERS_PATH = os.path.join(DATA_DIR, "label_encoders.json")
SCALER_PATH = os.path.join(DATA_DIR, "scaler_info.json")

df = pd.read_csv(RAW_PATH)

for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['int64','float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

y = df["abandoned"]
X = df.drop("abandoned", axis=1)

id_cols = ["session_id", "user_id"]
id_data = X[id_cols]
X = X.drop(id_cols, axis=1)

categorical_cols = [
    "day_of_week", "time_of_day", "device_type", "browser", 
    "referral_source", "location", "most_viewed_category"
]

binary_categorical_cols = [
    "return_user", "has_viewed_shipping_info",
    "discount_applied", "free_shipping_eligible", "if_payment_page_reached"
]

numerical_cols = [col for col in X.columns if col not in categorical_cols and col not in binary_categorical_cols]

label_encoders = {}
for col in categorical_cols:
    unique_vals = X[col].unique()
    mapping = {val: idx for idx, val in enumerate(unique_vals)}
    label_encoders[col] = mapping
    X[col] = X[col].map(mapping)

scaler_info = {}
for col in numerical_cols:
    mean_val = X[col].mean()
    std_val = X[col].std()
    scaler_info[col] = {"mean": mean_val, "std": std_val}
    X[col] = (X[col] - mean_val) / std_val if std_val != 0 else 0

os.makedirs("data/processed", exist_ok=True)
processed_df = pd.concat([id_data, X, y], axis=1)
processed_df.to_csv(PROCESSED_PATH, index=False)

with open(ENCODERS_PATH, "w") as f:
    json.dump(label_encoders, f)

with open(SCALER_PATH, "w") as f:
    json.dump(scaler_info, f)

print(f"✅ Preprocessing complete! Saved to {PROCESSED_PATH}")

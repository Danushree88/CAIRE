import pandas as pd
import json

df = pd.read_csv("cart_abandonment_dataset.csv")

# Missing Values Handling
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['int64','float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

# Separate Features & Target
y = df["abandoned"]
X = df.drop("abandoned", axis=1)

id_cols = ["session_id", "user_id"]
id_data = X[id_cols]
X = X.drop(id_cols, axis=1)

# Encode Categorical
categorical_cols = [
    "day_of_week", "time_of_day", "device_type", "browser", 
    "referral_source", "location", "most_viewed_category"
]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

label_encoders = {}
for col in categorical_cols:
    unique_vals = X[col].unique()
    mapping = {val: idx for idx, val in enumerate(unique_vals)}
    label_encoders[col] = mapping
    X[col] = X[col].map(mapping)

# Standardize Numerical
scaler_info = {}
for col in numerical_cols:
    mean_val = X[col].mean()
    std_val = X[col].std()
    scaler_info[col] = {"mean": mean_val, "std": std_val}
    X[col] = (X[col] - mean_val) / std_val if std_val != 0 else 0

# Save Preprocessed Data
processed_df = pd.concat([id_data, X, y], axis=1)
processed_df.to_csv("cart_abandonment_preprocessed.csv", index=False)

# Save encoders & scalers
with open("label_encoders.json", "w") as f:
    json.dump(label_encoders, f)

with open("scaler_info.json", "w") as f:
    json.dump(scaler_info, f)

print("Preprocessing complete. Saved as cart_abandonment_preprocessed.csv")

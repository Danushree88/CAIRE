import pandas as pd
import numpy as np
import json

#  Load preprocessed data
df = pd.read_csv("cart_abandonment_preprocessed.csv")

with open("label_encoders.json", "r") as f:
    label_encoders = json.load(f)

# Feature Engineering
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

# Cyclical Encoding
df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"].astype(float) / 7)
df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"].astype(float) / 7)

df["time_sin"] = np.sin(2 * np.pi * df["time_of_day"].astype(float) / 4)
df["time_cos"] = np.cos(2 * np.pi * df["time_of_day"].astype(float) / 4)


# Save Featured Data

df.to_csv("cart_abandonment_featured.csv", index=False)

print("Feature engineering complete. Saved as cart_abandonment_featured.csv")

print(df[["day_of_week","day_sin","day_cos","time_of_day","time_sin","time_cos"]].head())
print(df[["day_of_week","time_of_day"]].dtypes)

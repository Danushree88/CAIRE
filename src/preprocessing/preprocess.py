import pandas as pd
import numpy as np
import json
import os

class CartAbandonmentPreprocessor:
    def __init__(self, raw_path, processed_path, encoders_path, scaler_path):
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.encoders_path = encoders_path
        self.scaler_path = scaler_path
        self.label_encoders = {}
        self.scaler_info = {}
        self.df = None
        self.X = None
        self.y = None
        self.id_data = None

        self.categorical_cols = [
            "day_of_week", "time_of_day", "device_type", "browser",
            "referral_source", "location", "most_viewed_category"
        ]
        self.binary_categorical_cols = [
            "return_user", "has_viewed_shipping_info", "discount_applied",
            "free_shipping_eligible", "if_payment_page_reached"
        ]
        self.to_standardize = ["session_duration", "num_pages_viewed", "scroll_depth"]

    def load_data(self):
        self.df = pd.read_csv(self.raw_path)

    def handle_missing_values(self):
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                if self.df[col].dtype in ['int64', 'float64']:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                else:
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)

    def split_features_target(self):
        self.y = self.df["abandoned"]
        self.X = self.df.drop("abandoned", axis=1)

        id_cols = ["session_id", "user_id"]
        self.id_data = self.X[id_cols]
        self.X = self.X.drop(id_cols, axis=1)

    def encode_categoricals(self):
        for col in self.categorical_cols:
            unique_vals = self.X[col].unique()
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            self.label_encoders[col] = mapping
            self.X[col] = self.X[col].map(mapping)

    def scale_features(self):
        # Standardize
        for col in self.to_standardize:
            mean_val = self.X[col].mean()
            std_val = self.X[col].std()
            self.scaler_info[col] = {"mean": mean_val, "std": std_val}
            self.X[col] = (self.X[col] - mean_val) / std_val if std_val != 0 else 0

        # Log + Standardize cart_value
        if "cart_value" in self.X.columns:
            self.X["cart_value"] = np.log1p(self.X["cart_value"])
            mean_val = self.X["cart_value"].mean()
            std_val = self.X["cart_value"].std()
            self.scaler_info["cart_value"] = {"mean": mean_val, "std": std_val}
            self.X["cart_value"] = (self.X["cart_value"] - mean_val) / std_val if std_val != 0 else 0

        # Log transform shipping_fee
        if "shipping_fee" in self.X.columns:
            self.X["shipping_fee"] = np.log1p(self.X["shipping_fee"])

    def save_processed_data(self):
        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
        processed_df = pd.concat([self.id_data, self.X, self.y], axis=1)
        processed_df.to_csv(self.processed_path, index=False)

        with open(self.encoders_path, "w") as f:
            json.dump(self.label_encoders, f)

        with open(self.scaler_path, "w") as f:
            json.dump(self.scaler_info, f)

    def run(self):
        self.load_data()
        self.handle_missing_values()
        self.split_features_target()
        self.encode_categoricals()
        self.scale_features()
        self.save_processed_data()

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    RAW_PATH = os.path.join(DATA_DIR, "cart_abandonment_dataset.csv")
    PROCESSED_PATH = os.path.join(DATA_DIR, "cart_abandonment_preprocessed.csv")
    ENCODERS_PATH = os.path.join(DATA_DIR, "label_encoders.json")
    SCALER_PATH = os.path.join(DATA_DIR, "scaler_info.json")

    preprocessor = CartAbandonmentPreprocessor(RAW_PATH, PROCESSED_PATH, ENCODERS_PATH, SCALER_PATH)
    preprocessor.run()

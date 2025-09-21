import pandas as pd
import numpy as np
import json
import os

class FeatureEngineer:
    def __init__(self, processed_path, encoders_path, featured_path):
        self.processed_path = processed_path
        self.encoders_path = encoders_path
        self.featured_path = featured_path
        self.df = pd.read_csv(processed_path)

        with open(encoders_path, "r") as f:
            self.label_encoders = json.load(f)

        self.pca_loadings_path = os.path.join("data", "pca_loadings.csv")
        self.pca_features = [
            "session_duration", "num_pages_viewed", "scroll_depth",
            "cart_value", "shipping_fee"
        ]
        self.loadings = None

    def create_features(self):
        df = self.df

        # Derived Features
        df["time_per_item"] = df.apply(
            lambda row: row["session_duration"] / row["num_items_carted"]
            if row["num_items_carted"] > 0 else 0,
            axis=1
        )

        weekend_labels = [
            self.label_encoders["day_of_week"]["Saturday"],
            self.label_encoders["day_of_week"]["Sunday"]
        ]
        df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x in weekend_labels else 0)

        df["pages_per_minute"] = df.apply(
            lambda row: row["num_pages_viewed"] / (row["session_duration"] / 60)
            if row["session_duration"] > 0 else 0,
            axis=1
        )

        df["scroll_ratio"] = df.apply(
            lambda row: row["scroll_depth"] / row["num_pages_viewed"]
            if row["num_pages_viewed"] > 0 else 0,
            axis=1
        )

        df["avg_item_price"] = df.apply(
            lambda row: row["cart_value"] / row["num_items_carted"]
            if row["num_items_carted"] > 0 else 0,
            axis=1
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

        self.df = df

    def apply_pca(self):
        df = self.df
        X = df[self.pca_features].to_numpy()

        # Covariance matrix & eigen decomposition
        cov_matrix = np.cov(X, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues & eigenvectors
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Project data to first 2 PCs
        X_pca = np.dot(X, eigenvectors[:, :2])
        df["pca1"] = X_pca[:, 0]
        df["pca2"] = X_pca[:, 1]
        self.df = df

        loadings = pd.DataFrame(
            eigenvectors[:, :2],
            index=self.pca_features,
            columns=["PC1", "PC2"]
        )
        self.loadings = loadings
        loadings.to_csv(self.pca_loadings_path)

    def correlation_report(self):
        if "abandoned" in self.df.columns:
            corr = self.df.corr(numeric_only=True)["abandoned"].sort_values(ascending=False)
            print("\nFeature correlations with target:")
            print(corr)

    def save_features(self):
        self.df.to_csv(self.featured_path, index=False)


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    PROCESSED_PATH = os.path.join(DATA_DIR, "cart_abandonment_preprocessed.csv")
    FEATURED_PATH = os.path.join(DATA_DIR, "cart_abandonment_featured.csv")
    ENCODERS_PATH = os.path.join(DATA_DIR, "label_encoders.json")

    fe = FeatureEngineer(PROCESSED_PATH, ENCODERS_PATH, FEATURED_PATH)
    fe.create_features()
    fe.apply_pca()
    fe.correlation_report()
    fe.save_features()


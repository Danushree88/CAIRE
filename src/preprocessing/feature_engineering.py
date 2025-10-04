import pandas as pd
import numpy as np
import json
import os

class FeatureEngineer:
    def __init__(self, processed_path, encoders_path, featured_path):
        self.processed_path = processed_path
        self.encoders_path = encoders_path
        self.featured_path = featured_path
        
        # Check if file exists before reading
        if not os.path.exists(processed_path):
            raise FileNotFoundError(f"Processed data file not found: {processed_path}")
        
        self.df = pd.read_csv(processed_path)

        if not os.path.exists(encoders_path):
            raise FileNotFoundError(f"Encoders file not found: {encoders_path}")
            
        with open(encoders_path, "r") as f:
            self.label_encoders = json.load(f)

        # PCA loadings should also be saved in data directory
        self.pca_loadings_path = os.path.join(os.path.dirname(processed_path), "pca_loadings.csv")
        # SAFE PCA features - only pre-abandonment behaviors
        self.pca_features = [
            "session_duration", "num_pages_viewed", "scroll_depth",
            "num_items_carted", "has_viewed_shipping_info"
        ]
        self.loadings = None

    def create_features(self):
        df = self.df

        # === SAFE DERIVED FEATURES (No data leakage) ===
        
        # Engagement intensity (safe alternative to pages_per_minute)
        df["engagement_intensity"] = df.apply(
            lambda row: row["num_pages_viewed"] / (row["session_duration"] + 1)
            if row["session_duration"] > 0 else 0,
            axis=1
        )

        # Scroll engagement (safe alternative to scroll_ratio)
        df["scroll_engagement"] = df.apply(
            lambda row: row["scroll_depth"] / max(row["num_pages_viewed"], 1),
            axis=1
        )

        # Weekend indicator
        weekend_labels = [
            self.label_encoders["day_of_week"]["Saturday"],
            self.label_encoders["day_of_week"]["Sunday"]
        ]
        df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x in weekend_labels else 0)

        # Cart complexity indicators
        df["has_multiple_items"] = (df["num_items_carted"] > 1).astype(int)
        df["has_high_engagement"] = (df["num_pages_viewed"] > df["num_pages_viewed"].median()).astype(int)
        
        # Research behavior indicator
        df["research_behavior"] = ((df["num_pages_viewed"] > 5) & 
                                  (df["has_viewed_shipping_info"] == 1)).astype(int)

        # Quick browse indicator
        df["quick_browse"] = ((df["session_duration"] < 300) &  # 5 minutes
                             (df["num_pages_viewed"] < 4)).astype(int)

        # User engagement score (composite metric)
        df["engagement_score"] = (
            (df["num_pages_viewed"] / df["num_pages_viewed"].max()) * 0.4 +
            (df["scroll_depth"] / df["scroll_depth"].max()) * 0.3 +
            (df["has_viewed_shipping_info"] * 0.3)
        )

        # Time-based engagement
        df["peak_hours"] = df["time_of_day"].isin([1, 2]).astype(int)  # Assuming 1,2 are peak hours
        df["returning_peak"] = df["return_user"] * df["peak_hours"]

        # === CYCLICAL ENCODING (Safe) ===
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"].astype(float) / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"].astype(float) / 7)

        df["time_sin"] = np.sin(2 * np.pi * df["time_of_day"].astype(float) / 4)
        df["time_cos"] = np.cos(2 * np.pi * df["time_of_day"].astype(float) / 4)

        self.df = df

    def apply_pca(self):
        df = self.df
        X = df[self.pca_features].to_numpy()

        # Standardize the data before PCA
        mean = X.mean(axis=0)       
        std = X.std(axis=0)        
        std[std == 0] = 1.0
        X_scaled = (X - mean) / std
        
        # Covariance matrix & eigen decomposition
        cov_matrix = np.cov(X_scaled, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues & eigenvectors
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Project data to first 2 PCs
        X_pca = np.dot(X_scaled, eigenvectors[:, :2])
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
        
        print("PCA applied with safe features:")
        print(loadings)

    def correlation_report(self):
        if "abandoned" in self.df.columns:
            corr = self.df.corr(numeric_only=True)["abandoned"].sort_values(ascending=False)
            print("\nFeature correlations with target (abandoned):")
            print("=" * 50)
            for feature, correlation in corr.items():
                if feature != "abandoned":
                    print(f"{feature:25} : {correlation:+.4f}")
            
            # Highlight strong predictors
            strong_predictors = corr[(abs(corr) > 0.1) & (corr.index != "abandoned")]
            if len(strong_predictors) > 0:
                print(f"\nStrong predictors (|corr| > 0.1):")
                for feature, correlation in strong_predictors.items():
                    print(f"  {feature:23} : {correlation:+.4f}")

    def feature_summary(self):
        """Print summary of created features"""
        new_features = [
            "engagement_intensity", "scroll_engagement", "is_weekend",
            "has_multiple_items", "has_high_engagement", "research_behavior",
            "quick_browse", "engagement_score", "peak_hours", "returning_peak",
            "day_sin", "day_cos", "time_sin", "time_cos", "pca1", "pca2"
        ]
        
        print(f"\nCreated {len(new_features)} new features:")
        for feature in new_features:
            if feature in self.df.columns:
                print(f"  ✓ {feature}")
            else:
                print(f"  ✗ {feature} (missing)")

    def save_features(self):
        self.df.to_csv(self.featured_path, index=False)
        print(f"\nFeatures saved to: {self.featured_path}")
        print(f"Final dataset shape: {self.df.shape}")


if __name__ == "__main__":
    # Get the current file directory (src/preprocessing/)
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Go up two levels to reach CAIRE/ then into data/
    PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")

    # Correct file paths
    PROCESSED_PATH = os.path.join(DATA_DIR, "cart_abandonment_preprocessed.csv")
    FEATURED_PATH = os.path.join(DATA_DIR, "cart_abandonment_featured.csv")
    ENCODERS_PATH = os.path.join(DATA_DIR, "label_encoders.json")

    print(f"Looking for files in: {DATA_DIR}")
    print(f"Processed path: {PROCESSED_PATH}")
    print(f"Encoders path: {ENCODERS_PATH}")
    
    # Check if files exist
    if not os.path.exists(PROCESSED_PATH):
        print(f"❌ ERROR: File not found - {PROCESSED_PATH}")
        print("Available files in data directory:")
        if os.path.exists(DATA_DIR):
            for file in os.listdir(DATA_DIR):
                print(f"  - {file}")
        else:
            print(f"Data directory doesn't exist: {DATA_DIR}")
    else:
        print("✅ All files found!")
        fe = FeatureEngineer(PROCESSED_PATH, ENCODERS_PATH, FEATURED_PATH)
        fe.create_features()
        fe.apply_pca()
        fe.correlation_report()
        fe.feature_summary()
        fe.save_features()
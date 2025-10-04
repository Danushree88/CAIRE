import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from eda import EDAApp
from src.preprocessing.preprocess import CartAbandonmentPreprocessor
from src.preprocessing.feature_engineering import FeatureEngineer


class BaseTab:
    def __init__(self, name):
        self.name = name

    @staticmethod
    @st.cache_data
    def load_data(file_path):
        return pd.read_csv(file_path)

    @staticmethod
    def ensure_data_directory():
        os.makedirs("data", exist_ok=True)


class EDATab(BaseTab):
    def __init__(self):
        super().__init__("EDA (Raw Data)")

    def run(self):
        st.header("Exploratory Data Analysis (Raw Dataset)")
        RAW_PATH = os.path.join("data", "cart_abandonment_dataset.csv")
        df = self.load_data(RAW_PATH)

        if df is None:
            st.error(f"Raw dataset not found at {RAW_PATH}")
            return

        app = EDAApp(df)
        app.dataset_overview()
        app.missing_values()

        st.subheader("Analysis")
        tabs = st.tabs(["Categorical", "Numerical", "Target", "Correlation"])

        with tabs[0]:
            col = st.selectbox("Select Categorical Column",
                               app.categorical_cols + app.binary_categorical_cols,
                               key="raw_cat")
            app.categorical_analysis(col)

        with tabs[1]:
            col = st.selectbox("Select Numerical Column", app.numerical_cols, key="raw_num")
            app.numerical_analysis(col)

        with tabs[2]:
            app.target_analysis()

        with tabs[3]:
            app.correlation_analysis()

        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Raw Data as CSV",
            data=csv,
            file_name="cart_abandonment_raw.csv",
            mime="text/csv",
        )


class PreprocessingTab(BaseTab):
    def __init__(self):
        super().__init__("Preprocessing")

    def run(self):
        st.header("Data Preprocessing")

        RAW_PATH = os.path.join("data", "cart_abandonment_dataset.csv")
        PROCESSED_PATH = os.path.join("data", "cart_abandonment_preprocessed.csv")
        ENCODERS_PATH = os.path.join("data", "label_encoders.json")
        SCALER_PATH = os.path.join("data", "scaler_info.json")

        df_raw = self.load_data(RAW_PATH)
        if df_raw is None:
            st.error(f"Raw dataset not found at {RAW_PATH}")
            return

        with st.spinner("Preprocessing data..."):
            preprocessor = CartAbandonmentPreprocessor(RAW_PATH, PROCESSED_PATH, ENCODERS_PATH, SCALER_PATH)
            preprocessor.run()

        df = self.load_data(PROCESSED_PATH)
        if df is None:
            st.error("Failed to load processed data")
            return

        app = EDAApp(df)
        app.dataset_overview()
        app.missing_values()

        st.subheader("Analysis")
        tabs = st.tabs(["Categorical", "Numerical", "Target", "Correlation"])

        with tabs[0]:
            col = st.selectbox("Select Categorical Column",
                               app.categorical_cols + app.binary_categorical_cols,
                               key="pre_cat")
            app.categorical_analysis(col)

        with tabs[1]:
            col = st.selectbox("Select Numerical Column", app.numerical_cols, key="pre_num")
            app.numerical_analysis(col)

        with tabs[2]:
            app.target_analysis()

        with tabs[3]:
            app.correlation_analysis()

        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Processed Data as CSV",
            data=csv,
            file_name="cart_abandonment_processed.csv",
            mime="text/csv",
        )


class FeatureEngineeringTab(BaseTab):
    def __init__(self):
        super().__init__("Feature Engineering")

    def run(self):
        st.header("Feature Engineering")

        PROCESSED_PATH = os.path.join("data", "cart_abandonment_preprocessed.csv")
        FEATURED_PATH = os.path.join("data", "cart_abandonment_featured.csv")
        ENCODERS_PATH = os.path.join("data", "label_encoders.json")
        PCA_LOADINGS_PATH = os.path.join("data", "pca_loadings.csv")

        df_processed = self.load_data(PROCESSED_PATH)
        if df_processed is None:
            st.error("Preprocessed dataset not found. Please run preprocessing first.")
            return

        with st.spinner("Engineering features..."):
            fe = FeatureEngineer(PROCESSED_PATH, ENCODERS_PATH, FEATURED_PATH)
            fe.create_features()
            fe.apply_pca()
            fe.correlation_report()
            fe.save_features()

        df = self.load_data(FEATURED_PATH)
        if df is None:
            st.error("Failed to load featured data")
            return

        sub_tabs = st.tabs(["New Features Overview", "PCA Results"])

        with sub_tabs[0]:
            st.subheader("Newly Added Features")
            new_features = [
                "engagement_intensity", "scroll_engagement", "is_weekend",
                "has_multiple_items", "has_high_engagement", "research_behavior",
                "quick_browse", "engagement_score", "peak_hours", "returning_peak",
                "day_sin", "day_cos", "time_sin", "time_cos", "pca1", "pca2"
            ]

            added = [f for f in new_features if f in df.columns]

            if added:
                feature_info = []
                for feature in added:
                    desc = df[feature].describe().to_dict()
                    feature_info.append({
                        "Feature": feature,
                        "Type": str(df[feature].dtype),
                        "Mean": round(desc.get("mean", 0), 3) if "mean" in desc else "-",
                        "Std": round(desc.get("std", 0), 3) if "std" in desc else "-",
                        "Min": round(desc.get("min", 0), 3) if "min" in desc else "-",
                        "Max": round(desc.get("max", 0), 3) if "max" in desc else "-",
                        "NA Count": int(df[feature].isna().sum())
                    })

                st.dataframe(pd.DataFrame(feature_info))

            else:
                st.info("No new features detected.")

            st.subheader("Full Dataset with Features")
            app = EDAApp(df)
            app.dataset_overview()
            app.missing_values()

            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Featured Data as CSV",
                data=csv,
                file_name="cart_abandonment_featured.csv",
                mime="text/csv",
            )

        with sub_tabs[1]:
            st.subheader("PCA Projection (2D)")

            if "pca1" in df.columns and "pca2" in df.columns:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(
                    data=df, x="pca1", y="pca2",
                    hue="abandoned" if "abandoned" in df.columns else None,
                    palette="Set2", alpha=0.7
                )
                ax.set_title("PCA Scatter Plot (pca1 vs pca2)")
                st.pyplot(fig)

                st.markdown("""
                **Inference from PCA:**
                - PCA1 captures the largest variance in user session behavior, strongly separating abandoned vs non-abandoned sessions.
                - PCA2 adds complementary variance but contributes less compared to PCA1.
                - The scatter plot shows clusters, suggesting that user behavior patterns differ significantly between the two groups.
                """)

                # Load PCA Feature Contributions
                if os.path.exists(PCA_LOADINGS_PATH):
                    loadings = pd.read_csv(PCA_LOADINGS_PATH, index_col=0)
                    st.subheader("PCA Feature Contributions")
                    st.write("**Contribution of original features to PC1 and PC2**")

                    st.dataframe(loadings.style.format("{:.4f}"))

                    st.markdown("**Top Contributors to PC1:**")
                    st.write(loadings["PC1"].abs().sort_values(ascending=False).head(5))

                    st.markdown("**Top Contributors to PC2:**")
                    st.write(loadings["PC2"].abs().sort_values(ascending=False).head(5))
                else:
                    st.info("PCA loadings not found. Please rerun feature engineering.")

            else:
                st.warning("PCA components not found in dataset.")

class CartAbandonmentDashboard:
    def __init__(self):
        self.tabs = [
            EDATab(),
            PreprocessingTab(),
            FeatureEngineeringTab()
        ]

    def run(self):
        st.set_page_config(page_title="Cart Abandonment Dashboard", layout="wide")
        st.title("Cart Abandonment Dashboard")

        BaseTab.ensure_data_directory()

        main_tabs = st.tabs([tab.name for tab in self.tabs])

        for tab, ui in zip(main_tabs, self.tabs):
            with tab:
                ui.run()


if __name__ == "__main__":
    dashboard = CartAbandonmentDashboard()
    dashboard.run()

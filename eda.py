import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from io import StringIO
import os

class EDAApp:
    def __init__(self, df, target="abandoned"):
        self.df = df
        self.target = target

        self.categorical_cols = [
            "day_of_week", "time_of_day", "device_type", "browser",
            "referral_source", "location", "most_viewed_category"
        ]
        self.binary_categorical_cols = [
            "return_user", "has_viewed_shipping_info",
            "discount_applied", "free_shipping_eligible", "if_payment_page_reached"
        ]
        self.numerical_cols = [
            "session_duration", "num_pages_viewed", "num_items_carted",
            "scroll_depth", "cart_value", "shipping_fee"
        ]

    def dataset_overview(self):
        st.header("Dataset Overview")
        st.write(self.df.head(10))
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", self.df.shape[0])
        c2.metric("Columns", self.df.shape[1])
        c3.metric("Missing Values", self.df.isnull().sum().sum())

        with st.expander("Dataset Info"):
            buffer = StringIO()
            self.df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)

    def missing_values(self):
        st.header("Missing Values")
        missing = self.df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)

        if not missing.empty:
            st.write(missing)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(self.df.isnull(), cbar=False, cmap="rocket", ax=ax)
            st.pyplot(fig)
        else:
            st.success("No missing values detected!")

    # Categorical Analysis
    def categorical_analysis(self, col):
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.countplot(data=self.df, x=col, palette="Set2", hue=self.df[col])
        plt.xticks(rotation=30)
        plt.title(f"Distribution of {col}")
        st.pyplot(fig)

        if self.target in self.df.columns:
            st.subheader(f"{col} vs Target ({self.target})")
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.countplot(x=self.target, data=self.df, hue=self.target, palette="pastel", legend=False)
            plt.xticks(rotation=30)
            st.pyplot(fig)

    # Numerical Analysis
    def numerical_analysis(self, col):
        data = self.df[col].dropna()
        skew_val = skew(data)
        kurt_val = kurtosis(data, fisher=False)
        mean_val = data.mean()
        median_val = data.median()
        q1, q3 = data.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = ((data < (q1 - 1.5*iqr)) | (data > (q3 + 1.5*iqr))).sum()

        st.write(f"**Summary of {col}:**")
        st.write({
            "Mean": round(mean_val, 2),
            "Median": round(median_val, 2),
            "Skewness": round(skew_val, 2),
            "Kurtosis": round(kurt_val, 2),
            "Outliers": int(outliers)
        })

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(data, kde=True, bins=30, ax=axes[0], color="skyblue")
        axes[0].set_title(f"{col} Distribution")
        sns.boxplot(x=data, ax=axes[1], color="lightcoral")
        axes[1].set_title(f"{col} Outliers")
        st.pyplot(fig)

        if self.target in self.df.columns:
            st.subheader(f"{col} vs Target ({self.target})")
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.kdeplot(data=self.df, x=col, hue=self.target, fill=True)
            st.pyplot(fig)

    # Target Analysis
    def target_analysis(self):
        if self.target in self.df.columns:
            st.subheader("Target Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x=self.target, data=self.df, color="skyblue")
            st.pyplot(fig)
        else:
            st.error(f"No target column '{self.target}' found!")

    # Correlation Analysis
    def correlation_analysis(self):
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(self.df[self.numerical_cols].corr(), annot=False, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

def main():
    st.set_page_config(page_title="Cart Abandonment EDA", layout="wide")
    st.title("Cart Abandonment EDA Dashboard")

    csv_path = os.path.join("data", "cart_abandonment_dataset.csv")

    if not os.path.exists(csv_path):
        st.error(f"CSV file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    app = EDAApp(df)

    app.dataset_overview()
    app.missing_values()

    eda_type = st.sidebar.radio("Select Analysis", ["Categorical", "Numerical", "Target", "Correlation"])

    if eda_type == "Categorical":
        col = st.sidebar.selectbox("Select Column", app.categorical_cols + app.binary_categorical_cols)
        app.categorical_analysis(col)

    elif eda_type == "Numerical":
        col = st.sidebar.selectbox("Select Column", app.numerical_cols)
        app.numerical_analysis(col)

    elif eda_type == "Target":
        app.target_analysis()

    elif eda_type == "Correlation":
        app.correlation_analysis()


if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import json
from datetime import datetime

# FIXED: Add path and import model classes FIRST
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Import model classes BEFORE loading the pickle file
from model import GradientBoostingClassifierManual, RandomForestManual, DecisionTreeClassifierManual, LogisticRegressionGD, KNNClassifier

# PATH CONFIGURATION FOR DEBUGGING
MODELS_DIR = BASE_DIR
DATA_DIR = os.path.join(BASE_DIR, "data")
"""
st.sidebar.markdown("---")
st.sidebar.write("**Path Debug:**")
st.sidebar.write(f"BASE_DIR: {BASE_DIR}")
st.sidebar.write(f"DATA_DIR: {DATA_DIR}")
st.sidebar.write(f"DATA_DIR exists: {os.path.exists(DATA_DIR)}")"""

model_path = os.path.join(MODELS_DIR, "final_manual_model.pkl")

def load_model():
    """Fixed model loading with proper class imports"""
    try:
        # Ensure model classes are available
        import model as model_module
        sys.modules['model'] = model_module
        sys.modules['__main__'] = model_module
        
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
            
        if isinstance(model_data, dict) and "model" in model_data:
            loaded_model = model_data["model"]
            feature_names = model_data.get("feature_names", [])
            best_params = model_data.get("best_params", {})
            
            # Store additional info as attributes
            loaded_model.feature_names = feature_names
            loaded_model.best_params = best_params
            
            st.success("✅ Model loaded successfully from dictionary!")
            st.info(f"📊 Model Type: {best_params.get('model_type', 'Unknown')}")
            
        else:
            loaded_model = model_data
            st.success("✅ Model loaded successfully!")
            
        return loaded_model
        
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        
        # Show available pickle files
        st.info("📁 Available .pkl files:")
        pkl_files = [f for f in os.listdir(BASE_DIR) if f.endswith('.pkl')]
        if pkl_files:
            for file in pkl_files:
                st.write(f"- `{file}`")
        else:
            st.write("No .pkl files found")
        return None

# Load the model
loaded_model = load_model()

# =========================================================
# Data Preprocessing Class (Replicates your pipeline)
# =========================================================
class PredictionPreprocessor:
    def __init__(self):
        # Load encoders and scalers from your training - CORRECTED PATHS
        self.encoders_path = os.path.join(DATA_DIR, "label_encoders.json")
        self.scaler_path = os.path.join(DATA_DIR, "scaler_info.json")
        self.pca_loadings_path = os.path.join(DATA_DIR, "pca_loadings.csv")
        
        self.label_encoders = {}
        self.scaler_info = {}
        self.pca_loadings = None
        
        self.load_preprocessing_artifacts()
    
    def load_preprocessing_artifacts(self):
        """Load the encoders and scalers used during training
        try:
            # Show file paths for debugging
            st.sidebar.markdown("---")
            st.sidebar.write("**Preprocessing Files:**")
            st.sidebar.write(f"Encoders: {self.encoders_path}")
            st.sidebar.write(f"Exists: {os.path.exists(self.encoders_path)}")
            st.sidebar.write(f"Scaler: {self.scaler_path}")
            st.sidebar.write(f"Exists: {os.path.exists(self.scaler_path)}")
            st.sidebar.write(f"PCA: {self.pca_loadings_path}")
            st.sidebar.write(f"Exists: {os.path.exists(self.pca_loadings_path)}")
            
            # Check if files exist first
            if not os.path.exists(self.encoders_path):
                st.error(f"❌ Encoders file not found: {self.encoders_path}")
                return
            
            with open(self.encoders_path, "r") as f:
                self.label_encoders = json.load(f)
            
            if not os.path.exists(self.scaler_path):
                st.error(f"❌ Scaler file not found: {self.scaler_path}")
                return
                
            with open(self.scaler_path, "r") as f:
                self.scaler_info = json.load(f)
                
            if os.path.exists(self.pca_loadings_path):
                self.pca_loadings = pd.read_csv(self.pca_loadings_path, index_col=0)
                st.sidebar.success("✅ PCA loadings loaded!")
            else:
                st.sidebar.warning("⚠️ PCA loadings file not found")
                
            st.sidebar.success("✅ Preprocessing artifacts loaded!")
            
            # Debug: Show what was loaded
            st.sidebar.write(f"Encoders: {list(self.label_encoders.keys())}")
            
        except Exception as e:
            st.error(f"❌ Error loading preprocessing artifacts: {e}")
            import traceback
            st.error(traceback.format_exc())"""
    
    def preprocess_raw_data(self, raw_data):
        """Replicate the exact preprocessing pipeline from training"""
        processed_data = raw_data.copy()
        
        # Handle missing values (same as training)
        for col in processed_data.columns:
            if processed_data[col].isnull().sum() > 0:
                if processed_data[col].dtype in ['int64', 'float64']:
                    processed_data[col].fillna(processed_data[col].median(), inplace=True)
                else:
                    processed_data[col].fillna(processed_data[col].mode()[0], inplace=True)
        
        return processed_data
    
    def encode_categoricals(self, data):
        encoded_data = data.copy()
        
        categorical_cols = ["day_of_week", "time_of_day", "device_type", "browser", "referral_source", "location", "most_viewed_category"]
        
        for col in categorical_cols:
            if col in encoded_data.columns and col in self.label_encoders:
                encoded_data[col] = encoded_data[col].map(self.label_encoders[col])
                encoded_data[col] = encoded_data[col].fillna(-1)
        
        # ADD THIS: Convert all categorical columns to numeric
        for col in categorical_cols:
            if col in encoded_data.columns:
                encoded_data[col] = pd.to_numeric(encoded_data[col], errors='coerce').fillna(-1)
        
        return encoded_data
    
    def scale_features(self, data):
        """Apply the same scaling as training"""
        scaled_data = data.copy()
        
        # Standardize features
        to_standardize = ["session_duration", "num_pages_viewed", "scroll_depth"]
        for col in to_standardize:
            if col in scaled_data.columns and col in self.scaler_info:
                mean_val = self.scaler_info[col]["mean"]
                std_val = self.scaler_info[col]["std"]
                scaled_data[col] = (scaled_data[col] - mean_val) / std_val if std_val != 0 else 0
        
        # Log + Standardize cart_value
        if "cart_value" in scaled_data.columns and "cart_value" in self.scaler_info:
            scaled_data["cart_value"] = np.log1p(scaled_data["cart_value"])
            mean_val = self.scaler_info["cart_value"]["mean"]
            std_val = self.scaler_info["cart_value"]["std"]
            scaled_data["cart_value"] = (scaled_data["cart_value"] - mean_val) / std_val if std_val != 0 else 0
        
        # Log transform shipping_fee
        if "shipping_fee" in scaled_data.columns:
            scaled_data["shipping_fee"] = np.log1p(scaled_data["shipping_fee"])
        
        return scaled_data
    
    def engineer_features(self, data):
        """Replicate the exact feature engineering from training"""
        df = data.copy()
        
        # Check if encoders are loaded
        if not self.label_encoders:
            st.error("❌ Label encoders not loaded. Cannot engineer features.")
            return df
        
        # === SAFE DERIVED FEATURES (Same as training) ===
        
        # Engagement intensity
        df["engagement_intensity"] = df.apply(
            lambda row: row["num_pages_viewed"] / (row["session_duration"] + 1)
            if row["session_duration"] > 0 else 0,
            axis=1
        )

        # Scroll engagement
        df["scroll_engagement"] = df.apply(
            lambda row: row["scroll_depth"] / max(row["num_pages_viewed"], 1),
            axis=1
        )

        # Weekend indicator - SAFE CHECK
        if "day_of_week" in self.label_encoders:
            weekend_labels = [
                self.label_encoders["day_of_week"].get("Saturday", -1),
                self.label_encoders["day_of_week"].get("Sunday", -1)
            ]
            df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x in weekend_labels else 0)
        else:
            df["is_weekend"] = 0

        # Cart complexity indicators
        df["has_multiple_items"] = (df["num_items_carted"] > 1).astype(int)
        
        # Use median from training data
        training_median_pages = 8
        df["has_high_engagement"] = (df["num_pages_viewed"] > training_median_pages).astype(int)
        
        # Research behavior indicator
        df["research_behavior"] = ((df["num_pages_viewed"] > 5) & 
                                  (df["has_viewed_shipping_info"] == 1)).astype(int)

        # Quick browse indicator
        df["quick_browse"] = ((df["session_duration"] < 300) & 
                             (df["num_pages_viewed"] < 4)).astype(int)

        # User engagement score (composite metric)
        max_pages = 100
        max_scroll = 100
        df["engagement_score"] = (
            (df["num_pages_viewed"] / max_pages) * 0.4 +
            (df["scroll_depth"] / max_scroll) * 0.3 +
            (df["has_viewed_shipping_info"] * 0.3)
        )

        # Time-based engagement
        df["peak_hours"] = df["time_of_day"].isin([1, 2]).astype(int)
        df["returning_peak"] = df["return_user"] * df["peak_hours"]

        # === CYCLICAL ENCODING ===
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"].astype(float) / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"].astype(float) / 7)

        df["time_sin"] = np.sin(2 * np.pi * df["time_of_day"].astype(float) / 4)
        df["time_cos"] = np.cos(2 * np.pi * df["time_of_day"].astype(float) / 4)
        return df

    def transform_raw_to_engineered(self, raw_data):
        """Complete pipeline: Raw → Preprocessed → Encoded → Scaled → Engineered"""
        try:
            # Step 1: Preprocess (handle missing values)
            processed = self.preprocess_raw_data(raw_data)
            
            # Step 2: Encode categoricals
            encoded = self.encode_categoricals(processed)
            
            # Step 3: Scale features
            scaled = self.scale_features(encoded)
            
            # Step 4: Engineer features
            engineered = self.engineer_features(scaled)
            
            return engineered
            
        except Exception as e:
            st.error(f"❌ Error in transformation pipeline: {e}")
            import traceback
            st.error(traceback.format_exc())
            return raw_data  # Return original data as fallback

# =========================================================
# Updated Prediction Engine Class - FIXED FOR GRADIENT BOOSTING
# =========================================================
class PredictionEngine:
    def __init__(self):
        self.model = loaded_model
        self.preprocessor = PredictionPreprocessor()
        self.feature_names = self._get_feature_names()
        self.model_type = type(self.model).__name__ if self.model else None

    def _get_feature_names(self):
        if not self.model:
            return []
        
        # First try to get feature names from the loaded model data
        if hasattr(self.model, 'feature_names') and self.model.feature_names:
            return self.model.feature_names
        elif hasattr(self.model, 'best_params') and 'feature_names' in self.model.best_params:
            return self.model.best_params['feature_names']
        else:
            # Return the actual feature names from your engineered dataset
            st.warning("⚠️ Using default feature names from engineered dataset")
            return [
                'return_user', 'day_of_week', 'time_of_day', 'session_duration', 
                'num_pages_viewed', 'num_items_carted', 'has_viewed_shipping_info', 
                'scroll_depth', 'cart_value', 'discount_applied', 'shipping_fee', 
                'free_shipping_eligible', 'device_type', 'browser', 'referral_source', 
                'location', 'if_payment_page_reached', 'most_viewed_category',
                'engagement_intensity', 'scroll_engagement', 'is_weekend', 
                'has_multiple_items', 'has_high_engagement', 'research_behavior', 
                'quick_browse', 'engagement_score', 'peak_hours', 'returning_peak', 
                'day_sin', 'day_cos', 'time_sin', 'time_cos', 'pca1', 'pca2'
            ]

    def create_raw_data_inputs(self):
        """Create input fields for ORIGINAL raw features"""
        st.subheader("📋 Enter Raw Session Data")
        
        raw_features = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Session basics
            raw_features['session_duration'] = st.number_input(
                "Session Duration (seconds)", 0, 3600, 300, 30,
                help="How long the user stayed on site"
            )
            raw_features['num_pages_viewed'] = st.number_input(
                "Pages Viewed", 1, 100, 8, 1,
                help="Number of different pages visited"
            )
            raw_features['num_items_carted'] = st.number_input(
                "Number of Items in Cart", 0, 50, 2,
                help="Total items added to cart"
            )
            raw_features['scroll_depth'] = st.slider(
                "Scroll Depth (%)", 0, 100, 50,
                help="How far user scrolled (0-100%)"
            )
            raw_features['cart_value'] = st.number_input(
                "Cart Value ($)", 0.0, 10000.0, 500.0, 50.0,
                help="Total value of items in cart"
            )
            raw_features['shipping_fee'] = st.number_input(
                "Shipping Fee ($)", 0.0, 500.0, 10.0, 5.0,
                help="Shipping cost for the order"
            )
        
        with col2:
            # User and session attributes
            raw_features['return_user'] = st.selectbox(
                "Return User", [0, 1], format_func=lambda x: "New User" if x == 0 else "Return User"
            )
            raw_features['has_viewed_shipping_info'] = st.selectbox(
                "Viewed Shipping Info", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
            )
            raw_features['discount_applied'] = st.selectbox(
                "Discount Applied", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
            )
            raw_features['free_shipping_eligible'] = st.selectbox(
                "Free Shipping Eligible", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
            )
            raw_features['if_payment_page_reached'] = st.selectbox(
                "Reached Payment Page", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
            )
            
            # Categorical features
            raw_features['day_of_week'] = st.selectbox(
                "Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            )
            raw_features['time_of_day'] = st.selectbox(
                "Time of Day", ["Morning", "Afternoon", "Evening", "Night"]
            )
            raw_features['device_type'] = st.selectbox(
                "Device Type", ["Mobile", "Desktop", "Tablet"]
            )
        
        with st.expander("Additional Features (Optional)"):
            col3, col4 = st.columns(2)
            with col3:
                raw_features['browser'] = st.selectbox(
                    "Browser", ["Chrome", "Safari", "Firefox", "Edge", "Opera"]
                )
                raw_features['referral_source'] = st.selectbox(
                    "Referral Source", ["Direct", "Search Engine", "Social Media", "Email Campaign", "Ads"]
                )
            with col4:
                raw_features['location'] = st.selectbox(
                "Location", ["Mumbai, Maharashtra", "Delhi, Delhi", "Bangalore, Karnataka", 
               "Hyderabad, Telangana", "Chennai, Tamil Nadu", "Kolkata, West Bengal"]
                )
                raw_features['most_viewed_category'] = st.selectbox(
                    "Most Viewed Category", ["Electronics", "Clothing", "Home & Kitchen", 
                                           "Beauty", "Sports", "Automotive", "Books"]
                )
        
        return raw_features

    def render_single_prediction(self):
        """Render single prediction interface using RAW data"""
        st.markdown("### 🔮 Single Session Prediction (Raw Data)")
        
        if not self.model:
            st.error("Model not available. Please ensure final_manual_model.pkl exists.")
            return
        
        # Get raw data inputs
        raw_features = self.create_raw_data_inputs()

        # Predict button
        if st.button("🎯 Predict Abandonment Probability", type="primary", use_container_width=True):
            with st.spinner("Processing data and making prediction..."):
                self.make_prediction_from_raw(raw_features)

    def make_prediction_from_raw(self, raw_features):
        """Complete pipeline: Raw → Engineered → Prediction"""
        try:
            # Convert to DataFrame for processing
            raw_df = pd.DataFrame([raw_features])
            
            # Transform through complete pipeline
            engineered_df = self.preprocessor.transform_raw_to_engineered(raw_df)
            
            # DEBUG: Show feature engineering
            with st.expander("🔍 Data Transformation Details"):
                st.write("**Original Raw Features:**")
                st.json(raw_features)
                
                st.write("**Engineered Features:**")
                st.dataframe(engineered_df)
                
                # Check feature alignment
                st.write("**Feature Alignment:**")
                aligned_features = []
                for feature in self.feature_names:
                    if feature in engineered_df.columns:
                        value = engineered_df[feature].iloc[0]
                        aligned_features.append((feature, value))
                    else:
                        aligned_features.append((feature, "MISSING"))
                
                st.write("First 15 features:")
                for feature, value in aligned_features[:15]:
                    st.write(f"  - {feature}: {value}")
            
            # Create feature vector in correct order
            feature_vector = []
            for feature in self.feature_names:
                if feature in engineered_df.columns:
                    feature_vector.append(engineered_df[feature].iloc[0])
                else:
                    feature_vector.append(0)  # Default for missing features
            
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            # Get prediction probability - FIXED FOR GRADIENT BOOSTING
            if hasattr(self.model, 'predict_proba'):
                raw_probs = self.model.predict_proba(feature_vector)
                
                # DEBUG: Show raw probabilities
                with st.expander("🎯 Probability Details"):
                    st.write(f"Raw probabilities: {raw_probs}")
                    st.write(f"Raw probabilities type: {type(raw_probs)}")
                    if hasattr(raw_probs, 'shape'):
                        st.write(f"Raw probabilities shape: {raw_probs.shape}")
                
                prob_abandon = self._extract_probability(raw_probs)
            else:
                # Fallback for models without predict_proba
                prediction = self.model.predict(feature_vector)
                prob_abandon = float(prediction[0]) if prediction[0] in [0, 1] else 0.5
                st.warning("⚠️ Using predict() instead of predict_proba() - probabilities may not be accurate")
            
            prob_percent = prob_abandon * 100
            
            # Display results
            self.display_prediction_results(prob_percent, prob_abandon)
                
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            import traceback
            st.text(traceback.format_exc())

    def _extract_probability(self, raw_probs):
        """Robust probability extraction for GradientBoostingClassifierManual"""
        try:
            # Handle GradientBoostingClassifierManual specifically
            if isinstance(self.model, GradientBoostingClassifierManual):
                # For manual gradient boosting, it might return a single probability
                if isinstance(raw_probs, (int, float, np.number)):
                    return float(raw_probs)
                elif hasattr(raw_probs, '__len__') and len(raw_probs) == 1:
                    return float(raw_probs[0])
                elif hasattr(raw_probs, '__len__') and len(raw_probs) == 2:
                    # Assume [prob_class_0, prob_class_1]
                    return float(raw_probs[1])
                else:
                    # Try to extract from array
                    return float(raw_probs.flat[0])
            
            # For other models, use standard extraction
            if hasattr(raw_probs, 'shape'):
                if raw_probs.shape[1] == 2:
                    # [[prob_class_0, prob_class_1]]
                    return float(raw_probs[0, 1])
                elif raw_probs.shape[1] == 1:
                    # [[prob_class_1]]
                    return float(raw_probs[0, 0])
            elif hasattr(raw_probs, '__len__') and len(raw_probs) == 2:
                # [prob_class_0, prob_class_1]
                return float(raw_probs[1])
            else:
                # Single probability value
                return float(raw_probs[0])
                
        except Exception as e:
            st.warning(f"⚠️ Probability extraction warning: {e}. Using default 0.5")
            return 0.5

    def display_prediction_results(self, prob_percent, prob_abandon):
        """Display prediction results"""
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Abandonment Probability", 
                f"{prob_percent:.1f}%",
                delta=f"High risk" if prob_percent > 50 else "Low risk",
                delta_color="inverse"
            )
        
        with col2:
            prediction = 1 if prob_abandon > 0.5 else 0
            status = "🚨 LIKELY TO ABANDON" if prediction == 1 else "✅ LIKELY TO COMPLETE"
            st.metric("Prediction", status)
        
        with col3:
            confidence = max(prob_abandon, 1-prob_abandon) * 100
            st.metric("Model Confidence", f"{confidence:.1f}%")
        
        # Risk level indicator
        st.markdown("### 📊 Risk Assessment")
        if prob_percent >= 70:
            st.error("🔴 **HIGH RISK**: Strong likelihood of cart abandonment. Immediate recovery action recommended.")
            st.info("💡 **Suggested Action**: Send immediate discount email, free shipping offer, or personal follow-up")
        elif prob_percent >= 40:
            st.warning("🟡 **MEDIUM RISK**: Moderate chance of abandonment. Consider proactive engagement.")
            st.info("💡 **Suggested Action**: Send reminder email, highlight product benefits, or offer chat support")
        else:
            st.success("🟢 **LOW RISK**: Likely to complete purchase. Standard monitoring recommended.")
            st.info("💡 **Suggested Action**: Normal follow-up sequence, focus on customer satisfaction")

    def render_batch_prediction(self):
        """Render batch prediction interface for RAW data"""
        st.markdown("### 📊 Batch Prediction (Raw Data)")
        
        if not self.model:
            st.error("Model not available for batch predictions.")
            return
        
        st.info("""
        **Upload a CSV file with RAW session data for batch predictions.**
        The file should contain the same raw features as the original dataset.
        """)
        
        uploaded_file = st.file_uploader(
            "Upload RAW CSV file for batch prediction", 
            type=['csv'],
            help="CSV file with raw session data (same format as original dataset)"
        )
        
        if uploaded_file is not None:
            try:
                # Read uploaded data
                batch_data = pd.read_csv(uploaded_file)
                st.success(f"✅ File loaded: {len(batch_data)} records")
                
                # Show data preview
                with st.expander("📋 Raw Data Preview"):
                    st.dataframe(batch_data.head(), use_container_width=True)
                
                if st.button("🚀 Run Batch Predictions", type="primary"):
                    self.process_batch_predictions_raw(batch_data)
                        
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")

    def process_batch_predictions_raw(self, raw_batch_data):
        """Process batch predictions from raw data"""
        try:
            with st.spinner("Processing data and running predictions..."):
                # Transform raw data through complete pipeline
                engineered_data = self.preprocessor.transform_raw_to_engineered(raw_batch_data)
                
                # Prepare features for model
                X_batch = engineered_data[self.feature_names]
                
                # Make predictions
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(X_batch)
                    abandonment_probs = np.array([self._extract_probability(prob) for prob in probabilities])
                else:
                    predictions = self.model.predict(X_batch)
                    abandonment_probs = predictions.astype(float)
                    st.warning("⚠️ Using predict() instead of predict_proba() for batch predictions")
                
                predictions_binary = (abandonment_probs > 0.5).astype(int)
                
                # Create results dataframe
                results_df = raw_batch_data.copy()
                results_df['abandonment_probability'] = abandonment_probs
                results_df['predicted_abandonment'] = predictions_binary
                results_df['risk_level'] = results_df['abandonment_probability'].apply(
                    lambda x: 'HIGH' if x >= 0.7 else 'MEDIUM' if x >= 0.4 else 'LOW'
                )
                
                st.markdown("### 📈 Batch Prediction Results")        
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Sessions", len(results_df))
                
                with col2:
                    high_risk = (results_df['risk_level'] == 'HIGH').sum()
                    st.metric("High Risk Sessions", high_risk)
                
                with col3:
                    abandon_rate = results_df['predicted_abandonment'].mean() * 100
                    st.metric("Predicted Abandonment Rate", f"{abandon_rate:.1f}%")
                
                with col4:
                    avg_prob = results_df['abandonment_probability'].mean() * 100
                    st.metric("Average Risk", f"{avg_prob:.1f}%")
                
                # Show results table
                st.dataframe(results_df, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Prediction Results",
                    data=csv,
                    file_name=f"abandonment_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"❌ Batch prediction error: {e}")
            import traceback
            st.text(traceback.format_exc())

    def debug_model(self):
        """Debug function to identify model issues"""
        st.markdown("### 🐛 Debug Information")
        
        if not self.model:
            st.error("❌ No model loaded")
            return
            
        st.write(f"**Model Type**: {type(self.model)}")
        st.write(f"**Model Class**: {self.model.__class__.__name__}")
        st.write(f"**Feature Names**: {self.feature_names}")
        st.write(f"**Number of Features**: {len(self.feature_names)}")
        
        # Show best parameters if available
        if hasattr(self.model, 'best_params'):
            st.write("**Best Parameters**:")
            st.json(self.model.best_params)
        
        # Check available methods
        methods = [method for method in dir(self.model) if not method.startswith('_') and callable(getattr(self.model, method))]
        st.write(f"**Available Methods**: {methods}")
        
        # Test prediction with dummy data
        try:
            dummy_features = np.zeros((1, len(self.feature_names)))
            st.write(f"**Dummy input shape**: {dummy_features.shape}")
            
            if hasattr(self.model, 'predict_proba'):
                st.success("✅ Model has predict_proba method")
                probs = self.model.predict_proba(dummy_features)
                st.write(f"**Raw probabilities**: {probs}")
                st.write(f"**Probability shape**: {getattr(probs, 'shape', 'No shape')}")
                st.write(f"**Probability type**: {type(probs)}")
                
                # Test the actual probability extraction
                test_prob = self._extract_probability(probs)
                st.write(f"**Extracted probability**: {test_prob}")
                
            else:
                st.error("❌ Model has no predict_proba method")
                if hasattr(self.model, 'predict'):
                    st.info("ℹ️ Model has predict method, will use that instead")
                    pred = self.model.predict(dummy_features)
                    st.write(f"**Raw prediction**: {pred}")
                
        except Exception as e:
            st.error(f"❌ Prediction test failed: {e}")
            import traceback
            st.text(traceback.format_exc())

    def render_model_info(self):
        """Render model information"""
        st.markdown("### 🔧 Model Information")
        
        if not self.model:
            st.error("Model not loaded. Please check if final_manual_model.pkl exists.")
            return
        
        st.success("✅ Prediction model loaded and ready!")
        
        # Model details
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Details")
            st.write(f"**Model Type**: {self.model_type}")
            st.write(f"**Features Used**: {len(self.feature_names)}")
            
            # Show best parameters for Gradient Boosting
            if hasattr(self.model, 'best_params') and self.model.best_params:
                st.write("**Best Parameters**:")
                params = self.model.best_params.get('parameters', {})
                for key, value in params.items():
                    st.write(f"  - {key}: {value}")
            
            # Show performance if available
            if hasattr(self.model, 'best_params') and self.model.best_params:
                performance = self.model.best_params.get('performance', {})
                if performance:
                    st.write("**Test Performance**:")
                    for metric, score in performance.items():
                        st.write(f"  - {metric}: {score:.4f}")
            
        with col2:
            st.subheader("Prediction Guidelines")
            st.write("**🔴 HIGH RISK**: >70% - Immediate action needed")
            st.write("**🟡 MEDIUM RISK**: 40-70% - Proactive engagement") 
            st.write("**🟢 LOW RISK**: <40% - Standard monitoring")
            st.write("")
            st.write(f"**Model Class**: {self.model.__class__.__name__}")
            st.write(f"**Has predict_proba**: {'✅ Yes' if hasattr(self.model, 'predict_proba') else '❌ No'}")
            st.write(f"**Has predict**: {'✅ Yes' if hasattr(self.model, 'predict') else '❌ No'}")
        
        # Feature descriptions
        with st.expander("📋 Feature Descriptions"):
            st.write("The model uses engineered features created from raw session data.")
            st.write("**Input**: Raw session data → **Processing**: Preprocessing + Feature Engineering → **Output**: Prediction")

    def test_model_with_extreme_cases(self):
        """Test model with obvious abandonment vs completion cases"""
        st.markdown("### 🧪 Model Test with Extreme Cases")
        
        # Case 1: Likely to complete
        complete_case = {
            'session_duration': 1200,  # Long session
            'num_pages_viewed': 15,    # Many pages
            'num_items_carted': 3,     # Multiple items
            'scroll_depth': 80,        # High engagement
            'cart_value': 250.0,       # Good value
            'shipping_fee': 0.0,       # Free shipping
            'return_user': 1,          # Returning user
            'has_viewed_shipping_info': 1,
            'discount_applied': 1,
            'free_shipping_eligible': 1,
            'if_payment_page_reached': 1,
            'day_of_week': 'Saturday',
            'time_of_day': 'Evening',
            'device_type': 'Desktop',
            'browser': 'Chrome',
            'referral_source': 'Direct',
            'location': 'Mumbai, Maharashtra',
            'most_viewed_category': 'Electronics'
        }
        
        # Case 2: Likely to abandon
        abandon_case = {
            'session_duration': 60,    # Very short
            'num_pages_viewed': 2,     # Few pages
            'num_items_carted': 1,     # Single item
            'scroll_depth': 10,        # Low engagement
            'cart_value': 25.0,        # Low value
            'shipping_fee': 15.0,      # High shipping
            'return_user': 0,          # New user
            'has_viewed_shipping_info': 0,
            'discount_applied': 0,
            'free_shipping_eligible': 0,
            'if_payment_page_reached': 0,
            'day_of_week': 'Monday',
            'time_of_day': 'Morning',
            'device_type': 'Mobile',
            'browser': 'Chrome',
            'referral_source': 'Ads',
            'location': 'Mumbai, Maharashtra',
            'most_viewed_category': 'Clothing'
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Test Completion Case"):
                self.make_prediction_from_raw(complete_case)
        
        with col2:
            if st.button("Test Abandonment Case"):
                self.make_prediction_from_raw(abandon_case)

    def run(self):
        """Main method to run the prediction engine"""
        st.header("🎯 Cart Abandonment Prediction Engine")
        st.markdown("Predict which users are likely to abandon their shopping carts in real-time")
        
        # Add debug option in sidebar
        with st.sidebar:
            st.markdown("---")
            if st.button("🛠️ Debug Model", use_container_width=True):
                self.debug_model()
            if st.button("🧪 Test Extreme Cases", use_container_width=True):
                self.test_model_with_extreme_cases()
        
        if not self.model:
            st.error("""
            **Model not loaded**. Please ensure:
            - `final_manual_model.pkl` exists in the current directory
            - The model file was created with compatible classes
            - All required classes are defined in `model.py`
            """)
            
            # Show available pickle files
            st.info("📁 Available .pkl files in directory:")
            pkl_files = [f for f in os.listdir(BASE_DIR) if f.endswith('.pkl')]
            if pkl_files:
                for file in pkl_files:
                    st.write(f"- `{file}`")
            else:
                st.write("No .pkl files found")
            return
        
        # Create tabs for different prediction modes
        pred_tabs = st.tabs(["Single Prediction (Raw Data)", "Batch Prediction (Raw Data)", "Model Info"])
        
        with pred_tabs[0]:
            self.render_single_prediction()
            
        with pred_tabs[1]:
            self.render_batch_prediction()
            
        with pred_tabs[2]:
            self.render_model_info()

# Main execution
def main():
    st.set_page_config(
        page_title="Cart Abandonment Predictor",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🛒 Cart Abandonment Prediction System")
    
    # Initialize the prediction engine
    engine = PredictionEngine()
    
    if not engine.model:
        st.error("""
        **Model not loaded**. Please ensure:
        1. You're running the correct file (`streamlit run prediction_tab.py`)
        2. `final_manual_model.pkl` exists in the directory
        3. All model classes are defined in `model.py`
        """)
        return
    
    # Show model info
    if hasattr(engine.model, 'best_params'):
        best_params = engine.model.best_params
        model_type = best_params.get('model_type', 'Unknown')
        st.success(f"✅ **{model_type}** loaded successfully!")
        
        # Show performance if available
        performance = best_params.get('performance', {})
        if performance:
            st.info(f"📊 **Test Performance**: ROC-AUC: {performance.get('roc_auc', 'N/A'):.4f} | "
                   f"F1: {performance.get('f1', 'N/A'):.4f}")
    else:
        st.success(f"✅ Model loaded: {engine.model.__class__.__name__}")
    
    # Run the prediction engine
    engine.run()

if __name__ == "__main__":
    main()
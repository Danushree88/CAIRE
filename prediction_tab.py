import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import datetime

# -----------------------------
# Define directories
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = BASE_DIR  # assuming model is in the same folder

# -----------------------------
# Import custom model classes
# -----------------------------
from model import GradientBoostingClassifierManual, RandomForestManual, DecisionTreeClassifierManual, LogisticRegressionGD, KNNClassifier

# -----------------------------
# Register module for unpickling
# -----------------------------
import model as model_module
sys.modules['model'] = model_module

# -----------------------------
# Load model with better error handling
# -----------------------------
model_path = os.path.join(MODELS_DIR, "final_manual_model.pkl")

def load_model():
    """Load model with proper error handling"""
    try:
        # First try normal loading
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        
        # Extract model and feature names from the saved dictionary
        if isinstance(model_data, dict) and "model" in model_data:
            loaded_model = model_data["model"]
            if "feature_names" in model_data:
                loaded_model.feature_names = model_data["feature_names"]
            st.success("✅ Model loaded successfully from dictionary!")
        else:
            loaded_model = model_data
            st.success("✅ Model loaded successfully!")
            
        return loaded_model
        
    except (ModuleNotFoundError, AttributeError) as e:
        st.warning(f"⚠️ First load attempt failed: {e}. Trying alternative method...")
        try:
            # Register the module and try again
            import model
            sys.modules['__main__'] = model
            sys.modules['model'] = model
            
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
            
            # Extract model and feature names
            if isinstance(model_data, dict) and "model" in model_data:
                loaded_model = model_data["model"]
                if "feature_names" in model_data:
                    loaded_model.feature_names = model_data["feature_names"]
                st.success("✅ Model loaded successfully with alternative method!")
            else:
                loaded_model = model_data
                st.success("✅ Model loaded successfully with alternative method!")
                
            return loaded_model
            
        except Exception as e2:
            st.error(f"❌ Alternative loading failed: {e2}")
            return None
            
    except FileNotFoundError:
        st.error(f"❌ Model file not found at: {model_path}")
        st.info("📁 Files in directory:")
        for file in os.listdir(BASE_DIR):
            if file.endswith('.pkl'):
                st.write(f"- `{file}`")
        return None
        
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

loaded_model = load_model()

# =========================================================
# Prediction Engine Class
# =========================================================
class PredictionEngine:
    def __init__(self):
        self.model = loaded_model
        self.feature_names = self._get_feature_names()
        self.model_type = type(self.model).__name__ if self.model else None

    def _get_feature_names(self):
        """Safely get feature names from model"""
        if not self.model:
            return []
        
        # Try different ways to get feature names
        if hasattr(self.model, 'feature_names') and self.model.feature_names:
            return self.model.feature_names
        elif hasattr(self.model, 'feature_names_'):
            return self.model.feature_names_
        elif hasattr(self.model, 'get_feature_names'):
            return self.model.get_feature_names()
        else:
            # Return default feature names based on common features
            st.warning("⚠️ Using default feature names - model may not have feature_names attribute")
            return [
                'engagement_score', 'num_items_carted', 'cart_value', 
                'session_duration', 'num_pages_viewed', 'scroll_depth',
                'return_user', 'if_payment_page_reached', 'discount_applied', 
                'has_viewed_shipping_info', 'device_mobile', 'device_desktop',
                'time_of_day_morning', 'time_of_day_afternoon', 'time_of_day_evening'
            ]

    # -----------------------------
    # Feature descriptions
    # -----------------------------
    def get_feature_descriptions(self):
        descriptions = {
            'engagement_score': 'User engagement level (-1 to 1)',
            'num_items_carted': 'Number of items in cart',
            'cart_value': 'Total value of cart (normalized)',
            'session_duration': 'Duration of session in seconds',
            'num_pages_viewed': 'Number of pages viewed',
            'scroll_depth': 'How far user scrolled (0-1)',
            'return_user': 'Is returning user (0/1)',
            'if_payment_page_reached': 'Reached payment page (0/1)',
            'discount_applied': 'Discount applied (0/1)',
            'has_viewed_shipping_info': 'Viewed shipping info (0/1)'
        }
        
        # Add device and time features dynamically
        for feature in self.feature_names:
            if feature.startswith('device_') and feature not in descriptions:
                descriptions[feature] = f'Using {feature.replace("device_", "")} device (0/1)'
            elif feature.startswith('time_of_day_') and feature not in descriptions:
                descriptions[feature] = f'Session in {feature.replace("time_of_day_", "")} (0/1)'
        
        return descriptions

    def create_feature_inputs(self):
        """Create input fields for all features used in the model"""
        feature_values = {}
        
        st.subheader("📋 Enter Session Features")
        
        # Group features by category
        numeric_features = [f for f in self.feature_names if f not in ['return_user', 'if_payment_page_reached', 'discount_applied', 'has_viewed_shipping_info'] 
                           and not f.startswith('device_') and not f.startswith('time_of_day_')]
        
        binary_features = [f for f in self.feature_names if f in ['return_user', 'if_payment_page_reached', 'discount_applied', 'has_viewed_shipping_info']]
        
        device_features = [f for f in self.feature_names if f.startswith('device_')]
        time_features = [f for f in self.feature_names if f.startswith('time_of_day_')]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Numeric features
            for feature in numeric_features:
                if feature == 'engagement_score':
                    feature_values[feature] = st.slider(
                        "Engagement Score", -1.0, 1.0, 0.0, 0.1,
                        help="User engagement level from -1 (low) to 1 (high)"
                    )
                elif feature == 'cart_value':
                    feature_values[feature] = st.number_input(
                        "Cart Value", 0.0, 10000.0, 500.0, 50.0,
                        help="Total value of items in cart"
                    )
                elif feature == 'num_items_carted':
                    feature_values[feature] = st.number_input(
                        "Number of Items in Cart", 1, 50, 2,
                        help="Total items added to cart"
                    )
                elif feature == 'session_duration':
                    feature_values[feature] = st.number_input(
                        "Session Duration (seconds)", 0, 3600, 300, 30,
                        help="How long the user stayed on site"
                    )
                elif feature == 'num_pages_viewed':
                    feature_values[feature] = st.number_input(
                        "Pages Viewed", 1, 100, 8, 1,
                        help="Number of different pages visited"
                    )
                elif feature == 'scroll_depth':
                    feature_values[feature] = st.slider(
                        "Scroll Depth", 0.0, 1.0, 0.5, 0.1,
                        help="How far user scrolled (0=none, 1=full page)"
                    )
                else:
                    # Generic numeric input for other features
                    feature_values[feature] = st.number_input(
                        f"{feature.replace('_', ' ').title()}", 
                        0.0, 1000.0, 0.0, 0.1
                    )
        
        with col2:
            # Binary features
            for feature in binary_features:
                if feature == 'return_user':
                    feature_values[feature] = st.selectbox(
                        "Return User", [0, 1], format_func=lambda x: "New User" if x == 0 else "Return User"
                    )
                elif feature == 'if_payment_page_reached':
                    feature_values[feature] = st.selectbox(
                        "Reached Payment Page", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
                    )
                elif feature == 'discount_applied':
                    feature_values[feature] = st.selectbox(
                        "Discount Applied", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
                    )
                elif feature == 'has_viewed_shipping_info':
                    feature_values[feature] = st.selectbox(
                        "Viewed Shipping Info", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
                    )
            
            # Device type (handle one-hot encoded features)
            if device_features:
                device_options = [f.replace('device_', '') for f in device_features]
                selected_device = st.selectbox("Device Type", device_options)
                for device_feature in device_features:
                    feature_values[device_feature] = 1 if device_feature == f"device_{selected_device}" else 0
            
            # Time of day (handle one-hot encoded features)
            if time_features:
                time_options = [f.replace('time_of_day_', '') for f in time_features]
                selected_time = st.selectbox("Time of Day", time_options)
                for time_feature in time_features:
                    feature_values[time_feature] = 1 if time_feature == f"time_of_day_{selected_time}" else 0
        
        return feature_values

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

    def render_single_prediction(self):
        """Render single prediction interface"""
        st.markdown("### 🔮 Single Session Prediction")
        
        if not self.model:
            st.error("Model not available. Please ensure final_manual_model.pkl exists.")
            return
        
        feature_values = self.create_feature_inputs()

        # Predict button
        if st.button("🎯 Predict Abandonment Probability", type="primary", use_container_width=True):
            self.make_prediction(feature_values)

    def make_prediction(self, feature_values):
        """Make prediction based on input features"""
        try:
            # Create feature vector in correct order
            feature_vector = []
            for feature in self.feature_names:
                feature_vector.append(feature_values.get(feature, 0))
            
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            # Get prediction probability with robust handling
            if hasattr(self.model, 'predict_proba'):
                raw_probs = self.model.predict_proba(feature_vector)
                prob_abandon = self._extract_probability(raw_probs)
            else:
                # Fallback to predict method
                prediction = self.model.predict(feature_vector)
                prob_abandon = float(prediction[0]) if prediction[0] in [0, 1] else 0.5
                st.warning("⚠️ Using predict() instead of predict_proba() - probabilities may not be accurate")
            
            prob_percent = prob_abandon * 100
            
            # Display results
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
            
            # Feature importance insight
            st.markdown("### 🔍 Key Influencing Factors")
            top_features = self.get_top_influencing_factors(feature_values, prob_percent)
            for feature, impact, value in top_features:
                st.write(f"• **{feature}**: {impact} (Current: {value})")
                
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            st.info("💡 Check the debug information for more details")
            import traceback
            st.text(traceback.format_exc())

    def _extract_probability(self, raw_probs):
        """Extract abandonment probability from various probability formats"""
        try:
            # Handle GradientBoostingClassifierManual's predict_proba issue
            if hasattr(self.model, '__class__') and self.model.__class__.__name__ == 'GradientBoostingClassifierManual':
                # For your GBM implementation, predict_proba might return single values
                if isinstance(raw_probs, (int, float, np.number)):
                    return float(raw_probs)
                elif hasattr(raw_probs, '__len__') and len(raw_probs) == 1:
                    return float(raw_probs[0])
            
            # Convert to numpy array for consistent handling
            if hasattr(raw_probs, 'shape'):
                probs_array = raw_probs
            else:
                probs_array = np.array(raw_probs)
            
            # Handle different shapes
            if probs_array.shape == (1, 2):
                # Standard sklearn format: [[prob_class0, prob_class1]]
                return float(probs_array[0, 1])
            elif probs_array.shape == (1, 1):
                # Single probability output
                return float(probs_array[0, 0])
            elif probs_array.shape == (2,):
                # Single sample with two classes
                return float(probs_array[1])
            elif probs_array.shape == (1,):
                # Single probability
                return float(probs_array[0])
            elif len(probs_array) == 1 and hasattr(probs_array[0], '__len__'):
                # Nested array
                inner = probs_array[0]
                if len(inner) == 2:
                    return float(inner[1])
                else:
                    return float(inner[0])
            else:
                # Fallback: use first element
                return float(probs_array.flat[0])
                
        except Exception as e:
            st.warning(f"⚠️ Probability extraction warning: {e}. Using default 0.5")
            return 0.5

    def get_top_influencing_factors(self, feature_values, prob_percent):
        """Identify top factors influencing the prediction"""
        factors = []
        
        # Analyze key features
        if feature_values.get('if_payment_page_reached', 0) == 0 and prob_percent > 50:
            factors.append(("Payment Page Not Reached", "High abandonment risk", "No"))
        
        if feature_values.get('engagement_score', 0) < 0:
            factors.append(("Low Engagement Score", "Increased abandonment likelihood", f"{feature_values.get('engagement_score', 0):.2f}"))
            
        if feature_values.get('return_user', 0) == 0:
            factors.append(("New User", "Higher abandonment tendency", "Yes"))
            
        if feature_values.get('discount_applied', 0) == 0 and feature_values.get('cart_value', 0) > 1000:
            factors.append(("High Cart Value, No Discount", "Price sensitivity risk", f"${feature_values.get('cart_value', 0):.2f}"))
            
        if feature_values.get('session_duration', 0) < 60:
            factors.append(("Short Session Duration", "Lack of engagement", f"{feature_values.get('session_duration', 0)}s"))
        
        if feature_values.get('num_pages_viewed', 0) < 3:
            factors.append(("Few Pages Viewed", "Limited exploration", f"{feature_values.get('num_pages_viewed', 0)} pages"))
            
        if len(factors) == 0:
            factors.append(("Balanced Feature Profile", "Moderate risk factors", "Good"))
            
        return factors[:4]

    def render_batch_prediction(self):
        """Render batch prediction interface"""
        st.markdown("### 📊 Batch Prediction")
        
        if not self.model:
            st.error("Model not available for batch predictions.")
            return
        
        st.info("""
        **Upload a CSV file with session data for batch predictions.**
        The file should contain the same features used during model training.
        """)
        
        uploaded_file = st.file_uploader(
            "Upload CSV file for batch prediction", 
            type=['csv'],
            help="CSV file with session data features"
        )
        
        if uploaded_file is not None:
            try:
                # Read uploaded data
                batch_data = pd.read_csv(uploaded_file)
                st.success(f"✅ File loaded: {len(batch_data)} records")
                
                # Show data preview
                with st.expander("📋 Data Preview"):
                    st.dataframe(batch_data.head(), use_container_width=True)
                
                # Check for required features
                missing_features = set(self.feature_names) - set(batch_data.columns)
                if missing_features:
                    st.warning(f"⚠️ Missing features: {list(missing_features)}")
                    st.info("Please ensure your CSV contains all required features.")
                else:
                    if st.button("🚀 Run Batch Predictions", type="primary"):
                        self.process_batch_predictions(batch_data)
                        
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")

    def process_batch_predictions(self, batch_data):
        """Process batch predictions"""
        try:
            with st.spinner("Running predictions..."):
                # Prepare features
                X_batch = batch_data[self.feature_names]
                
                # Make predictions
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(X_batch)
                    # Extract abandonment probabilities
                    abandonment_probs = np.array([self._extract_probability(prob) for prob in probabilities])
                else:
                    # Fallback to predict method
                    predictions = self.model.predict(X_batch)
                    abandonment_probs = predictions.astype(float)
                    st.warning("⚠️ Using predict() instead of predict_proba() for batch predictions")
                
                predictions_binary = (abandonment_probs > 0.5).astype(int)
                
                # Create results dataframe
                results_df = batch_data.copy()
                results_df['abandonment_probability'] = abandonment_probs
                results_df['predicted_abandonment'] = predictions_binary
                results_df['risk_level'] = results_df['abandonment_probability'].apply(
                    lambda x: 'HIGH' if x >= 0.7 else 'MEDIUM' if x >= 0.4 else 'LOW'
                )
                
                # Display results
                st.markdown("### 📈 Batch Prediction Results")
                
                # Summary statistics
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
            st.write(f"**Available Methods**:")
            methods = [method for method in dir(self.model) if not method.startswith('_') and callable(getattr(self.model, method))]
            for method in methods[:6]:
                st.write(f"  - {method}()")
            
            # Show first few features
            st.write(f"**Features (first 10)**:")
            for feature in self.feature_names[:10]:
                st.write(f"  - {feature}")
            if len(self.feature_names) > 10:
                st.write(f"  - ... and {len(self.feature_names) - 10} more")
        
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
            descriptions = self.get_feature_descriptions()
            for feature in self.feature_names:
                description = descriptions.get(feature, "No description available")
                st.write(f"**{feature}**: {description}")

    def run(self):
        """Main method to run the prediction engine"""
        st.header("🎯 Cart Abandonment Prediction Engine")
        st.markdown("Predict which users are likely to abandon their shopping carts in real-time")
        
        # Add debug option in sidebar
        with st.sidebar:
            st.markdown("---")
            if st.button("🛠️ Debug Model", use_container_width=True):
                self.debug_model()
        
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
        pred_tabs = st.tabs(["Single Prediction", "Batch Prediction", "Model Info"])
        
        with pred_tabs[0]:
            self.render_single_prediction()
            
        with pred_tabs[1]:
            self.render_batch_prediction()
            
        with pred_tabs[2]:
            self.render_model_info()

# =========================================================
# Main execution
# =========================================================
def main():
    st.set_page_config(
        page_title="Cart Abandonment Predictor",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # App title
    st.markdown('<div class="main-header">🛒 Cart Abandonment Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time ML-powered insights to reduce cart abandonment</div>', unsafe_allow_html=True)
    
    # Initialize and run prediction engine
    engine = PredictionEngine()
    engine.run()

if __name__ == "__main__":
    main()
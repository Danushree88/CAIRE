import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import json
from datetime import datetime

# ==================== SETUP ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TEST_DATA_DIR = os.path.join(BASE_DIR, "test_data")
MODELS_DIR = BASE_DIR

# Add the project root to Python path
sys.path.append(BASE_DIR)

# ==================== IMPORT MODEL CLASSES ====================
try:
    # Import custom model classes
    from model import (
        GradientBoostingClassifierManual, 
        RandomForestManual, 
        DecisionTreeClassifierManual, 
        LogisticRegressionGD, 
        KNNClassifier
    )
    
    # Register module for unpickling
    import model as model_module
    sys.modules['model'] = model_module
    sys.modules['__main__'] = model_module
    
except ImportError as e:
    st.error(f"❌ Failed to import model classes: {e}")
    st.stop()

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_trained_model():
    """Load the pickled trained model"""
    possible_paths = [
        os.path.join(BASE_DIR, "final_manual_model.pkl"),
        os.path.join(BASE_DIR, "saved_models", "final_manual_model.pkl"),
        os.path.join(os.path.dirname(BASE_DIR), "final_manual_model.pkl"),
        os.path.join(os.path.dirname(BASE_DIR), "saved_models", "final_manual_model.pkl"),
        "final_manual_model.pkl"
    ]
    
    for model_path in possible_paths:
        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    model_data = pickle.load(f)
                
                if isinstance(model_data, dict):
                    model = model_data["model"]
                    feature_names = model_data.get("feature_names", [])
                    best_params = model_data.get("best_params", {})
                    return model, feature_names, best_params
                else:
                    return model_data, [], {}
                    
            except Exception as e:
                st.error(f"❌ Failed to load model from {model_path}: {e}")
                continue
    
    st.error("❌ Model file not found in any expected location")
    return None, [], {}

# ==================== LOAD TEST DATA ====================
@st.cache_data
def load_test_data():
    """Load the preprocessed test data"""
    test_data_path = os.path.join(TEST_DATA_DIR, "test_data_for_prediction.csv")
    
    if not os.path.exists(test_data_path):
        st.error(f"❌ Test data file not found at: {test_data_path}")
        return None
    
    try:
        test_data = pd.read_csv(test_data_path)
        return test_data
    except Exception as e:
        st.error(f"❌ Failed to load test data: {e}")
        return None

# ==================== PREDICTION ENGINE ====================
class PredictionEngine:
    """Handles predictions using preprocessed data"""
    
    def __init__(self):
        self.data_dir = DATA_DIR
    
    def make_prediction(self, model, feature_names, user_data):
        """Make prediction for a single user using preprocessed data"""
        try:
            # Create feature vector aligned with model's expected features
            feature_vector = []
            for feature in feature_names:
                if feature in user_data:
                    feature_vector.append(user_data[feature])
                else:
                    feature_vector.append(0)  # Missing features default to 0
            
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            # Get prediction probability
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(feature_vector)
                
                # Handle different probability formats
                if isinstance(prob, np.ndarray):
                    if prob.ndim == 2 and prob.shape[1] == 2:
                        prob_abandon = float(prob[0, 1])
                    elif prob.ndim == 1:
                        prob_abandon = float(prob[0])
                    else:
                        prob_abandon = float(prob.flat[0])
                else:
                    prob_abandon = float(prob)
            else:
                # Fallback to binary prediction
                pred = model.predict(feature_vector)
                prob_abandon = float(pred[0])
                
            return prob_abandon
            
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            return 0.5  # Default to uncertain

    def run_batch_predictions(self, model, feature_names, test_data):
        """Run predictions for all users in test data"""
        results = []
        
        for idx, user_row in test_data.iterrows():
            try:
                user_data = user_row.to_dict()
                prob_abandon = self.make_prediction(model, feature_names, user_data)
                
                results.append({
                    'session_id': user_data.get('session_id', f'Session_{idx}'),
                    'user_id': user_data.get('user_id', f'User_{idx}'),
                    'abandonment_probability': prob_abandon,
                    'predicted_abandonment': 1 if prob_abandon > 0.5 else 0,
                    'risk_level': 'HIGH' if prob_abandon >= 0.7 else 'MEDIUM' if prob_abandon >= 0.4 else 'LOW'
                })
                
            except Exception as e:
                st.warning(f"Error processing user {idx}: {e}")
                results.append({
                    'session_id': f'Session_{idx}',
                    'user_id': f'User_{idx}',
                    'abandonment_probability': 0.5,
                    'predicted_abandonment': 0,
                    'risk_level': 'MEDIUM'
                })
        
        return pd.DataFrame(results)

    def run(self):
        """Main method to run the prediction interface"""
        st.title("🛒 Cart Abandonment Prediction System")
        st.markdown("Predict which users are likely to abandon their shopping carts")
        
        # Load model
        model, feature_names, best_params = load_trained_model()
        
        if model is None:
            st.error("Model not loaded. Please ensure the model file exists.")
            return
        
        # Load test data
        test_data = load_test_data()
        
        if test_data is None:
            st.error("Test data not loaded. Please ensure test_data_for_prediction.csv exists.")
            return
        
        # Show model info in sidebar
        with st.sidebar:
            st.header("ℹ️ Model Information")
            if best_params:
                model_type = best_params.get("model_type", "Unknown")
                st.success(f"**Model**: {model_type}")
                
                performance = best_params.get("performance", {})
                if performance:
                    st.metric("ROC-AUC", f"{performance.get('roc_auc', 0):.3f}")
                    st.metric("F1 Score", f"{performance.get('f1', 0):.3f}")
            
            st.info(f"**Features**: {len(feature_names)}")
            st.success(f"**Test Data**: {len(test_data)} sessions")
            
            # Show data overview
            st.header("📊 Data Overview")
            if 'abandoned' in test_data.columns:
                actual_abandoned = test_data['abandoned'].sum()
                st.metric("Actual Abandoned", actual_abandoned)
            if 'return_user' in test_data.columns:
                return_users = test_data['return_user'].sum()
                st.metric("Return Users", return_users)
        
        # Main interface
        user_selection_ui(model, feature_names, test_data, self)

# ==================== USER SELECTION UI ====================
def user_selection_ui(model, feature_names, test_data, prediction_engine):
    """UI for selecting and predicting specific users from test data"""
    
    st.subheader("👥 User Selection & Prediction")
    
    # Display test data overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Users", len(test_data))
    with col2:
        abandoned_count = test_data['abandoned'].sum() if 'abandoned' in test_data.columns else "N/A"
        st.metric("Actual Abandoned", abandoned_count)
    with col3:
        return_users = test_data['return_user'].sum() if 'return_user' in test_data.columns else "N/A"
        st.metric("Return Users", return_users)
    with col4:
        avg_cart = test_data['cart_value'].mean() if 'cart_value' in test_data.columns else "N/A"
        st.metric("Avg Cart Value", f"${avg_cart:.2f}" if isinstance(avg_cart, (int, float)) else avg_cart)
    
    # User selection
    st.markdown("### 🎯 Select User for Detailed Prediction")
    
    # Create user selection options - SIMPLIFIED
    user_options = []
    for idx, row in test_data.iterrows():
        user_id = row.get('user_id', f'User_{idx}')
        session_id = row.get('session_id', f'Session_{idx}')
        
        # Simple display text with only user and session ID
        display_text = f"User: {user_id} | Session: {session_id}"
        
        user_options.append((idx, display_text))
    
    # User selection dropdown
    selected_user_idx = st.selectbox(
        "Choose a user to analyze:",
        options=[opt[0] for opt in user_options],
        format_func=lambda x: next(opt[1] for opt in user_options if opt[0] == x),
        key="user_selector"
    )
    
    # Get selected user data
    selected_user_data = test_data.iloc[selected_user_idx].to_dict()
    
    # Display user details and make prediction
    if selected_user_data:
        display_user_prediction(model, feature_names, selected_user_data, prediction_engine)
    
    # Batch predictions section - FIXED
    st.markdown("---")
    st.subheader("📊 Batch Predictions")
    
    if st.button("🚀 Run Predictions for All Users", type="primary", use_container_width=True):
        with st.spinner(f"Running predictions for {len(test_data)} users..."):
            try:
                # Run batch predictions
                batch_results = prediction_engine.run_batch_predictions(model, feature_names, test_data)
                
                # Merge with original test data to get all columns
                final_results = test_data.merge(
                    batch_results[['session_id', 'user_id', 'abandonment_probability', 'predicted_abandonment', 'risk_level']], 
                    on=['session_id', 'user_id'], 
                    how='left'
                )
                
                # Display batch results
                display_batch_results(final_results)
                
            except Exception as e:
                st.error(f"❌ Batch prediction failed: {e}")

def display_user_prediction(model, feature_names, user_data, prediction_engine):
    """Display detailed prediction for a single user"""
    
    st.markdown("---")
    st.subheader(f"🎯 Prediction for User: {user_data.get('user_id', 'Unknown')}")
    
    # SIMPLIFIED User information - Only show IDs
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**User Information**")
        st.write(f"**Session ID**: {user_data.get('session_id', 'N/A')}")
        st.write(f"**User ID**: {user_data.get('user_id', 'N/A')}")
        if 'abandoned' in user_data:
            st.write(f"**Actual Outcome**: {'🚨 Abandoned' if user_data['abandoned'] == 1 else '✅ Completed'}")
    
    with col2:
        st.markdown("**Key Metrics**")
        st.write(f"**Return User**: {'Yes' if user_data.get('return_user', 0) == 1 else 'No'}")
        st.write(f"**Cart Value**: ${user_data.get('cart_value', 0):.2f}")
        st.write(f"**Engagement Score**: {user_data.get('engagement_score', 'N/A')}")
    
    # Make prediction
    prob_abandon = prediction_engine.make_prediction(model, feature_names, user_data)
    prob_percent = prob_abandon * 100
    
    # Display prediction results
    st.markdown("### 📊 Prediction Results")
    
    pred_col1, pred_col2, pred_col3 = st.columns(3)
    
    with pred_col1:
        st.metric(
            "Abandonment Probability", 
            f"{prob_percent:.1f}%",
            delta="High Risk" if prob_percent > 50 else "Low Risk",
            delta_color="inverse"
        )
    
    with pred_col2:
        status = "🚨 LIKELY TO ABANDON" if prob_abandon > 0.5 else "✅ LIKELY TO COMPLETE"
        st.metric("Prediction", status)
    
    with pred_col3:
        confidence = max(prob_abandon, 1-prob_abandon) * 100
        st.metric("Confidence", f"{confidence:.1f}%")
    
    # Risk assessment and recommendations
    st.markdown("### 🚨 Risk Assessment & Recommendations")
    
    if prob_percent >= 70:
        st.error("🔴 **HIGH RISK** - Strong likelihood of abandonment")
        st.info("""
        **💡 Recommended Actions:**
        - Send immediate discount offer (10-15%)
        - Offer free shipping
        - Personal executive follow-up
        - Stock availability alerts
        """)
    elif prob_percent >= 40:
        st.warning("🟡 **MEDIUM RISK** - Moderate chance of abandonment")
        st.info("""
        **💡 Recommended Actions:**
        - Send reminder email in 6 hours
        - Highlight product benefits
        - Offer chat support
        - Social proof notifications
        """)
    else:
        st.success("🟢 **LOW RISK** - Likely to complete purchase")
        st.info("""
        **💡 Recommended Actions:**
        - Standard follow-up sequence
        - Cross-sell recommendations
        - Loyalty program invitation
        """)
    
    # Show feature values (collapsible)
    with st.expander("🔍 View Detailed Feature Values"):
        # Create a better formatted feature display
        important_features = [
            'num_pages_viewed', 'num_items_carted', 'scroll_depth', 'session_duration',
            'engagement_intensity', 'has_viewed_shipping_info', 'discount_applied',
            'if_payment_page_reached', 'engagement_score'
        ]
        
        feature_data = []
        for feature in important_features:
            if feature in user_data:
                value = user_data[feature]
                # Format values nicely
                if 'duration' in feature:
                    display_value = f"{value:.1f}s"
                elif 'scroll' in feature:
                    display_value = f"{value:.1f}%"
                elif 'cart_value' in feature:
                    display_value = f"${value:.2f}"
                elif feature in ['engagement_score', 'engagement_intensity']:
                    display_value = f"{value:.2f}"
                else:
                    display_value = value
                
                feature_data.append({"Feature": feature.replace('_', ' ').title(), "Value": display_value})
        
        if feature_data:
            st.dataframe(pd.DataFrame(feature_data), use_container_width=True)
        else:
            st.info("No detailed feature data available")

def display_batch_results(results_df):
    """Display batch prediction results"""
    
    st.markdown("### 📈 Batch Prediction Summary")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_users = len(results_df)
        st.metric("Total Users", total_users)
    
    with col2:
        high_risk = (results_df['risk_level'] == 'HIGH').sum()
        st.metric("High Risk Users", high_risk)
    
    with col3:
        predicted_abandon = results_df['predicted_abandonment'].sum()
        st.metric("Predicted to Abandon", predicted_abandon)
    
    with col4:
        avg_prob = results_df['abandonment_probability'].mean() * 100
        st.metric("Average Risk", f"{avg_prob:.1f}%")
    
    # Risk distribution
    st.markdown("#### Risk Level Distribution")
    risk_counts = results_df['risk_level'].value_counts()
    
    risk_col1, risk_col2, risk_col3 = st.columns(3)
    with risk_col1:
        st.metric("🟢 Low Risk", risk_counts.get('LOW', 0))
    with risk_col2:
        st.metric("🟡 Medium Risk", risk_counts.get('MEDIUM', 0))
    with risk_col3:
        st.metric("🔴 High Risk", risk_counts.get('HIGH', 0))
    
    # Display results table
    st.markdown("#### Detailed Results")
    
    # Create display columns - SIMPLIFIED
    display_columns = ['session_id', 'user_id', 'abandonment_probability', 'predicted_abandonment', 'risk_level']
    
    # Add actual outcome if available
    if 'abandoned' in results_df.columns:
        results_df['actual_outcome'] = results_df['abandoned'].apply(lambda x: 'Abandoned' if x == 1 else 'Completed')
        display_columns.append('actual_outcome')
    
    # Add cart value if available
    if 'cart_value' in results_df.columns:
        display_columns.append('cart_value')
    
    # Format the display dataframe
    display_df = results_df[display_columns].copy()
    
    # Format percentages
    if 'abandonment_probability' in display_df.columns:
        display_df['abandonment_probability'] = (display_df['abandonment_probability'] * 100).round(1).astype(str) + '%'
    
    # Format cart value
    if 'cart_value' in display_df.columns:
        display_df['cart_value'] = display_df['cart_value'].apply(lambda x: f"${x:.2f}" if isinstance(x, (int, float)) else x)
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Download results
    csv = results_df.to_csv(index=False)
    st.download_button(
        "📥 Download All Predictions",
        csv,
        f"cart_abandonment_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        "text/csv",
        use_container_width=True
    )
    
    # High-risk users section
    high_risk_users = results_df[results_df['risk_level'] == 'HIGH']
    if not high_risk_users.empty:
        st.markdown("#### 🚨 High-Risk Users (Priority Action Required)")
        
        high_risk_display = high_risk_users[['session_id', 'user_id', 'abandonment_probability', 'cart_value']].copy()
        high_risk_display['abandonment_probability'] = (high_risk_display['abandonment_probability'] * 100).round(1).astype(str) + '%'
        if 'cart_value' in high_risk_display.columns:
            high_risk_display['cart_value'] = high_risk_display['cart_value'].apply(lambda x: f"${x:.2f}" if isinstance(x, (int, float)) else x)
        
        st.dataframe(high_risk_display, use_container_width=True)

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    # Initialize and run the prediction engine
    engine = PredictionEngine()
    engine.run()
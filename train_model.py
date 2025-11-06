import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from prediction_tab import GradientBoostingClassifierManual

def train_and_save_model():
    """Train model on featured dataset and save it"""
    
    # Load your featured dataset
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    FEATURED_PATH = os.path.join(DATA_DIR, "cart_abandonment_featured.csv")
    
    if not os.path.exists(FEATURED_PATH):
        print(f"❌ Featured dataset not found: {FEATURED_PATH}")
        print("Please run feature engineering first.")
        return
    
    # Load data
    df = pd.read_csv(FEATURED_PATH)
    
    # Prepare features and target
    X = df.drop(['session_id', 'user_id', 'abandoned'], axis=1, errors='ignore')
    y = df['abandoned']
    
    feature_names = X.columns.tolist()
    
    print(f"📊 Training on {X.shape[0]} samples with {X.shape[1]} features")
    print(f"🎯 Target distribution: {y.mean():.1%} abandonment rate")
    
    # Use simpler parameters for faster, more stable training
    model = GradientBoostingClassifierManual(
        n_estimators=30,  # Reduced for faster training
        learning_rate=0.1,
        max_depth=3,      # Reduced depth
        min_samples_split=20
    )
    
    print("🔄 Training model... (this may take a few minutes)")
    model.fit(X, y)
    
    # Save model
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'training_timestamp': datetime.now().isoformat(),
        'training_samples': X.shape[0],
        'abandonment_rate': y.mean()
    }
    
    model_path = os.path.join(BASE_DIR, "trained_gb_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✅ Model saved to: {model_path}")
    print(f"📈 Features: {len(feature_names)}")
    print(f"🎯 Abandonment rate: {y.mean():.1%}")
    print(f"🌳 Number of trees: {len(model.trees)}")

if __name__ == "__main__":
    train_and_save_model()
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import math
import time
import json
import joblib
import sys
import os

# Page config
st.set_page_config(
    page_title="ShopEasy - Test Data Store",
    page_icon="🛒",
    layout="wide"
)

# ============================================================================
# LOAD LABEL ENCODERS FROM JSON FILE
# ============================================================================

try:
    with open('data/label_encoders.json', 'r') as f:
        label_encoders = json.load(f)
    
    DAY_OF_WEEK_MAP = label_encoders.get('day_of_week', {})
    TIME_OF_DAY_MAP = label_encoders.get('time_of_day', {})
    DEVICE_TYPE_MAP = label_encoders.get('device_type', {})
    BROWSER_MAP = label_encoders.get('browser', {})
    REFERRAL_SOURCE_MAP = label_encoders.get('referral_source', {})
    LOCATION_MAP = label_encoders.get('location', {})
    CATEGORY_MAP = label_encoders.get('most_viewed_category', {})
    
except FileNotFoundError:
    DAY_OF_WEEK_MAP = {"Saturday": 0, "Thursday": 1, "Monday": 2, "Sunday": 3, "Wednesday": 4, "Friday": 5, "Tuesday": 6}
    TIME_OF_DAY_MAP = {"Evening": 0, "Afternoon": 1, "Morning": 2, "Night": 3}
    DEVICE_TYPE_MAP = {"Mobile": 0, "Desktop": 1, "Tablet": 2}
    BROWSER_MAP = {"Opera": 0, "Safari": 1, "Chrome": 2, "Edge": 3, "Firefox": 4}
    REFERRAL_SOURCE_MAP = {"Search Engine": 0, "Ads": 1, "Direct": 2, "Social Media": 3, "Email Campaign": 4}
    LOCATION_MAP = {"Nagpur, Maharashtra": 0, "Mumbai, Maharashtra": 1, "Delhi, Delhi": 2}
    CATEGORY_MAP = {"Electronics": 0, "Clothing": 1, "Home & Kitchen": 2}

FEATURE_ORDER = [
    'return_user', 'day_of_week', 'time_of_day', 'session_duration', 'num_pages_viewed',
    'num_items_carted', 'has_viewed_shipping_info', 'scroll_depth', 'cart_value',
    'discount_applied', 'shipping_fee', 'free_shipping_eligible', 'device_type', 'browser',
    'referral_source', 'location', 'if_payment_page_reached', 'most_viewed_category',
    'engagement_intensity', 'scroll_engagement', 'is_weekend', 'has_multiple_items',
    'has_high_engagement', 'research_behavior', 'quick_browse', 'engagement_score',
    'peak_hours', 'returning_peak', 'day_sin', 'day_cos', 'time_sin', 'time_cos', 'pca1', 'pca2'
]

PRODUCTS = [
    {"id": 1, "name": "Smartphone", "price": 299.99, "category": "Electronics", "description": "Latest 5G smartphone with amazing camera", "image": "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=400&h=300&fit=crop"},
    {"id": 2, "name": "Car Accessories", "price": 899.99, "category": "Automotive", "description": "Premium car mats and organizers", "image": "https://images.unsplash.com/photo-1549317661-bd32c8ce0db2?w=400&h=300&fit=crop"},
    {"id": 3, "name": "Cookware Set", "price": 1499.99, "category": "Home & Kitchen", "description": "12-piece non-stick cookware set", "image": "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=400&h=300&fit=crop"},
    {"id": 4, "name": "Skincare Kit", "price": 499.99, "category": "Beauty", "description": "Complete skincare routine set", "image": "https://images.unsplash.com/photo-1556228578-8c89e6adf883?w=400&h=300&fit=crop"},
    {"id": 5, "name": "Novel Collection", "price": 1029.99, "category": "Books", "description": "Pack of bestselling fiction novels", "image": "https://images.unsplash.com/photo-1544716278-ca5e3f4abd8c?w=400&h=300&fit=crop"},
    {"id": 6, "name": "T-Shirt", "price": 1009.99, "category": "Clothing", "description": "100% cotton comfortable t-shirt", "image": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400&h=300&fit=crop"},
    {"id": 7, "name": "Action Figure", "price": 2400.99, "category": "Toys", "description": "Collectible action figure with accessories", "image": "https://images.unsplash.com/photo-1587654780291-39c9404d746b?w=400&h=300&fit=crop"},
    {"id": 8, "name": "Running Shoes", "price": 7900.99, "category": "Sports", "description": "Professional running shoes with cushioning", "image": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400&h=300&fit=crop"},
    {"id": 9, "name": "Organic Snacks", "price": 1400.99, "category": "Groceries", "description": "Healthy organic snack pack", "image": "https://images.unsplash.com/photo-1542838132-92c53300491e?w=400&h=300&fit=crop"},
    {"id": 10, "name": "Laptop", "price": 89900.99, "category": "Electronics", "description": "High-performance laptop for professionals", "image": "https://images.unsplash.com/photo-1496181133206-80ce9b88a853?w=400&h=300&fit=crop"}
]

# Valid promo codes with discount percentages
VALID_PROMO_CODES = {
    "SAVE10": 10,
    "SAVE20": 20,
    "WELCOME": 15,
    "SUMMER50": 50
}

# ============================================================================
# INITIALIZATION
# ============================================================================

if 'session_data' not in st.session_state:
    st.session_state.session_data = {
        'session_id': f"S{int(time.time())}",
        'user_id': f"U{np.random.randint(1000, 9999)}",
        'items': [],
        'start_time': datetime.now(),
        'events': [],
        'cart_value': 0.0,
        'pages_viewed': [],  # Changed to list to track order and count
        'current_page': 'home',  # Track current page
        'categories_viewed': set(),
        'scroll_depth': 0.0,  # Real scroll depth percentage
        'view_product_count': 0,
        'shipping_viewed': False,
        'payment_reached': False,
        'return_user': np.random.choice([0, 1], p=[0.7, 0.3]),
        'device_type': np.random.choice(list(DEVICE_TYPE_MAP.keys())),
        'browser': np.random.choice(list(BROWSER_MAP.keys())),
        'referral_source': np.random.choice(list(REFERRAL_SOURCE_MAP.keys())),
        'location': np.random.choice(list(LOCATION_MAP.keys())),
        'discount_applied': 0,
        'discount_percentage': 0,
        'discount_code': None,
        'viewed_products': [],
        'time_on_page': {}  # Track time spent on each page
    }

if 'all_sessions' not in st.session_state:
    st.session_state.all_sessions = []

if 'viewing_product' not in st.session_state:
    st.session_state.viewing_product = None

if 'promo_message' not in st.session_state:
    st.session_state.promo_message = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_time_of_day(hour):
    """Convert hour to time category"""
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 22:
        return "Evening"
    else:
        return "Night"

def log_event(event_type, product_id=None, product_name=None, category=None, page_type=None):
    """Log user interaction"""
    event = {
        'timestamp': datetime.now().isoformat(),
        'event_type': event_type,
        'product_id': product_id,
        'product_name': product_name,
        'category': category,
        'page_type': page_type
    }
    st.session_state.session_data['events'].append(event)
    
    if page_type:
        if page_type not in st.session_state.session_data['pages_viewed']:
            st.session_state.session_data['pages_viewed'].append(page_type)
    if category:
        st.session_state.session_data['categories_viewed'].add(category)

def navigate_page(page_name):
    """Track page navigation"""
    st.session_state.session_data['current_page'] = page_name
    log_event('page_view', None, None, None, page_name)

def add_to_cart(product):
    """Add product to cart"""
    st.session_state.session_data['items'].append(product)
    st.session_state.session_data['cart_value'] += product['price']
    log_event('add_to_cart', product['id'], product['name'], product['category'], 'product')

def remove_from_cart(product_id):
    """Remove product from cart by ID"""
    for item in st.session_state.session_data['items']:
        if item['id'] == product_id:
            st.session_state.session_data['cart_value'] -= item['price']
            st.session_state.session_data['items'].remove(item)
            log_event('remove_from_cart', product_id, item['name'], item['category'], 'cart')
            break

def view_product(product):
    """View product details"""
    if product not in st.session_state.session_data['viewed_products']:
        st.session_state.session_data['viewed_products'].append(product)
    st.session_state.session_data['view_product_count'] += 1
    log_event('view_product', product['id'], product['name'], product['category'], 'product_detail')

def apply_promo_code(code):
    """Validate and apply promo code"""
    code = code.upper().strip()
    if code in VALID_PROMO_CODES:
        st.session_state.session_data['discount_applied'] = 1
        st.session_state.session_data['discount_percentage'] = VALID_PROMO_CODES[code]
        st.session_state.session_data['discount_code'] = code
        log_event('promo_applied', None, None, None, 'checkout')
        return True, VALID_PROMO_CODES[code]
    return False, 0

def update_scroll_depth(page_scroll_percent):
    """Update realistic scroll depth based on actual page scrolling"""
    st.session_state.session_data['scroll_depth'] = max(
        st.session_state.session_data['scroll_depth'], 
        page_scroll_percent
    )
    if page_scroll_percent > 0:
        log_event('scroll', None, None, None, st.session_state.session_data['current_page'])

def calculate_features(abandoned):
    """Calculate all features for model input"""
    current_time = datetime.now()
    session_duration = (current_time - st.session_state.session_data['start_time']).total_seconds()
    
    # Get time info
    day_name = current_time.strftime("%A")
    hour = current_time.hour
    time_of_day = get_time_of_day(hour)
    
    # Count interactions
    add_events = len([e for e in st.session_state.session_data['events'] if e['event_type'] == 'add_to_cart'])
    view_events = len([e for e in st.session_state.session_data['events'] if e['event_type'] == 'view_product'])
    
    # Basic metrics - REALISTIC VALUES
    num_pages_viewed = len(st.session_state.session_data['pages_viewed'])  # Now accurate
    num_items_carted = len(st.session_state.session_data['items'])
    scroll_depth = st.session_state.session_data['scroll_depth']  # Real scroll depth
    cart_value_scaled = st.session_state.session_data['cart_value']
    shipping_fee = 0 if cart_value_scaled >= 200 else 99
    free_shipping_eligible = 1 if cart_value_scaled >= 200 else 0

    
    # Most viewed category
    if st.session_state.session_data['categories_viewed']:
        category_counts = {}
        for event in st.session_state.session_data['events']:
            if event['category']:
                category_counts[event['category']] = category_counts.get(event['category'], 0) + 1
        most_viewed_category = max(category_counts, key=category_counts.get) if category_counts else "Electronics"
    else:
        most_viewed_category = "Electronics"
    
    # Engineered features
    total_actions = len(st.session_state.session_data['events'])
    engagement_intensity = total_actions / max(session_duration / 60, 1)
    scroll_engagement = min(1.0, scroll_depth / 100.0)
    
    is_weekend = 1 if current_time.weekday() >= 5 else 0
    has_multiple_items = 1 if len(st.session_state.session_data['items']) > 1 else 0
    has_high_engagement = 1 if engagement_intensity > 1.5 else 0
    research_behavior = 1 if (view_events > 3 and num_items_carted > 0) else 0
    quick_browse = 1 if (session_duration < 120 and view_events <= 2) else 0
    engagement_score = min(10, total_actions * 0.5 + scroll_depth * 0.05)
    peak_hours = 1 if 9 <= hour <= 18 else 0
    returning_peak = 1 if (st.session_state.session_data['return_user'] and peak_hours) else 0
    
    # Cyclical encoding
    day_sin = math.sin(2 * math.pi * current_time.weekday() / 7)
    day_cos = math.cos(2 * math.pi * current_time.weekday() / 7)
    time_sin = math.sin(2 * math.pi * hour / 24)
    time_cos = math.cos(2 * math.pi * hour / 24)
    
    # PCA components
    pca1 = np.random.normal(0, 1)
    pca2 = np.random.normal(0, 1)
    
    # Build features dictionary
    features = {
        'return_user': st.session_state.session_data['return_user'],
        'day_of_week': DAY_OF_WEEK_MAP.get(day_name, 0),
        'time_of_day': TIME_OF_DAY_MAP.get(time_of_day, 0),
        'session_duration': max(60, min(5000, session_duration)),
        'num_pages_viewed': num_pages_viewed,
        'num_items_carted': num_items_carted,
        'has_viewed_shipping_info': int(st.session_state.session_data['shipping_viewed']),
        'scroll_depth': scroll_depth,
        'cart_value': cart_value_scaled,
        'discount_applied': st.session_state.session_data['discount_applied'],
        'shipping_fee': shipping_fee,
        'free_shipping_eligible': free_shipping_eligible,
        'device_type': DEVICE_TYPE_MAP.get(st.session_state.session_data['device_type'], 0),
        'browser': BROWSER_MAP.get(st.session_state.session_data['browser'], 0),
        'referral_source': REFERRAL_SOURCE_MAP.get(st.session_state.session_data['referral_source'], 0),
        'location': LOCATION_MAP.get(st.session_state.session_data['location'], 0),
        'if_payment_page_reached': 1 if st.session_state.session_data['payment_reached'] else 0,
        'most_viewed_category': CATEGORY_MAP.get(most_viewed_category, 0),
        'engagement_intensity': engagement_intensity,
        'scroll_engagement': scroll_engagement,
        'is_weekend': is_weekend,
        'has_multiple_items': has_multiple_items,
        'has_high_engagement': has_high_engagement,
        'research_behavior': research_behavior,
        'quick_browse': quick_browse,
        'engagement_score': engagement_score,
        'peak_hours': peak_hours,
        'returning_peak': returning_peak,
        'day_sin': day_sin,
        'day_cos': day_cos,
        'time_sin': time_sin,
        'time_cos': time_cos,
        'pca1': pca1,
        'pca2': pca2,
        'abandoned': abandoned
    }
    
    return features

def save_session_data(abandoned):
    """Save session to CSV"""
    features = calculate_features(abandoned)
    
    ordered_features = {feature: features[feature] for feature in FEATURE_ORDER if feature in features}
    ordered_features['abandoned'] = abandoned
    
    os.makedirs('test_data', exist_ok=True)
    
    df = pd.DataFrame([ordered_features])
    csv_file = 'test_data/test_data_for_prediction.csv'
    
    if os.path.exists(csv_file):
        existing_df = pd.read_csv(csv_file)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_csv(csv_file, index=False)
    st.session_state.all_sessions.append(ordered_features)
    
    return csv_file

def start_new_session():
    """Start fresh session"""
    st.session_state.session_data = {
        'session_id': f"S{int(time.time())}",
        'user_id': f"U{np.random.randint(1000, 9999)}",
        'items': [],
        'start_time': datetime.now(),
        'events': [],
        'cart_value': 0.0,
        'pages_viewed': [],
        'current_page': 'home',
        'categories_viewed': set(),
        'scroll_depth': 0.0,
        'view_product_count': 0,
        'shipping_viewed': False,
        'payment_reached': False,
        'return_user': np.random.choice([0, 1], p=[0.7, 0.3]),
        'device_type': np.random.choice(list(DEVICE_TYPE_MAP.keys())),
        'browser': np.random.choice(list(BROWSER_MAP.keys())),
        'referral_source': np.random.choice(list(REFERRAL_SOURCE_MAP.keys())),
        'location': np.random.choice(list(LOCATION_MAP.keys())),
        'discount_applied': 0,
        'discount_percentage': 0,
        'discount_code': None,
        'viewed_products': [],
        'time_on_page': {}
    }
    st.session_state.viewing_product = None
    st.session_state.promo_message = None


# Add this after your existing imports
import joblib
import sys
import os

# Add the segmentation model path
SEGMENTATION_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'enhanced_segmenter.pkl')

class RecoveryStrategyManager:
    """Manages recovery strategies based on customer segments"""
    
    def __init__(self):
        self.segment_strategies = {
            "High-Value Loyalists": {
                "priority": "Low",
                "strategies": [
                    "🎯 Personalized thank you message with early access to new collections",
                    "💎 VIP loyalty program with exclusive benefits",
                    "📧 Regular updates about products matching their preferences",
                    "🎁 Surprise free shipping or small gifts on next purchase"
                ],
                "timing": "24-48 hours after visit",
                "channel": "Email + Mobile App Notification"
            },
            "At-Risk Converters": {
                "priority": "Very High",
                "strategies": [
                    "🔥 Limited-time discount (10-15%) on abandoned items",
                    "🚚 Free expedited shipping offer",
                    "📞 Personal shopping assistant offer",
                    "⏰ Stock availability alerts for items in cart"
                ],
                "timing": "2-6 hours after abandonment",
                "channel": "Email + SMS + Push Notification"
            },
            "Engaged Researchers": {
                "priority": "High", 
                "strategies": [
                    "📚 Product comparison guides or buying guides",
                    "🎥 Video demonstrations of products they viewed",
                    "💬 Live chat support invitation",
                    "⭐ Customer reviews and testimonials highlight"
                ],
                "timing": "12-24 hours after visit",
                "channel": "Email + Retargeting Ads"
            },
            "Price-Sensitive Shoppers": {
                "priority": "Medium",
                "strategies": [
                    "💰 Tiered discounts based on cart value",
                    "🎟️ Additional promo codes for next purchase",
                    "📦 Free shipping threshold reduction",
                    "🔄 Price drop alerts for watched items"
                ],
                "timing": "6-12 hours after abandonment", 
                "channel": "Email + Browser Push"
            },
            "Casual Browsers": {
                "priority": "Low",
                "strategies": [
                    "🌐 Personalized product recommendations",
                    "📢 New arrival notifications",
                    "🏆 Social proof and trending products",
                    "🔔 Re-engagement campaign after 7 days"
                ],
                "timing": "48-72 hours after visit",
                "channel": "Email only"
            }
        }
    
    def get_recovery_strategy(self, segment_name, user_data=None):
        """Get recovery strategy for specific segment"""
        segment_info = self.segment_strategies.get(segment_name, {})
        
        if not segment_info:
            return {
                "segment": "Unknown",
                "priority": "Medium",
                "message": "Continue with standard engagement strategy",
                "strategies": ["Standard follow-up email after 24 hours"]
            }
        
        return {
            "segment": segment_name,
            "priority": segment_info["priority"],
            "message": f"Targeted recovery for {segment_name}",
            "strategies": segment_info["strategies"],
            "timing": segment_info["timing"],
            "channel": segment_info["channel"],
            "user_context": user_data
        }

# Initialize the recovery manager
recovery_manager = RecoveryStrategyManager()

import sys
import os

import sys
import os

def load_or_train_segmenter():
    """Load or train segmentation model - FIXED PATH"""
    try:
        # Get the current directory (where your Streamlit app is running)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"📁 Current directory: {current_dir}")
        
        # Build path to segmentation module
        segmentation_dir = os.path.join(current_dir, 'src', 'segmentation')
        print(f"📁 Segmentation directory: {segmentation_dir}")
        
        # Check if the directory exists
        if not os.path.exists(segmentation_dir):
            print(f"❌ Segmentation directory not found: {segmentation_dir}")
            return None
            
        # Add to Python path
        if segmentation_dir not in sys.path:
            sys.path.insert(0, segmentation_dir)
            print(f"✅ Added to path: {segmentation_dir}")
        
        # Try to import
        try:
            from segment import EnhancedCustomerSegmenter
            print("✅ Successfully imported EnhancedCustomerSegmenter")
        except ImportError as e:
            print(f"❌ Import failed: {e}")
            return None
        
        # Load training data
        data_path = os.path.join(current_dir, 'data', 'cart_abandonment_featured.csv')
        print(f"📊 Data path: {data_path}")
        
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            print(f"✅ Loaded training data: {df.shape}")
            
            segmenter = EnhancedCustomerSegmenter(n_segments=5)
            segmenter.fit(df)
            print("✅ Segmentation model trained successfully")
            return segmenter
        else:
            print(f"❌ Training data not found at: {data_path}")
            return None
            
    except Exception as e:
        print(f"❌ Error in load_or_train_segmenter: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_nan_features(features):
    """Comprehensive NaN debugging"""
    print("🔍 COMPREHENSIVE NaN DEBUG:")
    nan_features = []
    all_features_checked = []
    
    for key, value in features.items():
        is_nan = pd.isna(value) or value is None
        status = "❌ NaN" if is_nan else "✅ OK"
        all_features_checked.append(f"{key}: {value} ({status})")
        
        if is_nan:
            nan_features.append(key)
            print(f"   ❌ {key}: {value} (NaN/None)")
    
    print(f"📊 Total features: {len(features)}, NaN features: {len(nan_features)}")
    if nan_features:
        print(f"🚨 CRITICAL: Found NaN in: {nan_features}")
    
    return nan_features

def aggressive_feature_cleaning(features):
    """Aggressive cleaning to eliminate ALL NaN values"""
    cleaned = features.copy()
    
    # Define safe defaults for EVERY feature
    safe_defaults = {
        # Binary features
        'return_user': 0, 'discount_applied': 0, 'has_viewed_shipping_info': 0,
        'if_payment_page_reached': 0, 'free_shipping_eligible': 0, 'is_weekend': 0,
        'has_multiple_items': 0, 'has_high_engagement': 0, 'research_behavior': 0,
        'quick_browse': 0, 'peak_hours': 0, 'returning_peak': 0,
        
        # Count features
        'num_pages_viewed': 1, 'num_items_carted': 0,
        
        # Continuous features
        'session_duration': 120.0, 'scroll_depth': 25.0, 'cart_value': 50.0,
        'shipping_fee': 99.0, 'engagement_intensity': 0.5, 'scroll_engagement': 0.25,
        'engagement_score': 5.0, 'day_sin': 0.0, 'day_cos': 1.0, 'time_sin': 0.0,
        'time_cos': 1.0, 'pca1': 0.0, 'pca2': 0.0,
        
        # Encoded categorical features
        'day_of_week': 0, 'time_of_day': 0, 'device_type': 0, 'browser': 0,
        'referral_source': 0, 'location': 0, 'most_viewed_category': 0
    }
    
    # Apply defaults to ALL features to ensure no NaN
    for feature in cleaned.keys():
        if pd.isna(cleaned[feature]) or cleaned[feature] is None:
            default_val = safe_defaults.get(feature, 0)
            print(f"🔄 Replacing NaN in {feature} with {default_val}")
            cleaned[feature] = default_val
    
    return cleaned

def predict_customer_segment(features):
    """Predict segment with GUARANTEED no NaN values"""
    try:
        print("🎯 Starting segmentation prediction...")
        
        # Step 1: Debug NaN features
        nan_features = debug_nan_features(features)
        
        # Step 2: Aggressive cleaning
        cleaned_features = aggressive_feature_cleaning(features)
        
        # Step 3: Convert to DataFrame with additional safety
        features_df = pd.DataFrame([cleaned_features])
        
        # Step 4: Final nuclear option - fill ALL remaining NaN
        nan_before = features_df.isna().sum().sum()
        if nan_before > 0:
            print(f"⚠️ Nuclear cleanup: Found {nan_before} NaN, filling ALL with 0")
            features_df = features_df.fillna(0)
        
        nan_after = features_df.isna().sum().sum()
        print(f"📊 Final NaN check - Before: {nan_before}, After: {nan_after}")
        
        if nan_after > 0:
            print("❌ CRITICAL: Still have NaN after cleaning - using fallback")
            return simple_segment_fallback(features)
        
        print(f"✅ Features clean: {features_df.shape}, No NaN values")
        
        # Step 5: Load segmenter
        segmenter = load_or_train_segmenter()
        
        if segmenter is None:
            print("⚠️ Segmenter failed, using fallback")
            return simple_segment_fallback(features)
        
        print("✅ Segmenter loaded successfully")
        
        # Step 6: Predict
        segment_label = segmenter.predict_segment(features_df)[0]
        print(f"📊 Raw segment label: {segment_label}")
        
        # Step 7: Get segment name
        if (hasattr(segmenter, 'segment_profiles') and 
            segmenter.segment_profiles and 
            segment_label in segmenter.segment_profiles):
            segment_name = segmenter.segment_profiles[segment_label]['segment_name']
            print(f"🎯 SUCCESS: Predicted segment: {segment_name}")
            return segment_name
        
        # Fallback mapping
        segment_names = {
            0: "High-Value Loyalists", 1: "At-Risk Converters", 
            2: "Engaged Researchers", 3: "Price-Sensitive Shoppers", 
            4: "Casual Browsers"
        }
        segment_name = segment_names.get(segment_label, "Casual Browsers")
        print(f"🎯 Fallback mapping: {segment_name}")
        return segment_name
        
    except Exception as e:
        print(f"❌ Segmentation crashed: {e}")
        return simple_segment_fallback(features)

def simple_segment_fallback(features):
    """Simple fallback when model fails"""
    print("🔄 Using simple segment fallback")
    try:
        cart_value = features.get('cart_value', 0)
        return_user = features.get('return_user', 0)
        engagement = features.get('engagement_score', 0)
        pages_viewed = features.get('num_pages_viewed', 0)
        discount_applied = features.get('discount_applied', 0)
        payment_reached = features.get('if_payment_page_reached', 0)
        
        print(f"📊 Fallback analysis - Cart: {cart_value}, Return: {return_user}, Engagement: {engagement}")
        
        # Your actual segmentation logic
        if return_user and cart_value > 100 and engagement > 7:
            return "High-Value Loyalists"
        elif not return_user and cart_value > 100:
            return "At-Risk Converters"
        elif engagement > 7 and pages_viewed > 3:
            return "Engaged Researchers"
        elif discount_applied:
            return "Price-Sensitive Shoppers"
        else:
            return "Casual Browsers"
    except Exception as e:
        print(f"❌ Fallback error: {e}")
        return "Casual Browsers"

def trigger_recovery_strategy(session_data, abandonment_reason="cart_abandonment"):
    """
    Trigger appropriate recovery strategy based on user segment
    """
    try:
        # Calculate features for segmentation
        features = calculate_features(abandoned=1)  # Using your existing function
        
        # Predict segment
        segment = predict_customer_segment(features)
        
        # Get recovery strategy
        user_context = {
            "cart_value": session_data['cart_value'],
            "items_count": len(session_data['items']),
            "products_viewed": len(session_data['viewed_products']),
            "session_duration": (datetime.now() - session_data['start_time']).total_seconds(),
            "abandonment_reason": abandonment_reason
        }
        
        strategy = recovery_manager.get_recovery_strategy(segment, user_context)
        
        # Log the recovery trigger
        log_event('recovery_triggered', None, None, None, 'system')
        
        return strategy
        
    except Exception as e:
        print(f"Recovery strategy error: {e}")
def handle_session_end(action_type):
    """Handle session end with recovery strategy - FIXED"""
    if len(st.session_state.session_data['items']) > 0:
        if action_type == "abandon":
            # Calculate features WITH abandoned=1 (like your training data)
            features = calculate_features(abandoned=1)  # THIS IS CRITICAL
            
            # Then predict segment
            recovery_strategy = trigger_recovery_strategy(
                st.session_state.session_data, 
                "cart_abandonment"
            )
            
            # Save as abandoned
            csv_file = save_session_data(abandoned=1)
            st.success("📊 Cart abandoned - saved!")
            
            # Store the strategy
            st.session_state.current_recovery_strategy = recovery_strategy
            
        elif action_type == "purchase":
            # For purchase, use abandoned=0
            features = calculate_features(abandoned=0)
            st.session_state.session_data['payment_reached'] = True
            csv_file = save_session_data(abandoned=0)
            st.success("🎉 Purchase complete - saved!")
            st.session_state.current_recovery_strategy = None

def display_recovery_ui(strategy=None):
    """Display recovery strategy in the UI - PERSISTENT"""
    # Use strategy from parameter or from session state
    if strategy is None:
        strategy = st.session_state.get('current_recovery_strategy')
    
    if not strategy:
        return
    
    st.sidebar.divider()
    st.sidebar.header("🎯 Recovery Strategy")
    
    # Color code based on priority
    priority_colors = {
        "Very High": "#ff4444",
        "High": "#ffaa00", 
        "Medium": "#ffdd00",
        "Low": "#44cc44"
    }
    
    color = priority_colors.get(strategy["priority"], "#4444ff")
    
    st.sidebar.markdown(f"""
    <div style="padding: 15px; border-radius: 8px; border-left: 5px solid {color}; background-color: #f8f9fa; margin: 10px 0;">
        <h4 style="margin: 0 0 8px 0; color: {color};">{strategy['segment']}</h4>
        <p style="margin: 4px 0; font-size: 0.9em;"><strong>Priority:</strong> {strategy['priority']}</p>
        <p style="margin: 4px 0; font-size: 0.85em;"><strong>Timing:</strong> {strategy['timing']}</p>
        <p style="margin: 4px 0; font-size: 0.85em;"><strong>Channel:</strong> {strategy['channel']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.write("**Recommended Actions:**")
    for i, action in enumerate(strategy["strategies"][:3]):
        st.sidebar.markdown(f"<div style='margin: 8px 0; padding-left: 10px; border-left: 2px solid {color};'>{action}</div>", unsafe_allow_html=True)
    
    # Add a note about continuing
    st.sidebar.info("💡 Click '🔄 Start New Session' to continue testing")


def modified_abandon_cart_flow():
    """Enhanced abandon cart flow with recovery strategies"""
    
    st.sidebar.divider()
    st.sidebar.header("🏁 End Session")
    
    if st.sidebar.button("🔄 New Session", width='stretch', type="secondary"):
        start_new_session()
        st.rerun()
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.sidebar.button("🚪 Abandon", width='stretch'):
            if len(st.session_state.session_data['items']) > 0:
                # Trigger recovery strategy before saving
                recovery_strategy = trigger_recovery_strategy(
                    st.session_state.session_data, 
                    "cart_abandonment"
                )
                
                # Save session data
                csv_file = save_session_data(abandoned=1)
                
                # Show recovery insights
                st.sidebar.success("📊 Cart abandoned - saved!")
                
                # Display recovery strategy
                if recovery_strategy:
                    display_recovery_ui(recovery_strategy)
                
                st.balloons()
                time.sleep(3)
                start_new_session()
                st.rerun()
            else:
                st.sidebar.warning("❌ Add items to cart first!")
    
    with col2:
        if st.sidebar.button("💰 Purchase", type="primary", width='stretch'):
            if len(st.session_state.session_data['items']) > 0:
                st.session_state.session_data['payment_reached'] = True
                csv_file = save_session_data(abandoned=0)
                st.sidebar.success("🎉 Purchase complete - saved!")
                st.balloons()
                time.sleep(2)
                start_new_session()
                st.rerun()
            else:
                st.sidebar.warning("❌ Add items to cart first!")

# Add real-time segment insights in the main area
def display_segment_insights():
    """Display real-time segment insights based on current behavior"""
    
    if len(st.session_state.session_data['items']) > 0:
        features = calculate_features(abandoned=0)
        segment = predict_customer_segment(features)
        
        # Show segment insights in main area
        st.divider()
        st.subheader("🎯 Real-time Customer Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Segment", segment)
        
        with col2:
            cart_value = st.session_state.session_data['cart_value']
            if cart_value > 100:
                st.metric("Value Tier", "High")
            elif cart_value > 50:
                st.metric("Value Tier", "Medium") 
            else:
                st.metric("Value Tier", "Low")
        
        with col3:
            engagement = features.get('engagement_score', 0)
            if engagement > 7:
                st.metric("Engagement", "High")
            elif engagement > 4:
                st.metric("Engagement", "Medium")
            else:
                st.metric("Engagement", "Low")
        
        # Show proactive recovery suggestions
        if segment == "At-Risk Converters":
            st.info("💡 **Proactive Tip:** Consider applying a promo code to encourage conversion!")
        elif segment == "Engaged Researchers":
            st.info("💡 **Proactive Tip:** View shipping info to help with purchase decision!")
        elif segment == "Price-Sensitive Shoppers":
            st.info("💡 **Proactive Tip:** Check available discounts for better value!")


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("🛒 ShopEasy - Test Data Store")
    st.markdown("### Realistic shopping simulation with auto-saved features")
    
    # Your existing sidebar session info code remains the same
    with st.sidebar:
        st.header("👤 Session Info")
        st.write(f"**Session:** `{st.session_state.session_data['session_id']}`")
        st.write(f"**User:** `{st.session_state.session_data['user_id']}`")
        st.write(f"**Return:** {'Yes ✅' if st.session_state.session_data['return_user'] else 'No ❌'}")
        st.write(f"**Device:** {st.session_state.session_data['device_type']}")
        st.write(f"**Source:** {st.session_state.session_data['referral_source']}")
        
        session_duration = (datetime.now() - st.session_state.session_data['start_time']).total_seconds()
        mins = int(session_duration // 60)
        secs = int(session_duration % 60)
        st.write(f"**Duration:** {mins}m {secs}s")
        
        st.divider()
        st.header("📊 Stats")
        st.metric("Cart Value", f"₹{st.session_state.session_data['cart_value']:.2f}")
        st.metric("Items", len(st.session_state.session_data['items']))
        st.metric("Pages Viewed", len(st.session_state.session_data['pages_viewed']))
        st.metric("Product Views", st.session_state.session_data['view_product_count'])
        st.metric("Scroll Depth", f"{st.session_state.session_data['scroll_depth']:.1f}%")
        st.metric("Total Actions", len(st.session_state.session_data['events']))
        
        if st.session_state.session_data['discount_applied']:
            st.success(f"✅ Promo: {st.session_state.session_data['discount_code']} (-{st.session_state.session_data['discount_percentage']}%)")
        
        st.divider()
        st.header("🏁 End Session")

        if st.button("🔄 New Session", width='stretch', type="secondary"):
            start_new_session()
            st.session_state.show_new_session_btn = False
            st.session_state.current_recovery_strategy = None
            st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚪 Abandon", width='stretch'):
                handle_session_end("abandon")

        with col2:
            if st.button("💰 Purchase", width='stretch', type="primary"):
                handle_session_end("purchase")

        # Show "Continue" button only after abandonment/purchase
        if st.session_state.get('show_new_session_btn', False):
            st.divider()
            if st.button("🔄 Start New Session", width='stretch', type="primary"):
                start_new_session()
                st.session_state.show_new_session_btn = False
                st.session_state.current_recovery_strategy = None
                st.rerun()

# Display recovery strategy ONLY ONCE - remove the duplicate call below this

        # Always display recovery strategy if it exists (PERSISTENT)
        if st.session_state.get('current_recovery_strategy'):
            display_recovery_ui()  # This will use the strategy from session state
        
        st.divider()
        st.write(f"**Total Sessions:** {len(st.session_state.all_sessions)}")
    
    # Check if viewing product details
    if st.session_state.viewing_product:
        product = st.session_state.viewing_product
        navigate_page('product_detail')
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(product['image'], width='stretch')
            st.markdown(f"""
            ### {product['name']}
            **Price:** ₹{product['price']:.2f}
            **Category:** {product['category']}
            **Description:** {product['description']}
            **Rating:** ⭐⭐⭐⭐⭐ (4.8/5)
            **In Stock:** ✅ Yes
            """)
        
        with col2:
            st.write("")
            if st.button("🛒 Add to Cart", width='stretch', type="primary", key="add_detail"):
                add_to_cart(product)
                st.success(f"✅ Added to cart!")
                time.sleep(1)
                st.session_state.viewing_product = None
                st.rerun()
            
            if st.button("💖 Wishlist", width='stretch'):
                st.info("❤️ Added to wishlist!")
            
            if st.button("← Back", width='stretch'):
                st.session_state.viewing_product = None
                st.rerun()
        
        # Simulated scroll depth tracker
        st.divider()
        scroll_val = st.slider("📜 Scroll depth on this page", 0, 100, int(st.session_state.session_data['scroll_depth']), key="product_scroll")
        update_scroll_depth(scroll_val)
        
        st.divider()
        similar = [p for p in PRODUCTS if p['category'] == product['category'] and p['id'] != product['id']][:3]
        if similar:
            st.write("**Similar Products:**")
            cols = st.columns(3)
            for i, sim_prod in enumerate(similar):
                with cols[i]:
                    st.image(sim_prod['image'], width='stretch')
                    st.write(f"**{sim_prod['name']}**")
                    st.write(f"₹{sim_prod['price']:.0f}")
                    if st.button("👁️ View", key=f"sim_{sim_prod['id']}", width='stretch'):
                        view_product(sim_prod)
                        st.session_state.viewing_product = sim_prod
                        st.rerun()
        return
    
    # Main shopping page
    navigate_page('home')
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("🛍️ Browse Products")
        
        # Simulated scroll depth on main page
        scroll_val = st.slider("📜 Scroll depth", 0, 100, int(st.session_state.session_data['scroll_depth']), key="main_scroll")
        update_scroll_depth(scroll_val)
        
        # Category filter
        category_filter = st.selectbox("Filter by Category", ["All"] + list(set([p['category'] for p in PRODUCTS])))
        filtered_products = PRODUCTS if category_filter == "All" else [p for p in PRODUCTS if p['category'] == category_filter]
        
        for i in range(0, len(filtered_products), 3):
            cols = st.columns(3)
            for j, product in enumerate(filtered_products[i:i+3]):
                with cols[j]:
                    st.image(product['image'], width='stretch')
                    st.subheader(product['name'])
                    st.write(f"**₹{product['price']:.0f}**")
                    st.caption(product['category'])
                    
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if st.button("🛒 Add", key=f"add_{product['id']}", width='stretch'):
                            add_to_cart(product)
                            st.toast(f"✅ Added {product['name']}")
                    with btn_col2:
                        if st.button("👁️ View", key=f"view_{product['id']}", width='stretch'):
                            view_product(product)
                            st.session_state.viewing_product = product
                            st.rerun()
    
    with col2:
        st.header("🛒 Cart")
        
        if st.session_state.session_data['items']:
            total = st.session_state.session_data['cart_value']
            discount = 0
            
            if st.session_state.session_data['discount_applied']:
                discount_pct = st.session_state.session_data['discount_percentage']
                discount = total * (discount_pct / 100)
                final_total = total - discount
                st.write(f"**Subtotal:** ₹{total:.2f}")
                st.success(f"**Discount (-{discount_pct}%):** -₹{discount:.2f}")
                st.write(f"**Total:** ₹{final_total:.2f}")
            else:
                st.write(f"**Total:** ₹{total:.2f}")
            
            cart_value_scaled = total * 100
            if cart_value_scaled >= 20000:
                st.success("🎉 FREE SHIPPING!")
            else:
                remaining = (20000 - cart_value_scaled) / 100
                st.info(f"Add ₹{remaining:.2f} for free shipping")
            
            st.divider()
            
            # Checkout section
            st.write("**Checkout:**")
            
            with st.expander("🚚 Shipping Info", expanded=False):
                st.session_state.session_data['shipping_viewed'] = True
                log_event('view_shipping', None, None, None, 'shipping')
                st.markdown("""
                **Standard Delivery:** 3-5 business days
                **Express Delivery:** 1-2 business days (₹100 extra)
                **Free Shipping:** On orders above ₹200
                """)
                st.success("✅ Shipping info viewed")
            
            with st.expander("🎟️ Promo Code", expanded=False):
                promo_code = st.text_input("Enter promo code", placeholder="e.g., SAVE10", key="promo_input_real")
                if st.button("Apply", key="apply_promo"):
                    if promo_code:
                        valid, discount_pct = apply_promo_code(promo_code)
                        if valid:
                            st.session_state.promo_message = f"✅ Applied! {promo_code} (-{discount_pct}%)"
                        else:
                            st.session_state.promo_message = "❌ Invalid promo code. Try: SAVE10, SAVE20, WELCOME, SUMMER50"
                
                if st.session_state.promo_message:
                    if "✅" in st.session_state.promo_message:
                        st.success(st.session_state.promo_message)
                    else:
                        st.error(st.session_state.promo_message)
            
            st.divider()
            
            for item in st.session_state.session_data['items']:
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"• {item['name']}")
                    st.caption(f"₹{item['price']:.2f}")
                with col_b:
                    if st.button("❌", key=f"rm_{item['id']}_{id(item)}", width='stretch'):
                        remove_from_cart(item['id'])
                        st.rerun()
        else:
            st.info("🛍️ Your cart is empty")
            st.caption("Browse products and add items to get started!")
    
    # Feature preview at bottom
    st.divider()
    st.subheader("📋 Real-time Features")
    
    with st.expander("View calculated features for model", expanded=False):
        features = calculate_features(abandoned=0)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Duration (s)", f"{features['session_duration']:.0f}")
        with col2:
            st.metric("Pages Viewed", features['num_pages_viewed'])
        with col3:
            st.metric("Items Carted", features['num_items_carted'])
        with col4:
            st.metric("Engagement", f"{features['engagement_score']:.1f}/10")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Scroll Depth", f"{features['scroll_depth']:.1f}%")
        with col2:
            st.metric("Cart (₹)", f"{features['cart_value']:.0f}")
        with col3:
            st.metric("Discount", f"{'Yes ✅' if features['discount_applied'] else 'No ❌'}")
        with col4:
            st.metric("Shipping Info", f"{'Yes ✅' if features['has_viewed_shipping_info'] else 'No ❌'}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Intensity", f"{features['engagement_intensity']:.2f}")
        with col2:
            st.metric("Weekend", "✅" if features['is_weekend'] else "❌")
        with col3:
            st.metric("Peak Hours", "✅" if features['peak_hours'] else "❌")
        with col4:
            st.metric("Payment Page", "✅" if features['if_payment_page_reached'] else "❌")
        
        st.write("**Full Feature Set:**")
        features_df = pd.DataFrame([features])
        st.dataframe(features_df, width='stretch')
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Current Features",
                data=features_df.to_csv(index=False),
                file_name=f"session_features_{st.session_state.session_data['session_id']}.csv",
                mime="text/csv"
            )
            
if __name__ == "__main__":
    main()
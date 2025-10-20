import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import warnings
warnings.filterwarnings('ignore')

class EnhancedCustomerSegmenter:
    """Enhanced segmentation with better feature engineering and sensitivity"""
    
    def __init__(self, n_segments=5):
        self.n_segments = n_segments
        self.kmeans = None
        self.scaler = StandardScaler()
        self.segment_profiles = {}
        self.feature_columns = []
        self.global_metrics = {}
        
    def _engineer_segmentation_features(self, df):
        """Create enhanced features for better segmentation - FIXED"""
        # Basic features - ensure they exist
        required_features = [
            'engagement_score', 'num_items_carted', 'cart_value', 
            'session_duration', 'num_pages_viewed', 'scroll_depth',
            'return_user', 'if_payment_page_reached', 'discount_applied',
            'has_viewed_shipping_info', 'abandoned'
        ]
        
        # Check if all required features exist
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        features = df[required_features].copy()
        
        # Create enhanced behavioral features - FIXED LOGIC
        # Normalize before creating composite features to avoid scaling issues
        features['browsing_intensity'] = (
            self._normalize_feature(df['num_pages_viewed']) + 
            self._normalize_feature(df['session_duration'])
        ) / 2
        
        features['purchase_readiness'] = (
            df['if_payment_page_reached'] + 
            df['has_viewed_shipping_info']
        ) / 2
        
        # Use normalized values for multiplication to avoid extreme values
        cart_norm = self._normalize_feature(df['cart_value'])
        engagement_norm = self._normalize_feature(df['engagement_score'])
        features['value_engagement_ratio'] = cart_norm * engagement_norm
        
        # FIXED: research_behavior should compare to pages viewed median, not items
        features['research_behavior'] = (df['num_pages_viewed'] > df['num_pages_viewed'].median()).astype(int)
        
        # FIXED: purchase_intent with proper weights that sum to 1.0
        features['purchase_intent'] = (
            self._normalize_feature(df['num_items_carted']) * 0.25 +
            df['if_payment_page_reached'] * 0.25 +
            df['has_viewed_shipping_info'] * 0.25 +
            self._normalize_feature(df['engagement_score']) * 0.25
        )
        
        print(f"📊 Using {len(features.columns)} enhanced features for segmentation")
        return features
    
    def _normalize_feature(self, series):
        """Normalize a feature series to 0-1 range"""
        return (series - series.min()) / (series.max() - series.min())
    
    def _create_enhanced_segments(self, df, labels):
        """Create more sensitive segment profiles - FIXED"""
        df_segmented = df.copy()
        df_segmented['segment'] = labels
        
        # Calculate global metrics - FIXED: store for later use
        self.global_metrics = {
            'global_avg_cart_value': df['cart_value'].mean(),
            'global_avg_engagement': df['engagement_score'].mean(),
            'global_avg_abandonment': df['abandoned'].mean() * 100,
            'global_avg_return_rate': df['return_user'].mean() * 100,
            'global_avg_payment_reach': df['if_payment_page_reached'].mean() * 100,
            'global_avg_items': df['num_items_carted'].mean(),
            'global_avg_pages_viewed': df['num_pages_viewed'].mean(),
            'global_max_cart_value': df['cart_value'].max(),
            'global_min_cart_value': df['cart_value'].min()
        }
        
        segment_profiles = {}
        
        for segment_id in range(self.n_segments):
            segment_data = df_segmented[df_segmented['segment'] == segment_id]
            
            if len(segment_data) == 0:
                continue
                
            # Calculate comprehensive metrics
            profile = {
                'size': len(segment_data),
                'size_percentage': len(segment_data) / len(df) * 100,
                
                # Core metrics
                'abandonment_rate': segment_data['abandoned'].mean() * 100,
                'avg_cart_value': segment_data['cart_value'].mean(),
                'avg_engagement': segment_data['engagement_score'].mean(),
                'avg_items': segment_data['num_items_carted'].mean(),
                'avg_session_duration': segment_data['session_duration'].mean(),
                'avg_pages_viewed': segment_data['num_pages_viewed'].mean(),
                'avg_scroll_depth': segment_data['scroll_depth'].mean(),
                
                # User characteristics
                'return_user_rate': segment_data['return_user'].mean() * 100,
                'discount_sensitivity': segment_data['discount_applied'].mean() * 100,
                'payment_reach_rate': segment_data['if_payment_page_reached'].mean() * 100,
                'shipping_info_view_rate': segment_data['has_viewed_shipping_info'].mean() * 100,
            }
            
            # Add global metrics for comparison
            profile.update(self.global_metrics)
            
            # Enhanced segment identification
            profile.update(self._enhanced_segment_identification(profile))
            segment_profiles[segment_id] = profile
            
        return segment_profiles
    
    def _enhanced_segment_identification(self, profile):
        """Simplified segment identification focusing on 5 core segments - FIXED"""
        # Calculate relative scores (how far from global average)
        rel_abandonment = profile['abandonment_rate'] - profile['global_avg_abandonment']
        rel_cart_value = profile['avg_cart_value'] - profile['global_avg_cart_value']
        rel_engagement = profile['avg_engagement'] - profile['global_avg_engagement']
        rel_return_rate = profile['return_user_rate'] - profile['global_avg_return_rate']
        rel_payment_reach = profile['payment_reach_rate'] - profile['global_avg_payment_reach']
        
        # FOCUS ON 5 CORE SEGMENTS ONLY
        segment_scores = {
            'High-Value Loyalists': 0,      # High value, high loyalty, low abandonment
            'At-Risk Converters': 0,        # High value, low loyalty, medium abandonment  
            'Engaged Researchers': 0,       # Medium value, high engagement, research behavior
            'Price-Sensitive Shoppers': 0,  # Low value, high discount sensitivity
            'Casual Browsers': 0           # Low everything, minimal engagement
        }
        
        # High-Value Loyalists: High value, high loyalty, completes purchases
        if rel_cart_value > 0.05: segment_scores['High-Value Loyalists'] += 3
        if rel_return_rate > 5: segment_scores['High-Value Loyalists'] += 2
        if rel_abandonment < -5: segment_scores['High-Value Loyalists'] += 2
        if rel_payment_reach > 10: segment_scores['High-Value Loyalists'] += 1
        
        # At-Risk Converters: High value but might abandon, need conversion
        if rel_cart_value > 0.05: segment_scores['At-Risk Converters'] += 3
        if rel_return_rate < 0: segment_scores['At-Risk Converters'] += 2
        if rel_abandonment > 5: segment_scores['At-Risk Converters'] += 2
        if rel_payment_reach > 5: segment_scores['At-Risk Converters'] += 1
        
        # Engaged Researchers: High engagement, research behavior, medium value
        if rel_engagement > 0.2: segment_scores['Engaged Researchers'] += 3
        if profile['avg_pages_viewed'] > profile['global_avg_pages_viewed']: segment_scores['Engaged Researchers'] += 2
        if profile['shipping_info_view_rate'] > 50: segment_scores['Engaged Researchers'] += 1
        if abs(rel_cart_value) < 0.05: segment_scores['Engaged Researchers'] += 1
        
        # Price-Sensitive Shoppers: High discount sensitivity, lower value
        if profile['discount_sensitivity'] > 40: segment_scores['Price-Sensitive Shoppers'] += 3
        if rel_cart_value < 0: segment_scores['Price-Sensitive Shoppers'] += 2
        if rel_abandonment > 5: segment_scores['Price-Sensitive Shoppers'] += 1
        
        # Casual Browsers: Low engagement across all metrics
        if rel_engagement < -0.1: segment_scores['Casual Browsers'] += 3
        if rel_cart_value < -0.05: segment_scores['Casual Browsers'] += 2
        if rel_payment_reach < -10: segment_scores['Casual Browsers'] += 2
        if profile['avg_session_duration'] < profile['global_avg_items']: segment_scores['Casual Browsers'] += 1
        
        # Get the highest scoring segment
        best_segment = max(segment_scores, key=segment_scores.get)
        best_score = segment_scores[best_segment]
        
        # Fallback for low scores
        if best_score < 3:
            best_segment = self._simplified_segment_fallback(profile)
        
        segment_info = {
            'segment_name': best_segment,
            'description': self._get_segment_description(best_segment),
            'recovery_priority': self._get_segment_priority(best_segment, profile),
            'business_value': self._get_business_value(best_segment),
            'recovery_priority_score': self._calculate_enhanced_priority(profile),
            'segment_score': best_score
        }
        
        return segment_info
    
    def _simplified_segment_fallback(self, profile):
        """Simplified fallback logic for 5 segments"""
        # Check loyalty and value first
        if profile['return_user_rate'] > 50:
            if profile['avg_cart_value'] > profile['global_avg_cart_value']:
                return "High-Value Loyalists"
            else:
                return "Price-Sensitive Shoppers"  # Loyal but price sensitive
        
        # Check high value but new
        if profile['avg_cart_value'] > profile['global_avg_cart_value']:
            return "At-Risk Converters"
        
        # Check high engagement
        if profile['avg_engagement'] > profile['global_avg_engagement']:
            return "Engaged Researchers"
        
        # Check discount sensitivity
        if profile['discount_sensitivity'] > 40:
            return "Price-Sensitive Shoppers"
        
        # Default to casual browsers
        return "Casual Browsers"
    
    def _get_segment_description(self, segment_name):
        """Get segment descriptions for 5 core segments"""
        descriptions = {
            "High-Value Loyalists": "Frequent high-spending customers with low abandonment rates",
            "At-Risk Converters": "High-value new customers who need conversion encouragement",
            "Engaged Researchers": "Highly engaged users researching products before purchase",
            "Price-Sensitive Shoppers": "Customers highly responsive to discounts and promotions",
            "Casual Browsers": "Low-engagement users exploring with minimal intent"
        }
        return descriptions.get(segment_name, "Users with typical shopping behavior")
    
    def _get_segment_priority(self, segment_name, profile):
        """Get recovery priority based on 5 segments"""
        base_priority = {
            "At-Risk Converters": "Very High",
            "Engaged Researchers": "High",
            "Price-Sensitive Shoppers": "Medium",
            "High-Value Loyalists": "Low", 
            "Casual Browsers": "Low"
        }
        
        priority = base_priority.get(segment_name, "Medium")
        
        # Adjust based on actual abandonment rate
        if profile['abandonment_rate'] > 70:
            priority = "Very High"
        elif profile['abandonment_rate'] > 50 and priority == "Medium":
            priority = "High"
            
        return priority
    
    def _get_business_value(self, segment_name):
        """Get business value rating for 5 segments"""
        value_map = {
            "High-Value Loyalists": "Very High",
            "At-Risk Converters": "High",
            "Engaged Researchers": "Medium-High",
            "Price-Sensitive Shoppers": "Medium",
            "Casual Browsers": "Low"
        }
        return value_map.get(segment_name, "Medium")
    
    def _calculate_enhanced_priority(self, profile):
        """Enhanced priority scoring - FIXED"""
        weights = {
            'abandonment_rate': 0.35,
            'avg_cart_value': 0.20,
            'return_user_rate': 0.15,
            'payment_reach_rate': 0.15,
            'avg_engagement': 0.10,
            'discount_sensitivity': 0.05
        }
        
        abandonment_score = min(profile['abandonment_rate'] / 100, 1.0)
        
        cart_range = profile['global_max_cart_value'] - profile['global_min_cart_value']
        cart_value_score = (profile['avg_cart_value'] - profile['global_min_cart_value']) / cart_range
        
        return_user_score = profile['return_user_rate'] / 100
        payment_score = profile['payment_reach_rate'] / 100
        
        engagement_score = (profile['avg_engagement'] + 1) / 2 if profile['avg_engagement'] >= -1 else 0
        discount_score = profile['discount_sensitivity'] / 100
        
        priority_score = (
            weights['abandonment_rate'] * abandonment_score +
            weights['avg_cart_value'] * cart_value_score +
            weights['return_user_rate'] * return_user_score +
            weights['payment_reach_rate'] * payment_score +
            weights['avg_engagement'] * engagement_score +
            weights['discount_sensitivity'] * discount_score
        )
        
        return min(int(priority_score * 100), 100)
    
    def fit(self, df):
        """Perform enhanced customer segmentation"""
        try:
            # Use engineered features
            features_df = self._engineer_segmentation_features(df)
            self.feature_columns = features_df.columns.tolist()
            
            X = features_df.values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Perform K-means clustering
            self.kmeans = KMeans(n_clusters=self.n_segments, random_state=42, n_init=10)
            labels = self.kmeans.fit_predict(X_scaled)
            
            # Create enhanced segment profiles
            self.segment_profiles = self._create_enhanced_segments(df, labels)
            
            print("🎯 Enhanced segmentation completed!")
            return self
            
        except Exception as e:
            print(f"❌ Error in segmentation: {e}")
            raise
    
    def predict_segment(self, df):
        """Predict segments for new data"""
        if not hasattr(self, 'feature_columns') or not self.feature_columns:
            raise ValueError("Segmenter must be fitted before prediction")
        
        # Use the same feature engineering as during fit
        features_df = self._engineer_segmentation_features(df)
        
        # Ensure we have the same columns as during training
        missing_cols = set(self.feature_columns) - set(features_df.columns)
        if missing_cols:
            raise ValueError(f"Missing features for prediction: {missing_cols}")
        
        X = features_df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        return self.kmeans.predict(X_scaled)

def main():
    """Enhanced segmentation with simplified 5-segment approach"""
    try:
        # Get the correct paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        
        # Load featured dataset
        featured_path = os.path.join(project_root, 'data', 'cart_abandonment_featured.csv')
        
        print("=== Loading Dataset ===")
        print(f"Featured data: {featured_path}")
        
        if os.path.exists(featured_path):
            df = pd.read_csv(featured_path)
            print(f"✅ Featured dataset loaded: {df.shape}")
            print(f"📊 Dataset overview:")
            print(f"   - Abandonment rate: {df['abandoned'].mean():.1%}")
            print(f"   - Cart value range: {df['cart_value'].min():.3f} to {df['cart_value'].max():.3f}")
            print(f"   - Engagement score range: {df['engagement_score'].min():.3f} to {df['engagement_score'].max():.3f}")
            print(f"   - Return user rate: {df['return_user'].mean():.1%}")
        else:
            print("❌ Featured dataset not found!")
            return
        
        print("\n🔄 Performing Enhanced Customer Segmentation (5 Segments)...")
        
        # Initialize and fit enhanced segmenter
        segmenter = EnhancedCustomerSegmenter(n_segments=5)
        segmenter.fit(df)
        
        # Add segments to original dataframe
        df['segment'] = segmenter.predict_segment(df)
        
        print("\n" + "="*60)
        print("📊 SIMPLIFIED 5-SEGMENT CUSTOMER ANALYSIS")
        print("="*60)
        
        # Display enhanced segment analysis
        for segment_id, profile in segmenter.segment_profiles.items():
            print(f"\n🎯 Segment {segment_id}: {profile['segment_name']}")
            print(f"   📈 Size: {profile['size']} users ({profile['size_percentage']:.1f}%)")
            print(f"   💰 Avg Cart Value: {profile['avg_cart_value']:.3f} (normalized)")
            print(f"   🚫 Abandonment Rate: {profile['abandonment_rate']:.1f}%")
            print(f"   🔄 Return User Rate: {profile['return_user_rate']:.1f}%")
            print(f"   💳 Payment Reach Rate: {profile['payment_reach_rate']:.1f}%")
            print(f"   ⭐ Recovery Priority: {profile['recovery_priority']} ({profile['recovery_priority_score']}/100)")
            print(f"   📝 Description: {profile['description']}")
            print(f"   💼 Business Value: {profile['business_value']}")
            
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
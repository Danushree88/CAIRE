import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

def analyze_data_type(df):
    """Analyze if data is normalized (0-1) or actual values"""
    print("🔍 DATA TYPE ANALYSIS")
    print("=" * 40)
    
    cart_range = (df['cart_value'].min(), df['cart_value'].max())
    eng_range = (df['engagement_score'].min(), df['engagement_score'].max())
    
    print(f"💰 Cart Value: {cart_range[0]:.2f} to {cart_range[1]:.2f}")
    print(f"🎯 Engagement: {eng_range[0]:.2f} to {eng_range[1]:.2f}")
    
    # Check if data appears normalized
    is_normalized = (cart_range[1] <= 1.0 and eng_range[1] <= 1.0)
    
    if is_normalized:
        print("📊 DATA TYPE: NORMALIZED (0-1 scale)")
    else:
        print("📊 DATA TYPE: ACTUAL VALUES")
    
    return is_normalized

def normalize_customer_data(df):
    """Normalize raw customer data to 0-1 range for segmentation"""
    df_normalized = df.copy()
    
    # Normalize these key features to 0-1 range
    features_to_normalize = [
        'engagement_score', 'cart_value', 'session_duration',
        'num_pages_viewed', 'num_items_carted', 'scroll_depth'
    ]
    
    print("🔄 Normalizing raw data...")
    for feature in features_to_normalize:
        if feature in df.columns:
            min_val = df[feature].min()
            max_val = df[feature].max()
            
            print(f"   {feature}: {min_val:.2f} to {max_val:.2f}")
            
            if max_val > min_val:
                df_normalized[feature] = (df[feature] - min_val) / (max_val - min_val)
            else:
                df_normalized[feature] = 0.5  # Default if all values same
    
    return df_normalized

class RuleBasedCustomerSegmenter:
    """Rule-based segmentation for 5 core customer segments - CORRECTED"""
    
    def __init__(self, use_normalized_data=False):
        self.segment_profiles = {}
        self.global_metrics = {}
        self.use_normalized_data = use_normalized_data
        print(f"🎯 Segmenter initialized for {'NORMALIZED' if use_normalized_data else 'ACTUAL'} data")
        
    def calculate_global_metrics(self, df):
        """Calculate global metrics for comparison"""
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
        return self.global_metrics
    
    def segment_single_customer(self, customer_data):
        """
        Segment a single customer using CORRECT rule-based logic
        """
        # Extract key features safely with defaults
        cart_value = customer_data.get('cart_value', 0)
        engagement_score = customer_data.get('engagement_score', 0)
        return_user = customer_data.get('return_user', 0)
        num_pages_viewed = customer_data.get('num_pages_viewed', 0)
        num_items_carted = customer_data.get('num_items_carted', 0)
        abandoned = customer_data.get('abandoned', 0)
        if_payment_page_reached = customer_data.get('if_payment_page_reached', 0)
        discount_applied = customer_data.get('discount_applied', 0)
        
        print(f"📊 Customer Features:")
        print(f"   Cart Value: {cart_value:.2f}")
        print(f"   Engagement: {engagement_score:.2f}")
        print(f"   Return User: {return_user}")
        print(f"   Abandoned: {abandoned}")
        print(f"   Payment Reached: {if_payment_page_reached}")
        print(f"   Discount Applied: {discount_applied}")
        print(f"   Pages Viewed: {num_pages_viewed}")
        
        if self.use_normalized_data:
            return self._segment_normalized(
                cart_value, engagement_score, return_user, abandoned,
                if_payment_page_reached, discount_applied, num_pages_viewed, num_items_carted
            )
        else:
            return self._segment_actual(
                cart_value, engagement_score, return_user, abandoned,
                if_payment_page_reached, discount_applied, num_pages_viewed, num_items_carted
            )
    
    def _segment_normalized(self, cart_value, engagement, return_user, abandoned, 
                           payment_reached, discount_applied, pages_viewed, items_carted):
        """Segmentation for NORMALIZED data (0-1 scale)"""
        # 1. High-Value Loyalists
        if (return_user == 1 and 
            cart_value > 0.7 and      # High cart value
            engagement > 0.7 and      # High engagement  
            payment_reached == 1 and  # Reached payment
            abandoned == 0):          # Didn't abandon
            segment = "High-Value Loyalists"
            
        # 2. At-Risk Converters  
        elif (return_user == 0 and      # New customer
              abandoned == 1 and        # Abandoned cart
              cart_value > 0.5 and      # Good cart value
              payment_reached == 0):    # Didn't reach payment
            segment = "At-Risk Converters"
            
        # 3. Engaged Researchers
        elif (engagement >= 0.7 and     # High engagement
              pages_viewed > 0.6 and    # Many pages viewed
              items_carted > 0 and      # Added items to cart
              payment_reached == 0):    # Didn't complete purchase
            segment = "Engaged Researchers"
            
        # 4. Price-Sensitive Shoppers
        elif (discount_applied == 1 or   # Used discount
              cart_value < 0.3):         # Low spending
            segment = "Price-Sensitive Shoppers"
            
        # 5. Casual Browsers (default)
        else:
            segment = "Casual Browsers"
        
        print(f"🎯 Assigned Segment: {segment}")
        return segment
    
    def _segment_actual(self, cart_value, engagement, return_user, abandoned, 
                       payment_reached, discount_applied, pages_viewed, items_carted):
        """Segmentation for ACTUAL data values - CORRECTED THRESHOLDS"""
        # 1. High-Value Loyalists
        if (return_user == 1 and 
            cart_value > 5000 and      # High cart value (₹5000+)
            engagement > 7 and         # High engagement (7+/10)
            payment_reached == 1 and   # Reached payment
            abandoned == 0):           # Didn't abandon
            segment = "High-Value Loyalists"
            
        # 2. At-Risk Converters  
        elif (return_user == 0 and      # New customer
              abandoned == 1 and        # Abandoned cart
              cart_value > 1000 and     # Good cart value (₹1000+)
              payment_reached == 0):    # Didn't reach payment
            segment = "At-Risk Converters"
            
        # 3. Engaged Researchers
        elif (engagement >= 7 and       # High engagement (7+/10)
              pages_viewed > 6 and      # Many pages viewed (6+)
              items_carted > 0 and      # Added items to cart
              payment_reached == 0):    # Didn't complete purchase
            segment = "Engaged Researchers"
            
        # 4. Price-Sensitive Shoppers
        elif (discount_applied == 1 or   # Used discount
              cart_value < 500):         # Low spending (<₹500)
            segment = "Price-Sensitive Shoppers"
            
        # 5. Casual Browsers (default)
        else:
            segment = "Casual Browsers"
        
        print(f"🎯 Assigned Segment: {segment}")
        return segment
    
    def segment_dataset(self, df):
        """
        Segment entire dataset using rule-based logic
        """
        print("🔄 Performing rule-based segmentation on dataset...")
        
        # Calculate global metrics
        self.calculate_global_metrics(df)
        
        segments = []
        
        for idx, row in df.iterrows():
            customer_data = row.to_dict()
            segment = self.segment_single_customer(customer_data)
            segments.append(segment)
        
        df_segmented = df.copy()
        df_segmented['segment'] = segments
        
        # Create aggregate segment profiles
        self._create_aggregate_segment_profiles(df_segmented)
        
        return df_segmented
    
    def _create_aggregate_segment_profiles(self, df):
        """Create aggregate profiles for each segment"""
        segment_profiles = {}
        
        for segment_name in df['segment'].unique():
            segment_data = df[df['segment'] == segment_name]
            
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
                
                # Segment info
                'segment_name': segment_name,
                'description': self._get_segment_description(segment_name),
                'recovery_priority': self._get_aggregate_priority(segment_name, segment_data),
                'business_value': self._get_business_value(segment_name),
                'recovery_priority_score': self._calculate_aggregate_priority_score(segment_data)
            }
            
            segment_profiles[segment_name] = profile
        
        self.segment_profiles = segment_profiles
        return segment_profiles
    
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
    
    def _get_aggregate_priority(self, segment_name, segment_data):
        """Get recovery priority for segment aggregate"""
        base_priority = {
            "At-Risk Converters": "Very High",
            "Engaged Researchers": "High",
            "Price-Sensitive Shoppers": "Medium",
            "High-Value Loyalists": "Low",
            "Casual Browsers": "Low"
        }
        
        priority = base_priority.get(segment_name, "Medium")
        
        # Adjust based on aggregate abandonment rate
        abandonment_rate = segment_data['abandoned'].mean() * 100
        if abandonment_rate > 70:
            priority = "Very High"
        elif abandonment_rate > 50 and priority == "Medium":
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
    
    def _calculate_aggregate_priority_score(self, segment_data):
        """Calculate priority score for segment aggregate - CORRECTED"""
        weights = {
            'abandonment_rate': 0.35,
            'avg_cart_value': 0.25,
            'return_user_rate': 0.15, 
            'payment_reach_rate': 0.15,
            'avg_engagement': 0.10
        }
        
        abandonment_score = segment_data['abandoned'].mean() * 100
        
        # Scale cart value appropriately based on data type
        if self.use_normalized_data:
            cart_value_score = segment_data['cart_value'].mean() * 100  # 0-1 -> 0-100
            engagement_score = segment_data['engagement_score'].mean() * 100  # 0-1 -> 0-100
        else:
            # For actual values, scale cart value to reasonable range
            max_reasonable_cart = 10000  # Assume ₹10,000 as max reasonable cart
            cart_value_score = min(segment_data['cart_value'].mean() / max_reasonable_cart * 100, 100)
            engagement_score = segment_data['engagement_score'].mean() * 10  # 0-10 -> 0-100
        
        return_user_score = segment_data['return_user'].mean() * 100
        payment_score = segment_data['if_payment_page_reached'].mean() * 100
        
        priority_score = (
            weights['abandonment_rate'] * abandonment_score +
            weights['avg_cart_value'] * cart_value_score +
            weights['return_user_rate'] * return_user_score +
            weights['payment_reach_rate'] * payment_score +
            weights['avg_engagement'] * engagement_score
        )
        
        return min(int(priority_score), 100)

def analyze_segmentation_quality(df_segmented):
    """Analyze if segmentation makes sense"""
    print("\n" + "="*60)
    print("🔍 SEGMENTATION QUALITY ANALYSIS")
    print("="*60)
    
    # Check segment distribution
    segment_counts = df_segmented['segment'].value_counts()
    print(f"\n📊 Segment Distribution:")
    for segment, count in segment_counts.items():
        percentage = (count / len(df_segmented)) * 100
        print(f"   {segment}: {count} users ({percentage:.1f}%)")
    
    # Analyze key metrics by segment
    print(f"\n📈 Key Metrics by Segment:")
    metrics_by_segment = df_segmented.groupby('segment').agg({
        'cart_value': ['mean', 'std'],
        'engagement_score': ['mean', 'std'],
        'abandoned': 'mean',
        'return_user': 'mean',
        'if_payment_page_reached': 'mean',
        'num_pages_viewed': 'mean',
        'num_items_carted': 'mean'
    }).round(2)
    
    print(metrics_by_segment)
    
    # Check if segments make business sense
    print(f"\n✅ Business Logic Validation:")
    for segment in df_segmented['segment'].unique():
        segment_data = df_segmented[df_segmented['segment'] == segment]
        
        avg_cart = segment_data['cart_value'].mean()
        avg_abandonment = segment_data['abandoned'].mean() * 100
        avg_return = segment_data['return_user'].mean() * 100
        avg_payment_reach = segment_data['if_payment_page_reached'].mean() * 100
        
        print(f"\n🎯 {segment}:")
        print(f"   Avg Cart: ₹{avg_cart:.2f}")
        print(f"   Abandonment: {avg_abandonment:.1f}%")
        print(f"   Return Users: {avg_return:.1f}%")
        print(f"   Payment Reach: {avg_payment_reach:.1f}%")
        
        # Business logic validation
        if segment == "High-Value Loyalists":
            if avg_return > 50 and avg_abandonment < 30 and avg_payment_reach > 80:
                print("   ✅ Valid: High return, low abandonment, high payment completion")
            else:
                print("   ⚠️ Check: Should have high return, low abandonment")
                
        elif segment == "At-Risk Converters":
            if avg_return < 50 and avg_abandonment > 50 and avg_payment_reach < 50:
                print("   ✅ Valid: Low return, high abandonment, low payment reach")
            else:
                print("   ⚠️ Check: Should have low return, high abandonment")

def main():
    """Rule-based segmentation main function - CORRECTED"""
    try:
        # Load your actual data
        df = pd.read_csv('your_data.csv')  # Replace with your actual file path
        
        print("=== Loading Dataset ===")
        print(f"✅ Raw dataset loaded: {df.shape}")
        
        # FIRST: Analyze data type
        is_normalized = analyze_data_type(df)
        
        print(f"\n📊 Data summary:")
        print(f"   - Cart Value: {df['cart_value'].min():.2f} to {df['cart_value'].max():.2f}")
        print(f"   - Engagement: {df['engagement_score'].min():.2f} to {df['engagement_score'].max():.2f}")
        print(f"   - Abandonment Rate: {df['abandoned'].mean() * 100:.1f}%")
        print(f"   - Return Users: {df['return_user'].mean() * 100:.1f}%")
        print(f"   - Payment Reach: {df['if_payment_page_reached'].mean() * 100:.1f}%")
        
        # Initialize segmenter with correct data type
        segmenter = RuleBasedCustomerSegmenter(use_normalized_data=is_normalized)
        
        # PERFORM RULE-BASED SEGMENTATION
        print("\n🔄 Performing Rule-Based Customer Segmentation (5 Segments)...")
        
        # Segment the entire dataset
        df_segmented = segmenter.segment_dataset(df)
        
        # Display results
        print("\n" + "="*60)
        print("📊 RULE-BASED 5-SEGMENT CUSTOMER ANALYSIS")
        print("="*60)
        
        for segment_name, profile in segmenter.segment_profiles.items():
            print(f"\n🎯 Segment: {segment_name}")
            print(f"   📈 Size: {profile['size']} users ({profile['size_percentage']:.1f}%)")
            print(f"   💰 Avg Cart Value: ₹{profile['avg_cart_value']:.2f}")
            print(f"   🚫 Abandonment Rate: {profile['abandonment_rate']:.1f}%")
            print(f"   🔄 Return User Rate: {profile['return_user_rate']:.1f}%")
            print(f"   💳 Payment Reach Rate: {profile['payment_reach_rate']:.1f}%")
            print(f"   ⭐ Recovery Priority: {profile['recovery_priority']} ({profile['recovery_priority_score']}/100)")
            print(f"   📝 Description: {profile['description']}")
            print(f"   💼 Business Value: {profile['business_value']}")
        
        # Analyze segmentation quality
        analyze_segmentation_quality(df_segmented)
        
        # Save output
        output_path = 'rule_based_segmented_output.csv'
        df_segmented.to_csv(output_path, index=False)
        print(f"\n💾 Rule-based segmentation results saved to: {output_path}")

    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
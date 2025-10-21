import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import hdbscan
import time
import os
import warnings
warnings.filterwarnings('ignore')

class EnhancedCustomerSegmenter:
    def __init__(self, n_segments=5):
        self.n_segments = n_segments
        self.kmeans = None
        self.scaler = StandardScaler()
        self.segment_profiles = {}
        self.feature_columns = []
        self.global_metrics = {}
        self.assigned_segments = set()  # Track assigned segment names
        
    def _engineer_segmentation_features(self, df):
        required_features = [
            'engagement_score', 'num_items_carted', 'cart_value', 
            'session_duration', 'num_pages_viewed', 'scroll_depth',
            'return_user', 'if_payment_page_reached', 'discount_applied',
            'has_viewed_shipping_info', 'abandoned'
        ]
        
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        features = df[required_features].copy()
        features['browsing_intensity'] = (
            self._normalize_feature(df['num_pages_viewed']) + 
            self._normalize_feature(df['session_duration'])
        ) / 2
        
        features['purchase_readiness'] = (
            df['if_payment_page_reached'] + 
            df['has_viewed_shipping_info']
        ) / 2
        
        cart_norm = self._normalize_feature(df['cart_value'])
        engagement_norm = self._normalize_feature(df['engagement_score'])
        features['value_engagement_ratio'] = cart_norm * engagement_norm
        
        features['research_behavior'] = (df['num_pages_viewed'] > df['num_pages_viewed'].median()).astype(int)
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
        df_segmented = df.copy()
        df_segmented['segment'] = labels
        
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
            profile = {
                'size': len(segment_data),
                'size_percentage': len(segment_data) / len(df) * 100,
                
                'abandonment_rate': segment_data['abandoned'].mean() * 100,
                'avg_cart_value': segment_data['cart_value'].mean(),
                'avg_engagement': segment_data['engagement_score'].mean(),
                'avg_items': segment_data['num_items_carted'].mean(),
                'avg_session_duration': segment_data['session_duration'].mean(),
                'avg_pages_viewed': segment_data['num_pages_viewed'].mean(),
                'avg_scroll_depth': segment_data['scroll_depth'].mean(),
                'return_user_rate': segment_data['return_user'].mean() * 100,
                'discount_sensitivity': segment_data['discount_applied'].mean() * 100,
                'payment_reach_rate': segment_data['if_payment_page_reached'].mean() * 100,
                'shipping_info_view_rate': segment_data['has_viewed_shipping_info'].mean() * 100,
            }
            
            profile.update(self.global_metrics)
            
            profile.update(self._enhanced_segment_identification(profile, segment_id))
            segment_profiles[segment_id] = profile
            
        return segment_profiles
    
    def _enhanced_segment_identification(self, profile, segment_id):
        rel_abandonment = profile['abandonment_rate'] - profile['global_avg_abandonment']
        rel_cart_value = profile['avg_cart_value'] - profile['global_avg_cart_value']
        rel_engagement = profile['avg_engagement'] - profile['global_avg_engagement']
        rel_return_rate = profile['return_user_rate'] - profile['global_avg_return_rate']
        rel_payment_reach = profile['payment_reach_rate'] - profile['global_avg_payment_reach']

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
        
        if best_segment in self.assigned_segments and best_score < 6:
            sorted_segments = sorted(segment_scores.items(), key=lambda x: x[1], reverse=True)
            for segment_name, score in sorted_segments:
                if segment_name not in self.assigned_segments and score >= 3:
                    best_segment = segment_name
                    best_score = score
                    break
        
        if best_score < 3:
            best_segment = self._simplified_segment_fallback(profile)
            
        self.assigned_segments.add(best_segment)
        
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
        if profile['return_user_rate'] > 50:
            if profile['avg_cart_value'] > profile['global_avg_cart_value']:
                if "High-Value Loyalists" not in self.assigned_segments:
                    return "High-Value Loyalists"
                else:
                    return "At-Risk Converters" 
            else:
                return "Price-Sensitive Shoppers" 
        
        if profile['avg_cart_value'] > profile['global_avg_cart_value']:
            return "At-Risk Converters"
        
        if profile['avg_engagement'] > profile['global_avg_engagement']:
            return "Engaged Researchers"

        if profile['discount_sensitivity'] > 40:
            return "Price-Sensitive Shoppers"
        
        # Default to casual browsers
        return "Casual Browsers"
    
    def _get_segment_description(self, segment_name):
        descriptions = {
            "High-Value Loyalists": "Frequent high-spending customers with low abandonment rates",
            "At-Risk Converters": "High-value new customers who need conversion encouragement",
            "Engaged Researchers": "Highly engaged users researching products before purchase",
            "Price-Sensitive Shoppers": "Customers highly responsive to discounts and promotions",
            "Casual Browsers": "Low-engagement users exploring with minimal intent"
        }
        return descriptions.get(segment_name, "Users with typical shopping behavior")
    
    def _get_segment_priority(self, segment_name, profile):
        base_priority = {
            "At-Risk Converters": "Very High",
            "Engaged Researchers": "High",
            "Price-Sensitive Shoppers": "Medium",
            "High-Value Loyalists": "Low", 
            "Casual Browsers": "Low"
        }
        
        priority = base_priority.get(segment_name, "Medium")
        
        if profile['abandonment_rate'] > 70:
            priority = "Very High"
        elif profile['abandonment_rate'] > 50 and priority == "Medium":
            priority = "High"
            
        return priority
    
    def _get_business_value(self, segment_name):
        value_map = {
            "High-Value Loyalists": "Very High",
            "At-Risk Converters": "High",
            "Engaged Researchers": "Medium-High",
            "Price-Sensitive Shoppers": "Medium",
            "Casual Browsers": "Low"
        }
        return value_map.get(segment_name, "Medium")
    
    def _calculate_enhanced_priority(self, profile):
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
    
    def _evaluate_clustering(self, X_scaled, labels, method_name):
        try:
            silhouette = silhouette_score(X_scaled, labels)
            calinski = calinski_harabasz_score(X_scaled, labels)
            davies = davies_bouldin_score(X_scaled, labels)
            
            evaluation = {
                'method': method_name,
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski,
                'davies_bouldin_score': davies,
                'n_clusters': len(np.unique(labels))
            }
            
            print(f"📊 {method_name} Evaluation:")
            print(f"   Silhouette Score: {silhouette:.4f} (higher is better)")
            print(f"   Calinski-Harabasz: {calinski:.4f} (higher is better)")
            print(f"   Davies-Bouldin: {davies:.4f} (lower is better)")
            
            return evaluation
        except Exception as e:
            print(f"⚠️ Evaluation failed for {method_name}: {e}")
            return None
    
    def kmeans_from_scratch(self, X, max_iters=100, restarts=5):
        n_samples, n_features = X.shape
        best_inertia = np.inf
        best_labels, best_centroids = None, None

        for _ in range(restarts):
            # Random initialization
            init_idx = np.random.choice(n_samples, self.n_segments, replace=False)
            centroids = X[init_idx]

            for _ in range(max_iters):
                # Assign clusters
                distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
                labels = np.argmin(distances, axis=1)

                # Recompute centroids
                new_centroids = np.array([
                    X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
                    for i in range(self.n_segments)
                ])

                if np.allclose(centroids, new_centroids):
                    break
                centroids = new_centroids

            # Compute inertia
            inertia = sum(((X[labels == i] - centroids[i]) ** 2).sum() for i in range(self.n_segments))

            # Keep best
            if inertia < best_inertia:
                best_inertia, best_labels, best_centroids = inertia, labels, centroids

        self.centroids = best_centroids  # store centroids
        return best_labels, best_centroids, best_inertia

    # ------------------------
    #  Predict using KMeans
    # ------------------------
    def predict_segment(self, df):
        if not hasattr(self, 'feature_columns') or len(self.feature_columns) == 0:
            raise ValueError("Segmenter must be fitted before prediction")

        features_df = self._engineer_segmentation_features(df)
        missing_cols = set(self.feature_columns) - set(features_df.columns)
        if missing_cols:
            raise ValueError(f"Missing features for prediction: {missing_cols}")
        
        X = features_df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)

        if self.centroids is not None:
            distances = np.linalg.norm(X_scaled[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            return labels
        else:
            raise ValueError("No KMeans centroids found. Fit the model first.")

    def fit(self, df, method='auto'):
        """Perform enhanced segmentation with KMeans, GMM, or HDBSCAN"""
        try:
            print("\n🔄 Performing Enhanced Customer Segmentation (5 Segments)...")

            features_df = self._engineer_segmentation_features(df)
            self.feature_columns = features_df.columns
            X = features_df.values
            X_scaled = self.scaler.fit_transform(X)

            best_score = -1
            best_labels = None
            best_method = None
            evaluations = []

            # ------------------------
            # KMEANS
            # ------------------------
            print("\n🔍 Testing KMeans clustering...")
            start_time = time.time()
            kmeans_labels, kmeans_centroids, kmeans_inertia = self.kmeans_from_scratch(X_scaled, self.n_segments)
            print(f"[INFO] KMeans from scratch | Inertia: {kmeans_inertia:.2f}")
            kmeans_time = time.time() - start_time

            kmeans_eval = self._evaluate_clustering(X_scaled, kmeans_labels, "KMeans")
            if kmeans_eval:
                kmeans_eval['time'] = kmeans_time
                evaluations.append(kmeans_eval)
                if kmeans_eval['silhouette_score'] > best_score:
                    best_score = kmeans_eval['silhouette_score']
                    best_labels = kmeans_labels
                    best_method = 'kmeans'
                    self.kmeans = self  # keep class reference for predict

            # ------------------------
            # GMM
            # ------------------------
            print("\n🔍 Testing Gaussian Mixture Model...")
            start_time = time.time()
            gmm = GaussianMixture(n_components=self.n_segments, random_state=42)
            gmm_labels = gmm.fit_predict(X_scaled)
            gmm_time = time.time() - start_time

            gmm_eval = self._evaluate_clustering(X_scaled, gmm_labels, "GMM")
            if gmm_eval:
                gmm_eval['time'] = gmm_time
                evaluations.append(gmm_eval)
                if gmm_eval['silhouette_score'] > best_score:
                    best_score = gmm_eval['silhouette_score']
                    best_labels = gmm_labels
                    best_method = 'gmm'
                    self.kmeans = gmm  # reference GMM

            # ------------------------
            # HDBSCAN
            # ------------------------
            print("\n🔍 Testing HDBSCAN clustering...")
            start_time = time.time()
            clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=10)
            hdb_labels = clusterer.fit_predict(X_scaled)
            hdb_time = time.time() - start_time

            unique_clusters = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
            print(f"🌀 HDBSCAN found {unique_clusters} clusters (excluding noise)")

            if unique_clusters > 1:
                hdb_eval = self._evaluate_clustering(X_scaled[hdb_labels != -1], hdb_labels[hdb_labels != -1], "HDBSCAN")
                if hdb_eval:
                    hdb_eval['time'] = hdb_time
                    evaluations.append(hdb_eval)
                    if hdb_eval['silhouette_score'] > best_score:
                        best_score = hdb_eval['silhouette_score']
                        best_labels = hdb_labels
                        best_method = 'hdbscan'
                        self.kmeans = clusterer
            else:
                print("⚠️ HDBSCAN produced only noise or a single cluster — skipping evaluation.")

            # ------------------------
            # Manual override
            # ------------------------
            if method != 'auto':
                if method == 'kmeans':
                    best_labels = kmeans_labels
                    best_method = 'kmeans'
                    self.kmeans = self
                elif method == 'gmm':
                    best_labels = gmm_labels
                    best_method = 'gmm'
                    self.kmeans = gmm
                elif method == 'hdbscan':
                    best_labels = hdb_labels
                    best_method = 'hdbscan'
                    self.kmeans = clusterer

            print("\n✅ Selected:", best_method.upper())

            self.segment_profiles = self._create_enhanced_segments(df, best_labels)
            print("🎯 Enhanced segmentation completed!")
            print(f"📋 Assigned segments: {self.assigned_segments}")
            return self

        except Exception as e:
            print(f"Error in segmentation: {e}")
            raise

def main():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
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
            print("Featured dataset not found!")
            return
        
        print("\n🔄 Performing Enhanced Customer Segmentation (5 Segments)...")
        
        # Initialize and fit enhanced segmenter with algorithm comparison
        segmenter = EnhancedCustomerSegmenter(n_segments=5)
        
        # You can specify method: 'auto', 'kmeans', or 'gmm'
        segmenter.fit(df, method='auto')  # 'auto' will choose the best one
        
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
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
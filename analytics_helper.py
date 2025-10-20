# analytics_helper.py
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class AnalyticsHelper:
    @staticmethod
    def create_abandonment_trend_chart(df, days=30):
        """Create abandonment trend chart"""
        # Simulate time series data
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        abandonment_rates = np.random.uniform(20, 40, days)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, 
            y=abandonment_rates,
            mode='lines+markers',
            name='Abandonment Rate',
            line=dict(color='#ff6b6b', width=3),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title="Abandonment Rate Trend",
            xaxis_title="Date",
            yaxis_title="Abandonment Rate (%)",
            height=400,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_segment_distribution_chart(segments_data):
        """Create segment distribution chart"""
        fig = px.pie(
            segments_data,
            values='Size',
            names='Segment',
            title="Customer Segment Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig
    
    @staticmethod
    def create_feature_importance_chart(features, importance_scores):
        """Create feature importance chart"""
        fig = px.bar(
            x=importance_scores,
            y=features,
            orientation='h',
            title="Feature Importance in Abandonment Prediction",
            labels={'x': 'Importance Score', 'y': 'Features'},
            color=importance_scores,
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(height=400)
        return fig
    
    @staticmethod
    def calculate_recovery_metrics(campaign_data):
        """Calculate recovery campaign metrics"""
        metrics = {
            'total_campaigns': len(campaign_data),
            'total_revenue': sum(campaign_data['revenue']),
            'total_cost': sum(campaign_data['cost']),
            'avg_success_rate': np.mean(campaign_data['success_rate']),
            'total_roi': (sum(campaign_data['revenue']) - sum(campaign_data['cost'])) / sum(campaign_data['cost']) * 100
        }
        return metrics
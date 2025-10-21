# analytics_helper.py
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import scipy.stats as stats
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import os
import streamlit as st

# Remove the BaseTab dependency and create a standalone analytics class
class AnalyticsHelper:
    @staticmethod
    def create_abandonment_trend_chart(df, date_col='timestamp', target_col='abandoned', days=30):
        """Create abandonment trend chart with real data"""
        if date_col in df.columns and target_col in df.columns:
            # Convert to datetime and filter last N days
            df[date_col] = pd.to_datetime(df[date_col])
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_data = df[df[date_col] >= cutoff_date]
            
            # Group by date and calculate abandonment rate
            daily_rates = recent_data.groupby(recent_data[date_col].dt.date)[target_col].mean() * 100
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_rates.index, 
                y=daily_rates.values,
                mode='lines+markers',
                name='Abandonment Rate',
                line=dict(color='#ff6b6b', width=3),
                marker=dict(size=6),
                hovertemplate='Date: %{x}<br>Abandonment Rate: %{y:.1f}%<extra></extra>'
            ))
            
            # Add 7-day moving average
            if len(daily_rates) >= 7:
                moving_avg = daily_rates.rolling(window=7).mean()
                fig.add_trace(go.Scatter(
                    x=moving_avg.index,
                    y=moving_avg.values,
                    mode='lines',
                    name='7-Day Moving Avg',
                    line=dict(color='#2e86ab', width=2, dash='dash')
                ))
            
            fig.update_layout(
                title=f"Abandonment Rate Trend (Last {days} Days)",
                xaxis_title="Date",
                yaxis_title="Abandonment Rate (%)",
                height=400,
                showlegend=True
            )
        else:
            # Fallback to simulated data
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
                title="Abandonment Rate Trend (Simulated Data)",
                xaxis_title="Date",
                yaxis_title="Abandonment Rate (%)",
                height=400,
                showlegend=False
            )
        
        return fig

    @staticmethod
    def create_correlation_heatmap(df, numerical_features):
        """Create correlation heatmap for numerical features"""
        # Select only numerical features that exist in dataframe
        available_features = [f for f in numerical_features if f in df.columns]
        
        if len(available_features) >= 2:
            corr_matrix = df[available_features].corr()
            
            fig = ff.create_annotated_heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns.tolist(),
                y=corr_matrix.index.tolist(),
                annotation_text=corr_matrix.round(2).values,
                colorscale='RdBu_r',
                showscale=True,
                hoverinfo='z'
            )
            
            fig.update_layout(
                title="Feature Correlation Matrix",
                height=500,
                xaxis_title="Features",
                yaxis_title="Features"
            )
        else:
            # Create a placeholder if not enough features
            fig = go.Figure()
            fig.add_annotation(text="Not enough numerical features for correlation analysis")
            fig.update_layout(height=400)
        
        return fig

    @staticmethod
    def create_feature_importance_chart(model, feature_names, top_n=15):
        """Create feature importance chart from trained model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:top_n]
                
                fig = px.bar(
                    x=importances[indices],
                    y=[feature_names[i] for i in indices],
                    orientation='h',
                    title=f"Top {top_n} Feature Importances",
                    labels={'x': 'Importance Score', 'y': 'Features'},
                    color=importances[indices],
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(height=500, showlegend=False)
                
            elif hasattr(model, 'coef_'):
                # For linear models
                coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                indices = np.argsort(np.abs(coef))[::-1][:top_n]
                
                fig = px.bar(
                    x=coef[indices],
                    y=[feature_names[i] for i in indices],
                    orientation='h',
                    title=f"Top {top_n} Feature Coefficients",
                    labels={'x': 'Coefficient Value', 'y': 'Features'},
                    color=np.abs(coef[indices]),
                    color_continuous_scale='RdBu'
                )
                
                fig.update_layout(height=500, showlegend=False)
            else:
                raise AttributeError("Model doesn't have feature importances or coefficients")
                
        except Exception as e:
            # Fallback to random importances for demo
            features = feature_names[:top_n]
            importance_scores = np.random.uniform(0, 1, len(features))
            
            fig = px.bar(
                x=importance_scores,
                y=features,
                orientation='h',
                title="Feature Importance (Demo Data)",
                labels={'x': 'Importance Score', 'y': 'Features'},
                color=importance_scores,
                color_continuous_scale='Blues'
            )
            
            fig.update_layout(height=400)
        
        return fig

    @staticmethod
    def create_segmentation_analysis(df, segment_col, metrics_cols):
        """Create comprehensive segmentation analysis"""
        if segment_col not in df.columns:
            return None, None
            
        segment_stats = df.groupby(segment_col)[metrics_cols].agg(['mean', 'std', 'count']).round(2)
        
        # Create subplots for each metric
        fig = make_subplots(
            rows=len(metrics_cols), 
            cols=1,
            subplot_titles=[f"{col} by Segment" for col in metrics_cols]
        )
        
        colors = px.colors.qualitative.Set3
        
        for i, col in enumerate(metrics_cols):
            segment_means = df.groupby(segment_col)[col].mean()
            
            fig.add_trace(
                go.Bar(
                    x=segment_means.index,
                    y=segment_means.values,
                    name=col,
                    marker_color=colors[i % len(colors)],
                    text=segment_means.values.round(2),
                    textposition='auto'
                ),
                row=i+1, col=1
            )
        
        fig.update_layout(
            height=300 * len(metrics_cols),
            title_text="Segment Performance Analysis",
            showlegend=False
        )
        
        return fig, segment_stats

    @staticmethod
    def create_behavioral_patterns_chart(df, behavioral_features):
        """Analyze behavioral patterns leading to abandonment"""
        available_features = [f for f in behavioral_features if f in df.columns]
        
        if len(available_features) < 2:
            return None
            
        # Calculate correlation with abandonment
        if 'abandoned' in df.columns:
            correlations = df[available_features + ['abandoned']].corr()['abandoned'].drop('abandoned')
            correlations = correlations.sort_values(ascending=False)
            
            fig = px.bar(
                x=correlations.values,
                y=correlations.index,
                orientation='h',
                title="Behavioral Features Correlation with Abandonment",
                labels={'x': 'Correlation with Abandonment', 'y': 'Behavioral Features'},
                color=correlations.values,
                color_continuous_scale='RdBu_r'
            )
            
            fig.update_layout(height=400)
        else:
            # Create distribution comparison
            fig = make_subplots(rows=2, cols=2, subplot_titles=available_features[:4])
            
            for i, feature in enumerate(available_features[:4]):
                row = (i // 2) + 1
                col = (i % 2) + 1
                
                fig.add_trace(
                    go.Histogram(
                        x=df[feature].dropna(),
                        name=feature,
                        nbinsx=20
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(height=600, title_text="Behavioral Feature Distributions")
        
        return fig

    @staticmethod
    def create_time_based_analysis(df, time_col, target_col):
        """Analyze patterns based on time (hour, day, month)"""
        if time_col not in df.columns or target_col not in df.columns:
            return None
            
        df_copy = df.copy()
        df_copy[time_col] = pd.to_datetime(df_copy[time_col])
        
        # Extract time components
        df_copy['hour'] = df_copy[time_col].dt.hour
        df_copy['day_of_week'] = df_copy[time_col].dt.day_name()
        df_copy['month'] = df_copy[time_col].dt.month_name()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Abandonment Rate by Hour',
                'Abandonment Rate by Day of Week',
                'Abandonment Rate by Month',
                'Session Volume by Hour'
            ]
        )
        
        # Hourly abandonment rate
        hourly_rates = df_copy.groupby('hour')[target_col].mean() * 100
        fig.add_trace(
            go.Scatter(x=hourly_rates.index, y=hourly_rates.values, mode='lines+markers', name='Hourly Rate'),
            row=1, col=1
        )
        
        # Day of week abandonment rate
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_rates = df_copy.groupby('day_of_week')[target_col].mean().reindex(day_order) * 100
        fig.add_trace(
            go.Bar(x=daily_rates.index, y=daily_rates.values, name='Daily Rate'),
            row=1, col=2
        )
        
        # Monthly abandonment rate
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                      'July', 'August', 'September', 'October', 'November', 'December']
        monthly_rates = df_copy.groupby('month')[target_col].mean().reindex(month_order) * 100
        fig.add_trace(
            go.Bar(x=monthly_rates.index, y=monthly_rates.values, name='Monthly Rate'),
            row=2, col=1
        )
        
        # Session volume by hour
        hourly_volume = df_copy.groupby('hour').size()
        fig.add_trace(
            go.Scatter(x=hourly_volume.index, y=hourly_volume.values, mode='lines', name='Session Volume'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title_text="Time-Based Analysis", showlegend=False)
        return fig

    @staticmethod
    def calculate_business_impact(df, recovery_rate=0.15, avg_cart_value=150):
        """Calculate business impact of abandonment reduction"""
        total_abandoned = df['abandoned'].sum() if 'abandoned' in df.columns else len(df) * 0.3
        recoverable = total_abandoned * recovery_rate
        potential_revenue = recoverable * avg_cart_value
        
        metrics = {
            'total_abandoned_sessions': int(total_abandoned),
            'recoverable_sessions': int(recoverable),
            'potential_revenue': f"${potential_revenue:,.2f}",
            'recovery_rate': f"{recovery_rate * 100}%",
            'avg_cart_value': f"${avg_cart_value}"
        }
        
        return metrics

    @staticmethod
    def create_performance_dashboard(model, X_test, y_test, feature_names):
        """Create comprehensive model performance dashboard"""
        try:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = ff.create_annotated_heatmap(
                z=cm,
                x=['Predicted 0', 'Predicted 1'],
                y=['Actual 0', 'Actual 1'],
                annotation_text=cm,
                colorscale='Blues'
            )
            fig_cm.update_layout(title="Confusion Matrix")
            
            # Feature Importance
            fig_importance = AnalyticsHelper.create_feature_importance_chart(model, feature_names)
            
            # ROC Curve (if probabilities available)
            if y_pred_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'ROC curve (AUC = {roc_auc:.2f})',
                    line=dict(color='darkorange', width=2)
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Random classifier',
                    line=dict(color='navy', width=2, dash='dash')
                ))
                fig_roc.update_layout(
                    title="ROC Curve",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate"
                )
            else:
                fig_roc = None
            
            return {
                'confusion_matrix': fig_cm,
                'feature_importance': fig_importance,
                'roc_curve': fig_roc,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
        except Exception as e:
            st.error(f"Error creating performance dashboard: {e}")
            return None

    @staticmethod
    def generate_insights_report(df, model=None):
        """Generate automated insights report"""
        insights = []
        
        # Basic statistics
        if 'abandoned' in df.columns:
            abandonment_rate = df['abandoned'].mean() * 100
            insights.append(f"📊 Overall abandonment rate: {abandonment_rate:.1f}%")
        
        # High-value insights
        if 'cart_value' in df.columns:
            avg_cart_value = df['cart_value'].mean()
            high_value_threshold = avg_cart_value * 1.5
            high_value_abandonment = df[df['cart_value'] > high_value_threshold]['abandoned'].mean() * 100
            insights.append(f"💰 High-value cart abandonment: {high_value_abandonment:.1f}% (carts > ${high_value_threshold:.0f})")
        
        # Behavioral insights
        if all(col in df.columns for col in ['session_duration', 'abandoned']):
            short_sessions = df[df['session_duration'] < 300]['abandoned'].mean() * 100
            long_sessions = df[df['session_duration'] > 1800]['abandoned'].mean() * 100
            insights.append(f"⏱️ Short sessions (<5min) abandonment: {short_sessions:.1f}%")
            insights.append(f"🕒 Long sessions (>30min) abandonment: {long_sessions:.1f}%")
        
        # Device insights
        if 'device_type' in df.columns and 'abandoned' in df.columns:
            device_rates = df.groupby('device_type')['abandoned'].mean() * 100
            worst_device = device_rates.idxmax()
            worst_rate = device_rates.max()
            insights.append(f"📱 Highest abandonment on {worst_device}: {worst_rate:.1f}%")
        
        return insights


# Simple function to create header (replace the missing create_header function)
def create_header(title, description, icon):
    """Create a simple header for the analytics section"""
    st.markdown(f"# {icon} {title}")
    st.markdown(f"**{description}**")
    st.markdown("---")


# Standalone function to run analytics (instead of the AdvancedAnalyticsTab class)
def run_analytics_tab():
    """Run the analytics tab as a standalone function"""
    analytics_helper = AnalyticsHelper()
    
    create_header("Advanced Analytics", "Deep Dive Analysis & Insights", "📈")
    
    # Load sample data or use provided data
    df = load_sample_data()
    
    if df is not None:
        create_analytics_dashboard(df, analytics_helper)
    else:
        st.warning("No data available for analytics. Using demo mode.")
        create_demo_analytics(analytics_helper)


def load_sample_data():
    """Load sample data for analytics"""
    try:
        # Try to load from your data directory
        data_files = [f for f in os.listdir('data') if f.endswith('.csv')]
        if data_files:
            # Try to load the main dataset first
            if 'cart_abandonment_dataset.csv' in data_files:
                df = pd.read_csv(os.path.join('data', 'cart_abandonment_dataset.csv'))
            else:
                df = pd.read_csv(os.path.join('data', data_files[0]))
            return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
    return None


def create_analytics_dashboard(df, analytics_helper):
    """Create comprehensive analytics dashboard"""
    
    # Tab layout for different analytics views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", "🔍 Behavioral Patterns", 
        "⏰ Time Analysis", "📈 Performance"
    ])
    
    with tab1:
        create_overview_tab(df, analytics_helper)
    
    with tab2:
        create_behavioral_analysis_tab(df, analytics_helper)
    
    with tab3:
        create_time_analysis_tab(df, analytics_helper)
    
    with tab4:
        create_performance_tab(df, analytics_helper)


def create_overview_tab(df, analytics_helper):
    """Create overview analytics tab"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Abandonment trend
        st.subheader("Abandonment Trend")
        trend_fig = analytics_helper.create_abandonment_trend_chart(df)
        st.plotly_chart(trend_fig, use_container_width=True)
        
        # Business impact
        st.subheader("Business Impact Analysis")
        impact_metrics = analytics_helper.calculate_business_impact(df)
        for metric, value in impact_metrics.items():
            st.metric(metric.replace('_', ' ').title(), value)

    with col2:
        # Feature correlations
        st.subheader("Feature Correlations")
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        corr_fig = analytics_helper.create_correlation_heatmap(df, numerical_features)
        st.plotly_chart(corr_fig, use_container_width=True)
        
        # Automated insights
        st.subheader("Key Insights")
        insights = analytics_helper.generate_insights_report(df)
        for insight in insights:
            st.info(insight)


def create_behavioral_analysis_tab(df, analytics_helper):
    """Create behavioral patterns analysis tab"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Behavioral Patterns")
        behavioral_features = ['session_duration', 'num_pages_viewed', 'scroll_depth', 'num_items_carted']
        behavior_fig = analytics_helper.create_behavioral_patterns_chart(df, behavioral_features)
        if behavior_fig:
            st.plotly_chart(behavior_fig, use_container_width=True)
    
    with col2:
        st.subheader("Segment Analysis")
        # Example segmentation
        segments_data = pd.DataFrame({
            'Segment': ['High Risk', 'Medium Risk', 'Low Risk', 'Loyal'],
            'Size': [25, 35, 20, 20],
            'Abandonment_Rate': [75, 45, 15, 5]
        })
        
        segment_fig = px.pie(
            segments_data,
            values='Size',
            names='Segment',
            title="Customer Segments",
            color='Abandonment_Rate',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(segment_fig, use_container_width=True)


def create_time_analysis_tab(df, analytics_helper):
    """Create time-based analysis tab"""
    st.subheader("Time-Based Patterns")
    
    if 'timestamp' in df.columns and 'abandoned' in df.columns:
        time_fig = analytics_helper.create_time_based_analysis(df, 'timestamp', 'abandoned')
        if time_fig:
            st.plotly_chart(time_fig, use_container_width=True)
    else:
        st.info("Time-based analysis requires 'timestamp' and 'abandoned' columns")


def create_performance_tab(df, analytics_helper):
    """Create model performance tab"""
    st.subheader("Model Performance")
    
    # This would integrate with your actual model
    st.info("""
    **Performance metrics would include:**
    - Model accuracy and precision
    - Feature importance analysis
    - Confusion matrix
    - ROC curves
    - Cross-validation results
    """)
    
    # Demo feature importance
    st.subheader("Feature Importance (Demo)")
    demo_features = ['session_duration', 'num_pages_viewed', 'cart_value', 'return_user']
    demo_importance = np.random.uniform(0, 1, len(demo_features))
    
    fig = px.bar(
        x=demo_importance,
        y=demo_features,
        orientation='h',
        title="Demo Feature Importance",
        color=demo_importance,
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig, use_container_width=True)


def create_demo_analytics(analytics_helper):
    """Create demo analytics when no real data is available"""
    st.info("📊 Showing demo analytics with sample data")
    
    # Generate sample data
    np.random.seed(42)
    demo_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='H'),
        'session_duration': np.random.exponential(300, 1000),
        'num_pages_viewed': np.random.poisson(8, 1000),
        'cart_value': np.random.lognormal(5, 1, 1000),
        'abandoned': np.random.choice([0, 1], 1000, p=[0.7, 0.3]),
        'device_type': np.random.choice(['Mobile', 'Desktop', 'Tablet'], 1000),
        'return_user': np.random.choice([0, 1], 1000, p=[0.6, 0.4])
    })
    
    create_analytics_dashboard(demo_data, analytics_helper)
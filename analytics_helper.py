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

class AnalyticsHelper:
    def __init__(self):
        self.df = self.load_cart_abandonment_data()
    
    def load_cart_abandonment_data(self):
        try:
            file_path = 'data/cart_abandonment_dataset.csv'
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                st.success(f"✅ Loaded cart abandonment data: {len(df)} records")
                return df
            else:
                st.error("❌ Cart abandonment dataset not found")
                return None
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            return None
    
    def create_abandonment_overview(self):
        if self.df is None:
            return None
        
        total_sessions = len(self.df)
        abandonment_rate = self.df['abandoned'].mean() * 100
        total_abandoned = self.df['abandoned'].sum()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Sessions", f"{total_sessions:,}")
        with col2:
            st.metric("Abandonment Rate", f"{abandonment_rate:.1f}%")
        with col3:
            st.metric("Abandoned Carts", f"{total_abandoned:,}")
        with col4:
            avg_cart_value = self.df['cart_value'].mean() if 'cart_value' in self.df.columns else 0
            st.metric("Avg Cart Value", f"${avg_cart_value:.2f}")
        
        return self.df
    
    def create_trend_analysis(self):
        if self.df is None or 'timestamp' not in self.df.columns:
            st.warning("No timestamp data available for trend analysis")
            return
        
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df['date'] = self.df['timestamp'].dt.date
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['day_of_week'] = self.df['timestamp'].dt.day_name()
        
        daily_rates = self.df.groupby('date')['abandoned'].mean() * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_rates.index, 
            y=daily_rates.values,
            mode='lines+markers',
            name='Daily Abandonment Rate',
            line=dict(color='#ff6b6b', width=3)
        ))
        
        # Add 7-day moving average
        if len(daily_rates) >= 7:
            moving_avg = daily_rates.rolling(window=7, center=True).mean()
            fig.add_trace(go.Scatter(
                x=moving_avg.index,
                y=moving_avg.values,
                mode='lines',
                name='7-Day Moving Average',
                line=dict(color='#2e86ab', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title="Cart Abandonment Rate Trend",
            xaxis_title="Date",
            yaxis_title="Abandonment Rate (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Hourly and weekly patterns
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly pattern
            hourly_rates = self.df.groupby('hour')['abandoned'].mean() * 100
            fig_hourly = px.bar(
                x=hourly_rates.index,
                y=hourly_rates.values,
                title="Abandonment Rate by Hour of Day",
                labels={'x': 'Hour of Day', 'y': 'Abandonment Rate (%)'}
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        with col2:
            # Day of week pattern
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_rates = self.df.groupby('day_of_week')['abandoned'].mean().reindex(day_order) * 100
            fig_daily = px.bar(
                x=daily_rates.index,
                y=daily_rates.values,
                title="Abandonment Rate by Day of Week",
                labels={'x': 'Day of Week', 'y': 'Abandonment Rate (%)'}
            )
            st.plotly_chart(fig_daily, use_container_width=True)
    
    def create_feature_analysis(self):
        if self.df is None:
            return
        
        st.subheader("Feature Impact Analysis")
        numerical_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'abandoned' in numerical_features:
            numerical_features.remove('abandoned')
        if 'user_id' in numerical_features:
            numerical_features.remove('user_id')
        
        correlations = []
        for feature in numerical_features[:10]: 
            if feature != 'abandoned':
                corr = self.df[feature].corr(self.df['abandoned'])
                correlations.append({'feature': feature, 'correlation': corr})
        
        corr_df = pd.DataFrame(correlations).sort_values('correlation', key=abs, ascending=False)
        
        fig = px.bar(
            corr_df,
            x='correlation',
            y='feature',
            orientation='h',
            title="Feature Correlation with Cart Abandonment",
            color='correlation',
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Feature Distributions by Abandonment Status")
        selected_features = st.multiselect(
            "Select features to compare:",
            numerical_features[:8],
            default=numerical_features[:3] if len(numerical_features) >= 3 else numerical_features
        )
        
        if selected_features:
            for feature in selected_features:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_dist = px.histogram(
                        self.df,
                        x=feature,
                        color='abandoned',
                        barmode='overlay',
                        title=f"{feature} Distribution",
                        opacity=0.7
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                with col2:
                    # Box plot
                    fig_box = px.box(
                        self.df,
                        x='abandoned',
                        y=feature,
                        title=f"{feature} by Abandonment Status",
                        color='abandoned'
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
    
    def create_segmentation_analysis(self):
        if self.df is None:
            return
        
        st.subheader("Customer Segmentation Analysis")
        
        if all(col in self.df.columns for col in ['session_duration', 'cart_value']):
            # Define segments
            conditions = [
                (self.df['session_duration'] < 300) & (self.df['cart_value'] < 100),
                (self.df['session_duration'] >= 300) & (self.df['cart_value'] < 100),
                (self.df['session_duration'] < 300) & (self.df['cart_value'] >= 100),
                (self.df['session_duration'] >= 300) & (self.df['cart_value'] >= 100)
            ]
            segments = ['Quick Browser', 'Engaged Browser', 'Quick Buyer', 'Engaged Buyer']
            
            self.df['segment'] = np.select(conditions, segments, default='Other')
            
            # Analyze segments
            segment_analysis = self.df.groupby('segment').agg({
                'abandoned': 'mean',
                'session_duration': 'mean',
                'cart_value': 'mean',
                'user_id': 'count'
            }).round(2)
            
            segment_analysis = segment_analysis.rename(columns={
                'abandoned': 'abandonment_rate',
                'user_id': 'user_count'
            })
            
            st.dataframe(segment_analysis, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_segment_size = px.pie(
                    values=segment_analysis['user_count'],
                    names=segment_analysis.index,
                    title="Segment Distribution"
                )
                st.plotly_chart(fig_segment_size, use_container_width=True)
            
            with col2:
                fig_abandonment = px.bar(
                    x=segment_analysis.index,
                    y=segment_analysis['abandonment_rate'] * 100,
                    title="Abandonment Rate by Segment",
                    labels={'x': 'Segment', 'y': 'Abandonment Rate (%)'}
                )
                st.plotly_chart(fig_abandonment, use_container_width=True)
    
    def create_business_impact(self):
        if self.df is None:
            return
        
        st.subheader("Business Impact Analysis")
        
        total_abandoned = self.df['abandoned'].sum()
        avg_cart_value = self.df['cart_value'].mean() if 'cart_value' in self.df.columns else 100
        
        recovery_scenarios = [0.05, 0.10, 0.15, 0.20, 0.25]
        
        impact_data = []
        for recovery_rate in recovery_scenarios:
            recoverable_carts = total_abandoned * recovery_rate
            potential_revenue = recoverable_carts * avg_cart_value
            
            impact_data.append({
                'Recovery Rate': f"{recovery_rate * 100:.0f}%",
                'Recoverable Carts': int(recoverable_carts),
                'Potential Revenue': f"${potential_revenue:,.0f}"
            })
        
        impact_df = pd.DataFrame(impact_data)
        
        st.dataframe(impact_df, use_container_width=True)
        
        fig = px.bar(
            impact_df,
            x='Recovery Rate',
            y='Recoverable Carts',
            title="Potential Recovery by Scenario",
            text='Potential Revenue'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("💰 Key Revenue Opportunities")
        
        insights = [
            f"**Immediate Opportunity**: Recovering just 5% of abandoned carts could generate **${impact_data[0]['Potential Revenue']}** in additional revenue",
            f"**Realistic Goal**: A 15% recovery rate could bring in **${impact_data[2]['Potential Revenue']}**",
            f"**Ambitious Target**: With 25% recovery, potential revenue reaches **${impact_data[4]['Potential Revenue']}**"
        ]
        
        for insight in insights:
            st.info(insight)
    
    def create_predictive_insights(self):
        if self.df is None:
            return
        
        st.subheader("Predictive Insights")
        if all(col in self.df.columns for col in ['session_duration', 'cart_value', 'num_pages_viewed']):
            risk_factors = [
                ('Short Sessions (<5min)', self.df['session_duration'] < 300),
                ('High Cart Value (>$200)', self.df['cart_value'] > 200),
                ('Few Pages Viewed (<3)', self.df['num_pages_viewed'] < 3),
                ('New Users', self.df.get('return_user', 0) == 0)
            ]
            
            risk_analysis = []
            for factor_name, condition in risk_factors:
                if condition.any():
                    at_risk_count = condition.sum()
                    abandonment_rate = self.df[condition]['abandoned'].mean() * 100
                    risk_analysis.append({
                        'Risk Factor': factor_name,
                        'At Risk Users': at_risk_count,
                        'Abandonment Rate': f"{abandonment_rate:.1f}%"
                    })
            
            if risk_analysis:
                risk_df = pd.DataFrame(risk_analysis)
                st.dataframe(risk_df, use_container_width=True)
                
                # Highlight highest risk factors
                max_risk = risk_df.loc[risk_df['Abandonment Rate'].str.replace('%', '').astype(float).idxmax()]
                st.warning(f"🚨 **Highest Risk**: {max_risk['Risk Factor']} with {max_risk['Abandonment Rate']} abandonment rate")
    
    def run_comprehensive_analysis(self):
        if self.df is None:
            st.error("Unable to load cart abandonment data. Please check your data files.")
            return
        
        st.title("🛒 Cart Abandonment Analytics Dashboard")
        st.markdown("---")
        self.create_abandonment_overview()
        
        tab1, tab2, tab3, tab4= st.tabs([
             "🔍 Features", "👥 Segments", "💰 Business Impact", "🎯 Insights"
        ])
        
        with tab1:
            self.create_feature_analysis()
        
        with tab2:
            self.create_segmentation_analysis()
        
        with tab3:
            self.create_business_impact()
        
        with tab4:
            self.create_predictive_insights()
            

            st.subheader("Dataset Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Data Overview:**")
                st.write(f"- Total records: {len(self.df):,}")
                st.write(f"- Features: {len(self.df.columns)}")
                st.write(f"- Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            with col2:
                st.write("**Column Summary:**")
                for col in self.df.columns[:6]:  # Show first 6 columns
                    st.write(f"- {col}: {self.df[col].dtype}")

def run_analytics_dashboard():
    helper = AnalyticsHelper()
    helper.run_comprehensive_analysis()

if __name__ == "__main__":
    run_analytics_dashboard()
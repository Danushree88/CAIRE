import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pickle
import os
import sys
from pathlib import Path
import json
from analytics_helper import AnalyticsHelper

# Import custom styles
from custom_styles import (
    load_custom_css, create_header, create_section, 
    info_box, success_box, warning_box, error_box,
    create_stat_card, create_comparison_table, add_footer
)

class BaseTab:
    def __init__(self, name):
        self.name = name

    @staticmethod
    @st.cache_data
    def load_data(file_path):
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        return None

    @staticmethod
    def ensure_data_directory():
        os.makedirs("data", exist_ok=True)
        os.makedirs("saved_models", exist_ok=True)


class AdminDashboardTab(BaseTab):
    def __init__(self):
        super().__init__("📊 Admin Dashboard")

    def get_correct_paths(self):
        """Get correct file paths based on project structure"""
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        
        paths = {
            'featured_data': project_root / "data" / "cart_abandonment_featured.csv",
            'original_data': project_root / "data" / "cart_abandonment_dataset.csv", 
            'test_data': project_root / "test_data" / "test_data_for_prediction.csv",
            'model': project_root / "saved_models" / "final_manual_model.pkl",
            'models_dir': project_root / "src" / "models"
        }
        
        return paths

    def load_model_and_data(self):
        self.paths = self.get_correct_paths()
        self.df = self.load_data(self.paths['featured_data'])
        self.df1 = self.load_data(self.paths['original_data'])
        self.df2 = self.load_data(self.paths['test_data'])
        self.model = None
        self.feature_names = []
        possible_model_paths = [
            self.paths['model'],  
            Path("src/models/final_manual_model.pkl"), 
            Path("saved_models/final_manual_model.pkl"),
            Path("final_manual_model.pkl")
        ] 
        model_loaded = False
        for model_path in possible_model_paths:
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                        self.model = model_data['model']
                        self.feature_names = model_data.get('feature_names', [])
                    st.session_state.model_loaded = True
                    model_loaded = True
                    print(f"✅ Model loaded from: {model_path}")
                    break
                except Exception as e:
                    print(f"❌ Failed to load model from {model_path}: {e}")
        
        if not model_loaded:
            st.session_state.model_loaded = False
            print("❌ Model not found in any location")

    def calculate_real_metrics(self):
        if self.df1 is None:
            # Fallback to normalized data if original not available
            return self._calculate_metrics_from_normalized()
        
        # Use original dataset for accurate business metrics
        metrics = {
            'total_sessions': len(self.df1),
            'abandonment_rate': self.df1['abandoned'].mean() * 100,
            'avg_cart_value': self.df1['cart_value'].mean(),
            'total_abandoned': self.df1['abandoned'].sum(),
            'return_user_rate': self.df1['return_user'].mean() * 100,
            'avg_engagement': self.df1['num_pages_viewed'].mean(),  # Using pages viewed as engagement proxy
            'payment_reach_rate': self.df1['if_payment_page_reached'].mean() * 100,
            'avg_pages_viewed': self.df1['num_pages_viewed'].mean(),
            'avg_session_duration': self.df1['session_duration'].mean(),
            'recovery_potential': self.df1[self.df1['abandoned'] == 1]['cart_value'].sum(),
            'total_revenue_lost': self.df1[self.df1['abandoned'] == 1]['cart_value'].sum(),
            'avg_abandoned_cart_value': self.df1[self.df1['abandoned'] == 1]['cart_value'].mean(),
            'conversion_rate': (1 - self.df1['abandoned'].mean()) * 100
        }
        return metrics

    def render_dashboard_overview(self):
        """Render main dashboard with REAL data"""
        create_header("CAIRE Analytics", "Cart Abandonment Intelligence & Recovery Engine", "📊")
        
        if self.df is None:
            error_box("No data available. Please ensure featured data exists in the data folder.")
            return
        
        metrics = self.calculate_real_metrics()
        create_section("Key Performance Indicators", "📈")
        pastel_colors = {
            'blue': '#A7C7E7',     
            'red': '#F8BBD0',      
            'green': '#C8E6C9',     
            'yellow': '#FFEAA7',  
            'purple': '#D4BFF9',    
            'teal': '#B2EBF2',    
            'orange': '#FFCCBC',  
            'lime': '#E6EE9C'       
        }
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_stat_card(
                "Total Sessions", 
                f"{metrics['total_sessions']:,}",
                "👥", 
                pastel_colors['blue']
            )
        
        with col2:
            create_stat_card(
                "Abandonment Rate", 
                f"{metrics['abandonment_rate']:.1f}%",
                "📊", 
                pastel_colors['red']
            )
        
        with col3:
            cart_val = metrics['avg_cart_value']
            display_val = f"${cart_val:,.2f}" if cart_val > 10 else f"{cart_val:.3f}"
            create_stat_card(
                "Avg Cart Value", 
                display_val,
                "🛒",  
                pastel_colors['green']
            )
        
        with col4:
            create_stat_card(
                "Abandoned Carts", 
                f"{metrics['total_abandoned']:,}",
                "⚠️", 
                pastel_colors['yellow']
            )
        
       
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_stat_card(
                "Return Users", 
                f"{metrics['return_user_rate']:.1f}%",
                "🔄",  
                pastel_colors['purple']
            )
        
        with col2:
            create_stat_card(
                "Avg Engagement", 
                f"{metrics['avg_engagement']:.2f}",
                "📈", 
                pastel_colors['teal']
            )
        
        with col3:
            create_stat_card(
                "Payment Reach", 
                f"{metrics['payment_reach_rate']:.1f}%",
                "💳",
                pastel_colors['orange']
            )
        
        with col4:
            recovery_pot = metrics['recovery_potential']
            display_pot = f"${recovery_pot:,.0f}" if recovery_pot > 0 else "$0"
            create_stat_card(
                "Recovery Potential", 
                display_pot,
                "💰", 
                pastel_colors['lime']
            )

    def render_behavior_analysis(self):
        create_section("User Behavior Analysis", "🔍")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Abandonment by engagement level
            st.subheader("🎯 Abandonment by Engagement Level")
            
            if 'engagement_score' in self.df.columns:
                # Create engagement segments from real data
                engagement_bins = [-1, -0.5, 0, 0.5, 1]
                engagement_labels = ['Very Low', 'Low', 'Medium', 'High']
                
                self.df['engagement_level'] = pd.cut(
                    self.df['engagement_score'], 
                    bins=engagement_bins, 
                    labels=engagement_labels
                )
                
                engagement_abandonment = self.df.groupby('engagement_level', observed=True)['abandoned'].mean() * 100
                
                # Use pastel color scale
                fig = px.bar(
                    x=engagement_abandonment.index,
                    y=engagement_abandonment.values,
                    labels={'x': 'Engagement Level', 'y': 'Abandonment Rate (%)'},
                    color=engagement_abandonment.values,
                    color_continuous_scale='blues',  # Pastel blue scale
                    template='plotly_white'
                )
                
                # Customize appearance
                fig.update_traces(
                    marker_line_color='rgba(0,0,0,0.2)',
                    marker_line_width=1,
                    opacity=0.8
                )
                
                fig.update_layout(
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#2c3e50"),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Insight
                if not engagement_abandonment.empty:
                    max_segment = engagement_abandonment.idxmax()
                    max_rate = engagement_abandonment.max()
                    info_box(f"**Insight:** Highest abandonment ({max_rate:.1f}%) in **{max_segment}** engagement group")
        
        with col2:
            # Cart value distribution
            st.subheader("💰 Cart Value Distribution")
            cart_value_col = 'original_cart_value' if 'original_cart_value' in self.df.columns else 'cart_value'
            
            if cart_value_col in self.df.columns:
                fig = px.histogram(
                    self.df, 
                    x=cart_value_col,
                    nbins=20,
                    labels={cart_value_col: 'Cart Value'},
                    color_discrete_sequence=['#A7C7E7'],  # Pastel blue
                    template='plotly_white'
                )
                
                fig.update_traces(
                    marker_line_color='rgba(0,0,0,0.2)',
                    marker_line_width=1,
                    opacity=0.7
                )
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#2c3e50"),
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)

    def render_user_patterns(self):
        """Render user behavior patterns with pastel colors"""
        create_section("User Behavior Patterns", "📊")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Return vs New User Comparison
            st.subheader("👥 Return vs New User Behavior")
            
            if 'return_user' in self.df.columns:
                user_comparison = self.df.groupby('return_user').agg({
                    'abandoned': 'mean',
                    'engagement_score': 'mean',
                    'session_duration': 'mean'
                })
                user_comparison.index = ['New Users', 'Return Users']
                user_comparison['abandonment_rate'] = user_comparison['abandoned'] * 100
                
                fig = go.Figure()
                
                # Pastel colors for bars
                fig.add_trace(go.Bar(
                    name='Abandonment Rate (%)', 
                    x=user_comparison.index, 
                    y=user_comparison['abandonment_rate'],
                    marker_color='#F8BBD0',  # Pastel pink
                    marker_line_color='rgba(0,0,0,0.2)',
                    marker_line_width=1
                ))
                fig.add_trace(go.Bar(
                    name='Avg Engagement', 
                    x=user_comparison.index, 
                    y=user_comparison['engagement_score'] * 100,
                    marker_color='#A7C7E7',  # Pastel blue
                    marker_line_color='rgba(0,0,0,0.2)',
                    marker_line_width=1
                ))
                
                fig.update_layout(
                    barmode='group', 
                    showlegend=True,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#2c3e50"),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Payment Page Analysis
            st.subheader("💳 Payment Page Reach Impact")
            
            if 'if_payment_page_reached' in self.df.columns:
                payment_analysis = self.df.groupby('if_payment_page_reached').agg({
                    'abandoned': 'mean',
                    'cart_value': 'mean'
                })
                payment_analysis.index = ['Did Not Reach', 'Reached Payment']
                payment_analysis['abandonment_rate'] = payment_analysis['abandoned'] * 100
                
                # Use pastel color scale
                fig = px.bar(
                    payment_analysis,
                    y='abandonment_rate',
                    labels={'value': 'Abandonment Rate (%)'},
                    color='abandonment_rate',
                    color_continuous_scale='blues',  # Pastel blue scale
                    template='plotly_white'
                )
                
                fig.update_traces(
                    marker_line_color='rgba(0,0,0,0.2)',
                    marker_line_width=1,
                    opacity=0.8
                )
                
                fig.update_layout(
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#2c3e50"),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

    def render_recent_data(self):
        """Render recent session data"""
        create_section("Recent Session Data", "📋")
        

        data_source = self.df2 if self.df2 is not None else self.df
        
        if data_source is None:
            error_box("No dataset available to display")
            return
        
        # Define basic columns that should be available
        basic_cols = [
            'user_id', 'cart_value', 'num_items_carted', 
            'session_duration', 'abandoned', 'if_payment_page_reached'
        ]
        
        # Filter to only available columns
        available_cols = [col for col in basic_cols if col in data_source.columns]
        
        if available_cols:
            st.dataframe(
                data_source[available_cols].tail(15), 
                use_container_width=True,
                height=300
            )
        else:
            warning_box("Required columns not found in dataset")
            # Show what columns are available for debugging
            st.write("Available columns:", list(data_source.columns))

    def run(self):
        """Main method to run the admin dashboard"""
        self.load_model_and_data()
        
        # Navigation sidebar
        st.sidebar.title("🎯 CAIRE Analytics")
        st.sidebar.markdown("---")
        
        # Data status - REMOVED "Model not available" text
        st.sidebar.subheader("📊 Data Status")
        
        if self.df is not None:
            st.sidebar.success(f"✅ Data: {len(self.df):,} sessions")
        else:
            st.sidebar.error("❌ Data: Not loaded")
            
        # Only show model status if it exists, don't show negative status
        if hasattr(self, 'model') and self.model is not None:
            st.sidebar.success("🤖 Model: Ready for predictions")
        
        st.sidebar.markdown("---")
        
        # Quick actions
        st.sidebar.subheader("🚀 Quick Actions")
        if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
        
        if st.sidebar.button("📊 Generate Report", use_container_width=True):
            success_box("Report generation started...")
        
        # Main content
        self.render_dashboard_overview()
        self.render_behavior_analysis()
        self.render_user_patterns()
        self.render_recent_data()


class PredictionTab(BaseTab):
    def __init__(self):
        super().__init__("🎯 Predictions")

    def run(self):
        """Main method to run the prediction tab by importing from prediction_tab.py"""
        try:
            # Import the prediction component
            from prediction_tab import PredictionEngine
            
            # Initialize and run the prediction engine
            prediction_engine = PredictionEngine()
            prediction_engine.run()
            
        except ImportError as e:
            # Fallback UI if import fails
            error_box(f"Prediction module import failed: {e}")
            self.render_fallback_ui()
        except Exception as e:
            error_box(f"Prediction error: {e}")
            self.render_fallback_ui()

    def render_fallback_ui(self):
        """Render fallback UI when prediction system is not available"""
        create_header("Abandonment Predictions", "Real-time Prediction Engine", "🎯")
        
        info_box("""
        **Prediction Engine System**
        
        Our advanced prediction system uses your trained machine learning model to 
        provide real-time cart abandonment predictions with high accuracy.
        
        **Key Features:**
        🔮 **Single Session Prediction** - Real-time abandonment probability for individual sessions
        📊 **Batch Processing** - Upload CSV files for multiple predictions
        📈 **Risk Assessment** - Color-coded risk levels and actionable insights
        💡 **Feature Analysis** - Understand which factors influence predictions
        📥 **Export Results** - Download prediction results for further analysis
        """)
        
        # Show current model status
        st.markdown("---")
        st.subheader("🔍 System Status Check")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Check for model file
            model_paths = [
                "final_manual_model.pkl",
                "saved_models/final_manual_model.pkl",
                "../saved_models/final_manual_model.pkl"
            ]
            
            model_found = False
            for path in model_paths:
                if os.path.exists(path):
                    st.success(f"✅ Model found: {path}")
                    model_found = True
                    break
            
            if not model_found:
                st.error("❌ Model file not found")
        
        with col2:
            # Check for prediction_tab.py
            if os.path.exists("prediction_tab.py"):
                st.success("✅ Prediction module available")
            else:
                st.error("❌ Prediction module not found")
        
        # Quick setup assistance
        st.markdown("---")
        st.subheader("🚀 Quick Setup")
        
        if st.button("🔄 Check System Again", type="secondary"):
            st.rerun()


class SegmentAnalysisTab(BaseTab):
    def __init__(self):
        super().__init__("👥 Segments")

    def run(self):
        create_header("Enhanced Customer Segmentation", "Intelligent Behavior-Based Segments", "👥")
        
        # Try to import and use the segmentation tab
        try:
            # Import the segmentation component
            from segmentation_tab import SegmentAnalysisTab as SegmentationComponent
            
            # Initialize and run the segmentation tab
            segmentation_component = SegmentationComponent()
            segmentation_component.run()
            
        except ImportError as e:
            # Fallback UI if import fails
            error_box(f"Segmentation module import failed: {e}")
            self.render_fallback_ui()
        except Exception as e:
            error_box(f"Segmentation error: {e}")
            self.render_fallback_ui()

    def render_fallback_ui(self):
        """Render fallback UI when segmentation system is not available"""
        info_box("""
        **Customer Segmentation System**
        
        Our enhanced segmentation system uses advanced machine learning to identify 
        distinct customer groups based on their shopping behavior and characteristics.
        
        **Key Features:**
        🔍 **Behavioral Analysis** - Groups customers by engagement patterns
        🎯 **Smart Identification** - Automatically names segments meaningfully  
        📊 **Priority Scoring** - Ranks segments by recovery urgency
        💡 **Actionable Insights** - Provides segment-specific strategies
        📈 **Visual Analytics** - Interactive charts and comparisons
        """)
        
        # Show current data status
        featured_path = os.path.join("data", "cart_abandonment_featured.csv")
        if os.path.exists(featured_path):
            df = pd.read_csv(featured_path)
            st.success(f"✅ Featured dataset available: {len(df)} records, {len(df.columns)} features")
            
            # Show available features
            with st.expander("📋 Available Features in Dataset"):
                feature_cols = [col for col in df.columns if col not in ['session_id', 'user_id', 'abandoned']]
                st.write(f"**{len(feature_cols)} features available:**")
                cols = st.columns(3)
                for i, feature in enumerate(feature_cols):
                    cols[i % 3].write(f"• {feature}")
        else:
            warning_box("📋 Featured dataset not found. Please run feature engineering first.")
        
        # Quick manual segmentation option
        st.markdown("---")
        st.subheader("Quick Manual Segmentation")
        
        if st.button("🔄 Check System Availability", type="secondary"):
            st.rerun()


class AdvancedAnalyticsTab(BaseTab):
    def __init__(self):
        super().__init__("📈 Analytics")
        self.analytics_helper = AnalyticsHelper()

    def run(self):
        self.create_header("Advanced Analytics", "Deep Dive Analysis & Insights", "📈")
        
        # Load sample data or use provided data
        df = self.load_sample_data()
        
        if df is not None:
            self.create_analytics_dashboard(df)
        else:
            st.warning("No data available for analytics. Using demo mode.")
            self.create_demo_analytics()

    def create_header(self, title, description, icon):
        """Create header for the analytics section"""
        st.markdown(f"# {icon} {title}")
        st.markdown(f"**{description}**")
        st.markdown("---")

    def load_sample_data(self):
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

    def create_analytics_dashboard(self, df):
        """Create comprehensive analytics dashboard"""
        
        # Tab layout for different analytics views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview", "🔍 Behavioral Patterns", 
            "⏰ Time Analysis", "📈 Performance"
        ])
        
        with tab1:
            self.create_overview_tab(df)
        
        with tab2:
            self.create_behavioral_analysis_tab(df)
        
        with tab3:
            self.create_time_analysis_tab(df)
        
        with tab4:
            self.create_performance_tab(df)

    def create_overview_tab(self, df):
        """Create overview analytics tab"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Abandonment trend
            st.subheader("Abandonment Trend")
            trend_fig = self.analytics_helper.create_abandonment_trend_chart(df)
            st.plotly_chart(trend_fig, use_container_width=True)
            
            # Business impact
            st.subheader("Business Impact Analysis")
            impact_metrics = self.analytics_helper.calculate_business_impact(df)
            for metric, value in impact_metrics.items():
                st.metric(metric.replace('_', ' ').title(), value)

        with col2:
            # Feature correlations
            st.subheader("Feature Correlations")
            numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
            corr_fig = self.analytics_helper.create_correlation_heatmap(df, numerical_features)
            st.plotly_chart(corr_fig, use_container_width=True)
            
            # Automated insights
            st.subheader("Key Insights")
            insights = self.analytics_helper.generate_insights_report(df)
            for insight in insights:
                st.info(insight)

    def create_behavioral_analysis_tab(self, df):
        """Create behavioral patterns analysis tab"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Behavioral Patterns")
            behavioral_features = ['session_duration', 'num_pages_viewed', 'scroll_depth', 'num_items_carted']
            # Only use features that exist in the dataframe
            available_features = [f for f in behavioral_features if f in df.columns]
            behavior_fig = self.analytics_helper.create_behavioral_patterns_chart(df, available_features)
            if behavior_fig:
                st.plotly_chart(behavior_fig, use_container_width=True)
            else:
                st.info("Not enough behavioral features available for analysis")
        
        with col2:
            st.subheader("Segment Analysis")
            # Example segmentation
            segments_data = pd.DataFrame({
                'Segment': ['High Risk', 'Medium Risk', 'Low Risk', 'Loyal'],
                'Size': [25, 35, 20, 20],
                'Abandonment_Rate': [75, 45, 15, 5]
            })
            
            # CORRECTED: Use color_discrete_sequence instead of color_continuous_scale for pie charts
            segment_fig = px.pie(
                segments_data,
                values='Size',
                names='Segment',
                title="Customer Segments",
                color='Segment',  # Use Segment for coloring
                color_discrete_sequence=px.colors.sequential.RdYlGn_r  # Use discrete color sequence
            )
            st.plotly_chart(segment_fig, use_container_width=True)

    def create_time_analysis_tab(self, df):
        """Create time-based analysis tab"""
        st.subheader("Time-Based Patterns")
        
        if 'timestamp' in df.columns and 'abandoned' in df.columns:
            time_fig = self.analytics_helper.create_time_based_analysis(df, 'timestamp', 'abandoned')
            if time_fig:
                st.plotly_chart(time_fig, use_container_width=True)
            else:
                st.info("Could not generate time-based analysis")
        else:
            st.info("Time-based analysis requires 'timestamp' and 'abandoned' columns")

    def create_performance_tab(self, df):
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

    def create_demo_analytics(self):
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
        
        self.create_analytics_dashboard(demo_data)

class AdminDashboard:
    def __init__(self):
        self.tabs = [
            AdminDashboardTab(),
            PredictionTab(),
            SegmentAnalysisTab(),
            AdvancedAnalyticsTab()
        ]

    def run(self):
        st.set_page_config(
            page_title="CAIRE - Admin Dashboard",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        load_custom_css()

        st.markdown("""
        <div style='text-align: center; padding: 20px 0 10px 0;'>
            <h1 style='
                background: linear-gradient(135deg, #60a5fa, #3b82f6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-size: 2.5em;
                margin: 0;
            '>🎯 CAIRE - Admin Dashboard</h1>
            <p style='color: #94a3b8; font-size: 1.1em; margin: 10px 0;'>
                Cart Abandonment Intelligence & Recovery Engine
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Create tabs for different admin functionalities
        main_tabs = st.tabs([tab.name for tab in self.tabs])

        BaseTab.ensure_data_directory()

        for tab, ui in zip(main_tabs, self.tabs):
            with tab:
                ui.run()

        add_footer()


if __name__ == "__main__":
    dashboard = AdminDashboard()
    dashboard.run()
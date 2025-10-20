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
            'original_data': project_root / "data" / "cart_abandonment_original.csv", 
            'test_data': project_root / "test_data" / "test_data_for_prediction.csv",
            'model': project_root / "saved_models" / "final_manual_model.pkl",
            'models_dir': project_root / "src" / "models"
        }
        
        return paths

    def load_model_and_data(self):
        """Load model and data with proper error handling"""
        self.paths = self.get_correct_paths()
        
        # Load featured dataset
        self.df = self.load_data(self.paths['featured_data'])
        
        # Load model from correct location
        self.model = None
        self.feature_names = []
        
        # Try multiple possible model locations
        possible_model_paths = [
            self.paths['model'],  # Original path
            Path("src/models/final_manual_model.pkl"),  # Your actual location
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
        """Calculate actual metrics from your data"""
        if self.df is None:
            return {}
        
        # Use original cart values if available, otherwise use normalized
        cart_value_col = 'original_cart_value' if 'original_cart_value' in self.df.columns else 'cart_value'
        
        metrics = {
            'total_sessions': len(self.df),
            'abandonment_rate': self.df['abandoned'].mean() * 100,
            'avg_cart_value': self.df[cart_value_col].mean() if cart_value_col in self.df.columns else self.df['cart_value'].mean(),
            'total_abandoned': self.df['abandoned'].sum(),
            'return_user_rate': self.df['return_user'].mean() * 100,
            'avg_engagement': self.df['engagement_score'].mean(),
            'payment_reach_rate': self.df['if_payment_page_reached'].mean() * 100,
            'avg_pages_viewed': self.df['num_pages_viewed'].mean(),
            'avg_session_duration': self.df['session_duration'].mean(),
            'recovery_potential': self.df['abandoned'].sum() * self.df[cart_value_col].mean() if cart_value_col in self.df.columns else 0
        }
        
        return metrics

    def render_dashboard_overview(self):
        """Render main dashboard with REAL data"""
        create_header("CAIRE Analytics", "Cart Abandonment Intelligence & Recovery Engine", "📊")
        
        if self.df is None:
            error_box("No data available. Please ensure featured data exists in the data folder.")
            return
        
        metrics = self.calculate_real_metrics()
        
        # Key Metrics in a grid
        create_section("Key Performance Indicators", "📈")
        
        # Define pastel color palette
        pastel_colors = {
            'blue': '#A7C7E7',      # Soft blue
            'red': '#F8BBD0',       # Soft pink/red
            'green': '#C8E6C9',     # Soft green
            'yellow': '#FFEAA7',    # Soft yellow
            'purple': '#D4BFF9',    # Soft purple
            'teal': '#B2EBF2',      # Soft teal
            'orange': '#FFCCBC',    # Soft orange
            'lime': '#E6EE9C'       # Soft lime
        }
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_stat_card(
                "Total Sessions", 
                f"{metrics['total_sessions']:,}",
                "👥",  # Replaced fa-users with emoji
                pastel_colors['blue']
            )
        
        with col2:
            create_stat_card(
                "Abandonment Rate", 
                f"{metrics['abandonment_rate']:.1f}%",
                "📊",  # Replaced fa-chart-line with emoji
                pastel_colors['red']
            )
        
        with col3:
            cart_val = metrics['avg_cart_value']
            display_val = f"${cart_val:,.2f}" if cart_val > 10 else f"{cart_val:.3f}"
            create_stat_card(
                "Avg Cart Value", 
                display_val,
                "🛒",  # Replaced fa-shopping-cart with emoji
                pastel_colors['green']
            )
        
        with col4:
            create_stat_card(
                "Abandoned Carts", 
                f"{metrics['total_abandoned']:,}",
                "⚠️",  # Replaced fa-exclamation-triangle with emoji
                pastel_colors['yellow']
            )
        
        # Second row of metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_stat_card(
                "Return Users", 
                f"{metrics['return_user_rate']:.1f}%",
                "🔄",  # Replaced fa-user-check with emoji
                pastel_colors['purple']
            )
        
        with col2:
            create_stat_card(
                "Avg Engagement", 
                f"{metrics['avg_engagement']:.2f}",
                "📈",  # Replaced fa-chart-bar with emoji
                pastel_colors['teal']
            )
        
        with col3:
            create_stat_card(
                "Payment Reach", 
                f"{metrics['payment_reach_rate']:.1f}%",
                "💳",  # Replaced fa-credit-card with emoji
                pastel_colors['orange']
            )
        
        with col4:
            recovery_pot = metrics['recovery_potential']
            display_pot = f"${recovery_pot:,.0f}" if recovery_pot > 0 else "$0"
            create_stat_card(
                "Recovery Potential", 
                display_pot,
                "💰",  # Replaced fa-money-bill-wave with emoji
                pastel_colors['lime']
            )

    def render_behavior_analysis(self):
        """Render behavior analysis section with pastel colors"""
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
                
                # Customize appearance
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
        
        display_cols = ['session_id', 'user_id', 'cart_value', 'num_items_carted', 'abandoned']
        available_cols = [col for col in display_cols if col in self.df.columns]
        
        if available_cols:
            st.dataframe(
                self.df[available_cols].head(10), 
                use_container_width=True,
                height=300
            )
        else:
            warning_box("Required columns not found in dataset")

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

    def run(self):
        create_header("Advanced Analytics", "Deep Dive Analysis & Insights", "📈")
        
        # This would integrate with your friend's analytics system
        info_box("""
        **Analytics Features:**
        - Feature correlation analysis
        - Behavioral pattern discovery
        - Trend analysis over time
        - Predictive insights
        """)
        
        st.info("🔧 Analytics system integration in progress...")


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
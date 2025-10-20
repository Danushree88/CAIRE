import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from custom_styles import create_header, create_section, info_box, warning_box, success_box, error_box

class SegmentAnalysisTab:
    def __init__(self):
        self.name = "👥 Segments"
        self.segmenter = None
        self.df = None
        
    def load_data_and_segmenter(self):
        """Load data and initialize segmenter from recovery.py"""
        try:
            # Since segmentation_tab.py is in project root (CAIRE/)
            project_root = Path(__file__).parent  # CAIRE
            
            # Add the current directory to Python path
            sys.path.insert(0, str(project_root))
            
            # Import your segmentation class
            try:
                from src.recovery.recovery import EnhancedCustomerSegmenter
            except ImportError as import_err:
                st.warning(f"First import attempt failed: {import_err}. Trying alternative...")
                # Alternative: import directly from the file
                import importlib.util
                recovery_path = project_root / "src" / "recovery" / "recovery.py"
                if recovery_path.exists():
                    spec = importlib.util.spec_from_file_location("recovery_module", recovery_path)
                    recovery_module = importlib.util.module_from_spec(spec)
                    sys.modules["recovery_module"] = recovery_module
                    spec.loader.exec_module(recovery_module)
                    EnhancedCustomerSegmenter = recovery_module.EnhancedCustomerSegmenter
                else:
                    st.error(f"❌ Recovery.py not found at: {recovery_path}")
                    return None, None
            
            # Load featured data
            possible_paths = [
                project_root / "data" / "cart_abandonment_featured.csv",
                project_root / "cart_abandonment_featured.csv",
                Path("data/cart_abandonment_featured.csv"),
            ]
            
            self.df = None
            for featured_path in possible_paths:
                if featured_path.exists():
                    self.df = pd.read_csv(featured_path)
                    st.success(f"✅ Data loaded from: {featured_path}")
                    break
            
            # FIX: Check if df is None, not just truthy value
            if self.df is None:
                st.error("❌ Could not find featured data file in any location")
                return None, None
            
            # Check if required features exist
            required_features = [
                'engagement_score', 'num_items_carted', 'cart_value', 
                'session_duration', 'num_pages_viewed', 'scroll_depth',
                'return_user', 'if_payment_page_reached', 'discount_applied',
                'has_viewed_shipping_info', 'abandoned'
            ]
            
            missing_features = [f for f in required_features if f not in self.df.columns]
            if missing_features:
                st.error(f"❌ Missing required features: {missing_features}")
                return None, None
            
            # Initialize and fit segmenter
            self.segmenter = EnhancedCustomerSegmenter(n_segments=5)
            self.segmenter.fit(self.df)
            
            # Add segments to dataframe
            self.df['segment'] = self.segmenter.predict_segment(self.df)
            self.df['segment_name'] = self.df['segment'].map(
                {k: v['segment_name'] for k, v in self.segmenter.segment_profiles.items()}
            )
            
            st.success("✅ Segmentation completed successfully!")
            return self.df, self.segmenter
            
        except ImportError as e:
            st.error(f"❌ Could not import EnhancedCustomerSegmenter: {e}")
            return None, None
        except Exception as e:
            st.error(f"❌ Error in segmentation: {e}")
            import traceback
            st.code(traceback.format_exc())
            return None, None
        
    def render_segment_overview(self):
        """Render comprehensive segment overview"""
        create_section("🎯 Customer Segments Overview", "AI-Powered Behavioral Segmentation")
        
        if self.segmenter is None or self.df is None:
            warning_box("Segmentation data not available. Please check data files.")
            return
        
        # Key metrics summary
        col1, col2, col3, col4 = st.columns(4)
        
        total_segments = len(self.segmenter.segment_profiles)
        total_customers = len(self.df)
        avg_abandonment = self.df['abandoned'].mean() * 100
        high_priority_segments = sum(1 for profile in self.segmenter.segment_profiles.values() 
                                   if profile['recovery_priority'] in ['Very High', 'High'])
        
        with col1:
            st.metric("Total Segments", total_segments)
        with col2:
            st.metric("Total Customers", f"{total_customers:,}")
        with col3:
            st.metric("Avg Abandonment", f"{avg_abandonment:.1f}%")
        with col4:
            st.metric("High Priority Segments", high_priority_segments)
        
        # Segment distribution
        st.subheader("📊 Segment Distribution")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            segment_counts = self.df['segment_name'].value_counts()
            fig = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                title="Customer Distribution by Segment",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Segment Sizes**")
            for segment, count in segment_counts.items():
                percentage = (count / total_customers) * 100
                st.write(f"• **{segment}**: {count} ({percentage:.1f}%)")

    def render_segment_performance(self):
        """Render segment performance metrics"""
        create_section("📈 Segment Performance Metrics", "Key Behavioral Indicators")
        
        if self.segmenter is None:
            return
        
        # Create performance comparison chart
        segments_data = []
        for segment_id, profile in self.segmenter.segment_profiles.items():
            segments_data.append({
                'Segment': profile['segment_name'],
                'Abandonment Rate': profile['abandonment_rate'],
                'Avg Cart Value': profile['avg_cart_value'],
                'Avg Engagement': profile['avg_engagement'],
                'Return User Rate': profile['return_user_rate'],
                'Recovery Priority': profile['recovery_priority'],
                'Business Value': profile['business_value'],
                'Size': profile['size']
            })
        
        segments_df = pd.DataFrame(segments_data)
        
        # Performance comparison
        col1, col2 = st.columns(2)
        
        with col1:
            # Abandonment rate comparison
            fig = px.bar(
                segments_df,
                x='Segment',
                y='Abandonment Rate',
                title="Abandonment Rate by Segment",
                color='Abandonment Rate',
                color_continuous_scale='RdYlGn_r',
                text='Abandonment Rate'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cart value comparison
            fig = px.bar(
                segments_df,
                x='Segment',
                y='Avg Cart Value',
                title="Average Cart Value by Segment",
                color='Avg Cart Value',
                color_continuous_scale='Blues',
                text='Avg Cart Value'
            )
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics table
        st.subheader("📋 Detailed Segment Metrics")
        display_df = segments_df[['Segment', 'Size', 'Abandonment Rate', 'Avg Cart Value', 
                                'Avg Engagement', 'Return User Rate', 'Recovery Priority', 'Business Value']].copy()
        display_df['Abandonment Rate'] = display_df['Abandonment Rate'].round(1)
        display_df['Avg Cart Value'] = display_df['Avg Cart Value'].round(3)
        display_df['Avg Engagement'] = display_df['Avg Engagement'].round(3)
        display_df['Return User Rate'] = display_df['Return User Rate'].round(1)
        
        st.dataframe(display_df, use_container_width=True)

    def render_segment_insights(self):
        """Render AI-powered segment insights"""
        create_section("🔍 Deep Segment Insights", "AI-Generated Behavioral Analysis")
        
        if self.segmenter is None:
            return
        
        # Segment insights and characteristics
        for segment_id, profile in self.segmenter.segment_profiles.items():
            with st.expander(f"🎯 {profile['segment_name']} - {profile['recovery_priority']} Priority", expanded=False):
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**📝 {profile['description']}**")
                    
                    # Key metrics comparison to global average
                    st.write("**📊 Performance vs Global Average:**")
                    
                    metrics_comparison = [
                        ("Abandonment Rate", profile['abandonment_rate'], profile['global_avg_abandonment'], "lower"),
                        ("Cart Value", profile['avg_cart_value'], profile['global_avg_cart_value'], "higher"),
                        ("Engagement", profile['avg_engagement'], profile['global_avg_engagement'], "higher"),
                        ("Return Rate", profile['return_user_rate'], profile['global_avg_return_rate'], "higher")
                    ]
                    
                    for metric, segment_val, global_val, direction in metrics_comparison:
                        diff = segment_val - global_val
                        if direction == "lower":
                            icon = "✅" if diff < 0 else "⚠️"
                        else:
                            icon = "✅" if diff > 0 else "⚠️"
                        
                        if metric == "Abandonment Rate":
                            st.write(f"{icon} {metric}: {segment_val:.1f}% (Global: {global_val:.1f}%)")
                        elif metric == "Cart Value":
                            st.write(f"{icon} {metric}: {segment_val:.3f} (Global: {global_val:.3f})")
                        elif metric == "Engagement":
                            st.write(f"{icon} {metric}: {segment_val:.3f} (Global: {global_val:.3f})")
                        else:
                            st.write(f"{icon} {metric}: {segment_val:.1f}% (Global: {global_val:.1f}%)")
                
                with col2:
                    # Recovery potential
                    st.write("**🎯 Recovery Potential**")
                    st.metric("Priority Score", f"{profile['recovery_priority_score']}/100")
                    st.metric("Business Value", profile['business_value'])
                    st.metric("Segment Size", f"{profile['size']} users")


    def render_recovery_strategies(self):
        """Render actionable strategy implementation panel"""
        create_section("🛠️ Strategy Implementation", "Deploy & Track Recovery Actions")
        
        if self.segmenter is None:
            return
        
        # Strategy options for each segment
        strategy_options = {
            "High-Value New Abandoners": [
                "💎 VIP onboarding call within 1 hour",
                "🚀 Personal executive email follow-up", 
                "🎁 Exclusive welcome discount (15%)"
            ],
            "Checkout Abandoners": [
                "⏰ 3-step abandoned cart email sequence",
                "🔓 Simplify checkout process reminder",
                "🛡️ Security & trust badges emphasis"
            ],
            "Engaged Researchers": [
                "📚 Product expert consultation offer",
                "🎥 Detailed product demonstration videos",
                "💬 Live chat support promotion"
            ],
            "Price Sensitive Shoppers": [
                "💰 Tiered discounts based on cart value",
                "📦 Free shipping threshold education", 
                "🎯 Flash sale notifications"
            ],
            "Loyal High-Value Shoppers": [
                "⭐ VIP early access to new products",
                "🎫 Double loyalty points campaign",
                "📊 Personalized product recommendations"
            ],
            "Loyal Low-Value Shoppers": [
                "🔄 Loyalty program benefits reminder",
                "📈 Volume discount incentives",
                "🎁 Small surprise free gift"
            ],
            "Casual Browsers": [
                "👋 Welcome discount for first purchase",
                "📱 Mobile app download incentive",
                "🎯 Re-engagement email series"
            ],
            "Standard Shoppers": [
                "📧 Standard abandoned cart emails",
                "💡 Product recommendations",
                "🚚 Free shipping offer"
            ]
        }
        
        st.subheader("🎯 Select & Deploy Strategies")
        
        # Strategy selection and deployment
        for segment_id, profile in self.segmenter.segment_profiles.items():
            segment_name = profile['segment_name']
            
            with st.expander(f"🛠️ {segment_name} - {profile['size']} users", expanded=False):
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Strategy selection - UNIQUE KEY
                    selected_strategies = st.multiselect(
                        f"Choose strategies for {segment_name}:",
                        options=strategy_options.get(segment_name, strategy_options["Standard Shoppers"]),
                        default=[],
                        key=f"strategies_select_{segment_id}"  # UNIQUE KEY
                    )
                    
                    if selected_strategies:
                        # Implementation configuration - UNIQUE KEYS
                        st.write("**⚙️ Implementation Settings:**")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            start_date = st.date_input("Start Date", key=f"date_{segment_id}_{segment_name}")  # UNIQUE
                            assigned_team = st.selectbox("Assigned Team", 
                                                    ["Marketing", "Sales", "Customer Success", "Automated"],
                                                    key=f"team_{segment_id}_{segment_name}")  # UNIQUE
                        
                        with col_b:
                            budget = st.number_input("Budget Allocation ($)", 
                                                min_value=0, value=1000, 
                                                key=f"budget_{segment_id}_{segment_name}")  # UNIQUE
                            success_metric = st.selectbox("Success Metric",
                                                        ["Reduced Abandonment", "Increased Revenue", "Improved Conversion"],
                                                        key=f"metric_{segment_id}_{segment_name}")  # UNIQUE
                
                with col2:
                    # Quick actions - UNIQUE KEYS
                    st.write("**🚀 Quick Actions:**")
                    
                    if st.button(f"📋 Save Plan", key=f"save_{segment_id}_{segment_name}", use_container_width=True):  # UNIQUE
                        st.success(f"Strategy plan saved for {segment_name}!")
                    
                    if st.button(f"📧 Deploy Now", key=f"deploy_{segment_id}_{segment_name}", use_container_width=True):  # UNIQUE
                        st.success(f"Deploying {len(selected_strategies)} strategies for {segment_name}!")
                    
                    if st.button(f"📊 Track Progress", key=f"track_{segment_id}_{segment_name}", use_container_width=True):  # UNIQUE
                        st.success(f"Opening progress dashboard for {segment_name}!")
        
        # Bulk actions - UNIQUE KEYS
        st.markdown("---")
        st.subheader("📈 Bulk Operations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Deploy All High-Risk Strategies", type="primary", 
                        key="bulk_deploy_all", use_container_width=True):  # UNIQUE
                st.success("Deploying strategies for all high-abandonment segments!")
        
        with col2:
            if st.button("📋 Generate Implementation Report", 
                        key="bulk_generate_report", use_container_width=True):  # UNIQUE
                st.success("Implementation report generated!")
        
        with col3:
            if st.button("💰 Calculate Total Budget", 
                        key="bulk_calculate_budget", use_container_width=True):  # UNIQUE
                st.success("Total budget calculation completed!")

    def _calculate_expected_recovery(self, profile):
        """Calculate expected recovery rate based on segment characteristics"""
        base_rate = 15  # Base recovery rate
        
        # Adjust based on segment characteristics
        if profile['recovery_priority'] == "Very High":
            base_rate += 10
        elif profile['recovery_priority'] == "High":
            base_rate += 5
        
        if profile['avg_cart_value'] > profile['global_avg_cart_value']:
            base_rate += 5
        
        if profile['return_user_rate'] > profile['global_avg_return_rate']:
            base_rate += 3
        
        return f"{base_rate}-{base_rate + 15}"

    def _get_implementation_timeline(self, priority):
        """Get implementation timeline based on priority"""
        timelines = {
            "Very High": "1-3 days",
            "High": "3-7 days", 
            "Medium": "1-2 weeks",
            "Low": "2-4 weeks"
        }
        return timelines.get(priority, "1-2 weeks")

    def _get_resource_requirement(self, priority):
        """Get resource requirement based on priority"""
        resources = {
            "Very High": "High (Dedicated team)",
            "High": "Medium (Cross-functional)",
            "Medium": "Low (Marketing team)",
            "Low": "Minimal (Automated)"
        }
        return resources.get(priority, "Medium")

    def render_segment_actions(self):
        """Render actionable insights and next steps"""
        create_section("🚀 Recommended Actions", "Priority-Based Implementation Plan")
        
        if self.segmenter is None:
            return
        
        # Priority-based action plan
        st.subheader("🎯 Priority Action Plan")
        
        # Group segments by priority
        priority_groups = {}
        for segment_id, profile in self.segmenter.segment_profiles.items():
            priority = profile['recovery_priority']
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(profile)
        
        # Display actions by priority
        for priority in ["Very High", "High", "Medium", "Low"]:
            if priority in priority_groups:
                with st.expander(f"🔴 {priority} Priority Segments", expanded=(priority in ["Very High", "High"])):
                    for profile in priority_groups[priority]:
                        st.write(f"**{profile['segment_name']}**")
                        st.write(f"*Size: {profile['size']} users | Recovery Score: {profile['recovery_priority_score']}/100*")
                        
                        # Quick wins
                        st.write("**Quick Wins:**")
                        if profile['abandonment_rate'] > 50:
                            st.write("• Implement immediate abandoned cart email sequence")
                        if profile['avg_cart_value'] > profile['global_avg_cart_value']:
                            st.write("• Assign dedicated high-value customer support")
                        if profile['return_user_rate'] > 60:
                            st.write("• Launch loyalty program benefits campaign")
                        
                        st.write("")

    def run(self):
        """Main method to run enhanced segmentation tab"""
        create_header("AI-Powered Customer Segmentation", 
                    "Intelligent Behavioral Analysis & Recovery Strategies", "👥")
        
        # Load data and segmenter
        with st.spinner("🔄 Loading segmentation data and AI models..."):
            self.df, self.segmenter = self.load_data_and_segmenter()
        
        if self.segmenter and self.df is not None:
            success_box(f"✅ Successfully analyzed {len(self.df):,} customers across {len(self.segmenter.segment_profiles)} behavioral segments")
            
            # Create tabs for different segmentation views - ADDED IMPLEMENTATION TAB
            seg_tabs = st.tabs([
                "📊 Overview", 
                "📈 Performance", 
                "🔍 Deep Insights", 
                "💡 Strategies", 
                "🛠️ Implementation",  # NEW TAB
                "🚀 Actions"
            ])
            
            with seg_tabs[0]:
                self.render_segment_overview()
            
            with seg_tabs[1]:
                self.render_segment_performance()
            
            with seg_tabs[2]:
                self.render_segment_insights()
            
            with seg_tabs[3]:
                self.render_recovery_strategies()
            
            with seg_tabs[4]:  # NEW IMPLEMENTATION TAB
                self.render_strategy_implementation()
            
            with seg_tabs[5]:
                self.render_segment_actions()
        else:
            error_box("""
            **Unable to load segmentation system.** Please ensure:
            
            1. **Data File**: `data/cart_abandonment_featured.csv` exists with required features
            2. **Recovery Module**: `recovery.py` is available in the recovery directory
            3. **Dependencies**: All required packages are installed (scikit-learn, etc.)
            
            **Required Features**: engagement_score, cart_value, session_duration, num_pages_viewed, 
            scroll_depth, return_user, if_payment_page_reached, discount_applied, has_viewed_shipping_info, abandoned
            """)
            
            # Show available files for debugging
            with st.expander("🔧 Debug Information"):
                st.write("**Current Directory Files:**")
                current_files = os.listdir('.')
                st.write([f for f in current_files if f.endswith('.py') or f == 'data'])
                
                if os.path.exists('data'):
                    st.write("**Data Directory Files:**")
                    data_files = os.listdir('data')
                    st.write(data_files)
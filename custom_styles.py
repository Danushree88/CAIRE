# custom_styles.py
import streamlit as st
import pandas as pd

def load_custom_css():
    """Load custom CSS styles for the dashboard"""
    st.markdown("""
    <style>
    /* Main styling */
    .main {
        background-color: #f8fafc;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
    }
    
    /* Header styling */
    .dashboard-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    /* Info boxes */
    .info-box {
        background: #dbeafe;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #dcfce7;
        border-left: 4px solid #22c55e;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .error-box {
        background: #fee2e2;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Section headers */
    .section-header {
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
        color: #1e293b;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        margin-top: 3rem;
        border-top: 1px solid #e2e8f0;
    }
    </style>
    """, unsafe_allow_html=True)

def create_header(title, subtitle, icon):
    """Create a styled header"""
    st.markdown(f"""
    <div class="dashboard-header">
        <h1 style="margin: 0; font-size: 2.5em;">{icon} {title}</h1>
        <p style="margin: 0.5em 0 0 0; font-size: 1.2em; opacity: 0.9;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

def create_section(title, icon):
    """Create a section header"""
    st.markdown(f"""
    <div class="section-header">
        <h2 style="margin: 0; display: flex; align-items: center; gap: 0.5rem;">
            <span>{icon}</span>
            <span>{title}</span>
        </h2>
    </div>
    """, unsafe_allow_html=True)

def info_box(content):
    """Create an info box"""
    st.markdown(f"""
    <div class="info-box">
        <strong>💡 Info:</strong> {content}
    </div>
    """, unsafe_allow_html=True)

def success_box(content):
    """Create a success box"""
    st.markdown(f"""
    <div class="success-box">
        <strong>✅ Success:</strong> {content}
    </div>
    """, unsafe_allow_html=True)

def warning_box(content):
    """Create a warning box"""
    st.markdown(f"""
    <div class="warning-box">
        <strong>⚠️ Warning:</strong> {content}
    </div>
    """, unsafe_allow_html=True)

def error_box(content):
    """Create an error box"""
    st.markdown(f"""
    <div class="error-box">
        <strong>❌ Error:</strong> {content}
    </div>
    """, unsafe_allow_html=True)

def create_stat_card(title, value, icon, color):
    """Create a metric card"""
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: {color};">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div>
                <h3 style="margin: 0; color: #64748b; font-size: 0.9em; text-transform: uppercase;">
                    {title}
                </h3>
                <h1 style="margin: 0.5em 0 0 0; color: #1e293b; font-size: 1.8em;">
                    {value}
                </h1>
            </div>
            <div style="font-size: 2em; color: {color};">
                {icon}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_comparison_table(data):
    """Create a comparison table"""
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

def add_footer():
    """Add footer to the dashboard"""
    st.markdown("""
    <div class="footer">
        <p>🎯 <strong>CAIRE System</strong> - Cart Abandonment Intelligence & Recovery Engine</p>
        <p style="font-size: 0.9em;">© 2025 All rights reserved</p>
    </div>
    """, unsafe_allow_html=True)
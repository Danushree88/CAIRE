import streamlit as st

def load_custom_css():
    """Load comprehensive custom CSS for professional appearance"""
    custom_css = """
    <style>
    /* Root variables */
    :root {
        --primary: #3b82f6;
        --primary-dark: #1e40af;
        --secondary: #8b5cf6;
        --success: #10b981;
        --danger: #ef4444;
        --warning: #f59e0b;
        --info: #0ea5e9;
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
        --bg-tertiary: #334155;
        --text-primary: #e2e8f0;
        --text-secondary: #cbd5e1;
        --border: #475569;
    }

    /* Global styles */
    body {
        background-color: var(--bg-primary);
        color: var(--text-primary);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .main {
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 600;
    }

    h1 {
        background: linear-gradient(135deg, #60a5fa, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5em !important;
    }

    h2 {
        border-bottom: 2px solid var(--primary);
        padding-bottom: 10px;
    }

    /* Metric containers */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 20px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }

    [data-testid="metric-container"]:hover {
        border-color: var(--primary) !important;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.15);
        transform: translateY(-4px);
        transition: all 0.3s ease;
    }

    /* Buttons */
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3) !important;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.5) !important;
    }

    /* Tabs */
    [data-baseweb="tab-list"] {
        border-bottom: 2px solid var(--border);
        gap: 0;
    }

    [data-baseweb="tab"] {
        padding: 12px 24px !important;
        border-radius: 8px 8px 0 0 !important;
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        transition: all 0.3s ease;
    }

    [aria-selected="true"] [data-baseweb="tab"] {
        background-color: var(--primary) !important;
        color: white !important;
        border-bottom: 3px solid var(--primary) !important;
    }

    [data-baseweb="tab"]:hover {
        color: var(--text-primary) !important;
    }

    /* Input fields */
    .stSelectbox, .stMultiSelect, .stTextInput, .stNumberInput, .stTextArea, .stSlider {
        margin-bottom: 16px;
    }

    [data-baseweb="input"], [data-baseweb="select"] {
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }

    [data-baseweb="input"]:focus, [data-baseweb="select"]:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }

    /* Checkbox and Radio */
    [data-testid="stCheckbox"], [data-testid="stRadio"] {
        padding: 10px 0;
    }

    /* Expander */
    [data-testid="stExpander"] {
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        background-color: var(--bg-secondary) !important;
    }

    [data-testid="stExpander"] [data-testid="stExpanderToggleButton"] {
        color: var(--primary) !important;
    }

    /* Alert messages */
    [data-testid="stAlert"] {
        border-radius: 8px !important;
        padding: 16px !important;
        border-left: 4px solid !important;
    }

    [data-testid="stInfoMessage"] {
        background-color: rgba(14, 165, 233, 0.1) !important;
        border-left-color: var(--info) !important;
        color: var(--info) !important;
    }

    [data-testid="stSuccessMessage"] {
        background-color: rgba(16, 185, 129, 0.1) !important;
        border-left-color: var(--success) !important;
        color: var(--success) !important;
    }

    [data-testid="stWarningMessage"] {
        background-color: rgba(245, 158, 11, 0.1) !important;
        border-left-color: var(--warning) !important;
        color: var(--warning) !important;
    }

    [data-testid="stErrorMessage"] {
        background-color: rgba(239, 68, 68, 0.1) !important;
        border-left-color: var(--danger) !important;
        color: var(--danger) !important;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        background-color: var(--bg-secondary) !important;
        border-radius: 8px !important;
        overflow: hidden;
    }

    /* Progress bar */
    [data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
        border-right: 1px solid var(--border);
    }

    /* Divider */
    hr {
        border-color: var(--border) !important;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary);
    }

    /* Links */
    a {
        color: var(--info) !important;
        text-decoration: none;
    }

    a:hover {
        text-decoration: underline;
    }

    /* Code blocks */
    pre, code {
        background-color: var(--bg-tertiary) !important;
        color: #a5f3fc !important;
        border-radius: 6px !important;
        border: 1px solid var(--border) !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def create_header(title: str, subtitle: str = "", icon: str = ""):
    """Create a professional header section"""
    st.markdown(f"""
    <div style='text-align: center; margin-bottom: 30px; padding: 30px 0;'>
        <h1 style='margin: 0; font-size: 2.8em;'>{icon} {title}</h1>
        {f'<p style="color: #94a3b8; font-size: 1.1em; margin: 10px 0 0 0;">{subtitle}</p>' if subtitle else ''}
        <hr style='border: none; border-top: 2px solid #334155; margin: 20px 0; opacity: 0.5;'>
    </div>
    """, unsafe_allow_html=True)


def create_metric_row(metrics: dict):
    """Create a professional metric row with multiple metrics
    
    Args:
        metrics: dict with format {
            'label': 'value',
            'label2': 'value2'
        }
    """
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        with col:
            st.metric(label, value)


def create_section(title: str, icon: str = ""):
    """Create a professional section divider"""
    st.markdown(f"""
    <div style='margin: 30px 0 20px 0;'>
        <h2 style='
            font-size: 1.5em;
            color: #e2e8f0;
            border-bottom: 2px solid #3b82f6;
            padding-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        '>{icon} {title}</h2>
    </div>
    """, unsafe_allow_html=True)


def info_box(message: str, icon: str = "ℹ️"):
    """Create a professional info box"""
    st.markdown(f"""
    <div style='
        background-color: rgba(14, 165, 233, 0.1);
        border-left: 4px solid #0ea5e9;
        padding: 16px;
        border-radius: 8px;
        color: #0ea5e9;
        margin: 16px 0;
    '>{icon} {message}</div>
    """, unsafe_allow_html=True)


def success_box(message: str, icon: str = "✅"):
    """Create a professional success box"""
    st.markdown(f"""
    <div style='
        background-color: rgba(16, 185, 129, 0.1);
        border-left: 4px solid #10b981;
        padding: 16px;
        border-radius: 8px;
        color: #10b981;
        margin: 16px 0;
    '>{icon} {message}</div>
    """, unsafe_allow_html=True)


def warning_box(message: str, icon: str = "⚠️"):
    """Create a professional warning box"""
    st.markdown(f"""
    <div style='
        background-color: rgba(245, 158, 11, 0.1);
        border-left: 4px solid #f59e0b;
        padding: 16px;
        border-radius: 8px;
        color: #f59e0b;
        margin: 16px 0;
    '>{icon} {message}</div>
    """, unsafe_allow_html=True)


def error_box(message: str, icon: str = "❌"):
    """Create a professional error box"""
    st.markdown(f"""
    <div style='
        background-color: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #ef4444;
        padding: 16px;
        border-radius: 8px;
        color: #ef4444;
        margin: 16px 0;
    '>{icon} {message}</div>
    """, unsafe_allow_html=True)


def create_stat_card(label: str, value: str, change: str = "", icon: str = "📊"):
    """Create a professional stat card"""
    color = "#10b981" if change.startswith("+") else "#ef4444" if change.startswith("-") else "#64748b"
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    '>
        <div style='font-size: 2em; margin-bottom: 10px;'>{icon}</div>
        <div style='font-size: 0.9em; color: #94a3b8; margin-bottom: 8px;'>{label}</div>
        <div style='font-size: 2em; font-weight: bold; color: #60a5fa; margin-bottom: 8px;'>{value}</div>
        {f'<div style="font-size: 0.85em; color: {color};">{change}</div>' if change else ''}
    </div>
    """, unsafe_allow_html=True)


def create_comparison_table(df, highlight_max=True):
    """Create a styled comparison table"""
    if highlight_max:
        st.dataframe(
            df.style.highlight_max(axis=0, color='#3b82f6')
                   .highlight_min(axis=0, color='#ef4444')
                   .format(precision=3),
            use_container_width=True
        )
    else:
        st.dataframe(df, use_container_width=True)


def add_footer():
    """Add a professional footer"""
    st.markdown("""
    <hr style='border: none; border-top: 1px solid #334155; margin: 40px 0 20px 0;'>
    <div style='text-align: center; color: #64748b; font-size: 0.9em; padding: 20px 0;'>
        <p>🛒 Cart Abandonment Analytics Dashboard | © 2024 | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)
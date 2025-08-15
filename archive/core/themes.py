"""
Theme management and mobile-responsive styling
"""
import streamlit as st

def get_card_styles():
    """Get theme-appropriate card styles"""
    dark_mode = st.session_state.get("mmt_state_v2", {}).get("settings", {}).get("darkMode", False)
    blackout_mode = st.session_state.get("mmt_state_v2", {}).get("settings", {}).get("blackoutMode", True)
    
    if blackout_mode:
        return {
            'background': '#1a1a1a',
            'border': '1px solid #333333',
            'shadow': '0 4px 15px rgba(255, 255, 255, 0.05)',
            'text_color': '#ffffff',
            'muted_color': '#b0b0b0',
            'accent_color': '#58a6ff',
            'header_gradient': 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)'
        }
    elif dark_mode:
        return {
            'background': 'linear-gradient(135deg, #21262d 0%, #161b22 100%)',
            'border': '1px solid #30363d',
            'shadow': '0 4px 15px rgba(0, 0, 0, 0.3)',
            'text_color': '#f0f6fc',
            'muted_color': '#8b949e',
            'accent_color': '#58a6ff',
            'header_gradient': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
        }
    else:
        return {
            'background': 'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)',
            'border': '1px solid #e2e8f0',
            'shadow': '0 4px 15px rgba(0, 0, 0, 0.1)',
            'text_color': '#1e293b',
            'muted_color': '#64748b',
            'accent_color': '#3b82f6',
            'header_gradient': 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)'
        }

def apply_mobile_css():
    """Apply mobile-responsive CSS"""
    mobile_view = st.session_state.get("mobile_view", False)
    
    # Base mobile-responsive CSS
    mobile_css = """
    <style>
    /* Mobile-first responsive design */
    @media (max-width: 768px) {
        .main .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        
        /* Compact dashboard cards */
        .dashboard-card {
            padding: 1rem !important;
            margin: 0.5rem 0 !important;
            min-height: auto !important;
        }
        
        /* Smaller text on mobile */
        .dashboard-card .metric-number {
            font-size: 1.5rem !important;
        }
        
        .dashboard-card .metric-label {
            font-size: 0.875rem !important;
        }
        
        /* Compact spaced repetition cards */
        .spaced-card {
            padding: 1.5rem !important;
            margin: 1rem 0 !important;
            min-height: 150px !important;
        }
        
        .spaced-card .card-text {
            font-size: 1.1rem !important;
        }
        
        /* Compact page headers */
        .page-header {
            padding: 1rem !important;
            margin: 0.5rem 0 !important;
        }
        
        .page-header h1 {
            font-size: 1.5rem !important;
        }
        
        /* Smaller buttons */
        .stButton > button {
            padding: 0.4rem 0.8rem !important;
            font-size: 0.85rem !important;
        }
        
        /* Compact progress bars */
        .progress-container {
            padding: 1rem !important;
            margin: 0.5rem 0 !important;
        }
    }
    
    /* Tablet optimization */
    @media (min-width: 769px) and (max-width: 1024px) {
        .main .block-container {
            padding-top: 2rem !important;
        }
        
        .dashboard-card {
            padding: 1.5rem !important;
        }
    }
    </style>
    """
    
    return mobile_css

def apply_theme():
    """Apply complete theme with mobile responsiveness"""
    dark_mode = st.session_state.get("mmt_state_v2", {}).get("settings", {}).get("darkMode", False)
    blackout_mode = st.session_state.get("mmt_state_v2", {}).get("settings", {}).get("blackoutMode", True)
    
    # Get mobile CSS
    mobile_css = apply_mobile_css()
    
    if blackout_mode:
        # Blackout mode styling
        st.markdown(mobile_css + """
        <style>
        .stApp {
            background-color: #000000 !important;
        }
        .main .block-container {
            background-color: #000000 !important;
        }
        .stSelectbox > div > div {
            background-color: #1a1a1a !important;
            color: #ffffff !important;
            border: 1px solid #333333 !important;
        }
        .stButton > button {
            background: #2a2a2a !important;
            color: #ffffff !important;
            border: 1px solid #404040 !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button:hover {
            background: #3a3a3a !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(255,255,255,0.1) !important;
        }
        .stSidebar {
            background-color: #0a0a0a !important;
        }
        .stSidebar .stButton > button {
            background: #1a1a1a !important;
            color: #ffffff !important;
            border: 1px solid #333333 !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
    elif dark_mode:
        # Dark mode styling
        st.markdown(mobile_css + """
        <style>
        .stApp {
            background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%) !important;
        }
        .main .block-container {
            background: transparent !important;
        }
        .stButton > button {
            background: linear-gradient(135deg, #4c1d95 0%, #7c3aed 100%) !important;
            color: #ffffff !important;
            border: 1px solid #6d28d9 !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #5b21b6 0%, #8b5cf6 100%) !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3) !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
    else:
        # Light mode styling
        st.markdown(mobile_css + """
        <style>
        .stApp {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important;
        }
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
            color: #ffffff !important;
            border: 1px solid #2563eb !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
        }
        </style>
        """, unsafe_allow_html=True)

def page_header(title, subtitle=None):
    """Mobile-optimized page header"""
    styles = get_card_styles()
    mobile_view = st.session_state.get("mobile_view", False)
    
    header_padding = "1rem" if mobile_view else "2rem"
    title_size = "1.75rem" if mobile_view else "2.5rem"
    
    gradient = styles.get('header_gradient', 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)')
    
    header_html = f"""
    <div class="page-header" style="
        background: {gradient};
        padding: {header_padding};
        border-radius: 16px;
        text-align: center;
        margin: 0.5rem 0 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    ">
        <h1 style="
            color: white;
            margin: 0;
            font-size: {title_size};
            font-weight: 700;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        ">{title}</h1>
    """
    
    if subtitle:
        subtitle_size = "0.9rem" if mobile_view else "1.1rem"
        header_html += f"""
        <p style="
            color: rgba(255,255,255,0.9);
            margin: 0.5rem 0 0 0;
            font-size: {subtitle_size};
            text-shadow: 0 1px 2px rgba(0,0,0,0.2);
        ">{subtitle}</p>
        """
    
    header_html += "</div>"
    st.markdown(header_html, unsafe_allow_html=True)

"""
Mobile-optimized UI components
"""
import streamlit as st
from core.themes import get_card_styles

def create_mobile_dashboard_card(title, completed_count, total_count, subtitle):
    """Create a compact mobile-friendly dashboard card"""
    styles = get_card_styles()
    completion_icon = "✓" if (completed_count == total_count and total_count > 0) else "○"
    completion_color = "#22c55e" if (completed_count == total_count and total_count > 0) else styles['muted_color']
    
    st.markdown(f"""
    <div class="dashboard-card" style="
        background: {styles['background']};
        padding: 1rem;
        border-radius: 12px;
        border: {styles['border']};
        box-shadow: {styles['shadow']};
        text-align: center;
        margin: 0.5rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    ">
        <div style="text-align: left;">
            <div style="font-weight: 600; color: {styles['text_color']}; font-size: 1rem;">{title}</div>
            <div style="color: {styles['muted_color']}; font-size: 0.8rem;">{subtitle}</div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 1.2rem; color: {completion_color}; margin-bottom: 0.25rem;">{completion_icon}</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: {styles['accent_color']};">{completed_count}/{total_count}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_desktop_dashboard_card(title, completed_count, total_count, subtitle):
    """Create a desktop dashboard card"""
    styles = get_card_styles()
    completion_icon = "✓" if (completed_count == total_count and total_count > 0) else "○"
    completion_color = "#22c55e" if (completed_count == total_count and total_count > 0) else styles['muted_color']
    
    st.markdown(f"""
    <div style="
        background: {styles['background']};
        padding: 1.5rem;
        border-radius: 16px;
        border: {styles['border']};
        box-shadow: {styles['shadow']};
        text-align: center;
        transition: transform 0.2s ease;
    ">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem; color: {completion_color};">{completion_icon}</div>
        <div style="font-weight: 600; color: {styles['text_color']}; margin-bottom: 0.25rem;">{title}</div>
        <div style="font-size: 2rem; font-weight: 700; color: {styles['accent_color']}; margin-bottom: 0.5rem;">{completed_count}/{total_count}</div>
        <div style="color: {styles['muted_color']}; font-size: 0.875rem;">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

def create_spaced_repetition_card(card, mobile_view=False):
    """Create a spaced repetition card optimized for screen size"""
    styles = get_card_styles()
    card_padding = "1.5rem" if mobile_view else "2.5rem"
    card_height = "120px" if mobile_view else "200px"
    font_size = "1.1rem" if mobile_view else "1.25rem"
    
    st.markdown(f"""
    <div class="spaced-card" style="
        background: {styles['background']};
        padding: {card_padding};
        border-radius: 20px;
        border: {styles['border']};
        box-shadow: {styles['shadow']};
        margin: 1rem 0;
        text-align: center;
        min-height: {card_height};
        display: flex;
        flex-direction: column;
        justify-content: center;
    ">
        <div class="card-text" style="
            font-size: {font_size};
            font-weight: 600;
            color: {styles['text_color']};
            line-height: 1.5;
        ">{card["front"]}</div>
    </div>
    """, unsafe_allow_html=True)

def create_answer_card(card, mobile_view=False):
    """Create answer display card"""
    styles = get_card_styles()
    answer_padding = "1.5rem" if mobile_view else "2rem"
    answer_font = "1rem" if mobile_view else "1.125rem"
    
    st.markdown(f"""
    <div style="
        background: {styles['background']};
        padding: {answer_padding};
        border-radius: 16px;
        border: {styles['border']};
        margin: 1rem 0;
        text-align: center;
    ">
        <div style="
            font-size: {answer_font};
            color: {styles['text_color']};
            line-height: 1.5;
        ">{card["back"]}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show tags if available
    if card.get("tags"):
        tags_color = styles['muted_color']
        tags_bg = styles['background'].replace('linear-gradient', 'rgba').replace('135deg, ', '').replace(' 0%', '').replace(' 100%', '').replace(')', ', 0.5)')
        tags_html = " ".join(f'<span style="background: {tags_bg}; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.875rem; color: {tags_color}; margin: 0.25rem; border: 1px solid {styles["border"].split()[2]};">{t}</span>' for t in card["tags"])
        st.markdown(f'<div style="text-align: center; margin: 1rem 0;">{tags_html}</div>', unsafe_allow_html=True)

def create_grade_buttons(card, mobile_view=False, handle_grade_func=None):
    """Create grading buttons optimized for screen size"""
    st.markdown("### Rate your recall")
    
    if mobile_view:
        # Mobile: 2x2 grid of buttons
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)
        
        with row1_col1:
            if st.button("● Again", key="grade_0", use_container_width=True, help="Need to study more"):
                if handle_grade_func:
                    handle_grade_func(card, 0)
        with row1_col2:
            if st.button("● Hard", key="grade_3", use_container_width=True, help="Difficult recall"):
                if handle_grade_func:
                    handle_grade_func(card, 3)
        with row2_col1:
            if st.button("● Good", key="grade_4", use_container_width=True, help="Good recall"):
                if handle_grade_func:
                    handle_grade_func(card, 4)
        with row2_col2:
            if st.button("● Easy", key="grade_5", use_container_width=True, help="Perfect recall"):
                if handle_grade_func:
                    handle_grade_func(card, 5)
    else:
        # Desktop: Single row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("● Again", key="grade_0", use_container_width=True, help="Need to study more"):
                if handle_grade_func:
                    handle_grade_func(card, 0)
        with col2:
            if st.button("● Hard", key="grade_3", use_container_width=True, help="Difficult recall"):
                if handle_grade_func:
                    handle_grade_func(card, 3)
        with col3:
            if st.button("● Good", key="grade_4", use_container_width=True, help="Good recall"):
                if handle_grade_func:
                    handle_grade_func(card, 4)
        with col4:
            if st.button("● Easy", key="grade_5", use_container_width=True, help="Perfect recall"):
                if handle_grade_func:
                    handle_grade_func(card, 5)

def create_progress_summary(completed, total_activities):
    """Create a mobile-optimized progress summary"""
    styles = get_card_styles()
    completed_count = sum(completed.values())
    progress_pct = int((completed_count / total_activities) * 100)
    
    mobile_view = st.session_state.get("mobile_view", False)
    
    if mobile_view:
        # Compact mobile progress display
        st.markdown(f"""
        <div style="
            background: {styles['background']};
            padding: 1rem;
            border-radius: 12px;
            border: {styles['border']};
            box-shadow: {styles['shadow']};
            margin: 1rem 0;
            text-align: center;
        ">
            <div style="font-weight: 600; color: {styles['text_color']}; margin-bottom: 0.5rem;">Today's Progress</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: {styles['accent_color']}; margin-bottom: 0.5rem;">{completed_count}/{total_activities}</div>
            <div style="
                background: #f1f5f9;
                border-radius: 8px;
                height: 8px;
                overflow: hidden;
                margin-bottom: 0.5rem;
            ">
                <div style="
                    background: {styles['accent_color']};
                    height: 100%;
                    width: {progress_pct}%;
                    border-radius: 8px;
                    transition: width 0.3s ease;
                "></div>
            </div>
            <div style="color: {styles['muted_color']}; font-size: 0.8rem;">{progress_pct}% Complete</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Full desktop progress display
        progress_gradient = "linear-gradient(90deg, #58a6ff 0%, #238636 100%)" if st.session_state.get("mmt_state_v2", {}).get("settings", {}).get("darkMode", False) else "linear-gradient(90deg, #007aff 0%, #00d4ff 100%)"
        background_bar = "#21262d" if st.session_state.get("mmt_state_v2", {}).get("settings", {}).get("darkMode", False) else "#f1f5f9"
        
        st.markdown(f"""
        <div style="
            background: {styles['background']};
            padding: 2rem;
            border-radius: 16px;
            border: {styles['border']};
            box-shadow: {styles['shadow']};
            margin: 1.5rem 0;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <div style="font-weight: 600; color: {styles['text_color']};">Today's Activities</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: {styles['accent_color']};">{completed_count}/{total_activities}</div>
            </div>
            <div style="
                background: {background_bar};
                border-radius: 12px;
                height: 12px;
                overflow: hidden;
                margin-bottom: 0.5rem;
            ">
                <div style="
                    background: {progress_gradient};
                    height: 100%;
                    width: {progress_pct}%;
                    border-radius: 12px;
                    transition: width 0.3s ease;
                "></div>
            </div>
            <div style="color: {styles['muted_color']}; font-size: 0.875rem; text-align: center;">{progress_pct}% Complete</div>
        </div>
        """, unsafe_allow_html=True)

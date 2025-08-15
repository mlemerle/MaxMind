# MaxMind.py - Mobile-Optimized Cognitive Training Platform
# Run: streamlit run MaxMind.py

import streamlit as st
import json
import random
import time
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional

# Core imports
from core.state_management import initialize_state, get_state, save_state, check_daily_reset, mark_completed, is_completed_today
from core.themes import apply_theme, page_header, get_card_styles
from core.spaced_repetition import due_cards, schedule, add_card, remove_card, search_cards
from core.utils import today_iso, new_id
from components.ui_components import (
    create_mobile_dashboard_card, 
    create_desktop_dashboard_card,
    create_spaced_repetition_card,
    create_answer_card,
    create_grade_buttons,
    create_progress_summary
)

# Configuration
st.set_page_config(
    page_title="MaxMind Trainer", 
    page_icon="🧠", 
    layout="centered",
    initial_sidebar_state="auto"
)

# Initialize app
initialize_state()
apply_theme()

# Mobile view toggle in sidebar
with st.sidebar:
    st.markdown("# MaxMind")
    mobile_toggle = st.checkbox("📱 Mobile View", value=st.session_state.get("mobile_view", False))
    st.session_state["mobile_view"] = mobile_toggle
    
    # Navigation
    page = st.radio("Navigate", [
        "Dashboard", "Spaced Review", "N-Back", "Task Switching", 
        "Complex Span", "Go/No-Go", "Processing Speed", 
        "Mental Math", "Writing", "Forecasts", "Settings"
    ])
    st.session_state["page"] = page

# Main app functions
def page_dashboard():
    """Mobile-optimized dashboard"""
    page_header("Dashboard")
    check_daily_reset()
    
    # Get completion status
    completed = get_state()["daily"]["completed"]
    total_activities = len(completed)
    
    # Mobile vs Desktop layout
    mobile_view = st.session_state.get("mobile_view", False)
    
    if mobile_view:
        # Mobile: Single column stack
        st.markdown("### Today's Progress")
        
        due_count = len(due_cards())
        create_mobile_dashboard_card("Review", 1 if completed["review"] else 0, 1, f"{due_count} cards due")
        
        drill_count = sum([completed["nback"], completed["task_switching"], completed["complex_span"], 
                          completed["gng"], completed["processing_speed"]])
        create_mobile_dashboard_card("Drills", drill_count, 5, "cognitive exercises")
        
        learning_count = sum([completed["mental_math"], completed["writing"], completed["forecasts"], 
                             completed["topic_study"], completed["world_model_a"], completed["world_model_b"]])
        create_mobile_dashboard_card("Learning", learning_count, 6, "study sessions")
        
        # Quick action buttons
        st.markdown("### Quick Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Review", use_container_width=True):
                st.session_state["page"] = "Spaced Review"
                st.rerun()
        with col2:
            if st.button("Train N-Back", use_container_width=True):
                st.session_state["page"] = "N-Back"
                st.rerun()
    else:
        # Desktop: Three columns
        st.markdown("### Today's Progress")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            due_count = len(due_cards())
            create_desktop_dashboard_card("Review", 1 if completed["review"] else 0, 1, f"{due_count} cards due")
            if st.button("Start Review", use_container_width=True):
                st.session_state["page"] = "Spaced Review"
                st.rerun()
        
        with col2:
            drill_count = sum([completed["nback"], completed["task_switching"], completed["complex_span"], 
                              completed["gng"], completed["processing_speed"]])
            create_desktop_dashboard_card("Drills", drill_count, 5, "cognitive exercises")
            if st.button("Open Drills", use_container_width=True):
                st.session_state["page"] = "N-Back"
                st.rerun()
        
        with col3:
            learning_count = sum([completed["mental_math"], completed["writing"], completed["forecasts"], 
                                 completed["topic_study"], completed["world_model_a"], completed["world_model_b"]])
            create_desktop_dashboard_card("Learning", learning_count, 6, "study sessions")
    
    # Progress summary
    create_progress_summary(completed, total_activities)
    
    # Reset button
    if st.button("Reset Daily Progress", key="reset_progress"):
        for key in get_state()["daily"]["completed"]:
            get_state()["daily"]["completed"][key] = False
        save_state()
        st.success("Daily progress reset!")
        st.rerun()

def page_review():
    """Mobile-optimized spaced repetition"""
    page_header("Spaced Review")
    mobile_view = st.session_state.get("mobile_view", False)
    
    # Initialize review session
    if "review_queue" not in st.session_state:
        st.session_state["review_queue"] = due_cards()
        st.session_state["current_card"] = None
        st.session_state["show_back"] = False
    
    rq = st.session_state["review_queue"]
    
    # Check if done
    if not rq and not st.session_state.get("current_card"):
        st.success("All cards reviewed for today!")
        mark_completed("review")
        if st.button("Reload Cards", use_container_width=True):
            st.session_state.pop("review_queue", None)
            st.rerun()
        return
    
    # Get current card
    if st.session_state["current_card"] is None and rq:
        st.session_state["current_card"] = rq.pop(0)
        st.session_state["show_back"] = False
    
    card = st.session_state["current_card"]
    if card:
        # Show card front
        create_spaced_repetition_card(card, mobile_view)
        
        # Flip button
        if mobile_view:
            if st.button("Show Answer", key="flip_card", use_container_width=True):
                st.session_state["show_back"] = not st.session_state["show_back"]
        else:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Show Answer", key="flip_card", use_container_width=True):
                    st.session_state["show_back"] = not st.session_state["show_back"]
        
        # Show answer and grading
        if st.session_state["show_back"]:
            create_answer_card(card, mobile_view)
            create_grade_buttons(card, mobile_view, handle_grade)
        
        # Progress indicator
        total_cards = len(st.session_state.get("review_queue", [])) + (1 if card else 0)
        completed_cards = len(due_cards()) - total_cards
        st.caption(f"Progress: {completed_cards}/{len(due_cards())} cards completed")

def handle_grade(card, grade):
    """Handle card grading"""
    # Find the actual card in state and update it
    cards = get_state()["cards"]
    for c in cards:
        if c["id"] == card["id"]:
            schedule(c, grade)
            c.setdefault("history", []).append({"date": today_iso(), "grade": grade})
            break
    
    save_state()
    st.session_state["current_card"] = None
    st.session_state["show_back"] = False
    st.rerun()

def page_nback():
    """Simplified N-Back training"""
    page_header("Dual N-Back")
    
    st.markdown("""
    ### Train your working memory
    Track visual positions AND audio letters. Click when current stimulus matches N steps back.
    """)
    
    # Settings
    col1, col2, col3 = st.columns(3)
    with col1:
        n = st.selectbox("N-Level", [1, 2, 3], index=1)
    with col2:
        trials = st.selectbox("Trials", [15, 20, 30], index=1)
    with col3:
        isi = st.selectbox("Speed (ms)", [1800, 1500, 1200, 900], index=1)
    
    if st.button("Start N-Back Session", use_container_width=True):
        mark_completed("nback")
        st.success(f"N-Back {n} session completed!")
        st.info("Full N-Back implementation coming soon...")

def page_settings():
    """Settings page"""
    page_header("Settings")
    
    state = get_state()
    settings = state["settings"]
    
    # Theme settings
    st.markdown("### Theme")
    col1, col2 = st.columns(2)
    
    with col1:
        dark_mode = st.checkbox("Dark Mode", value=settings.get("darkMode", False))
        settings["darkMode"] = dark_mode
    
    with col2:
        blackout_mode = st.checkbox("Blackout Mode", value=settings.get("blackoutMode", True))
        settings["blackoutMode"] = blackout_mode
    
    # Spaced repetition settings
    st.markdown("### Spaced Repetition")
    col1, col2 = st.columns(2)
    
    with col1:
        new_limit = st.number_input("New cards per day", min_value=1, max_value=50, value=settings.get("newLimit", 10))
        settings["newLimit"] = new_limit
    
    with col2:
        review_limit = st.number_input("Review cards per day", min_value=1, max_value=200, value=settings.get("reviewLimit", 60))
        settings["reviewLimit"] = review_limit
    
    # Mobile settings
    st.markdown("### Mobile")
    mobile_default = st.checkbox("Default to mobile view", value=st.session_state.get("mobile_view", False))
    st.session_state["mobile_view"] = mobile_default
    
    if st.button("Save Settings", use_container_width=True):
        save_state()
        st.success("Settings saved!")
        st.rerun()

# Simple placeholder pages
def page_task_switching():
    page_header("Task Switching")
    st.info("Task switching drill coming soon...")
    if st.button("Mark Complete"):
        mark_completed("task_switching")
        st.success("Marked as complete!")

def page_complex_span():
    page_header("Complex Span")
    st.info("Complex span drill coming soon...")
    if st.button("Mark Complete"):
        mark_completed("complex_span")
        st.success("Marked as complete!")

def page_gng():
    page_header("Go/No-Go")
    st.info("Go/No-Go drill coming soon...")
    if st.button("Mark Complete"):
        mark_completed("gng")
        st.success("Marked as complete!")

def page_processing_speed():
    page_header("Processing Speed")
    st.info("Processing speed drill coming soon...")
    if st.button("Mark Complete"):
        mark_completed("processing_speed")
        st.success("Marked as complete!")

def page_mental_math():
    page_header("Mental Math")
    st.info("Mental math training coming soon...")
    if st.button("Mark Complete"):
        mark_completed("mental_math")
        st.success("Marked as complete!")

def page_writing():
    page_header("Writing")
    st.info("Writing sprints coming soon...")
    if st.button("Mark Complete"):
        mark_completed("writing")
        st.success("Marked as complete!")

def page_forecasts():
    page_header("Forecasting")
    st.info("Forecasting training coming soon...")
    if st.button("Mark Complete"):
        mark_completed("forecasts")
        st.success("Marked as complete!")

# Router
page = st.session_state.get("page", "Dashboard")

if page == "Dashboard":
    page_dashboard()
elif page == "Spaced Review":
    page_review()
elif page == "N-Back":
    page_nback()
elif page == "Task Switching":
    page_task_switching()
elif page == "Complex Span":
    page_complex_span()
elif page == "Go/No-Go":
    page_gng()
elif page == "Processing Speed":
    page_processing_speed()
elif page == "Mental Math":
    page_mental_math()
elif page == "Writing":
    page_writing()
elif page == "Forecasts":
    page_forecasts()
elif page == "Settings":
    page_settings()

# Auto-save and PWA features
save_state()

# PWA Configuration for iPhone home screen
st.markdown("""
<script>
// Add PWA meta tags for iPhone home screen support
if (!document.querySelector('meta[name="apple-mobile-web-app-capable"]')) {
    const head = document.head;
    
    // Apple PWA meta tags
    const appleCap = document.createElement('meta');
    appleCap.name = 'apple-mobile-web-app-capable';
    appleCap.content = 'yes';
    head.appendChild(appleCap);
    
    const appleStatus = document.createElement('meta');
    appleStatus.name = 'apple-mobile-web-app-status-bar-style';
    appleStatus.content = 'black-translucent';
    head.appendChild(appleStatus);
    
    const appleTitle = document.createElement('meta');
    appleTitle.name = 'apple-mobile-web-app-title';
    appleTitle.content = 'MaxMind Trainer';
    head.appendChild(appleTitle);
    
    // Viewport for mobile
    const viewport = document.querySelector('meta[name="viewport"]');
    if (viewport) {
        viewport.content = 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no';
    }
    
    // Theme color
    const themeColor = document.createElement('meta');
    themeColor.name = 'theme-color';
    themeColor.content = '#000000';
    head.appendChild(themeColor);
}
</script>
""", unsafe_allow_html=True)

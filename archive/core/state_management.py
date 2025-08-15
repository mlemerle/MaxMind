"""
State management with persistent storage
"""
import streamlit as st
from core.utils import today_iso, load_default_cards

# Import persistent storage
try:
    from storage import storage
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False

def get_default_state():
    """Get default application state"""
    return {
        "created": today_iso(),
        "settings": {"newLimit": 10, "reviewLimit": 60, "darkMode": False, "blackoutMode": True},
        "cards": load_default_cards(),
        "sessions": {},
        "nbackHistory": [],
        "stroopHistory": [],
        "mmHistory": [],
        "writingSessions": [],
        "forecasts": [],
        "adaptive": {
            "nback": {"skill": 1200.0, "last_level": None, "recent_scores": [], "sessions_today": 0},
            "stroop": {"skill": 1200.0, "last_level": None, "recent_scores": [], "sessions_today": 0},
            "complex_span": {"skill": 1200.0, "last_level": None, "recent_scores": [], "sessions_today": 0},
            "gng": {"skill": 1200.0, "last_level": None, "recent_scores": [], "sessions_today": 0},
            "processing_speed": {"skill": 1200.0, "last_level": None, "recent_scores": [], "sessions_today": 0},
            "K": 32.0,
            "target_min": 0.80,
            "target_max": 0.85,
            "base": 1100.0,
            "step": 50.0,
            "window_size": 5
        },
        "daily": {
            "last_reset": today_iso(),
            "completed": {
                "review": False,
                "nback": False,
                "task_switching": False,
                "complex_span": False,
                "gng": False,
                "processing_speed": False,
                "mental_math": False,
                "writing": False,
                "forecasts": False,
                "world_model_a": False,
                "world_model_b": False,
                "topic_study": False
            },
            "completion_history": {}
        },
        "topic_suggestions": {
            "current_topic": None,
            "study_history": [],
            "mastered_topics": [],
            "suggestion_queue": [],
            "last_suggestion_date": None
        },
        "world_model": {
            "current_tracks": ["probabilistic_reasoning", "systems_complexity"],
            "track_progress": {
                "probabilistic_reasoning": {"lesson": 0, "completed": []},
                "systems_complexity": {"lesson": 0, "completed": []},
                "decision_science": {"lesson": 0, "completed": []},
                "econ_institutions": {"lesson": 0, "completed": []},
                "scientific_foundations": {"lesson": 0, "completed": []},
                "history_data": {"lesson": 0, "completed": []},
                "computational_literacy": {"lesson": 0, "completed": []}
            },
            "lesson_history": []
        }
    }

def load_persistent_state():
    """Load state from persistent storage or initialize default"""
    if STORAGE_AVAILABLE:
        user_id = storage.get_user_id()
        saved_data = storage.load_user_data(user_id)
        if saved_data:
            return saved_data
    return get_default_state()

def save_persistent_state(state_data):
    """Save state to persistent storage"""
    if STORAGE_AVAILABLE:
        user_id = storage.get_user_id()
        storage.save_user_data(user_id, state_data)

def initialize_state():
    """Initialize application state"""
    KEY = "mmt_state_v2"
    if KEY not in st.session_state:
        st.session_state[KEY] = load_persistent_state()

def get_state():
    """Get current application state"""
    KEY = "mmt_state_v2"
    return st.session_state[KEY]

def save_state():
    """Save state both to session and persistent storage"""
    KEY = "mmt_state_v2"
    st.session_state[KEY] = st.session_state[KEY]  # explicit
    save_persistent_state(st.session_state[KEY])

def check_daily_reset():
    """Check if we need to reset daily progress (new day)"""
    state = get_state()
    today = today_iso()
    
    # Initialize daily tracking if it doesn't exist
    if "daily" not in state:
        state["daily"] = {
            "last_reset": today,
            "completed": {
                "review": False,
                "nback": False,
                "task_switching": False,
                "complex_span": False,
                "gng": False,
                "processing_speed": False,
                "mental_math": False,
                "writing": False,
                "forecasts": False,
                "world_model_a": False,
                "world_model_b": False,
                "topic_study": False
            },
            "completion_history": {}
        }
    
    # Initialize completion_history if it doesn't exist
    if "completion_history" not in state["daily"]:
        state["daily"]["completion_history"] = {}
    
    # Reset if it's a new day
    if state["daily"]["last_reset"] != today:
        # Save yesterday's completion before resetting
        yesterday = state["daily"]["last_reset"]
        completed_activities = state["daily"]["completed"]
        total_activities = len(completed_activities)
        completed_count = sum(completed_activities.values())
        completion_percentage = (completed_count / total_activities * 100) if total_activities > 0 else 0
        
        state["daily"]["completion_history"][yesterday] = {
            "completed": completed_activities.copy(),
            "total_count": total_activities,
            "completion_percentage": completion_percentage
        }
        
        # Reset for new day
        for key in state["daily"]["completed"]:
            state["daily"]["completed"][key] = False
        
        state["daily"]["last_reset"] = today
        save_state()

def mark_completed(activity: str):
    """Mark an activity as completed today"""
    check_daily_reset()
    get_state()["daily"]["completed"][activity] = True
    save_state()

def is_completed_today(activity: str) -> bool:
    """Check if an activity is completed today"""
    check_daily_reset()
    return get_state()["daily"]["completed"].get(activity, False)

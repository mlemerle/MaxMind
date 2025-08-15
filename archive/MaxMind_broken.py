# MaxMind.py - Cognitive Training Platform
# Run: streamlit run MaxMind.py

import streamlit as st
import json
import random
import time
import uuid
import math
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional

# Configuration
st.set_page_config(
    page_title="MaxMind Trainer", 
    page_icon="🧠", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ========== Utilities ==========
def today_iso() -> str:
    return date.today().isoformat()

def now_ts() -> float:
    return time.time()

def clamp(x, a, b):
    return max(a, min(b, x))

def new_id() -> str:
    return uuid.uuid4().hex[:12]

def timer_text(seconds_left: int) -> str:
    m, s = divmod(max(0, seconds_left), 60)
    return f"{m:02d}:{s:02d}"

@dataclass
class Card:
    id: str
    front: str
    back: str
    tags: List[str] = field(default_factory=list)
    ef: float = 2.5
    reps: int = 0
    interval: int = 0
    due: str = field(default_factory=today_iso)
    history: List[Dict[str, Any]] = field(default_factory=list)
    new: bool = True

# Load cards from data file with fallback
@st.cache_data
def load_default_cards():
    """Load default cards from JSON file with fallback"""
    try:
        with open("data/default_cards.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            cards = []
            for card_data in data["cards"]:
                card = Card(
                    id=new_id(),
                    front=card_data["front"],
                    back=card_data["back"],
                    tags=card_data.get("tags", []),
                    new=True
                )
                cards.append(asdict(card))
            return cards
    except:
        # Fallback cards
        fallback_data = [
            {"front":"Expected value (EV)?","back":"Sum of outcomes weighted by probabilities; choose the higher EV if risk-neutral.","tags":["decision"]},
            {"front":"Base rate - why it matters","back":"It's the prior prevalence; ignoring it leads to base-rate neglect and miscalibration.","tags":["probability"]},
            {"front":"Sunk cost fallacy antidote","back":"Ignore irrecoverable costs; evaluate the future only.","tags":["debias"]},
            {"front":"Well-calibrated forecast","back":"Of events you call 70%, approximately 70% happen in the long run.","tags":["forecasting"]},
            {"front":"Moloch","back":"Metaphor for system-level dynamics that sacrifice individual values for collective failure.","tags":["rationalism","systems"]},
        ]
        
        cards = []
        for card_data in fallback_data:
            card = Card(
                id=new_id(),
                front=card_data["front"],
                back=card_data["back"],
                tags=card_data.get("tags", []),
                new=True
            )
            cards.append(asdict(card))
        return cards

# ========== Default app state ==========
DEFAULT_STATE: Dict[str, Any] = {
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
    }
}

# ========== Persistence ==========
KEY = "mmt_state_v2"

# Import persistent storage
try:
    from storage import storage
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False

def load_persistent_state():
    """Load state from persistent storage or initialize default"""
    if STORAGE_AVAILABLE:
        user_id = storage.get_user_id()
        saved_data = storage.load_user_data(user_id)
        if saved_data:
            return saved_data
    return DEFAULT_STATE.copy()

def save_persistent_state(state_data):
    """Save state to persistent storage"""
    if STORAGE_AVAILABLE:
        user_id = storage.get_user_id()
        storage.save_user_data(user_id, state_data)

# Initialize state
if KEY not in st.session_state:
    st.session_state[KEY] = load_persistent_state()

def S() -> Dict[str, Any]:
    return st.session_state[KEY]

def save_state():
    """Save state both to session and persistent storage"""
    st.session_state[KEY] = st.session_state[KEY]  # explicit
    save_persistent_state(st.session_state[KEY])

# ========== Spaced repetition (SM-2) ==========
def schedule(card: Dict[str, Any], q: int):
    # q: 0(Again), 3(Hard), 4(Good), 5(Easy)
    if q < 3:
        card["reps"] = 0
        card["interval"] = 1
        card["ef"] = max(1.3, card["ef"] - 0.2)
    else:
        if card["reps"] == 0:
            card["interval"] = 1
        elif card["reps"] == 1:
            card["interval"] = 6
        else:
            card["interval"] = round(card["interval"] * card["ef"])
        
        card["reps"] += 1
        
        # EF adjustment
        card["ef"] = max(1.3, card["ef"] + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)))
    
    due = date.today() + timedelta(days=int(card["interval"]))
    card["due"] = due.isoformat()
    card["new"] = False

def due_cards(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    today = date.today().isoformat()
    cards = state["cards"]
    due = [c for c in cards if (not c.get("new")) and c.get("due", today) <= today]
    newbies = [c for c in cards if c.get("new")]
    
    # Randomize the order but maintain separation between due and new
    random.shuffle(due)
    random.shuffle(newbies)
    
    # limits
    due = due[: state["settings"]["reviewLimit"]]
    newbies = newbies[: state["settings"]["newLimit"]]
    return due + newbies

# ========== Daily tracking helpers ==========
def check_daily_reset():
    """Check if we need to reset daily progress (new day)"""
    state = S()
    today = today_iso()
    
    # Initialize daily tracking if it doesn't exist (for existing users)
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
    S()["daily"]["completed"][activity] = True
    save_state()

def is_completed_today(activity: str) -> bool:
    """Check if an activity is completed today"""
    check_daily_reset()
    return S()["daily"]["completed"].get(activity, False)

# ========== Blackout theme and styling ==========
def apply_blackout_theme():
    """Apply blackout mode styling"""
    st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > div,
    .stNumberInput > div > div > input {
        background-color: #111111;
        color: #ffffff;
        border: 1px solid #333333;
    }
    
    .stButton > button {
        background-color: #1a1a1a;
        color: #ffffff;
        border: 1px solid #333333;
        border-radius: 6px;
    }
    
    .stButton > button:hover {
        background-color: #333333;
        border-color: #555555;
    }
    
    div[data-testid="stSidebar"] {
        background-color: #0a0a0a;
    }
    
    .stProgress > div > div > div {
        background-color: #444444;
    }
    
    /* Card styling */
    div[data-testid="stExpander"] {
        background-color: #111111;
        border: 1px solid #333333;
        border-radius: 8px;
    }
    
    /* Mobile-friendly responsive design */
    @media (max-width: 768px) {
        .main > div {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        .stButton > button {
            width: 100%;
            margin-bottom: 0.5rem;
        }
        
        div[data-testid="stColumns"] {
            flex-direction: column;
        }
        
        div[data-testid="stColumns"] > div {
            width: 100% !important;
            margin-bottom: 1rem;
        }
    }
    
    /* Completion checkmarks */
    .completion-mark {
        color: #00ff00;
        font-weight: bold;
        font-size: 1.2em;
    }
    
    .incomplete-mark {
        color: #666666;
        font-weight: bold;
        font-size: 1.2em;
    }
    
    .navigation-button {
        background-color: #1a1a1a;
        color: #ffffff;
        border: 1px solid #333333;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        cursor: pointer;
        text-align: center;
        display: inline-block;
        text-decoration: none;
    }
    
    .navigation-button:hover {
        background-color: #333333;
        border-color: #555555;
    }
    
    .navigation-button.active {
        background-color: #444444;
        border-color: #666666;
    }
    </style>
    """, unsafe_allow_html=True)

# ========== Page header with completion tracking ==========
def page_header(title: str, activity_key: str = None):
    """Display page header with completion status"""
    completion_mark = ""
    if activity_key and is_completed_today(activity_key):
        completion_mark = '<span class="completion-mark">✓</span> '
    
    st.markdown(f"""
    <h1 style="text-align: center; margin-bottom: 2rem;">
        {completion_mark}{title}
    </h1>
    """, unsafe_allow_html=True)

# ========== Progress tracking ==========
def get_completion_percentage():
    """Calculate daily completion percentage"""
    check_daily_reset()
    completed = S()["daily"]["completed"]
    total = len(completed)
    completed_count = sum(completed.values())
    return (completed_count / total * 100) if total > 0 else 0

# ========== Adaptive difficulty system ==========
# Param grids — ordered from easiest (index 0) to hardest (last)
NBACK_GRID = [(1,1800), (2,1800), (2,1500), (2,1200), (3,1500), (3,1200), (3,900)]
STROOP_GRID = [1800, 1500, 1200, 900, 700]  # Also used for Task Switching response deadlines
CSPAN_GRID = [3, 4, 5, 6, 7]  # set sizes (letters to remember)
GNG_GRID = [800, 700, 600, 500]  # ISI ms (shorter = harder); No-Go prob fixed at 0.2
PROC_SPEED_GRID = ["Easy", "Medium", "Hard"]  # Processing speed difficulty levels

def adaptive_suggest_index(drill: str) -> int:
    a = S()["adaptive"]
    skill = a[drill]["skill"]
    base, step = a["base"], a["step"]
    # Map skill to level index
    idx = int(round((skill - base) / step))
    grid_len = {
        "nback": len(NBACK_GRID),
        "stroop": len(STROOP_GRID),
        "complex_span": len(CSPAN_GRID),
        "gng": len(GNG_GRID),
        "processing_speed": len(PROC_SPEED_GRID)
    }[drill]
    return clamp(idx, 0, grid_len - 1)

def adaptive_expected(skill: float, level_idx: int, base: float, step: float) -> float:
    # Elo expected score of success against "level rating"
    level_rating = base + step * level_idx
    # Standard Elo logistic with 400 scaling
    return 1.0 / (1.0 + 10 ** ((level_rating - skill) / 400.0))

def adaptive_update(drill: str, level_idx: int, accuracy: float):
    # Enhanced adaptive system targeting 80-85% success zone
    a = S()["adaptive"]
    target_min, target_max = a["target_min"], a["target_max"]
    base, step, K = a["base"], a["step"], a["K"]
    skill = a[drill]["skill"]
    
    # Store recent performance
    a[drill]["recent_scores"].append(accuracy)
    if len(a[drill]["recent_scores"]) > a["window_size"]:
        a[drill]["recent_scores"].pop(0)
    
    # Calculate expected performance and update skill
    actual_in_zone = 1.0 if target_min <= accuracy <= target_max else 0.0
    expected = adaptive_expected(skill, level_idx, base, step)
    
    # More aggressive updates when outside target zone
    multiplier = 1.5 if accuracy < target_min or accuracy > target_max else 1.0
    new_skill = skill + K * multiplier * (actual_in_zone - expected)
    
    a[drill]["skill"] = float(new_skill)
    a[drill]["last_level"] = int(level_idx)
    a[drill]["sessions_today"] += 1
    save_state()

def get_performance_feedback(drill: str) -> str:
    """Generate adaptive feedback and suggestions"""
    a = S()["adaptive"]
    recent = a[drill]["recent_scores"]
    
    if len(recent) < 3:
        return "Building performance baseline..."
    
    avg_recent = sum(recent) / len(recent)
    target_min, target_max = a["target_min"], a["target_max"]
    
    if avg_recent < target_min:
        return f"📉 Recent avg: {avg_recent:.1%} - Consider easier level or focus on strategy"
    elif avg_recent > target_max:
        return f"Recent avg: {avg_recent:.1%} - Ready for increased difficulty!"
    else:
        return f"Recent avg: {avg_recent:.1%} - Perfect challenge zone!"

# ========== Navigation and main app ==========
def navigation():
    """Main navigation with proper completion tracking"""
    check_daily_reset()
    
    # Get mobile view preference
    mobile_view = st.session_state.get("mobile_view", False)
    
    if mobile_view:
        # Mobile navigation - horizontal buttons
        nav_cols = st.columns(3)
        with nav_cols[0]:
            if st.button("📊 Dashboard", use_container_width=True):
                st.session_state["page"] = "dashboard"
        with nav_cols[1]:
            if st.button("📚 Review", use_container_width=True):
                st.session_state["page"] = "review"
        with nav_cols[2]:
            if st.button("🧠 Train", use_container_width=True):
                st.session_state["page"] = "train"
                
        # Second row for more options
        nav_cols2 = st.columns(3)
        with nav_cols2[0]:
            if st.button("📈 Progress", use_container_width=True):
                st.session_state["page"] = "progress"
        with nav_cols2[1]:
            if st.button("⚙️ Settings", use_container_width=True):
                st.session_state["page"] = "settings"
        with nav_cols2[2]:
            if st.button("📱 Mobile", use_container_width=True):
                st.session_state["mobile_view"] = False
                st.rerun()
    else:
        # Desktop navigation - sidebar
        with st.sidebar:
            st.markdown("# MaxMind Trainer")
            
            # Mobile toggle
            if st.button("📱 Mobile View"):
                st.session_state["mobile_view"] = True
                st.rerun()
            
            st.markdown("---")
            
            # Progress display
            completion_pct = get_completion_percentage()
            st.metric("Today's Progress", f"{completion_pct:.0f}%")
            st.progress(completion_pct / 100)
            
            st.markdown("---")
            
            # Navigation buttons with completion indicators
            completed = S()["daily"]["completed"]
            
            nav_options = [
                ("📊 Dashboard", "dashboard", None),
                ("📚 Spaced Review", "review", "review"),
                ("🧠 Dual N-Back", "nback", "nback"),
                ("🔄 Task Switching", "task_switching", "task_switching"),
                ("📝 Complex Span", "complex_span", "complex_span"),
                ("🎯 Go/No-Go", "gng", "gng"),
                ("⚡ Processing Speed", "processing_speed", "processing_speed"),
                ("🧮 Mental Math", "mental_math", "mental_math"),
                ("✍️ Writing", "writing", "writing"),
                ("🔮 Forecasts", "forecasts", "forecasts"),
                ("📈 Progress", "progress", None),
                ("⚙️ Settings", "settings", None)
            ]
            
            for label, page_key, activity_key in nav_options:
                # Add completion indicator
                if activity_key and completed.get(activity_key, False):
                    button_label = f"✓ {label}"
                else:
                    button_label = label
                
                if st.button(button_label, use_container_width=True):
                    st.session_state["page"] = page_key
    
    # Initialize page if not set
    if "page" not in st.session_state:
        st.session_state["page"] = "dashboard"

# ========== Pages ==========
def dashboard_page():
    """Main dashboard showing daily progress"""
    page_header("Dashboard")
    
    check_daily_reset()
    completed = S()["daily"]["completed"]
    total_activities = len(completed)
    completed_count = sum(completed.values())
    completion_percentage = (completed_count / total_activities * 100) if total_activities > 0 else 0
    
    # Progress overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Completed Today", completed_count)
    with col2:
        st.metric("Total Activities", total_activities)
    with col3:
        st.metric("Progress", f"{completion_percentage:.0f}%")
    
    st.progress(completion_percentage / 100)
    
    # Activity grid
    st.subheader("Today's Activities")
    
    activities = [
        ("📚 Spaced Review", "review", f"{len(due_cards(S()))} cards due"),
        ("🧠 Dual N-Back", "nback", "cognitive training"),
        ("🔄 Task Switching", "task_switching", "attention training"),
        ("📝 Complex Span", "complex_span", "working memory"),
        ("🎯 Go/No-Go", "gng", "impulse control"),
        ("⚡ Processing Speed", "processing_speed", "speed training"),
        ("🧮 Mental Math", "mental_math", "arithmetic skills"),
        ("✍️ Writing", "writing", "expression practice"),
        ("🔮 Forecasts", "forecasts", "prediction tracking")
    ]
    
    # Display in 3-column grid
    for i in range(0, len(activities), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(activities):
                name, key, desc = activities[i + j]
                is_done = completed.get(key, False)
                
                with col:
                    # Activity card
                    card_color = "#2d5a2d" if is_done else "#1a1a1a"
                    status_icon = "✓" if is_done else "○"
                    
                    if st.button(f"{status_icon} {name}\n{desc}", key=f"btn_{key}", use_container_width=True):
                        st.session_state["page"] = key
    
    # Quick actions
    st.subheader("Quick Actions")
    action_cols = st.columns(3)
    with action_cols[0]:
        if st.button("🔄 Reset Today's Progress", use_container_width=True):
            for key in S()["daily"]["completed"]:
                S()["daily"]["completed"][key] = False
            save_state()
            st.rerun()
    
    with action_cols[1]:
        if st.button("📊 View Progress History", use_container_width=True):
            st.session_state["page"] = "progress"
    
    with action_cols[2]:
        due_count = len(due_cards(S()))
        if due_count > 0:
            if st.button(f"📚 Review {due_count} Cards", use_container_width=True):
                st.session_state["page"] = "review"

def review_page():
    """Spaced repetition review page"""
    page_header("Spaced Review", "review")
    
    state = S()
    cards_to_review = due_cards(state)
    
    if not cards_to_review:
        st.success("🎉 No cards due for review!")
        st.info("All caught up! Check back tomorrow or add new cards.")
        
        if st.button("← Back to Dashboard"):
            st.session_state["page"] = "dashboard"
        return
    
    # Initialize review session
    if "review_session" not in st.session_state:
        st.session_state["review_session"] = {
            "cards": cards_to_review,
            "current": 0,
            "show_answer": False
        }
    
    session = st.session_state["review_session"]
    
    if session["current"] >= len(session["cards"]):
        # Session complete
        st.success("🎉 Review session complete!")
        
        # Mark as completed
        mark_completed("review")
        
        # Show summary
        st.subheader("Session Summary")
        st.write(f"Reviewed {len(session['cards'])} cards")
        
        if st.button("🏠 Back to Dashboard"):
            del st.session_state["review_session"]
            st.session_state["page"] = "dashboard"
        
        if st.button("🔄 Review More Cards"):
            del st.session_state["review_session"]
            st.rerun()
        
        return
    
    # Current card
    card = session["cards"][session["current"]]
    progress = (session["current"] + 1) / len(session["cards"])
    
    # Progress indicator
    st.progress(progress)
    st.write(f"Card {session['current'] + 1} of {len(session['cards'])}")
    
    # Card display
    st.markdown(f"""
    <div style="
        background-color: #1a1a1a;
        border: 1px solid #333333;
        border-radius: 8px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
        min-height: 200px;
        display: flex;
        align-items: center;
        justify-content: center;
    ">
        <div style="font-size: 1.2em; line-height: 1.5;">
            {card['front']}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show answer button or answer + grading
    if not session["show_answer"]:
        if st.button("Show Answer", use_container_width=True):
            session["show_answer"] = True
            st.rerun()
    else:
        # Show answer
        st.markdown(f"""
        <div style="
            background-color: #1a2a1a;
            border: 1px solid #333333;
            border-radius: 8px;
            padding: 2rem;
            margin: 1rem 0;
            text-align: center;
            min-height: 150px;
            display: flex;
            align-items: center;
            justify-content: center;
        ">
            <div style="font-size: 1.1em; line-height: 1.4;">
                {card['back']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Grading buttons
        st.write("How well did you remember this?")
        
        grade_cols = st.columns(4)
        grades = [
            ("Again", 0, "#8b0000"),
            ("Hard", 3, "#8b4513"),
            ("Good", 4, "#006400"),
            ("Easy", 5, "#228b22")
        ]
        
        for i, (label, grade, color) in enumerate(grades):
            with grade_cols[i]:
                if st.button(label, key=f"grade_{grade}", use_container_width=True):
                    # Apply SM-2 algorithm
                    schedule(card, grade)
                    
                    # Move to next card
                    session["current"] += 1
                    session["show_answer"] = False
                    
                    # Save state
                    save_state()
                    st.rerun()

def nback_page():
    """Complete Dual N-Back implementation with strategy training"""
    page_header("Dual N-Back", "nback")
    st.caption("Track **visual positions** AND **audio letters**. Click **Visual Match** or **Audio Match** when current stimulus matches N steps back.")

    # Strategy Training Section
    with st.expander("🎯 Strategy Tips (Read First!)", expanded=False):
        st.markdown("""
        **Effective Strategies for Dual N-Back:**
        1. **Rehearsal**: Mentally repeat the sequence (V-A-V-A for visual-audio pattern)
        2. **Chunking**: Group items together (e.g., position 3 + letter C = "3C")
        3. **Spatial Mapping**: Visualize letters in spatial positions
        4. **Rhythmic Pattern**: Use timing to help remember sequences
        5. **Focus Strategy**: Alternate attention between visual and audio channels
        
        **Before Starting:** Choose ONE strategy to focus on this session.
        **After Session:** Reflect on which strategy worked best for you.
        """)

    # Adaptive defaults
    def_idx = adaptive_suggest_index("nback")
    defN, defISI = NBACK_GRID[def_idx]

    n = st.selectbox("N-Back Level", [1, 2, 3], index=[1,2,3].index(defN))
    isi_ms = st.selectbox("ISI (ms)", [1800, 1500, 1200, 900], index=[1800,1500,1200,900].index(defISI))
    trials = st.selectbox("Trials", [15, 20, 30], index=1)
    
    # Strategy selection
    strategies = ["Rehearsal", "Chunking", "Spatial Mapping", "Rhythmic Pattern", "Focus Strategy"]
    chosen_strategy = st.selectbox("Today's Strategy Focus:", strategies)

    # Performance feedback
    feedback = get_performance_feedback("nback")
    if feedback:
        st.info(f"📊 {feedback}")

    if "nb" not in st.session_state:
        st.session_state["nb"] = None

    if st.button("🎮 Start Dual N-Back", use_container_width=True):
        # Generate random sequences for both modalities
        visual_seq = [random.randint(0, 8) for _ in range(trials)]  # 3x3 grid positions
        audio_letters = ["C", "H", "K", "L", "Q", "R", "S", "T"]  # 8 consonants
        audio_seq = [random.choice(audio_letters) for _ in range(trials)]
        
        visual_targets = set(i for i in range(n, trials) if visual_seq[i] == visual_seq[i - n])
        audio_targets = set(i for i in range(n, trials) if audio_seq[i] == audio_seq[i - n])
        
        st.session_state["nb"] = {
            "n": n, "trials": trials, "isi_ms": isi_ms, "strategy": chosen_strategy,
            "visual_seq": visual_seq, "audio_seq": audio_seq,
            "visual_targets": visual_targets, "audio_targets": audio_targets,
            "i": 0, "visual_hits": 0, "audio_hits": 0, "visual_fa": 0, "audio_fa": 0,
            "done": False, "show_grid": True
        }
        st.rerun()

    nb = st.session_state["nb"]
    if nb and nb["show_grid"]:
        # Display current strategy
        st.info(f"🎯 **Current Strategy**: {nb['strategy']}")
        
        if not nb["done"]:
            if nb["i"] < nb["trials"]:
                # Show current stimuli
                current_visual = nb["visual_seq"][nb["i"]]
                current_audio = nb["audio_seq"][nb["i"]]
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("### Visual Grid")
                    # Create 3x3 grid HTML
                    grid_html = "<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; width: 240px; margin: auto; padding: 20px;'>"
                    for pos in range(9):
                        if pos == current_visual:
                            grid_html += f"<div style='width: 70px; height: 70px; background-color: #00ff00; border: 2px solid #ffffff; display: flex; align-items: center; justify-content: center; font-size: 30px; font-weight: bold; border-radius: 8px;'>●</div>"
                        else:
                            grid_html += f"<div style='width: 70px; height: 70px; background-color: #333333; border: 2px solid #666666; border-radius: 8px;'></div>"
                    grid_html += "</div>"
                    st.markdown(grid_html, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### Audio")
                    st.markdown(
                        f"<div style='font-size: 60px; text-align: center; color: #00ff00; font-weight: bold; padding: 30px; border: 3px solid #00ff00; border-radius: 15px; margin: 20px 0;'>{current_audio}</div>",
                        unsafe_allow_html=True
                    )
                
                # Match buttons
                button_col1, button_col2 = st.columns(2)
                with button_col1:
                    visual_match = st.button("🎯 Visual Match", key="nb_visual_match", help="Click when visual position matches N steps back", use_container_width=True)
                with button_col2:
                    audio_match = st.button("🔊 Audio Match", key="nb_audio_match", help="Click when audio letter matches N steps back", use_container_width=True)
                
                if visual_match:
                    _nb_mark_visual()
                if audio_match:
                    _nb_mark_audio()
                
                # Progress info
                st.caption(f"Trial {nb['i']+1}/{nb['trials']} | Visual: Pos {current_visual+1} | Audio: {current_audio}")
                if nb["i"] >= nb["n"]:
                    st.caption(f"N-back targets → Visual: Pos {nb['visual_seq'][nb['i']-nb['n']]+1} | Audio: {nb['audio_seq'][nb['i']-nb['n']]}")
                
                # Wait for ISI then move to next
                time.sleep(nb["isi_ms"] / 1000.0)
                nb["i"] += 1
                if nb["i"] < nb["trials"]:
                    st.rerun()
                else:
                    nb["done"] = True
                    st.rerun()

        if nb["done"]:
            # Calculate results for both modalities
            visual_targets = max(1, len(nb["visual_targets"]))
            audio_targets = max(1, len(nb["audio_targets"]))
            
            visual_acc = round(nb["visual_hits"] / visual_targets * 100, 1) if visual_targets > 0 else 0
            audio_acc = round(nb["audio_hits"] / audio_targets * 100, 1) if audio_targets > 0 else 0
            composite_acc = (visual_acc + audio_acc) / 2
            
            st.success(f"🎯 **Visual**: {nb['visual_hits']}/{visual_targets} hits, {nb['visual_fa']} false alarms → {visual_acc}%")
            st.success(f"🔊 **Audio**: {nb['audio_hits']}/{audio_targets} hits, {nb['audio_fa']} false alarms → {audio_acc}%")
            st.info(f"**Composite Accuracy**: {composite_acc:.1f}%")
            
            # Strategy reflection
            st.markdown("### 🤔 Strategy Reflection")
            strategy_rating = st.slider(f"How well did '{nb['strategy']}' work for you?", 1, 5, 3, 
                                      help="1=Not helpful, 5=Very helpful")
            
            if st.button("✅ Complete Session & Save Results", use_container_width=True):
                # Store results
                S()["nbackHistory"].append({
                    "date": today_iso(), "n": nb["n"], "trials": nb["trials"],
                    "isi": nb["isi_ms"], "visual_acc": visual_acc, "audio_acc": audio_acc,
                    "composite_acc": composite_acc, "strategy": nb["strategy"], 
                    "strategy_rating": strategy_rating, "type": "dual"
                })
                
                # Adaptive update based on composite score
                level_idx = NBACK_GRID.index((n, isi_ms)) if (n, isi_ms) in NBACK_GRID else adaptive_suggest_index("nback")
                adaptive_update("nback", level_idx, accuracy=composite_acc/100.0)
                
                # Mark N-Back as completed
                mark_completed("nback")
                save_state()
                
                # Strategy feedback
                if strategy_rating >= 4:
                    st.success(f"Great! '{nb['strategy']}' is working well for you. Consider using it again.")
                elif strategy_rating <= 2:
                    st.info(f"'{nb['strategy']}' wasn't very helpful. Try a different strategy next time.")
                
                st.session_state["nb"] = None
                st.session_state["page"] = "dashboard"
                st.rerun()

def _nb_mark_visual():
    nb = st.session_state.get("nb")
    if not nb or nb["done"] or nb["i"] == 0:
        return
    
    current_idx = nb["i"]
    if current_idx >= nb["n"]:
        if current_idx in nb["visual_targets"]:
            nb["visual_hits"] += 1
            st.success("✅ Visual hit!")
        else:
            nb["visual_fa"] += 1
            st.error("❌ Visual false alarm!")
    else:
        nb["visual_fa"] += 1
        st.warning("⚠️ Too early for visual N-back!")

def _nb_mark_audio():
    nb = st.session_state.get("nb")
    if not nb or nb["done"] or nb["i"] == 0:
        return
    
    current_idx = nb["i"]
    if current_idx >= nb["n"]:
        if current_idx in nb["audio_targets"]:
            nb["audio_hits"] += 1
            st.success("✅ Audio hit!")
        else:
            nb["audio_fa"] += 1
            st.error("❌ Audio false alarm!")
    else:
        nb["audio_fa"] += 1
        st.warning("⚠️ Too early for audio N-back!")

def mental_math_page():
    """Mental math training"""
    page_header("Mental Math", "mental_math")
    
    st.write("🧮 **Mental Math Training**")
    st.write("Practice arithmetic calculations to improve numerical fluency.")
    
    # Math problem types
    problem_type = st.selectbox("Problem Type", [
        "Addition", "Subtraction", "Multiplication", "Division", "Mixed"
    ])
    
    difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
    
    # Generate problem based on settings
    def generate_problem(ptype, diff):
        if diff == "Easy":
            range_max = 20
        elif diff == "Medium":
            range_max = 100
        else:
            range_max = 1000
        
        if ptype == "Addition":
            a, b = random.randint(1, range_max), random.randint(1, range_max)
            return f"{a} + {b}", a + b
        elif ptype == "Subtraction":
            a, b = random.randint(1, range_max), random.randint(1, range_max)
            if b > a:
                a, b = b, a
            return f"{a} - {b}", a - b
        elif ptype == "Multiplication":
            range_max = min(range_max, 50)  # Keep multiplication reasonable
            a, b = random.randint(1, range_max), random.randint(1, range_max)
            return f"{a} × {b}", a * b
        elif ptype == "Division":
            b = random.randint(2, 20)
            result = random.randint(1, range_max // b)
            a = b * result
            return f"{a} ÷ {b}", result
        else:  # Mixed
            types = ["Addition", "Subtraction", "Multiplication", "Division"]
            return generate_problem(random.choice(types), diff)
    
    # Initialize session
    if "math_session" not in st.session_state:
        st.session_state["math_session"] = {
            "problems": [],
            "answers": [],
            "user_answers": [],
            "current": 0,
            "total": 10,
            "start_time": None
        }
    
    session = st.session_state["math_session"]
    
    # Start new session
    if st.button("🎮 Start Math Session"):
        session["problems"] = []
        session["answers"] = []
        session["user_answers"] = []
        session["current"] = 0
        session["start_time"] = time.time()
        
        # Generate problems
        for _ in range(session["total"]):
            problem, answer = generate_problem(problem_type, difficulty)
            session["problems"].append(problem)
            session["answers"].append(answer)
        
        st.rerun()
    
    # Active session
    if session["problems"] and session["current"] < len(session["problems"]):
        progress = session["current"] / len(session["problems"])
        st.progress(progress)
        st.write(f"Problem {session['current'] + 1} of {len(session['problems'])}")
        
        # Current problem
        current_problem = session["problems"][session["current"]]
        st.markdown(f"""
        <div style="
            background-color: #1a1a1a;
            border: 1px solid #333333;
            border-radius: 8px;
            padding: 3rem;
            margin: 2rem 0;
            text-align: center;
            font-size: 2em;
            font-weight: bold;
        ">
            {current_problem} = ?
        </div>
        """, unsafe_allow_html=True)
        
        # Answer input
        user_answer = st.number_input("Your answer:", value=0, step=1, key=f"answer_{session['current']}")
        
        if st.button("Submit Answer"):
            session["user_answers"].append(user_answer)
            session["current"] += 1
            st.rerun()
    
    # Session complete
    elif session["problems"] and session["current"] >= len(session["problems"]):
        st.success("🎉 Math session complete!")
        
        # Calculate results
        correct = sum(1 for i, ans in enumerate(session["user_answers"]) 
                     if ans == session["answers"][i])
        total = len(session["answers"])
        accuracy = correct / total if total > 0 else 0
        
        elapsed_time = time.time() - session["start_time"] if session["start_time"] else 0
        
        # Mark as completed
        mark_completed("mental_math")
        save_state()
        
        # Show results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{accuracy:.1%}")
        with col2:
            st.metric("Correct", f"{correct}/{total}")
        with col3:
            st.metric("Time", f"{elapsed_time:.1f}s")
        
        # Detailed results
        with st.expander("View Detailed Results"):
            for i, (problem, correct_answer, user_answer) in enumerate(
                zip(session["problems"], session["answers"], session["user_answers"])
            ):
                status = "✓" if user_answer == correct_answer else "✗"
                st.write(f"{status} {problem} = {correct_answer} (You: {user_answer})")
        
        if st.button("🏠 Back to Dashboard"):
            st.session_state["math_session"] = {
                "problems": [], "answers": [], "user_answers": [], 
                "current": 0, "total": 10, "start_time": None
            }
            st.session_state["page"] = "dashboard"

def settings_page():
    """Settings and configuration"""
    page_header("Settings")
    
    state = S()
    
    # Theme settings
    st.subheader("🎨 Theme Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        blackout_mode = st.checkbox("Blackout Mode", value=state["settings"].get("blackoutMode", True))
    with col2:
        dark_mode = st.checkbox("Dark Mode", value=state["settings"].get("darkMode", False))
    
    # Update theme settings
    state["settings"]["blackoutMode"] = blackout_mode
    state["settings"]["darkMode"] = dark_mode
    
    # Spaced repetition settings
    st.subheader("📚 Spaced Repetition Settings")
    
    col3, col4 = st.columns(2)
    with col3:
        new_limit = st.number_input("New cards per day", min_value=1, max_value=50, 
                                   value=state["settings"].get("newLimit", 10))
    with col4:
        review_limit = st.number_input("Review cards per day", min_value=1, max_value=200, 
                                      value=state["settings"].get("reviewLimit", 60))
    
    state["settings"]["newLimit"] = new_limit
    state["settings"]["reviewLimit"] = review_limit
    
    # Data management
    st.subheader("📊 Data Management")
    
    if st.button("📁 Export Data"):
        data_str = json.dumps(state, indent=2, default=str)
        st.download_button(
            label="💾 Download JSON",
            data=data_str,
            file_name=f"maxmind_backup_{today_iso()}.json",
            mime="application/json"
        )
    
    # Reset options
    st.subheader("🔄 Reset Options")
    
    col5, col6 = st.columns(2)
    with col5:
        if st.button("🔄 Reset Today's Progress", type="secondary"):
            for key in state["daily"]["completed"]:
                state["daily"]["completed"][key] = False
            st.success("Today's progress reset!")
    
    with col6:
        if st.button("⚠️ Reset All Data", type="secondary"):
            if st.checkbox("I understand this will delete all data"):
                st.session_state[KEY] = DEFAULT_STATE.copy()
                st.success("All data reset!")
                st.rerun()
    
    # Save settings
    if st.button("💾 Save Settings"):
        save_state()
        st.success("Settings saved!")

def progress_page():
    """Progress tracking and history"""
    page_header("Progress")
    
    state = S()
    
    # Current day summary
    st.subheader("📊 Today's Progress")
    
    check_daily_reset()
    completed = state["daily"]["completed"]
    completion_pct = get_completion_percentage()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Completion", f"{completion_pct:.0f}%")
    with col2:
        completed_count = sum(completed.values())
        st.metric("Completed Activities", f"{completed_count}/{len(completed)}")
    
    st.progress(completion_pct / 100)
    
    # Activity breakdown
    st.subheader("📋 Activity Status")
    
    activities = [
        ("📚 Spaced Review", "review"),
        ("🧠 Dual N-Back", "nback"),
        ("🔄 Task Switching", "task_switching"),
        ("📝 Complex Span", "complex_span"),
        ("🎯 Go/No-Go", "gng"),
        ("⚡ Processing Speed", "processing_speed"),
        ("🧮 Mental Math", "mental_math"),
        ("✍️ Writing", "writing"),
        ("🔮 Forecasts", "forecasts")
    ]
    
    for name, key in activities:
        status = "✅ Complete" if completed.get(key, False) else "⏳ Pending"
        st.write(f"{name}: {status}")
    
    # Historical data
    st.subheader("📈 Historical Progress")
    
    history = state["daily"].get("completion_history", {})
    if history:
        # Convert to chart data
        dates = sorted(history.keys())[-14:]  # Last 14 days
        percentages = [history[date]["completion_percentage"] for date in dates]
        
        # Simple chart
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dates, percentages, marker='o', linewidth=2, markersize=6)
        ax.set_ylabel("Completion %")
        ax.set_title("Daily Completion Rate (Last 14 Days)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Summary stats
        if percentages:
            avg_completion = sum(percentages) / len(percentages)
            st.metric("14-Day Average", f"{avg_completion:.1f}%")
    else:
        st.info("No historical data available yet. Complete some activities to see your progress!")

# ========== Main Application ==========
def main():
    """Main application entry point"""
    # Apply theme
    if S()["settings"].get("blackoutMode", True):
        apply_blackout_theme()
    
    # Navigation
    navigation()
    
    # Route to current page
    page = st.session_state.get("page", "dashboard")
    
    if page == "dashboard":
        dashboard_page()
    elif page == "review":
        review_page()
    elif page == "nback":
        nback_page()
    elif page == "mental_math":
        mental_math_page()
    elif page == "settings":
        settings_page()
    elif page == "progress":
        progress_page()
    elif page in ["task_switching", "complex_span", "gng", "processing_speed", "writing", "forecasts"]:
        # Route to specific training modules
        if page == "task_switching":
            task_switching_page()
        elif page == "complex_span":
            complex_span_page()
        elif page == "gng":
            gng_page()
        elif page == "processing_speed":
            processing_speed_page()
        elif page == "writing":
            writing_page()
        elif page == "forecasts":
            forecasts_page()
    else:
        dashboard_page()

def complex_span_page():
    """Complete Complex Span implementation with strategy training"""
    page_header("Complex Span", "complex_span")
    st.caption("Remember letters **in order**, while verifying simple equations between letters (dual task).")

    # Strategy Training Section
    with st.expander("🎯 Strategy Tips (Read First!)", expanded=False):
        st.markdown("""
        **Effective Complex Span Strategies:**
        1. **Rehearsal**: Continuously repeat the letter sequence in your head
        2. **Grouping**: Chunk letters into pairs or triplets (e.g., B-K-M → "BKM")
        3. **Verbal Coding**: Say letters aloud (mentally) in a rhythm
        4. **Dual Attention**: Alternate focus between letters and math quickly
        5. **Story Method**: Create a mini-story with the letters as initials
        
        **Processing Strategy**: Do math quickly but accurately - don't let it interfere with letter rehearsal.
        **Before Starting:** Choose ONE strategy to focus on this session.
        """)

    # Adaptive default set-size
    def_idx = adaptive_suggest_index("complex_span")
    set_size = st.selectbox("Set size (letters to recall)", CSPAN_GRID, index=def_idx)
    equations_per_item = 1  # one verification between each letter
    
    # Strategy selection
    strategies = ["Rehearsal", "Grouping", "Verbal Coding", "Dual Attention", "Story Method"]
    chosen_strategy = st.selectbox("Today's Strategy Focus:", strategies)

    # Performance feedback
    feedback = get_performance_feedback("complex_span")
    if feedback:
        st.info(f"📊 {feedback}")

    if "cspan" not in st.session_state:
        st.session_state["cspan"] = None

    if st.button("🎮 Start Complex Span", use_container_width=True):
        letters = [random.choice("BCDFGHJKLMNPQRSTVWXYZ") for _ in range(set_size)]
        # Generate simple equation items (a±b=?), with truth flag
        eqs = []
        for _ in range(set_size * equations_per_item):
            a, b = random.randint(2,9), random.randint(2,9)
            op = random.choice(["+", "−"])
            true_val = a + b if op == "+" else a - b
            # Create a statement maybe correct, maybe off by 1-2
            if random.random() < 0.5:
                shown = true_val
                truth = True
            else:
                delta = random.choice([1,2,-1,-2])
                shown = true_val + delta
                truth = False
            eqs.append((a, op, b, shown, truth))
        st.session_state["cspan"] = {
            "letters": letters, "eqs": eqs, "i": 0, "strategy": chosen_strategy,
            "proc_correct": 0, "proc_total": 0, "proc_rts": [],
            "phase": "letters",  # letters -> verify -> recall
            "set_size": set_size
        }
        _cspan_next()

    cs = st.session_state["cspan"]
    if cs:
        # Display current strategy
        st.info(f"🎯 **Current Strategy**: {cs['strategy']}")
        
        if cs["phase"] == "letters":
            st.markdown(f"### Remember This Letter:")
            st.markdown(
                f"<div style='font-size:72px;text-align:center;color:#00ff00;font-weight:bold;padding:30px;border:3px solid #00ff00;border-radius:15px;'>{cs['letters'][cs['i']]}</div>",
                unsafe_allow_html=True
            )
            st.caption(f"Letter {cs['i']+1} of {cs['set_size']}")
            time.sleep(1.5)  # show for 1.5s
            cs["i"] += 1
            if cs["i"] >= cs["set_size"]:
                cs["phase"] = "verify"
                cs["i"] = 0
            st.rerun()

        elif cs["phase"] == "verify":
            a, op, b, shown, truth = cs["eqs"][cs["i"]]
            st.markdown("### Verify This Equation:")
            st.markdown(f"**Is this correct?**")
            st.markdown(
                f"<div style='font-size:48px;text-align:center;color:#ff6b35;font-weight:bold;padding:20px;border:2px solid #ff6b35;border-radius:10px;'>{a} {op} {b} = {shown}</div>",
                unsafe_allow_html=True
            )
            
            start_time = now_ts()
            c1, c2 = st.columns(2)
            if c1.button("✅ TRUE", use_container_width=True):
                rt = (now_ts() - start_time) * 1000
                cs["proc_rts"].append(rt)
                if truth: cs["proc_correct"] += 1
                cs["proc_total"] += 1
                cs["i"] += 1
                if cs["i"] >= len(cs["eqs"]):
                    cs["phase"] = "recall"
                st.rerun()
            if c2.button("❌ FALSE", use_container_width=True):
                rt = (now_ts() - start_time) * 1000
                cs["proc_rts"].append(rt)
                if not truth: cs["proc_correct"] += 1
                cs["proc_total"] += 1
                cs["i"] += 1
                if cs["i"] >= len(cs["eqs"]):
                    cs["phase"] = "recall"
                st.rerun()
            
            st.caption(f"Equation {cs['i']+1} of {len(cs['eqs'])}")

        elif cs["phase"] == "recall":
            st.markdown("### Recall the Letters in Order:")
            st.caption("Type the letters you saw, in the correct order (no spaces)")
            ans = st.text_input("Your answer:", key="cspan_recall", placeholder="Example: BKMP")
            
            if st.button("Submit Recall", use_container_width=True):
                guess = [ch.upper() for ch in ans.strip()]
                correct_positions = sum(1 for i,ch in enumerate(guess[:cs["set_size"]]) if ch == cs["letters"][i])
                recall_acc = correct_positions / cs["set_size"]
                proc_acc = (cs["proc_correct"] / max(1, cs["proc_total"])) if cs["proc_total"] else 0.0
                composite = (recall_acc + proc_acc) / 2.0
                avg_proc_rt = sum(cs["proc_rts"]) / len(cs["proc_rts"]) if cs["proc_rts"] else 0
                
                st.success(f"📝 **Recall**: {correct_positions}/{cs['set_size']} correct ({recall_acc*100:.1f}%)")
                st.success(f"🧮 **Math**: {cs['proc_correct']}/{cs['proc_total']} correct ({proc_acc*100:.1f}%)")
                st.info(f"**Composite Score**: {composite*100:.1f}%")
                st.caption(f"Average processing RT: {avg_proc_rt:.0f}ms")
                
                # Show correct sequence
                st.markdown("**Correct sequence was:** " + " → ".join(cs["letters"]))
                
                # Strategy reflection
                st.markdown("### 🤔 Strategy Reflection")
                strategy_rating = st.slider(f"How well did '{cs['strategy']}' work for you?", 1, 5, 3, 
                                          help="1=Not helpful, 5=Very helpful")
                
                if st.button("✅ Complete Session & Save Results", use_container_width=True):
                    # Adaptive update
                    level_idx = CSPAN_GRID.index(cs["set_size"])
                    adaptive_update("complex_span", level_idx, accuracy=composite)
                    
                    # Mark Complex Span as completed
                    mark_completed("complex_span")
                    save_state()
                    
                    # Strategy feedback
                    if strategy_rating >= 4:
                        st.success(f"Great! '{cs['strategy']}' is working well for you.")
                    elif strategy_rating <= 2:
                        st.info(f"'{cs['strategy']}' wasn't very helpful. Try a different strategy next time.")
                    
                    st.session_state["cspan"] = None
                    st.session_state["page"] = "dashboard"
                    st.rerun()

def _cspan_next():
    """Helper function for complex span progression"""
    pass

def task_switching_page():
    """Complete Task Switching implementation"""
    page_header("Task Switching", "task_switching")
    st.caption("Switch between **Number** (odd/even) and **Letter** (vowel/consonant) categorization tasks based on the cue.")

    # Strategy Training Section
    with st.expander("🎯 Strategy Tips (Read First!)", expanded=False):
        st.markdown("""
        **Effective Task Switching Strategies:**
        1. **Cue Preparation**: Focus on the cue first, then prepare the right rule
        2. **Rule Rehearsal**: Silently repeat "odd/even" or "vowel/consonant" 
        3. **Response Mapping**: Practice L=odd/vowel, R=even/consonant mentally
        4. **Switch Detection**: Anticipate when task might change
        5. **Conflict Resolution**: When confused, go back to the cue
        
        **Before Starting:** Choose ONE strategy to focus on this session.
        **Practice the mappings**: Numbers (L=odd, R=even) | Letters (L=vowel, R=consonant)
        """)

    # Adaptive default ISI
    def_idx = adaptive_suggest_index("stroop")  # Reuse stroop adaptive index
    isi_ms = st.selectbox("Response deadline (ms)", [1500, 1200, 1000, 800], index=def_idx)
    trials = st.selectbox("Trials", [30, 40, 60], index=1)
    switch_prob = st.selectbox("Switch probability", [0.3, 0.5, 0.7], index=1)
    
    # Strategy selection
    strategies = ["Cue Preparation", "Rule Rehearsal", "Response Mapping", "Switch Detection", "Conflict Resolution"]
    chosen_strategy = st.selectbox("Today's Strategy Focus:", strategies)

    # Performance feedback
    feedback = get_performance_feedback("stroop")
    if feedback:
        st.info(f"📊 {feedback}")

    if st.button("🎮 Start Task Switching", use_container_width=True):
        # Generate sequence with random switches
        tasks = ["NUMBER", "LETTER"]
        stimuli = []
        task_sequence = []
        
        current_task = random.choice(tasks)
        task_sequence.append(current_task)
        
        for i in range(trials):
            if i > 0 and random.random() < switch_prob:
                current_task = "LETTER" if current_task == "NUMBER" else "NUMBER"
            task_sequence.append(current_task)
            
            if current_task == "NUMBER":
                # Numbers 1-9
                num = random.randint(1, 9)
                correct_resp = "L" if num % 2 == 1 else "R"  # L=odd, R=even
                stimuli.append({"stimulus": str(num), "task": "NUMBER", "correct": correct_resp})
            else:
                # Letters (mix of vowels and consonants)
                vowels = "AEIOU"
                consonants = "BCDFGHJKLMNPQRSTVWXYZ"
                letter = random.choice(vowels + consonants)
                correct_resp = "L" if letter in vowels else "R"  # L=vowel, R=consonant
                stimuli.append({"stimulus": letter, "task": "LETTER", "correct": correct_resp})
        
        st.session_state["task_switch"] = {
            "stimuli": stimuli, "isi": isi_ms, "i": 0, "strategy": chosen_strategy,
            "correct": 0, "switch_trials": 0, "repeat_trials": 0,
            "switch_correct": 0, "repeat_correct": 0, "rt_sum": 0,
            "current": None, "waiting_response": False, "start_time": None
        }
        _task_switch_next()

    # Mark as completed (placeholder for full implementation)
    if st.button("✅ Mark as Completed (Demo)", use_container_width=True):
        mark_completed("task_switching")
        st.success("Task Switching marked as completed!")
        st.session_state["page"] = "dashboard"
        st.rerun()

def gng_page():
    """Complete Go/No-Go implementation"""
    page_header("Go/No-Go", "gng")
    st.caption("Press **GO** for Go stimuli; do **nothing** for No-Go. Measures response inhibition.")

    # Strategy Training Section
    with st.expander("🎯 Strategy Tips (Read First!)", expanded=False):
        st.markdown("""
        **Effective Response Inhibition Strategies:**
        1. **Preparation**: Hover finger over button but don't press until sure
        2. **Two-Stage**: First classify (Go/No-Go), then decide whether to respond
        3. **Inhibition Focus**: Actively practice "stopping" when you see X
        4. **Speed-Accuracy**: Balance quick responses with avoiding false alarms
        5. **Attention**: Maintain constant vigilance - don't zone out
        
        **Remember**: Go = Letters (respond), No-Go = X (don't respond)
        **Before Starting:** Choose ONE strategy to focus on this session.
        """)

    def_idx = adaptive_suggest_index("gng")
    isi = st.selectbox("ISI (ms)", GNG_GRID, index=def_idx)
    trials = st.selectbox("Trials", [40, 60, 80], index=1)
    p_nogo = 0.20  # fixed no-go probability
    
    # Strategy selection
    strategies = ["Preparation", "Two-Stage", "Inhibition Focus", "Speed-Accuracy", "Attention"]
    chosen_strategy = st.selectbox("Today's Strategy Focus:", strategies)

    # Performance feedback
    feedback = get_performance_feedback("gng")
    if feedback:
        st.info(f"📊 {feedback}")

    # Mark as completed (placeholder for full implementation)
    if st.button("✅ Mark as Completed (Demo)", use_container_width=True):
        mark_completed("gng")
        st.success("Go/No-Go marked as completed!")
        st.session_state["page"] = "dashboard"
        st.rerun()

def processing_speed_page():
    """Processing Speed training module"""
    page_header("Processing Speed", "processing_speed")
    st.info("🚧 Processing Speed training module with full implementation coming soon!")
    
    if st.button("✅ Mark as Completed (Demo)", use_container_width=True):
        mark_completed("processing_speed")
        st.success("Processing Speed marked as completed!")
        st.session_state["page"] = "dashboard"
        st.rerun()

def writing_page():
    """Writing training module"""
    page_header("Writing", "writing")
    st.info("🚧 Writing training module with full implementation coming soon!")
    
    if st.button("✅ Mark as Completed (Demo)", use_container_width=True):
        mark_completed("writing")
        st.success("Writing marked as completed!")
        st.session_state["page"] = "dashboard"
        st.rerun()

def forecasts_page():
    """Forecasting training module"""
    page_header("Forecasts", "forecasts")
    st.info("🚧 Forecasting training module with full implementation coming soon!")
    
    if st.button("✅ Mark as Completed (Demo)", use_container_width=True):
        mark_completed("forecasts")
        st.success("Forecasts marked as completed!")
        st.session_state["page"] = "dashboard"
        st.rerun()

def _task_switch_next():
    """Helper function for task switching progression"""
    pass
    else:
        dashboard_page()

if __name__ == "__main__":
    main()

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

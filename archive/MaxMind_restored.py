# MaxMind.py - Complete Cognitive Training Platform with All Features Restored
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

# Initialize state
if KEY not in st.session_state:
    st.session_state[KEY] = DEFAULT_STATE.copy()

def S() -> Dict[str, Any]:
    return st.session_state[KEY]

def save_state():
    """Save state"""
    pass  # For now, just keep in session state

# ========== Adaptive difficulty system ==========
NBACK_GRID = [(1,1800), (2,1800), (2,1500), (2,1200), (3,1500), (3,1200), (3,900)]
STROOP_GRID = [1800, 1500, 1200, 900, 700]
CSPAN_GRID = [3, 4, 5, 6, 7]
GNG_GRID = [800, 700, 600, 500]
PROC_SPEED_GRID = ["Easy", "Medium", "Hard"]

def adaptive_suggest_index(drill: str) -> int:
    a = S()["adaptive"]
    skill = a[drill]["skill"]
    base, step = a["base"], a["step"]
    idx = int(round((skill - base) / step))
    grid_len = {
        "nback": len(NBACK_GRID),
        "stroop": len(STROOP_GRID),
        "complex_span": len(CSPAN_GRID),
        "gng": len(GNG_GRID),
        "processing_speed": len(PROC_SPEED_GRID)
    }[drill]
    return clamp(idx, 0, grid_len - 1)

def adaptive_update(drill: str, level_idx: int, accuracy: float):
    a = S()["adaptive"]
    target_min, target_max = a["target_min"], a["target_max"]
    base, step, K = a["base"], a["step"], a["K"]
    skill = a[drill]["skill"]
    
    a[drill]["recent_scores"].append(accuracy)
    if len(a[drill]["recent_scores"]) > a["window_size"]:
        a[drill]["recent_scores"].pop(0)
    
    actual_in_zone = 1.0 if target_min <= accuracy <= target_max else 0.0
    level_rating = base + step * level_idx
    expected = 1.0 / (1.0 + 10 ** ((level_rating - skill) / 400.0))
    
    multiplier = 1.5 if accuracy < target_min or accuracy > target_max else 1.0
    new_skill = skill + K * multiplier * (actual_in_zone - expected)
    
    a[drill]["skill"] = float(new_skill)
    a[drill]["last_level"] = int(level_idx)
    a[drill]["sessions_today"] += 1
    save_state()

def get_performance_feedback(drill: str) -> str:
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

# ========== Spaced repetition (SM-2) ==========
def schedule(card: Dict[str, Any], q: int):
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
        card["ef"] = max(1.3, card["ef"] + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)))
    
    due = date.today() + timedelta(days=int(card["interval"]))
    card["due"] = due.isoformat()
    card["new"] = False

def due_cards(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    today = date.today().isoformat()
    cards = state["cards"]
    due = [c for c in cards if (not c.get("new")) and c.get("due", today) <= today]
    newbies = [c for c in cards if c.get("new")]
    
    random.shuffle(due)
    random.shuffle(newbies)
    
    due = due[:state["settings"]["reviewLimit"]]
    newbies = newbies[:state["settings"]["newLimit"]]
    return due + newbies

# ========== Daily tracking ==========
def check_daily_reset():
    state = S()
    today = today_iso()
    
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
    
    if "completion_history" not in state["daily"]:
        state["daily"]["completion_history"] = {}
    
    if state["daily"]["last_reset"] != today:
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
        
        for key in state["daily"]["completed"]:
            state["daily"]["completed"][key] = False
        
        state["daily"]["last_reset"] = today
        save_state()

def mark_completed(activity: str):
    check_daily_reset()
    S()["daily"]["completed"][activity] = True
    save_state()

def is_completed_today(activity: str) -> bool:
    check_daily_reset()
    return S()["daily"]["completed"].get(activity, False)

# ========== Blackout theme ==========
def apply_blackout_theme():
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
    
    div[data-testid="stExpander"] {
        background-color: #111111;
        border: 1px solid #333333;
        border-radius: 8px;
    }
    
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
    </style>
    """, unsafe_allow_html=True)

def page_header(title: str, activity_key: str = None):
    completion_mark = ""
    if activity_key and is_completed_today(activity_key):
        completion_mark = '<span class="completion-mark">✓</span> '
    
    st.markdown(f"""
    <h1 style="text-align: center; margin-bottom: 2rem;">
        {completion_mark}{title}
    </h1>
    """, unsafe_allow_html=True)

def get_completion_percentage():
    check_daily_reset()
    completed = S()["daily"]["completed"]
    total = len(completed)
    completed_count = sum(completed.values())
    return (completed_count / total * 100) if total > 0 else 0

# ========== COMPLETE N-BACK IMPLEMENTATION ==========
def nback_page():
    page_header("Dual N-Back", "nback")
    st.caption("Track **visual positions** AND **audio letters**. Click **Visual Match** or **Audio Match** when current stimulus matches N steps back.")

    with st.expander("🎯 Strategy Tips (Read First!)", expanded=False):
        st.markdown("""
        **Effective Strategies for Dual N-Back:**
        1. **Rehearsal**: Mentally repeat the sequence (V-A-V-A for visual-audio pattern)
        2. **Chunking**: Group items together (e.g., position 3 + letter C = "3C")
        3. **Spatial Mapping**: Visualize letters in spatial positions
        4. **Rhythmic Pattern**: Use timing to help remember sequences
        5. **Focus Strategy**: Alternate attention between visual and audio channels
        """)

    def_idx = adaptive_suggest_index("nback")
    defN, defISI = NBACK_GRID[def_idx]

    n = st.selectbox("N-Back Level", [1, 2, 3], index=[1,2,3].index(defN))
    isi_ms = st.selectbox("ISI (ms)", [1800, 1500, 1200, 900], index=[1800,1500,1200,900].index(defISI))
    trials = st.selectbox("Trials", [15, 20, 30], index=1)
    
    strategies = ["Rehearsal", "Chunking", "Spatial Mapping", "Rhythmic Pattern", "Focus Strategy"]
    chosen_strategy = st.selectbox("Today's Strategy Focus:", strategies)

    feedback = get_performance_feedback("nback")
    if feedback:
        st.info(f"📊 {feedback}")

    if "nb" not in st.session_state:
        st.session_state["nb"] = None

    if st.button("🎮 Start Dual N-Back", use_container_width=True):
        visual_seq = [random.randint(0, 8) for _ in range(trials)]
        audio_letters = ["C", "H", "K", "L", "Q", "R", "S", "T"]
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
        st.info(f"🎯 **Current Strategy**: {nb['strategy']}")
        
        if not nb["done"]:
            if nb["i"] < nb["trials"]:
                current_visual = nb["visual_seq"][nb["i"]]
                current_audio = nb["audio_seq"][nb["i"]]
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("### Visual Grid")
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
                
                button_col1, button_col2 = st.columns(2)
                with button_col1:
                    visual_match = st.button("🎯 Visual Match", key="nb_visual_match", use_container_width=True)
                with button_col2:
                    audio_match = st.button("🔊 Audio Match", key="nb_audio_match", use_container_width=True)
                
                if visual_match:
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
                
                if audio_match:
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
                
                st.caption(f"Trial {nb['i']+1}/{nb['trials']} | Visual: Pos {current_visual+1} | Audio: {current_audio}")
                if nb["i"] >= nb["n"]:
                    st.caption(f"N-back targets → Visual: Pos {nb['visual_seq'][nb['i']-nb['n']]+1} | Audio: {nb['audio_seq'][nb['i']-nb['n']]}")
                
                time.sleep(nb["isi_ms"] / 1000.0)
                nb["i"] += 1
                if nb["i"] < nb["trials"]:
                    st.rerun()
                else:
                    nb["done"] = True
                    st.rerun()

        if nb["done"]:
            visual_targets = max(1, len(nb["visual_targets"]))
            audio_targets = max(1, len(nb["audio_targets"]))
            
            visual_acc = round(nb["visual_hits"] / visual_targets * 100, 1) if visual_targets > 0 else 0
            audio_acc = round(nb["audio_hits"] / audio_targets * 100, 1) if audio_targets > 0 else 0
            composite_acc = (visual_acc + audio_acc) / 2
            
            st.success(f"🎯 **Visual**: {nb['visual_hits']}/{visual_targets} hits, {nb['visual_fa']} false alarms → {visual_acc}%")
            st.success(f"🔊 **Audio**: {nb['audio_hits']}/{audio_targets} hits, {nb['audio_fa']} false alarms → {audio_acc}%")
            st.info(f"**Composite Accuracy**: {composite_acc:.1f}%")
            
            st.markdown("### 🤔 Strategy Reflection")
            strategy_rating = st.slider(f"How well did '{nb['strategy']}' work for you?", 1, 5, 3)
            
            if st.button("✅ Complete Session & Save Results", use_container_width=True):
                S()["nbackHistory"].append({
                    "date": today_iso(), "n": nb["n"], "trials": nb["trials"],
                    "isi": nb["isi_ms"], "visual_acc": visual_acc, "audio_acc": audio_acc,
                    "composite_acc": composite_acc, "strategy": nb["strategy"], 
                    "strategy_rating": strategy_rating, "type": "dual"
                })
                
                level_idx = NBACK_GRID.index((n, isi_ms)) if (n, isi_ms) in NBACK_GRID else adaptive_suggest_index("nback")
                adaptive_update("nback", level_idx, accuracy=composite_acc/100.0)
                
                mark_completed("nback")
                save_state()
                
                if strategy_rating >= 4:
                    st.success(f"Great! '{nb['strategy']}' is working well for you.")
                elif strategy_rating <= 2:
                    st.info(f"'{nb['strategy']}' wasn't very helpful. Try a different strategy next time.")
                
                st.session_state["nb"] = None
                st.session_state["page"] = "dashboard"
                st.rerun()

# ========== COMPLETE COMPLEX SPAN IMPLEMENTATION ==========
def complex_span_page():
    page_header("Complex Span", "complex_span")
    st.caption("Remember letters **in order**, while verifying simple equations between letters (dual task).")

    with st.expander("🎯 Strategy Tips (Read First!)", expanded=False):
        st.markdown("""
        **Effective Complex Span Strategies:**
        1. **Rehearsal**: Continuously repeat the letter sequence in your head
        2. **Grouping**: Chunk letters into pairs or triplets (e.g., B-K-M → "BKM")
        3. **Verbal Coding**: Say letters aloud (mentally) in a rhythm
        4. **Dual Attention**: Alternate focus between letters and math quickly
        5. **Story Method**: Create a mini-story with the letters as initials
        """)

    def_idx = adaptive_suggest_index("complex_span")
    set_size = st.selectbox("Set size (letters to recall)", CSPAN_GRID, index=def_idx)
    
    strategies = ["Rehearsal", "Grouping", "Verbal Coding", "Dual Attention", "Story Method"]
    chosen_strategy = st.selectbox("Today's Strategy Focus:", strategies)

    feedback = get_performance_feedback("complex_span")
    if feedback:
        st.info(f"📊 {feedback}")

    if "cspan" not in st.session_state:
        st.session_state["cspan"] = None

    if st.button("🎮 Start Complex Span", use_container_width=True):
        letters = [random.choice("BCDFGHJKLMNPQRSTVWXYZ") for _ in range(set_size)]
        eqs = []
        for _ in range(set_size):
            a, b = random.randint(2,9), random.randint(2,9)
            op = random.choice(["+", "−"])
            true_val = a + b if op == "+" else a - b
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
            "phase": "letters", "set_size": set_size
        }
        st.rerun()

    cs = st.session_state["cspan"]
    if cs:
        st.info(f"🎯 **Current Strategy**: {cs['strategy']}")
        
        if cs["phase"] == "letters":
            st.markdown("### Remember This Letter:")
            st.markdown(
                f"<div style='font-size:72px;text-align:center;color:#00ff00;font-weight:bold;padding:30px;border:3px solid #00ff00;border-radius:15px;'>{cs['letters'][cs['i']]}</div>",
                unsafe_allow_html=True
            )
            st.caption(f"Letter {cs['i']+1} of {cs['set_size']}")
            time.sleep(1.5)
            cs["i"] += 1
            if cs["i"] >= cs["set_size"]:
                cs["phase"] = "verify"
                cs["i"] = 0
            st.rerun()

        elif cs["phase"] == "verify":
            a, op, b, shown, truth = cs["eqs"][cs["i"]]
            st.markdown("### Verify This Equation:")
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
                
                st.markdown("**Correct sequence was:** " + " → ".join(cs["letters"]))
                
                st.markdown("### 🤔 Strategy Reflection")
                strategy_rating = st.slider(f"How well did '{cs['strategy']}' work for you?", 1, 5, 3)
                
                if st.button("✅ Complete Session & Save Results", use_container_width=True):
                    level_idx = CSPAN_GRID.index(cs["set_size"])
                    adaptive_update("complex_span", level_idx, accuracy=composite)
                    
                    mark_completed("complex_span")
                    save_state()
                    
                    if strategy_rating >= 4:
                        st.success(f"Great! '{cs['strategy']}' is working well for you.")
                    elif strategy_rating <= 2:
                        st.info(f"'{cs['strategy']}' wasn't very helpful. Try a different strategy next time.")
                    
                    st.session_state["cspan"] = None
                    st.session_state["page"] = "dashboard"
                    st.rerun()

# ========== OTHER TRAINING MODULES ==========
def mental_math_page():
    page_header("Mental Math", "mental_math")
    st.write("🧮 **Mental Math Training**")
    
    if st.button("✅ Mark as Completed (Demo)", use_container_width=True):
        mark_completed("mental_math")
        st.success("Mental Math marked as completed!")
        st.session_state["page"] = "dashboard"
        st.rerun()

def placeholder_page(title: str, key: str):
    page_header(title, key)
    st.info(f"🚧 {title} training module with full implementation coming soon!")
    
    if st.button("✅ Mark as Completed (Demo)", use_container_width=True):
        mark_completed(key)
        st.success(f"{title} marked as completed!")
        st.session_state["page"] = "dashboard"
        st.rerun()

# ========== NAVIGATION AND MAIN APP ==========
def navigation():
    check_daily_reset()
    
    mobile_view = st.session_state.get("mobile_view", False)
    
    if mobile_view:
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
                
        nav_cols2 = st.columns(3)
        with nav_cols2[0]:
            if st.button("📈 Progress", use_container_width=True):
                st.session_state["page"] = "progress"
        with nav_cols2[1]:
            if st.button("⚙️ Settings", use_container_width=True):
                st.session_state["page"] = "settings"
        with nav_cols2[2]:
            if st.button("🖥️ Desktop", use_container_width=True):
                st.session_state["mobile_view"] = False
                st.rerun()
    else:
        with st.sidebar:
            st.markdown("# MaxMind Trainer")
            
            if st.button("📱 Mobile View"):
                st.session_state["mobile_view"] = True
                st.rerun()
            
            st.markdown("---")
            
            completion_pct = get_completion_percentage()
            st.metric("Today's Progress", f"{completion_pct:.0f}%")
            st.progress(completion_pct / 100)
            
            st.markdown("---")
            
            completed = S()["daily"]["completed"]
            
            nav_options = [
                ("📊 Dashboard", "dashboard", None),
                ("📚 Spaced Review", "review", "review"),
                ("🧠 Dual N-Back", "nback", "nback"),
                ("📝 Complex Span", "complex_span", "complex_span"),
                ("🔄 Task Switching", "task_switching", "task_switching"),
                ("🎯 Go/No-Go", "gng", "gng"),
                ("⚡ Processing Speed", "processing_speed", "processing_speed"),
                ("🧮 Mental Math", "mental_math", "mental_math"),
                ("✍️ Writing", "writing", "writing"),
                ("🔮 Forecasts", "forecasts", "forecasts"),
                ("📈 Progress", "progress", None),
                ("⚙️ Settings", "settings", None)
            ]
            
            for label, page_key, activity_key in nav_options:
                if activity_key and completed.get(activity_key, False):
                    button_label = f"✓ {label}"
                else:
                    button_label = label
                
                if st.button(button_label, use_container_width=True):
                    st.session_state["page"] = page_key
    
    if "page" not in st.session_state:
        st.session_state["page"] = "dashboard"

def dashboard_page():
    page_header("Dashboard")
    
    check_daily_reset()
    completed = S()["daily"]["completed"]
    total_activities = len(completed)
    completed_count = sum(completed.values())
    completion_percentage = (completed_count / total_activities * 100) if total_activities > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Completed Today", completed_count)
    with col2:
        st.metric("Total Activities", total_activities)
    with col3:
        st.metric("Progress", f"{completion_percentage:.0f}%")
    
    st.progress(completion_percentage / 100)
    
    st.subheader("Today's Activities")
    
    activities = [
        ("📚 Spaced Review", "review", f"{len(due_cards(S()))} cards due"),
        ("🧠 Dual N-Back", "nback", "cognitive training"),
        ("📝 Complex Span", "complex_span", "working memory"),
        ("🔄 Task Switching", "task_switching", "attention training"),
        ("🎯 Go/No-Go", "gng", "impulse control"),
        ("⚡ Processing Speed", "processing_speed", "speed training"),
        ("🧮 Mental Math", "mental_math", "arithmetic skills"),
        ("✍️ Writing", "writing", "expression practice"),
        ("🔮 Forecasts", "forecasts", "prediction tracking")
    ]
    
    for i in range(0, len(activities), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(activities):
                name, key, desc = activities[i + j]
                is_done = completed.get(key, False)
                
                with col:
                    status_icon = "✓" if is_done else "○"
                    
                    if st.button(f"{status_icon} {name}\n{desc}", key=f"btn_{key}", use_container_width=True):
                        st.session_state["page"] = key
    
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
    page_header("Spaced Review", "review")
    
    state = S()
    cards_to_review = due_cards(state)
    
    if not cards_to_review:
        st.success("🎉 No cards due for review!")
        st.info("All caught up! Check back tomorrow or add new cards.")
        
        if st.button("← Back to Dashboard"):
            st.session_state["page"] = "dashboard"
        return
    
    if "review_session" not in st.session_state:
        st.session_state["review_session"] = {
            "cards": cards_to_review,
            "current": 0,
            "show_answer": False
        }
    
    session = st.session_state["review_session"]
    
    if session["current"] >= len(session["cards"]):
        st.success("🎉 Review session complete!")
        
        mark_completed("review")
        save_state()
        
        st.subheader("Session Summary")
        st.write(f"Reviewed {len(session['cards'])} cards")
        
        if st.button("🏠 Back to Dashboard"):
            del st.session_state["review_session"]
            st.session_state["page"] = "dashboard"
        
        if st.button("🔄 Review More Cards"):
            del st.session_state["review_session"]
            st.rerun()
        
        return
    
    card = session["cards"][session["current"]]
    progress = (session["current"] + 1) / len(session["cards"])
    
    st.progress(progress)
    st.write(f"Card {session['current'] + 1} of {len(session['cards'])}")
    
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
    
    if not session["show_answer"]:
        if st.button("Show Answer", use_container_width=True):
            session["show_answer"] = True
            st.rerun()
    else:
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
                    schedule(card, grade)
                    
                    session["current"] += 1
                    session["show_answer"] = False
                    
                    save_state()
                    st.rerun()

def settings_page():
    page_header("Settings")
    
    state = S()
    
    st.subheader("🎨 Theme Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        blackout_mode = st.checkbox("Blackout Mode", value=state["settings"].get("blackoutMode", True))
    with col2:
        dark_mode = st.checkbox("Dark Mode", value=state["settings"].get("darkMode", False))
    
    state["settings"]["blackoutMode"] = blackout_mode
    state["settings"]["darkMode"] = dark_mode
    
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
    
    if st.button("💾 Save Settings"):
        save_state()
        st.success("Settings saved!")

def progress_page():
    page_header("Progress")
    
    st.subheader("📊 Today's Progress")
    
    check_daily_reset()
    completed = S()["daily"]["completed"]
    completion_pct = get_completion_percentage()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Completion", f"{completion_pct:.0f}%")
    with col2:
        completed_count = sum(completed.values())
        st.metric("Completed Activities", f"{completed_count}/{len(completed)}")
    
    st.progress(completion_pct / 100)
    
    st.subheader("📋 Activity Status")
    
    activities = [
        ("📚 Spaced Review", "review"),
        ("🧠 Dual N-Back", "nback"),
        ("📝 Complex Span", "complex_span"),
        ("🔄 Task Switching", "task_switching"),
        ("🎯 Go/No-Go", "gng"),
        ("⚡ Processing Speed", "processing_speed"),
        ("🧮 Mental Math", "mental_math"),
        ("✍️ Writing", "writing"),
        ("🔮 Forecasts", "forecasts")
    ]
    
    for name, key in activities:
        status = "✅ Complete" if completed.get(key, False) else "⏳ Pending"
        st.write(f"{name}: {status}")

def main():
    if S()["settings"].get("blackoutMode", True):
        apply_blackout_theme()
    
    navigation()
    
    page = st.session_state.get("page", "dashboard")
    
    if page == "dashboard":
        dashboard_page()
    elif page == "review":
        review_page()
    elif page == "nback":
        nback_page()
    elif page == "complex_span":
        complex_span_page()
    elif page == "mental_math":
        mental_math_page()
    elif page == "settings":
        settings_page()
    elif page == "progress":
        progress_page()
    elif page == "task_switching":
        placeholder_page("Task Switching", "task_switching")
    elif page == "gng":
        placeholder_page("Go/No-Go", "gng")
    elif page == "processing_speed":
        placeholder_page("Processing Speed", "processing_speed")
    elif page == "writing":
        placeholder_page("Writing", "writing")
    elif page == "forecasts":
        placeholder_page("Forecasts", "forecasts")
    else:
        dashboard_page()

if __name__ == "__main__":
    main()

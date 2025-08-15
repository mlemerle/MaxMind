# MaxMind.py
# Run: streamlit run MaxMind.py
# Deps: pip install streamlit graphviz matplotlib

from __future__ import annotations
import json, math, random, time, uuid
from dataclasses import dataclass, asdict, field
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
from graphviz import Digraph
import matplotlib.pyplot as plt

# Import persistent storage
try:
    from storage import storage
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False

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

# ========== Data model ==========
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

# Your comprehensive card collection
DEFAULT_SEED: List[Dict[str, Any]] = [
    # Original seed cards
    {"front":"Expected value (EV)?","back":"Sum of outcomes weighted by probabilities; choose the higher EV if risk-neutral.","tags":["decision"]},
    {"front":"Base rate — why it matters","back":"It's the prior prevalence; ignoring it → base-rate neglect & miscalibration.","tags":["probability"]},
    {"front":"Backdoor path (DAG)","back":"Non-causal path X→…←…→Y; block with the right covariates.","tags":["causal"]},
    {"front":"Sunk cost fallacy antidote","back":"Ignore irrecoverable costs; evaluate the future only.","tags":["debias"]},
    {"front":"Fermi first step","back":"Define target precisely; decompose; use reference classes.","tags":["estimation"]},
    {"front":"Well-calibrated forecast","back":"Of events you call 70%, ≈70% happen in the long run.","tags":["forecasting"]},
    
    # Extended rationalist/systems thinking collection
    {"front":"Moloch","back":"Metaphor for any system-level dynamic (competition, incentives, multipolar traps) that sacrifices individual values for collective failure; coined from Ginsberg's Howl and used by Scott Alexander","tags":["rationalism","systems"]},
    {"front":"Malthusian Trap","back":"When population growth outpaces productivity, wealth per capita falls back to subsistence despite short-term gains.","tags":["economics","systems"]},
    {"front":"Bayesian","back":"Epistemic stance that beliefs are probabilistic and should be updated by Bayes' Rule whenever new evidence arrives.","tags":["rationalism","probability"]},
    {"front":"Two-income Trap","back":"As more households rely on two pay-checks, bidding wars for fixed goods (housing, schools) erase the extra income, leaving families no safer.","tags":["economics","systems"]},
    {"front":"Multipolar Trap","back":"Scenario with many agents where unilateral defection is rewarded, cooperation punished, and no single actor can enforce better outcomes (e.g., overfishing, arms races).","tags":["systems","game-theory"]},
    {"front":"Objectivism","back":"Ayn Rand's philosophy: objective reality, reason, rational self-interest, laissez-faire capitalism.","tags":["philosophy","economics"]},
    {"front":"Utilitarianism","back":"Moral theory: action is right iff it maximizes aggregate well-being (classically 'the greatest happiness').","tags":["philosophy","ethics"]},
    {"front":"Orthogonality Thesis","back":"In AI theory: an agent's intelligence level is largely independent of its final goals; any goal can pair with any capability.","tags":["ai-safety","rationalism"]},
    {"front":"Goodhart's Law","back":"When a measure becomes a target, it ceases to be a good measure.","tags":["systems","measurement"]},
    {"front":"Chesterton's Fence","back":"Don't remove an old rule until you understand why it was built.","tags":["decision-making","systems"]},
    {"front":"Pascal's Mugging","back":"Low-probability, astronomically high-payoff scenarios that hijack expected-utility reasoning.","tags":["rationalism","decision"]},
    {"front":"Instrumental Convergence","back":"Diverse goals still imply similar sub-goals (resource acquisition, self-preservation).","tags":["ai-safety","rationalism"]},
    {"front":"Paperclip Maximizer","back":"Thought experiment: mis-aligned super-intelligence turns everything into paperclips; illustrates value mis-specification.","tags":["ai-safety","rationalism"]},
    
    # Systems Thinking
    {"front":"Stock","back":"An accumulation at a point in time (water in a tank, money in a bank).","tags":["systems","thinking"]},
    {"front":"Flow","back":"Rate that adds to or subtracts from a stock (inflow, outflow).","tags":["systems","thinking"]},
    {"front":"Feedback Loop","back":"Circular causality where a change feeds back to influence itself. Reinforcing loops amplify change; Balancing loops resist change, seek equilibrium.","tags":["systems","thinking"]},
    {"front":"Leverage Point","back":"Place in a system where a small shift yields big change; Meadows' famous list ranks parameters < goals < paradigms.","tags":["systems","thinking"]},
    {"front":"Resilience","back":"System's capacity to absorb shock and still function.","tags":["systems","thinking"]},
    {"front":"Emergence","back":"Qualitatively new behaviour appears at higher levels & cannot be fully reduced to parts (Anderson's 'More is Different').","tags":["systems","philosophy"]},
    
    # Cybernetics
    {"front":"Cybernetics","back":"The study of control and communication in the animal and the machine. From Greek κυβερνήτης (steersman, governor, pilot, or rudder).","tags":["cybernetics","systems"]},
    {"front":"Ashby's Law of Requisite Variety","back":"A controller must possess at least as much variety as the system it seeks to regulate.","tags":["cybernetics","systems"]},
    {"front":"Homeostasis","back":"Self-regulating process that keeps a variable near a set-point through negative feedback.","tags":["cybernetics","biology"]},
    
    # Epistemology (Deutsch)
    {"front":"Explanation","back":"Statement about what is there, what it does, and how and why.","tags":["epistemology","philosophy"]},
    {"front":"Fallibilism","back":"The recognition that there are no authoritative sources of knowledge, nor any reliable means of justifying knowledge as true or probable.","tags":["epistemology","philosophy"]},
    {"front":"Good / bad explanation","back":"An explanation that is hard / easy to vary while still accounting for what it purports to account for.","tags":["epistemology","philosophy"]},
    {"front":"The Jump to Universality","back":"Small changes in a system (DNA code, alphabets, computers) can yield unbounded creative potential.","tags":["epistemology","philosophy"]},
    
    # Cognitive Science
    {"front":"Bounded Rationality","back":"Herbert Simon's idea that cognitive limits force humans to 'satisfice' instead of optimise.","tags":["cognitive-science","decision"]},
    {"front":"System 1 / System 2","back":"Kahneman's shorthand: fast, automatic heuristics vs. slow, deliberative reasoning.","tags":["cognitive-science","thinking"]},
    {"front":"Scope Insensitivity","back":"Cognitive bias where intuition scales poorly with magnitude (e.g., pay same to save 2 birds or 2,000).","tags":["cognitive-science","bias"]},
    
    # Game Theory
    {"front":"Nash Equilibrium","back":"Strategy profile where no player can gain by unilaterally deviating.","tags":["game-theory","economics"]},
    {"front":"Pareto Optimal","back":"Outcome where no player can be made better off without making another worse off.","tags":["game-theory","economics"]},
    {"front":"Zero-Sum vs. Non-Zero-Sum","back":"In zero-sum, one's gain equals another's loss; in non-zero, aggregate payoffs can expand or shrink.","tags":["game-theory","economics"]},
    
    # Inspirational Quotes
    {"front":"'Everything that is not forbidden by the laws of nature is achievable—given the right knowledge.'","back":"David Deutsch - expressing the optimistic view that knowledge can overcome any solvable problem.","tags":["quotes","optimism"]},
    {"front":"'Between stimulus and response there is a space… In that space is our power to choose.'","back":"Viktor Frankl - on human agency and the power of conscious choice.","tags":["quotes","psychology"]},
    {"front":"'The most important words a man can say are: I will do better.'","back":"Dalinar, Oathbringer - on continuous improvement and taking responsibility.","tags":["quotes","improvement"]},
    {"front":"'Control is the hallmark of true strength'","back":"A man's emotions are what define him, and control is the hallmark of true strength. To lack feeling is to be dead, but to act on every feeling is to be a child. - Way of Kings","tags":["quotes","self-control"]},
]

def make_cards(seed: List[Dict[str, Any]]) -> List[Card]:
    return [Card(id=new_id(), front=s["front"], back=s["back"], tags=s.get("tags", [])) for s in seed]

# ========== Default app state ==========
DEFAULT_STATE: Dict[str, Any] = {
    "created": today_iso(),
    "settings": {"newLimit": 10, "reviewLimit": 60, "darkMode": False, "blackoutMode": True},
    "cards": [asdict(c) for c in make_cards(DEFAULT_SEED)],
    "sessions": {},
    "nbackHistory": [],
    "stroopHistory": [],
    "mmHistory": [],
    "writingSessions": [],
    "forecasts": [],
    # NEW: adaptive controller state (per-drill Elo skill)
    "adaptive": {
        # skill ~1200 baseline; difficulty is mapped via param grids below
        "nback": {"skill": 1200.0, "last_level": None, "recent_scores": [], "sessions_today": 0},
        "stroop": {"skill": 1200.0, "last_level": None, "recent_scores": [], "sessions_today": 0},
        "complex_span": {"skill": 1200.0, "last_level": None, "recent_scores": [], "sessions_today": 0},
        "gng": {"skill": 1200.0, "last_level": None, "recent_scores": [], "sessions_today": 0},
        "processing_speed": {"skill": 1200.0, "last_level": None, "recent_scores": [], "sessions_today": 0},
        "K": 32.0,        # Elo K-factor
        "target_min": 0.80,  # target range 80-85%
        "target_max": 0.85,
        "base": 1100.0,   # base rating for easiest level
        "step": 50.0,     # rating step per level increment
        "window_size": 5  # sessions to consider for auto-adjustment
    },
    # NEW: daily completion tracking with 60-day history
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
        "completion_history": {}  # date -> {"completed_count": X, "total_count": Y, "percentage": Z}
    },
    # NEW: AI Topic Suggestion System
    "topic_suggestions": {
        "current_topic": None,
        "study_history": [],  # Track what's been studied
        "mastered_topics": [],  # Topics that have been integrated into World Model
        "suggestion_queue": [],  # Pre-generated suggestions
        "last_suggestion_date": None
    },
    # NEW: World-Model Learning system
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

# ========== Persistence ==========
KEY = "mmt_state_v2"

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

def export_json() -> str:
    return json.dumps(st.session_state[KEY], indent=2)

def import_json(txt: str):
    data = json.loads(txt)
    st.session_state[KEY] = data
    save_persistent_state(data)  # Also save to persistent storage
    st.success("Imported data.")
    st.rerun()

# ========== Spaced repetition (SM-2) ==========
def schedule(card: Dict[str, Any], q: int):
    # q: 0(Again), 3(Hard), 4(Good), 5(Easy)
    if q < 3:
        card["reps"] = 0
        card["interval"] = 0
        card["ef"] = max(1.3, card.get("ef", 2.5) - 0.2)
    else:
        ef = card.get("ef", 2.5)
        ef = clamp(ef + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)), 1.3, 2.8)
        card["ef"] = ef
        card["reps"] = int(card.get("reps", 0)) + 1
        if card["reps"] == 1:
            card["interval"] = 1
        elif card["reps"] == 2:
            card["interval"] = 6
        else:
            card["interval"] = int(round(card.get("interval", 1) * ef))
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

def add_card(front: str, back: str, tags: List[str] = None) -> None:
    """Add a new card to the deck"""
    if tags is None:
        tags = []
    
    new_card = Card(
        id=new_id(),
        front=front.strip(),
        back=back.strip(), 
        tags=tags,
        new=True
    )
    
    S()["cards"].append(asdict(new_card))
    save_state()

def remove_card(card_id: str) -> bool:
    """Remove a card by ID, returns True if found and removed"""
    cards = S()["cards"]
    for i, card in enumerate(cards):
        if card["id"] == card_id:
            cards.pop(i)
            save_state()
            return True
    return False

def search_cards(query: str) -> List[Dict[str, Any]]:
    """Search cards by front, back, or tags"""
    query = query.lower().strip()
    if not query:
        return S()["cards"]
    
    results = []
    for card in S()["cards"]:
        # Search in front, back, and tags
        if (query in card["front"].lower() or 
            query in card["back"].lower() or 
            any(query in tag.lower() for tag in card.get("tags", []))):
            results.append(card)
    
    return results

# ========== Adaptive engine (per-drill Elo) ==========
# Param grids — ordered from easiest (index 0) to hardest (last)
NBACK_GRID: List[Tuple[int,int]] = [ (1,1800), (2,1800), (2,1500), (2,1200), (3,1500), (3,1200), (3,900) ]
STROOP_GRID: List[int] = [1800, 1500, 1200, 900, 700]  # Also used for Task Switching response deadlines
CSPAN_GRID: List[int] = [3, 4, 5, 6, 7]  # set sizes (letters to remember)
GNG_GRID: List[int] = [800, 700, 600, 500]  # ISI ms (shorter = harder); No-Go prob fixed at 0.2
PROC_SPEED_GRID: List[str] = ["Easy", "Medium", "Hard"]  # Processing speed difficulty levels

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
        return f"Recent avg: {avg_recent:.1%} - Consider easier level or focus on strategy"
    elif avg_recent > target_max:
        return f"Recent avg: {avg_recent:.1%} - Ready for increased difficulty!"
    else:
        return f"Recent avg: {avg_recent:.1%} - Perfect challenge zone!"

def suggest_difficulty_adjustment(drill: str) -> str:
    """Smart difficulty suggestions based on recent performance"""
    a = S()["adaptive"]
    recent = a[drill]["recent_scores"]
    
    if len(recent) < 3:
        return ""
    
    avg_recent = sum(recent) / len(recent)
    target_min, target_max = a["target_min"], a["target_max"]
    
    if avg_recent < target_min - 0.05:  # More than 5% below target
        return "Suggestion: Try an easier level to build confidence"
    elif avg_recent > target_max + 0.05:  # More than 5% above target
        return "Suggestion: Challenge yourself with a harder level"
    
    return ""

# ========== UI helpers ==========
def page_header(title: str):
    """Clean Apple-style page header with theme support"""
    dark_mode = S().get("settings", {}).get("darkMode", False)
    blackout_mode = S().get("settings", {}).get("blackoutMode", False)
    
    if blackout_mode:
        gradient = "linear-gradient(135deg, #1a1a1a 0%, #000000 100%)"
        shadow = "0 8px 32px rgba(255, 255, 255, 0.1)"
    elif dark_mode:
        gradient = "linear-gradient(135deg, #4c1d95 0%, #312e81 50%, #1e1b4b 100%)"
        shadow = "0 8px 32px rgba(124, 58, 237, 0.3)"
    else:
        gradient = "linear-gradient(135deg, #3b82f6 0%, #2563eb 50%, #1d4ed8 100%)"
        shadow = "0 8px 32px rgba(59, 130, 246, 0.3)"
    
    st.markdown(f"""
    <div style="
        background: {gradient};
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: {shadow};
    ">
        <h1 style="
            color: white;
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.02em;
        ">{title}</h1>
    </div>
    """, unsafe_allow_html=True)

def two_cols(a=0.5, b=0.5):
    return st.columns([a, b])

def mobile_responsive_columns(desktop_cols, mobile_cols=None):
    """Create responsive columns that adapt to mobile screens"""
    # For mobile, default to single column if not specified
    if mobile_cols is None:
        mobile_cols = 1
    
    # Use CSS to detect screen size and adjust layout
    # This is a workaround since Streamlit doesn't have built-in responsive detection
    return st.columns(desktop_cols)

def mobile_metrics_layout(metrics_data):
    """Display metrics in a mobile-friendly layout"""
    # On mobile, display metrics in 2x2 grid instead of 1x4 row
    if len(metrics_data) == 4:
        # Desktop: 4 columns, Mobile: 2x2 grid
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        
        with col1:
            st.metric(metrics_data[0]["label"], metrics_data[0]["value"])
        with col2:
            st.metric(metrics_data[1]["label"], metrics_data[1]["value"])
        with col3:
            st.metric(metrics_data[2]["label"], metrics_data[2]["value"])
        with col4:
            st.metric(metrics_data[3]["label"], metrics_data[3]["value"])
    else:
        # Fallback to standard columns
        cols = st.columns(len(metrics_data))
        for i, metric in enumerate(metrics_data):
            with cols[i]:
                st.metric(metric["label"], metric["value"])

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
                "review": False, "nback": False, "task_switching": False,
                "complex_span": False, "gng": False, "processing_speed": False,
                "mental_math": False, "writing": False, "forecasts": False,
                "world_model_a": False, "world_model_b": False, "topic_study": False
            },
            "completion_history": {}
        }
        save_state()
        return
    
    # Initialize completion_history if it doesn't exist
    if "completion_history" not in state["daily"]:
        state["daily"]["completion_history"] = {}
    
    # Reset if it's a new day
    if state["daily"]["last_reset"] != today:
        # Save yesterday's completion before reset
        yesterday = state["daily"]["last_reset"]
        completed = state["daily"]["completed"]
        total_activities = len(completed)
        completed_count = sum(completed.values())
        percentage = round((completed_count / total_activities) * 100, 1)
        
        state["daily"]["completion_history"][yesterday] = {
            "completed_count": completed_count,
            "total_count": total_activities,
            "percentage": percentage
        }
        
        # Clean up old history (keep only last 60 days)
        from datetime import datetime, timedelta
        cutoff_date = (datetime.fromisoformat(today) - timedelta(days=60)).isoformat()[:10]
        state["daily"]["completion_history"] = {
            date: data for date, data in state["daily"]["completion_history"].items()
            if date >= cutoff_date
        }
        
        # Reset for new day
        state["daily"]["last_reset"] = today
        for key in state["daily"]["completed"]:
            state["daily"]["completed"][key] = False
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

# ========== World-Model Learning Curriculum ==========
WORLD_MODEL_TRACKS = {
    "probabilistic_reasoning": {
        "name": "Probabilistic Reasoning & Stats",
        "lessons": [
            {
                "title": "Base Rates & Bayes' Theorem",
                "content": "The base rate is the prior probability of an event. Bayes' theorem: P(A|B) = P(B|A) × P(A) / P(B). Most errors come from ignoring base rates when updating beliefs.",
                "example": "Medical test: 1% disease rate, 95% accurate test. If positive, actual disease probability is only ~16%, not 95%!",
                "questions": [
                    "Why do base rates matter in medical testing?",
                    "What happens when you ignore prior probabilities?",
                    "How does Bayes' theorem help calibrate beliefs?"
                ],
                "transfer": "Apply to: job interview assessments, investment decisions, or diagnostic reasoning in your field."
            },
            {
                "title": "Causal vs Correlational Thinking",
                "content": "Correlation ≠ causation. Confounding variables can create spurious correlations. Use randomized experiments, natural experiments, or causal diagrams (DAGs) to identify true causal relationships.",
                "example": "Ice cream sales correlate with drowning deaths. Confound: summer weather causes both. Temperature is the true causal variable.",
                "questions": [
                    "What are three ways to test causation?",
                    "How can confounding variables mislead us?",
                    "When should you be skeptical of correlational claims?"
                ],
                "transfer": "Apply to: business metrics, health claims, or policy proposals you encounter."
            },
            {
                "title": "Reference Classes & Outside View",
                "content": "The outside view uses reference classes (similar past cases) for prediction. Inside view focuses on unique details of current case. Outside view typically more accurate for forecasting.",
                "example": "Planning fallacy: projects take longer than expected. Outside view: check how long similar projects actually took. Inside view: focus on this project's specifics.",
                "questions": [
                    "What is a reference class?",
                    "When does outside view beat inside view?",
                    "How do you choose the right reference class?"
                ],
                "transfer": "Apply to: project timeline estimates, startup success rates, or personal goal achievement."
            }
        ]
    },
    "systems_complexity": {
        "name": "Systems & Complexity",
        "lessons": [
            {
                "title": "Stocks, Flows, and Feedback Loops",
                "content": "Stock = accumulation (water in bathtub). Flow = rate of change (faucet/drain). Feedback loops: reinforcing (amplify) vs balancing (stabilize). Systems behavior emerges from structure.",
                "example": "Population (stock) changes via births/immigration (inflow) and deaths/emigration (outflow). Feedback: crowding reduces birth rates.",
                "questions": [
                    "What's the difference between stock and flow?",
                    "How do reinforcing loops create exponential growth?",
                    "Why do balancing loops seek equilibrium?"
                ],
                "transfer": "Apply to: personal finance, team dynamics, or organizational change initiatives."
            },
            {
                "title": "Leverage Points and Intervention",
                "content": "Meadows' leverage points (ascending power): parameters < buffers < regulating loops < self-organization < goals < paradigms < transcending paradigms. Higher leverage = bigger impact with less effort.",
                "example": "Company problems: changing bonuses (parameter) vs changing culture (paradigm). Paradigm shifts have much higher leverage but are harder to implement.",
                "questions": [
                    "What are the highest-leverage intervention points?",
                    "Why are paradigm shifts so powerful?",
                    "How do you identify leverage points in a system?"
                ],
                "transfer": "Apply to: organizational change, personal habit formation, or social problem-solving."
            }
        ]
    },
    "decision_science": {
        "name": "Decision Science",
        "lessons": [
            {
                "title": "Expected Value & Decision Trees",
                "content": "Expected Value = Σ(probability × outcome). Decision trees map choices, uncertainties, and outcomes. Choose the path with highest expected value if risk-neutral.",
                "example": "Job offer: 70% chance $80k, 30% chance $60k. EV = 0.7×80k + 0.3×60k = $74k. Compare to certain $70k offer.",
                "questions": [
                    "How do you calculate expected value?",
                    "When might you choose lower EV option?",
                    "What role does risk tolerance play?"
                ],
                "transfer": "Apply to: career decisions, investment choices, or business strategy options."
            },
            {
                "title": "Calibration & Forecasting",
                "content": "Calibration: if you say 70% confidence, you should be right ~70% of the time. Well-calibrated > overconfident. Track predictions to improve metacognition.",
                "example": "Weather forecaster says 30% rain. Good calibration = it rains ~3 out of 10 times when they say 30%. Most people are overconfident.",
                "questions": [
                    "What does it mean to be well-calibrated?",
                    "How can you improve forecasting accuracy?",
                    "Why is overconfidence problematic?"
                ],
                "transfer": "Apply to: business projections, personal planning, or investment decisions."
            }
        ]
    }
}

def get_current_lesson(track: str) -> Dict[str, Any]:
    """Get the current lesson for a track"""
    progress = S()["world_model"]["track_progress"][track]
    lessons = WORLD_MODEL_TRACKS[track]["lessons"]
    
    if progress["lesson"] >= len(lessons):
        # Cycle back to beginning with higher difficulty
        progress["lesson"] = 0
        save_state()
    
    return lessons[progress["lesson"]]

def advance_lesson(track: str):
    """Move to next lesson in track"""
    progress = S()["world_model"]["track_progress"][track]
    progress["lesson"] += 1
    progress["completed"].append(today_iso())
    save_state()

# ========== AI Topic Suggestion System ==========
TOPIC_KNOWLEDGE_BASE = {
    # Probability & Statistics
    "bayes_theorem": {
        "title": "Bayes' Theorem & Medical Testing",
        "category": "probability", 
        "difficulty": "medium",
        "description": "Learn how to properly update beliefs with new evidence and avoid base rate neglect.",
        "content": """
        **Bayes' Theorem**: P(A|B) = P(B|A) × P(A) / P(B)
        
        **Key Insight**: The posterior probability depends heavily on the prior (base rate).
        
        **Medical Example**: 
        - Disease affects 1% of population (base rate)
        - Test is 95% accurate (sensitivity & specificity)
        - If you test positive, what's the probability you have the disease?
        
        **Common Wrong Answer**: 95%
        **Correct Answer**: ~16%
        
        **Why**: The low base rate means most positive tests are false positives!
        """,
        "questions": [
            "Why does a 95% accurate test not mean 95% chance of disease if positive?",
            "How does base rate affect the interpretation of evidence?",
            "When is it safe to ignore base rates?"
        ],
        "applications": [
            "Medical diagnosis and testing",
            "Criminal justice and evidence",
            "Investment and market analysis",
            "A/B testing interpretation"
        ],
        "prerequisites": [],
        "follow_up": ["conditional_probability", "base_rate_neglect"]
    },
    
    "monte_carlo": {
        "title": "Monte Carlo Methods",
        "category": "computation",
        "difficulty": "hard", 
        "description": "Use random sampling to solve complex probabilistic problems.",
        "content": """
        **Monte Carlo Method**: Use random sampling to approximate solutions to mathematical problems.
        
        **Core Idea**: When analytical solutions are impossible, simulate thousands of random scenarios.
        
        **Example - Estimating π**:
        1. Draw random points in a unit square
        2. Count how many fall inside a quarter circle
        3. Ratio approximates π/4
        
        **Applications**:
        - Financial risk modeling
        - Physics simulations
        - Machine learning (dropout, MCMC)
        - Project timeline estimation
        """,
        "questions": [
            "When should you use Monte Carlo vs analytical methods?",
            "How does sample size affect accuracy?",
            "What are the limitations of Monte Carlo methods?"
        ],
        "applications": [
            "Portfolio risk assessment",
            "Climate modeling",
            "Drug discovery simulations",
            "Game AI decision making"
        ],
        "prerequisites": ["basic_probability"],
        "follow_up": ["markov_chains", "bayesian_inference"]
    },
    
    "network_effects": {
        "title": "Network Effects & Metcalfe's Law", 
        "category": "systems",
        "difficulty": "medium",
        "description": "Understand how network value scales and creates winner-take-all dynamics.",
        "content": """
        **Network Effects**: Product becomes more valuable as more people use it.
        
        **Metcalfe's Law**: Network value ∝ n² (number of connections)
        
        **Types of Network Effects**:
        1. **Direct**: More users = more value (phone networks)
        2. **Indirect**: More users = more complementary products (platforms)
        3. **Data**: More users = better algorithms (search engines)
        4. **Social**: More users = more status/utility (social media)
        
        **Critical Mass**: Point where network effects become self-reinforcing.
        """,
        "questions": [
            "Why do network effects create winner-take-all markets?", 
            "How can late entrants compete against network effects?",
            "What factors determine network effect strength?"
        ],
        "applications": [
            "Platform business strategy",
            "Technology adoption curves", 
            "Social movement dynamics",
            "Standard-setting competitions"
        ],
        "prerequisites": [],
        "follow_up": ["platform_strategy", "switching_costs"]
    },
    
    "cognitive_load": {
        "title": "Cognitive Load Theory",
        "category": "psychology",
        "difficulty": "easy",
        "description": "How working memory limitations affect learning and decision-making.",
        "content": """
        **Cognitive Load Theory**: Working memory has limited capacity (~7±2 items).
        
        **Three Types of Load**:
        1. **Intrinsic**: Inherent difficulty of material
        2. **Extraneous**: Poor presentation/irrelevant info
        3. **Germane**: Processing that builds schemas
        
        **Design Implications**:
        - Minimize extraneous load (clean interfaces)
        - Manage intrinsic load (chunking, scaffolding)
        - Optimize germane load (meaningful practice)
        
        **Miller's Rule**: Chunk information into groups of 7±2 items.
        """,
        "questions": [
            "How can you reduce cognitive load in presentations?",
            "Why does multitasking hurt performance?", 
            "When should you provide more vs less information?"
        ],
        "applications": [
            "Interface design",
            "Educational materials",
            "Training programs",
            "Decision support systems"
        ],
        "prerequisites": [],
        "follow_up": ["dual_process_theory", "attention_management"]
    }
}

def get_daily_topic_suggestion() -> Dict[str, Any]:
    """Generate or retrieve today's topic suggestion"""
    state = S()
    today = today_iso()
    
    # Initialize topic suggestions if needed
    if "topic_suggestions" not in state:
        state["topic_suggestions"] = {
            "current_topic": None,
            "study_history": [],
            "mastered_topics": [],
            "suggestion_queue": [],
            "last_suggestion_date": None
        }
        save_state()
    
    ts = state["topic_suggestions"]
    
    # Generate new suggestion if it's a new day
    if ts["last_suggestion_date"] != today:
        # Get topics not yet studied
        studied_keys = {item["topic_key"] for item in ts["study_history"]}
        available_topics = [key for key in TOPIC_KNOWLEDGE_BASE.keys() if key not in studied_keys]
        
        if not available_topics:
            # Reset if all topics studied - start over with increased difficulty
            available_topics = list(TOPIC_KNOWLEDGE_BASE.keys())
        
        # Simple selection: pick by difficulty progression
        easy_topics = [k for k in available_topics if TOPIC_KNOWLEDGE_BASE[k]["difficulty"] == "easy"]
        medium_topics = [k for k in available_topics if TOPIC_KNOWLEDGE_BASE[k]["difficulty"] == "medium"]
        hard_topics = [k for k in available_topics if TOPIC_KNOWLEDGE_BASE[k]["difficulty"] == "hard"]
        
        # Progress from easy → medium → hard
        study_count = len(ts["study_history"])
        if study_count < 3 and easy_topics:
            topic_key = random.choice(easy_topics)
        elif study_count < 8 and medium_topics:
            topic_key = random.choice(medium_topics) 
        elif hard_topics:
            topic_key = random.choice(hard_topics)
        else:
            topic_key = random.choice(available_topics)
        
        ts["current_topic"] = topic_key
        ts["last_suggestion_date"] = today
        save_state()
    
    current_key = ts["current_topic"]
    if current_key and current_key in TOPIC_KNOWLEDGE_BASE:
        return {
            "key": current_key,
            **TOPIC_KNOWLEDGE_BASE[current_key]
        }
    
    # Fallback
    fallback_key = "bayes_theorem"
    ts["current_topic"] = fallback_key
    save_state()
    return {
        "key": fallback_key,
        **TOPIC_KNOWLEDGE_BASE[fallback_key]
    }

def complete_topic_study(topic_key: str, understanding_rating: int, notes: str = ""):
    """Mark a topic as studied and save progress"""
    state = S()
    ts = state["topic_suggestions"]
    
    study_record = {
        "topic_key": topic_key,
        "date": today_iso(),
        "understanding_rating": understanding_rating,  # 1-5 scale
        "notes": notes,
        "title": TOPIC_KNOWLEDGE_BASE.get(topic_key, {}).get("title", "Unknown Topic")
    }
    
    ts["study_history"].append(study_record)
    
    # If well understood (4-5 rating), consider for World Model integration
    if understanding_rating >= 4:
        ts["mastered_topics"].append(topic_key)
    
    # Mark topic study as completed for the day
    mark_completed("topic_study")
    save_state()

def get_topics_for_world_model_integration() -> List[Dict[str, Any]]:
    """Get mastered topics that can be integrated into World Model tracks"""
    state = S()
    ts = state.get("topic_suggestions", {})
    mastered = ts.get("mastered_topics", [])
    
    integration_candidates = []
    for topic_key in mastered:
        if topic_key in TOPIC_KNOWLEDGE_BASE:
            topic_data = TOPIC_KNOWLEDGE_BASE[topic_key]
            integration_candidates.append({
                "key": topic_key,
                "title": topic_data["title"], 
                "category": topic_data["category"],
                "applications": topic_data["applications"]
            })
    
    return integration_candidates

def get_completion_status() -> Dict[str, bool]:
    """Get completion status for all activities"""
    check_daily_reset()
    return S()["daily"]["completed"].copy()

def generate_60_day_calendar():
    """Generate a 60-day forward-looking calendar with completion tracking"""
    from datetime import datetime, timedelta
    import calendar
    
    today = datetime.fromisoformat(today_iso())
    completion_history = S()["daily"]["completion_history"]
    
    # Generate next 60 days
    days = []
    for i in range(-7, 53):  # Show last week + next 60 days
        date = today + timedelta(days=i)
        date_str = date.isoformat()[:10]
        
        # Get completion data
        if date_str in completion_history:
            completion_data = completion_history[date_str]
        elif date_str == today_iso():
            # Today's current progress
            completed = S()["daily"]["completed"]
            total_activities = len(completed)
            completed_count = sum(completed.values())
            percentage = round((completed_count / total_activities) * 100, 1)
            completion_data = {
                "completed_count": completed_count,
                "total_count": total_activities,
                "percentage": percentage
            }
        else:
            completion_data = None
        
        days.append({
            "date": date,
            "date_str": date_str,
            "is_today": date_str == today_iso(),
            "is_past": date < today,
            "is_future": date > today,
            "completion": completion_data
        })
    
    return days

def render_calendar_grid():
    """Render a visual calendar grid showing completion status"""
    days = generate_60_day_calendar()
    
    st.markdown("### 📅 60-Day Practice Calendar")
    st.caption("Track your daily cognitive training consistency")
    
    # Group by weeks
    weeks = []
    current_week = []
    
    for day in days:
        if len(current_week) == 7:
            weeks.append(current_week)
            current_week = []
        current_week.append(day)
    
    if current_week:
        weeks.append(current_week)
    
    # Render calendar
    for week in weeks:
        cols = st.columns(7)
        for i, day in enumerate(week):
            if i < len(cols):
                with cols[i]:
                    date_obj = day["date"]
                    day_name = date_obj.strftime("%a")[:2]
                    day_num = date_obj.day
                    
                    # Determine color and emoji based on completion
                    if day["completion"]:
                        percentage = day["completion"]["percentage"]
                        if percentage == 100:
                            emoji = "●"
                            color = "green"
                        elif percentage >= 80:
                            emoji = "●"
                            color = "orange"
                        elif percentage >= 50:
                            emoji = "●"
                            color = "orange"
                        else:
                            emoji = "●"
                            color = "red"
                        display_text = f"{emoji}\n{day_name} {day_num}\n{percentage}%"
                    elif day["is_today"]:
                        emoji = "●"
                        color = "blue"
                        display_text = f"{emoji}\n{day_name} {day_num}\nTODAY"
                    elif day["is_future"]:
                        emoji = "○"
                        color = "gray"
                        display_text = f"{emoji}\n{day_name} {day_num}\n—"
                    else:
                        emoji = "○"
                        color = "gray"
                        display_text = f"{emoji}\n{day_name} {day_num}\nMissed"
                    
                    # Create a button-like display
                    st.markdown(
                        f"<div style='text-align: center; padding: 5px; margin: 2px; border: 1px solid #ddd; border-radius: 5px; font-size: 10px; height: 60px; display: flex; flex-direction: column; justify-content: center;'>"
                        f"{display_text}"
                        f"</div>",
                        unsafe_allow_html=True
                    )
    
    # Statistics
    past_days = [d for d in days if d["is_past"] and d["completion"]]
    if past_days:
        total_past = len(past_days)
        perfect_days = len([d for d in past_days if d["completion"]["percentage"] == 100])
        good_days = len([d for d in past_days if d["completion"]["percentage"] >= 80])
        avg_completion = sum(d["completion"]["percentage"] for d in past_days) / total_past
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Perfect Days", f"{perfect_days}/{total_past}")
        with col2:
            st.metric("Good Days (≥80%)", f"{good_days}/{total_past}")
        with col3:
            st.metric("Consistency Rate", f"{(good_days/total_past)*100:.1f}%")
        with col4:
            st.metric("Avg Completion", f"{avg_completion:.1f}%")

# ========== Pages ==========
def page_dashboard():
    page_header("Today")
    check_daily_reset()  # Ensure daily progress is reset if new day
    
    # Automatically integrate mastered topics into spaced repetition
    integrated_count = integrate_mastered_topics()
    if integrated_count > 0:
        st.success(f"Automatically integrated {integrated_count} mastered topics into spaced repetition!")
    
    # Get completion status
    completed = get_completion_status()
    
    # Daily Progress Bar at the top
    total_activities = len(completed)
    completed_count = sum(completed.values())
    progress_pct = int((completed_count / total_activities) * 100)
    
    styles = get_card_styles()
    progress_gradient = "linear-gradient(90deg, #58a6ff 0%, #238636 100%)" if S().get("settings", {}).get("darkMode", False) else "linear-gradient(90deg, #007aff 0%, #00d4ff 100%)"
    background_bar = "#21262d" if S().get("settings", {}).get("darkMode", False) else "#f1f5f9"
    
    st.markdown(f"""
    <div style="
        background: {styles['background']};
        padding: 1.5rem;
        border-radius: 16px;
        border: {styles['border']};
        box-shadow: {styles['shadow']};
        margin-bottom: 2rem;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <div style="font-weight: 600; color: {styles['text_color']};">Today's Progress</div>
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
    
    # Apple-style metrics
    st.markdown("""
    <div style="display: flex; gap: 1rem; margin-bottom: 2rem;">
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        dc = len(due_cards(S()))
        review_check = "✓" if completed["review"] else "○"
        styles = get_card_styles()
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
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem; color: {'#22c55e' if completed['review'] else styles['muted_color']};">{review_check}</div>
            <div style="font-weight: 600; color: {styles['text_color']}; margin-bottom: 0.25rem;">Spaced Review</div>
            <div style="font-size: 2rem; font-weight: 700; color: {styles['accent_color']}; margin-bottom: 0.5rem;">{dc}</div>
            <div style="color: {styles['muted_color']}; font-size: 0.875rem;">cards due</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Review", key="review_btn", use_container_width=True):
            st.session_state["page"] = "Spaced Review"
            st.rerun()
        if st.button("Review Details", key="toggle_review", use_container_width=True):
            st.session_state.show_review_details = not st.session_state.show_review_details
    
    with col2:
        drill_checks = {
            "nback": "✓" if completed["nback"] else "○",
            "task_switching": "✓" if completed["task_switching"] else "○", 
            "complex_span": "✓" if completed["complex_span"] else "○",
            "gng": "✓" if completed["gng"] else "○",
            "processing_speed": "✓" if completed["processing_speed"] else "○"
        }
        completed_drills = sum(1 for v in drill_checks.values() if v == "✓")
        st.markdown(f"""
        <div style="
            background: {styles['background']};
            padding: 1.5rem;
            border-radius: 16px;
            border: {styles['border']};
            box-shadow: {styles['shadow']};
            text-align: center;
        ">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem; color: {'#22c55e' if completed_drills == 5 else styles['muted_color']};">{'✓' if completed_drills == 5 else '○'}</div>
            <div style="font-weight: 600; color: {styles['text_color']}; margin-bottom: 0.25rem;">Cognitive Drills</div>
            <div style="font-size: 2rem; font-weight: 700; color: {styles['accent_color']}; margin-bottom: 0.5rem;">{completed_drills}/5</div>
            <div style="color: {styles['muted_color']}; font-size: 0.875rem;">completed today</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Drills", key="drills_btn", use_container_width=True):
            st.session_state["page"] = "N-Back"
            st.rerun()
        if st.button("Drill Details", key="toggle_drills", use_container_width=True):
            st.session_state.show_drills_details = not st.session_state.show_drills_details
    
    with col3:
        other_checks = {
            "writing": "✓" if completed["writing"] else "○",
            "forecasts": "✓" if completed["forecasts"] else "○",
            "mental_math": "✓" if completed["mental_math"] else "○",
            "topic_study": "✓" if completed["topic_study"] else "○"
        }
        wm_checks = {
            "world_model_a": "✓" if completed["world_model_a"] else "○",
            "world_model_b": "✓" if completed["world_model_b"] else "○"
        }
        learning_completed = sum(1 for v in list(other_checks.values()) + list(wm_checks.values()) if v == "✓")
        all_learning_complete = learning_completed == 6
        st.markdown(f"""
        <div style="
            background: {styles['background']};
            padding: 1.5rem;
            border-radius: 16px;
            border: {styles['border']};
            box-shadow: {styles['shadow']};
            text-align: center;
        ">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem; color: {'#22c55e' if all_learning_complete else styles['muted_color']};">{'✓' if all_learning_complete else '○'}</div>
            <div style="font-weight: 600; color: {styles['text_color']}; margin-bottom: 0.25rem;">Learning</div>
            <div style="font-size: 2rem; font-weight: 700; color: {styles['accent_color']}; margin-bottom: 0.5rem;">{learning_completed}/6</div>
            <div style="color: {styles['muted_color']}; font-size: 0.875rem;">activities done</div>
        </div>
        """, unsafe_allow_html=True)
        
        col_topic, col_world = st.columns(2)
        with col_topic:
            if st.button("Study Topic", key="topic_btn", use_container_width=True):
                st.session_state["page"] = "Topic Study"
                st.rerun()
        with col_world:
            if st.button("World Model", key="world_btn", use_container_width=True):
                st.session_state["page"] = "World Model"
                st.rerun()
        
        if st.button("Learning Details", key="toggle_learning", use_container_width=True):
            st.session_state.show_learning_details = not st.session_state.show_learning_details

    # Initialize expandable sections state
    if "show_review_details" not in st.session_state:
        st.session_state.show_review_details = False
    if "show_drills_details" not in st.session_state:
        st.session_state.show_drills_details = False
    if "show_learning_details" not in st.session_state:
        st.session_state.show_learning_details = False

    # Expandable details for each section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.show_review_details:
            due_cards_list = due_cards(S())
            with st.expander("Spaced Review Tasks", expanded=True):
                if due_cards_list:
                    st.write(f"**{len(due_cards_list)} cards due for review:**")
                    for i, card in enumerate(due_cards_list[:5]):  # Show first 5
                        status = "Done" if completed["review"] else "Pending"
                        card_text = card.get("front", "Unknown card")[:50]
                        if len(card.get("front", "")) > 50:
                            card_text += "..."
                        if st.button(f"{status}: {card_text}", key=f"card_{i}", use_container_width=True):
                            st.session_state["page"] = "Spaced Review"
                            st.rerun()
                    if len(due_cards_list) > 5:
                        st.write(f"... and {len(due_cards_list) - 5} more cards")
                        if st.button("View All Cards", key="view_all_cards", use_container_width=True):
                            st.session_state["page"] = "Spaced Review"
                            st.rerun()
                else:
                    st.write("No cards due for review today!")
                    if st.button("Go to Spaced Review", key="go_spaced_review", use_container_width=True):
                        st.session_state["page"] = "Spaced Review"
                        st.rerun()
                    
    with col2:
        if st.session_state.show_drills_details:
            with st.expander("Cognitive Drill Tasks", expanded=True):
                drill_tasks = {
                    "nback": ("Dual N-Back", "N-Back"),
                    "task_switching": ("Task Switching", "Task Switching"), 
                    "complex_span": ("Complex Span", "Complex Span"),
                    "gng": ("Go/No-Go", "Go/No-Go"),
                    "processing_speed": ("Processing Speed", "Processing Speed")
                }
                for key, (name, page) in drill_tasks.items():
                    status = "Done" if completed[key] else "Pending"
                    if st.button(f"{status}: {name}", key=f"drill_{key}", use_container_width=True):
                        st.session_state["page"] = page
                        st.rerun()
                    
    with col3:
        if st.session_state.show_learning_details:
            with st.expander("Learning Activities", expanded=True):
                learning_tasks = {
                    "topic_study": ("Topic Study", "Topic Study"),
                    "writing": ("Writing Exercise", "Writing"),
                    "forecasts": ("Forecasting", "Forecasts"),
                    "mental_math": ("Mental Math", "Mental Math"),
                    "world_model_a": ("World Model A", "World Model"),
                    "world_model_b": ("World Model B", "World Model")
                }
                for key, (name, page) in learning_tasks.items():
                    status = "Done" if completed[key] else "Pending"
                    if st.button(f"{status}: {name}", key=f"learning_{key}", use_container_width=True):
                        st.session_state["page"] = page
                        st.rerun()

    # Clean reset button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Reset Daily Progress", key="reset_progress", use_container_width=True):
            for key in S()["daily"]["completed"]:
                S()["daily"]["completed"][key] = False
            save_state()
            st.success("Daily progress reset!")
            st.rerun()

    st.markdown("### Adaptive Suggestions (targeting 80-85% accuracy)")
    nb_idx = adaptive_suggest_index("nback")
    ts_idx = adaptive_suggest_index("stroop")  # Reuse for task switching
    cspan_idx = adaptive_suggest_index("complex_span")
    gng_idx = adaptive_suggest_index("gng")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"**Dual N-Back** → N={NBACK_GRID[nb_idx][0]}, ISI={NBACK_GRID[nb_idx][1]}ms")
        nb_feedback = get_performance_feedback("nback")
        if nb_feedback:
            st.caption(nb_feedback)
        nb_suggestion = suggest_difficulty_adjustment("nback")
        if nb_suggestion:
            st.info(nb_suggestion)
            
        st.write(f"**Complex Span** → Set size={CSPAN_GRID[cspan_idx]}")
        cs_feedback = get_performance_feedback("complex_span")
        if cs_feedback:
            st.caption(cs_feedback)
    
    with col_b:
        st.write(f"**Task Switching** → Response time={STROOP_GRID[ts_idx]}ms")
        ts_feedback = get_performance_feedback("stroop")  # Reuse stroop feedback
        if ts_feedback:
            st.caption(ts_feedback)
            
        st.write(f"**Go/No-Go** → ISI={GNG_GRID[gng_idx]}ms")
        gng_feedback = get_performance_feedback("gng")
        if gng_feedback:
            st.caption(gng_feedback)

    # Progress
    st.markdown("### Overall Progress")
    learned = len([c for c in S()["cards"] if not c.get("new")])
    total = len(S()["cards"])
    st.write(f"Cards learned: **{learned}/{total}**")
    if S()["nbackHistory"]:
        recent_nb = S()["nbackHistory"][-5:]
        nb_scores = [h.get('composite_acc', h.get('acc', 0)) for h in recent_nb]
        st.write("Dual N-Back recent acc:",
                 ", ".join(f"{score:.1f}%" for score in nb_scores))
    if S()["stroopHistory"]:
        recent_ts = [h for h in S()["stroopHistory"][-5:] if h.get('type') == 'task_switching']
        if recent_ts:
            st.write("Task Switching recent acc:",
                     ", ".join(f"{h['acc']}%" for h in recent_ts))
    
    # Today's Topic Preview
    if not is_completed_today("topic_study"):
        st.markdown("### Today's Suggested Topic")
        topic = get_daily_topic_suggestion()
        difficulty_symbols = {"easy": "●", "medium": "●", "hard": "●"}
        
        with st.container(border=True):
            col_topic, col_button = st.columns([3, 1])
            with col_topic:
                st.markdown(f"**{topic['title']}** {difficulty_symbols.get(topic['difficulty'], '●')}")
                st.caption(topic['description'])
            with col_button:
                if st.button("Start Study"):
                    st.session_state["page"] = "Topic Study"
                    st.rerun()
    
    # 60-Day Calendar
    render_calendar_grid()

def page_review():
    page_header("Spaced Repetition")
    st.caption("Flip → grade: Again/Hard/Good/Easy (SM-2).")

    if "review_queue" not in st.session_state:
        st.session_state["review_queue"] = [c.copy() for c in due_cards(S())]
        st.session_state["current_card"] = None
        st.session_state["show_back"] = False

    rq = st.session_state["review_queue"]
    if not rq and not st.session_state.get("current_card"):
        st.success("All done for now. Nice work!")
        # Mark review as completed when queue is empty
        mark_completed("review")
        if st.button("Reload due cards"):
            st.session_state.pop("review_queue", None)
            st.rerun()
        return

    if st.session_state["current_card"] is None and rq:
        st.session_state["current_card"] = rq.pop(0)
        st.session_state["show_back"] = False

    c = st.session_state["current_card"]
    if c:
        # Apple-style card display with theme support
        styles = get_card_styles()
        st.markdown(f"""
        <div style="
            background: {styles['background']};
            padding: 2.5rem;
            border-radius: 20px;
            border: {styles['border']};
            box-shadow: {styles['shadow']};
            margin: 2rem 0;
            text-align: center;
            min-height: 200px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
            <div style="
                font-size: 1.25rem;
                font-weight: 600;
                color: {styles['text_color']};
                margin-bottom: 1.5rem;
                line-height: 1.6;
            ">{c["front"]}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Flip button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Show Answer", key="flip_card", use_container_width=True):
                st.session_state["show_back"] = not st.session_state["show_back"]
        
        if st.session_state["show_back"]:
            # Dark mode answer background
            answer_bg = "linear-gradient(135deg, #21262d 0%, #161b22 100%)" if S().get("settings", {}).get("darkMode", False) else "linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)"
            answer_border = "#30363d" if S().get("settings", {}).get("darkMode", False) else "#cbd5e1"
            
            st.markdown(f"""
            <div style="
                background: {answer_bg};
                padding: 2rem;
                border-radius: 16px;
                border: 1px solid {answer_border};
                margin: 1rem 0;
                text-align: center;
            ">
                <div style="
                    font-size: 1.125rem;
                    color: {styles['muted_color']};
                    line-height: 1.6;
                ">{c["back"]}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if c.get("tags"):
                tags_html = " ".join(f'<span style="background: #e2e8f0; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.875rem; color: #64748b; margin: 0.25rem;">{t}</span>' for t in c["tags"])
                st.markdown(f'<div style="text-align: center; margin: 1rem 0;">{tags_html}</div>', unsafe_allow_html=True)

            # Apple-style grade buttons
            st.markdown("### Rate your recall")
            cols = st.columns(4)
            
            button_styles = [
                ("Again", "●", "#ef4444", "Need to study more"),
                ("Hard", "●", "#f59e0b", "Difficult recall"),
                ("Good", "●", "#10b981", "Good recall"),
                ("Easy", "●", "#3b82f6", "Perfect recall")
            ]
            
            for i, (col, (label, emoji, color, desc)) in enumerate(zip(cols, button_styles)):
                with col:
                    if st.button(f"{emoji} {label}", key=f"grade_{i}", help=desc, use_container_width=True):
                        handle_grade(c, [0, 3, 4, 5][i])

        st.caption(f"{len(rq)} left in queue")

def handle_grade(card: Dict[str, Any], q: int):
    true_card = next((x for x in S()["cards"] if x["id"] == card["id"]), None)
    if true_card:
        schedule(true_card, q)
        true_card.setdefault("history", []).append({"date": today_iso(), "q": q})
        save_state()
    st.session_state["current_card"] = None
    st.session_state["show_back"] = False
    st.rerun()

# ----- Card Management -----
def page_card_management():
    page_header("Card Management")
    st.caption("Add, remove, and search your spaced repetition cards.")
    
    tab1, tab2, tab3 = st.tabs(["Add New Card", "Search & Manage", "Statistics"])
    
    with tab1:
        st.markdown("### Add New Card")
        with st.form("add_card_form"):
            front = st.text_area("Front (Question)", height=100, placeholder="What is Bayes' theorem?")
            back = st.text_area("Back (Answer)", height=100, placeholder="P(A|B) = P(B|A) × P(A) / P(B)")
            tags_input = st.text_input("Tags (comma-separated)", placeholder="probability, statistics, bayes")
            
            if st.form_submit_button("Add Card"):
                if front.strip() and back.strip():
                    tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]
                    add_card(front, back, tags)
                    st.success("Card added successfully!")
                    st.rerun()
                else:
                    st.error("Both front and back are required.")
    
    with tab2:
        st.markdown("### Search & Manage Cards")
        search_query = st.text_input("Search cards (front, back, or tags):", placeholder="bayes theorem")
        
        if search_query or st.button("Show All Cards"):
            results = search_cards(search_query)
            
            st.write(f"Found {len(results)} cards")
            
            # Pagination
            items_per_page = 10
            total_pages = (len(results) + items_per_page - 1) // items_per_page
            
            if total_pages > 1:
                page_num = st.selectbox("Page", range(1, total_pages + 1))
                start_idx = (page_num - 1) * items_per_page
                end_idx = start_idx + items_per_page
                page_results = results[start_idx:end_idx]
            else:
                page_results = results
            
            # Display cards
            for i, card in enumerate(page_results):
                with st.expander(f"Card {i+1}: {card['front'][:50]}..."):
                    st.write("**Front:**", card["front"])
                    st.write("**Back:**", card["back"])
                    
                    if card.get("tags"):
                        st.write("**Tags:**", ", ".join(f"`{tag}`" for tag in card["tags"]))
                    
                    # Card stats
                    st.write("**Stats:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Repetitions", card.get("reps", 0))
                    with col2:
                        st.metric("Interval (days)", card.get("interval", 0))
                    with col3:
                        st.metric("Ease Factor", f"{card.get('ef', 2.5):.2f}")
                    
                    # Actions
                    col_edit, col_delete = st.columns(2)
                    
                    with col_edit:
                        if st.button(f"Edit Card {i+1}", key=f"edit_{card['id']}"):
                            st.session_state[f"editing_{card['id']}"] = True
                            st.rerun()
                    
                    with col_delete:
                        if st.button(f"❌ Delete", key=f"delete_{card['id']}", type="secondary"):
                            if remove_card(card["id"]):
                                st.success("Card deleted!")
                                st.rerun()
                    
                    # Edit form (if editing)
                    if st.session_state.get(f"editing_{card['id']}", False):
                        with st.form(f"edit_form_{card['id']}"):
                            st.markdown("**Edit Card:**")
                            new_front = st.text_area("Front", value=card["front"], key=f"front_{card['id']}")
                            new_back = st.text_area("Back", value=card["back"], key=f"back_{card['id']}")
                            current_tags = ", ".join(card.get("tags", []))
                            new_tags = st.text_input("Tags", value=current_tags, key=f"tags_{card['id']}")
                            
                            col_save, col_cancel = st.columns(2)
                            with col_save:
                                if st.form_submit_button("Save Changes"):
                                    # Update card
                                    card["front"] = new_front.strip()
                                    card["back"] = new_back.strip()
                                    card["tags"] = [tag.strip() for tag in new_tags.split(",") if tag.strip()]
                                    save_state()
                                    st.session_state[f"editing_{card['id']}"] = False
                                    st.success("Card updated!")
                                    st.rerun()
                            
                            with col_cancel:
                                if st.form_submit_button("Cancel"):
                                    st.session_state[f"editing_{card['id']}"] = False
                                    st.rerun()
    
    with tab3:
        st.markdown("### Card Statistics")
        cards = S()["cards"]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_cards = len(cards)
            st.metric("Total Cards", total_cards)
        
        with col2:
            new_cards = len([c for c in cards if c.get("new", True)])
            st.metric("New Cards", new_cards)
        
        with col3:
            learning_cards = len([c for c in cards if not c.get("new", True) and c.get("reps", 0) < 3])
            st.metric("Learning", learning_cards)
        
        with col4:
            mature_cards = len([c for c in cards if c.get("reps", 0) >= 3])
            st.metric("Mature", mature_cards)
        
        # Tag distribution
        st.markdown("### Tag Distribution")
        tag_counts = {}
        for card in cards:
            for tag in card.get("tags", []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        if tag_counts:
            sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
            for tag, count in sorted_tags[:10]:  # Top 10 tags
                st.write(f"**{tag}**: {count} cards")
        else:
            st.write("No tags found")

# ----- Topic Study Page -----
def page_topic_study():
    page_header("Daily Topic Study")
    st.caption("Build your knowledge base with AI-suggested topics that integrate into your World Model.")
    
    # Get today's suggested topic
    topic = get_daily_topic_suggestion()
    
    # Check if already completed today
    if is_completed_today("topic_study"):
        st.success("✅ **Topic study completed for today!**")
        st.info("Come back tomorrow for a new topic suggestion.")
        
        # Show today's completed topic
        ts = S()["topic_suggestions"]
        if ts["study_history"]:
            latest = ts["study_history"][-1]
            if latest["date"] == today_iso():
                st.markdown(f"### Today's Topic: {latest['title']}")
                rating = latest["understanding_rating"]
                stars = "*" * rating + "-" * (5 - rating)
                st.write(f"**Your Rating**: {stars} ({rating}/5)")
                if latest["notes"]:
                    st.write(f"**Your Notes**: {latest['notes']}")
        return
    
    # Display today's topic
    st.markdown(f"### Today's Topic: {topic['title']}")
    
    # Topic metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        difficulty_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}
        st.write(f"**Difficulty**: {difficulty_emoji.get(topic['difficulty'], '⚪')} {topic['difficulty'].title()}")
    with col2:
        st.write(f"**Category**: {topic['category'].title()}")
    with col3:
        if topic.get("prerequisites"):
            st.write(f"**Prerequisites**: {', '.join(topic['prerequisites'])}")
        else:
            st.write("**Prerequisites**: None")
    
    st.markdown(f"**Overview**: {topic['description']}")
    
    # Main content
    with st.expander("Study Material", expanded=True):
        st.markdown(topic['content'])
    
    # Self-assessment questions
    with st.expander("🤔 Check Your Understanding"):
        for i, question in enumerate(topic['questions'], 1):
            st.write(f"**{i}.** {question}")
        
        st.info("**Tip**: Try to answer these questions mentally before moving on.")
    
    # Applications
    with st.expander("🔧 Real-World Applications"):
        for app in topic['applications']:
            st.write(f"• {app}")
    
    # Completion form
    st.markdown("### Complete Your Study")
    with st.form("complete_topic_study"):
        understanding = st.slider(
            "How well do you understand this topic?", 
            1, 5, 3,
            help="1=Confused, 2=Somewhat unclear, 3=Basic understanding, 4=Good grasp, 5=Could teach others"
        )
        
        notes = st.text_area(
            "Personal notes or insights (optional)",
            placeholder="Key takeaways, connections to other topics, questions for further study...",
            height=100
        )
        
        if st.form_submit_button("Complete Topic Study"):
            complete_topic_study(topic['key'], understanding, notes)
            st.success("Topic study completed! Great work!")
            st.rerun()
    
    # Study progress
    st.markdown("### Your Progress")
    ts = S()["topic_suggestions"]
    total_studied = len(ts.get("study_history", []))
    total_mastered = len(ts.get("mastered_topics", []))
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Topics Studied", total_studied)
    with col2:
        st.metric("Topics Mastered (4+ rating)", total_mastered)
    
    # Recent study history
    if ts.get("study_history"):
        st.markdown("### Recent Studies")
        recent = ts["study_history"][-5:]  # Last 5 studies
        for study in reversed(recent):
            rating_stars = "*" * study["understanding_rating"] + "-" * (5 - study["understanding_rating"])
            st.write(f"**{study['date']}**: {study['title']} - {rating_stars}")
    
    # Manual integration option
    if st.button("Integrate Mastered Topics into Spaced Repetition"):
        integrated_count = integrate_mastered_topics()
        if integrated_count > 0:
            st.success(f"✅ Integrated {integrated_count} mastered topics into your spaced repetition deck!")
        else:
            st.info("No new mastered topics to integrate.")
        st.rerun()

def handle_grade(card: Dict[str, Any], q: int):
    true_card = next((x for x in S()["cards"] if x["id"] == card["id"]), None)
    if true_card:
        schedule(true_card, q)
        true_card.setdefault("history", []).append({"date": today_iso(), "q": q})
        save_state()
    st.session_state["current_card"] = None
    st.session_state["show_back"] = False
    st.rerun()

# ----- Enhanced Dual N-Back with Strategy Training -----
def page_nback():
    page_header("Dual N-Back (Visual + Audio)")
    st.caption("Track **visual positions** AND **audio letters**. Click **Visual Match** or **Audio Match** when current stimulus matches N steps back.")

    # Strategy Training Section
    with st.expander("Strategy Tips (Read First!)", expanded=False):
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

    n = st.selectbox("N", [1, 2, 3], index=[1,2,3].index(defN))
    isi_ms = st.selectbox("ISI (ms)", [1800, 1500, 1200, 900], index=[1800,1500,1200,900].index(defISI))
    trials = st.selectbox("Trials", [15, 20, 30], index=1)
    
    # Strategy selection
    strategies = ["Rehearsal", "Chunking", "Spatial Mapping", "Rhythmic Pattern", "Focus Strategy"]
    chosen_strategy = st.selectbox("Today's Strategy Focus:", strategies)

    if "nb" not in st.session_state:
        st.session_state["nb"] = None

    if st.button("Start Dual N-Back"):
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
        st.info(f"**Current Strategy**: {nb['strategy']}")
        
        # Display 3x3 grid and audio
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Visual Grid")
            grid_container = st.container()
            
        with col2:
            st.markdown("### Audio")
            audio_container = st.container()
        
        # Match buttons
        button_col1, button_col2 = st.columns(2)
        with button_col1:
            visual_match = st.button("Visual Match", key="nb_visual_match", help="Click when visual position matches N steps back")
        with button_col2:
            audio_match = st.button("🔊 Audio Match", key="nb_audio_match", help="Click when audio letter matches N steps back")
        
        if visual_match:
            _nb_mark_visual()
        if audio_match:
            _nb_mark_audio()
        
        if not nb["done"]:
            if nb["i"] < nb["trials"]:
                # Show current stimuli
                current_visual = nb["visual_seq"][nb["i"]]
                current_audio = nb["audio_seq"][nb["i"]]
                
                with grid_container:
                    # Create 3x3 grid HTML
                    grid_html = "<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 5px; width: 200px; margin: auto;'>"
                    for pos in range(9):
                        if pos == current_visual:
                            grid_html += f"<div style='width: 60px; height: 60px; background-color: #ff4444; border: 2px solid #000; display: flex; align-items: center; justify-content: center; font-size: 24px; font-weight: bold;'>●</div>"
                        else:
                            grid_html += f"<div style='width: 60px; height: 60px; background-color: #f0f0f0; border: 2px solid #000;'></div>"
                    grid_html += "</div>"
                    st.markdown(grid_html, unsafe_allow_html=True)
                
                with audio_container:
                    st.markdown(
                        f"<div style='font-size: 48px; text-align: center; color: #2E8B57; font-weight: bold; padding: 20px; border: 2px solid #2E8B57; border-radius: 10px;'>{current_audio}</div>",
                        unsafe_allow_html=True
                    )
                
                # Progress info
                st.caption(f"Trial {nb['i']+1}/{nb['trials']} | Visual: Pos {current_visual+1} | Audio: {current_audio}")
                if nb["i"] >= n:
                    st.caption(f"N-back targets → Visual: Pos {nb['visual_seq'][nb['i']-n]+1} | Audio: {nb['audio_seq'][nb['i']-n]}")
                
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
            
            st.success(f"**Visual**: {nb['visual_hits']}/{visual_targets} hits, {nb['visual_fa']} false alarms → {visual_acc}%")
            st.success(f"🔊 **Audio**: {nb['audio_hits']}/{audio_targets} hits, {nb['audio_fa']} false alarms → {audio_acc}%")
            st.info(f"**Composite Accuracy**: {composite_acc:.1f}%")
            
            # Strategy reflection
            st.markdown("### 🤔 Strategy Reflection")
            strategy_rating = st.slider(f"How well did '{nb['strategy']}' work for you?", 1, 5, 3, 
                                      help="1=Not helpful, 5=Very helpful")
            
            if st.button("Complete Session & Save Results"):
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
                st.rerun()

    # Academic Research References
    st.markdown("---")
    st.caption("**Research Evidence**: Dual N-Back training has been shown to improve fluid intelligence and working memory capacity (Jaeggi et al., 2008; Au et al., 2015). Studies demonstrate transfer effects to other cognitive tasks and sustained improvements with consistent practice.")

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
        st.warning("Too early for visual N-back!")

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
        st.warning("Too early for audio N-back!")

# ----- Processing Speed Training -----
def page_processing_speed():
    page_header("Processing Speed Training")
    st.caption("Rapid visual search, pattern matching, and symbol coding tasks. Speed and accuracy both matter.")

    # Strategy Training Section
    with st.expander("Strategy Tips (Read First!)", expanded=False):
        st.markdown("""
        **Effective Processing Speed Strategies:**
        1. **Visual Scanning**: Use systematic left-to-right, top-to-bottom search patterns
        2. **Feature Focus**: Focus on distinctive features rather than whole symbols
        3. **Peripheral Vision**: Use your peripheral vision to spot targets quickly
        4. **Rhythm Method**: Develop a steady rhythm rather than rushing randomly
        5. **Chunking**: Group similar items together mentally to process faster
        
        **Before Starting:** Choose ONE strategy to focus on this session.
        **Goal**: Balance speed with accuracy - going too fast hurts performance.
        """)

    # Task selection
    task_type = st.selectbox("Task Type", ["Symbol Search", "Pattern Matching", "Visual Comparison"])
    difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
    duration = st.selectbox("Duration (minutes)", [1, 2, 3], index=1)
    
    # Strategy selection
    strategies = ["Visual Scanning", "Feature Focus", "Peripheral Vision", "Rhythm Method", "Chunking"]
    chosen_strategy = st.selectbox("Today's Strategy Focus:", strategies)

    if "proc_speed" not in st.session_state:
        st.session_state["proc_speed"] = None

    if st.button("Start Processing Speed Training"):
        # Set difficulty parameters
        if difficulty == "Easy":
            grid_size, distractors, time_pressure = 6, 3, 3000
        elif difficulty == "Medium":
            grid_size, distractors, time_pressure = 8, 5, 2500
        else:  # Hard
            grid_size, distractors, time_pressure = 10, 7, 2000
        
        st.session_state["proc_speed"] = {
            "task_type": task_type,
            "difficulty": difficulty,
            "strategy": chosen_strategy,
            "duration": duration,
            "start_time": now_ts(),
            "end_time": now_ts() + (duration * 60),
            "trials_completed": 0,
            "correct_responses": 0,
            "total_rt": 0,
            "current_trial": None,
            "trial_start": None,
            "grid_size": grid_size,
            "distractors": distractors,
            "time_pressure": time_pressure
        }
        _proc_speed_new_trial()

    ps = st.session_state["proc_speed"]
    if ps:
        # Display current strategy and progress
        st.info(f"**Strategy**: {ps['strategy']} | **Task**: {ps['task_type']} ({ps['difficulty']})")
        
        # Time remaining
        time_left = max(0, int(ps["end_time"] - now_ts()))
        st.metric("Time Remaining", timer_text(time_left))
        
        if time_left > 0 and ps["current_trial"]:
            _proc_speed_display_trial()
        elif time_left <= 0:
            _proc_speed_finish()

def _proc_speed_new_trial():
    """Generate a new processing speed trial"""
    ps = st.session_state["proc_speed"]
    if not ps:
        return
    
    task_type = ps["task_type"]
    
    if task_type == "Symbol Search":
        # Generate target symbol and grid of symbols
        symbols = ["◆", "▲", "●", "■", "★", "♦", "▼", "◐", "◑", "◒", "◓", "♠", "♣", "♥"]
        target = random.choice(symbols)
        
        # Create grid with target present 50% of the time
        target_present = random.choice([True, False])
        grid_symbols = []
        
        if target_present:
            # Add target once
            grid_symbols.append(target)
            # Fill rest with distractors
            distractors = [s for s in symbols if s != target]
            grid_symbols.extend(random.choices(distractors, k=ps["grid_size"]-1))
        else:
            # Only distractors
            distractors = [s for s in symbols if s != target]
            grid_symbols = random.choices(distractors, k=ps["grid_size"])
        
        random.shuffle(grid_symbols)
        
        ps["current_trial"] = {
            "type": "symbol_search",
            "target": target,
            "grid": grid_symbols,
            "target_present": target_present
        }
    
    elif task_type == "Pattern Matching":
        # Generate pattern and comparison options
        patterns = ["╋", "╬", "┼", "╪", "┿", "╂", "╄", "╆", "╊", "╈"]
        target_pattern = random.choice(patterns)
        
        # Create options (one match, rest different)
        options = [target_pattern]  # Correct match
        distractors = [p for p in patterns if p != target_pattern]
        options.extend(random.sample(distractors, min(3, len(distractors))))
        random.shuffle(options)
        
        correct_idx = options.index(target_pattern)
        
        ps["current_trial"] = {
            "type": "pattern_matching",
            "target": target_pattern,
            "options": options,
            "correct_idx": correct_idx
        }
    
    elif task_type == "Visual Comparison":
        # Generate two strings to compare
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        length = random.randint(4, 8)
        string1 = ''.join(random.choices(chars, k=length))
        
        # 50% chance they're the same
        if random.choice([True, False]):
            string2 = string1  # Same
            match = True
        else:
            # Different - change 1-2 characters
            string2 = list(string1)
            positions = random.sample(range(length), random.randint(1, 2))
            for pos in positions:
                string2[pos] = random.choice([c for c in chars if c != string1[pos]])
            string2 = ''.join(string2)
            match = False
        
        ps["current_trial"] = {
            "type": "visual_comparison",
            "string1": string1,
            "string2": string2,
            "match": match
        }
    
    ps["trial_start"] = now_ts()
    st.rerun()

def _proc_speed_display_trial():
    """Display the current trial"""
    ps = st.session_state["proc_speed"]
    trial = ps["current_trial"]
    
    if trial["type"] == "symbol_search":
        st.markdown("### Find the Target Symbol")
        st.markdown(f"**Target**: {trial['target']}")
        
        # Display grid
        cols_per_row = 4
        rows = [trial["grid"][i:i+cols_per_row] for i in range(0, len(trial["grid"]), cols_per_row)]
        
        for row in rows:
            cols = st.columns(len(row))
            for i, symbol in enumerate(row):
                cols[i].markdown(f"<div style='font-size:24px;text-align:center;'>{symbol}</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        if col1.button("✅ PRESENT", use_container_width=True):
            _proc_speed_respond(True)
        if col2.button("❌ ABSENT", use_container_width=True):
            _proc_speed_respond(False)
    
    elif trial["type"] == "pattern_matching":
        st.markdown("### Match the Pattern")
        st.markdown(f"**Target Pattern**: {trial['target']}")
        st.markdown("**Which option matches?**")
        
        cols = st.columns(len(trial["options"]))
        for i, option in enumerate(trial["options"]):
            if cols[i].button(f"{option}", key=f"opt_{i}", use_container_width=True):
                _proc_speed_respond(i)
    
    elif trial["type"] == "visual_comparison":
        st.markdown("### Compare the Strings")
        st.markdown(f"**String 1**: `{trial['string1']}`")
        st.markdown(f"**String 2**: `{trial['string2']}`")
        st.markdown("**Are they the same?**")
        
        col1, col2 = st.columns(2)
        if col1.button("✅ SAME", use_container_width=True):
            _proc_speed_respond(True)
        if col2.button("❌ DIFFERENT", use_container_width=True):
            _proc_speed_respond(False)

def _proc_speed_respond(response):
    """Handle user response"""
    ps = st.session_state["proc_speed"]
    trial = ps["current_trial"]
    
    # Calculate reaction time
    rt = (now_ts() - ps["trial_start"]) * 1000
    ps["total_rt"] += rt
    
    # Check correctness
    correct = False
    if trial["type"] == "symbol_search":
        correct = response == trial["target_present"]
    elif trial["type"] == "pattern_matching":
        correct = response == trial["correct_idx"]
    elif trial["type"] == "visual_comparison":
        correct = response == trial["match"]
    
    if correct:
        ps["correct_responses"] += 1
        st.success(f"✅ Correct! ({rt:.0f}ms)")
    else:
        st.error(f"❌ Wrong! ({rt:.0f}ms)")
    
    ps["trials_completed"] += 1
    
    # Brief pause then next trial
    time.sleep(0.3)
    
    # Check if time is up
    if now_ts() >= ps["end_time"]:
        _proc_speed_finish()
    else:
        _proc_speed_new_trial()

def _proc_speed_finish():
    """Finish processing speed session"""
    ps = st.session_state["proc_speed"]
    
    accuracy = (ps["correct_responses"] / max(1, ps["trials_completed"])) * 100
    avg_rt = ps["total_rt"] / max(1, ps["correct_responses"])  # RT for correct responses only
    throughput = ps["trials_completed"] / (ps["duration"] * 60)  # trials per second
    
    st.success(f"**Accuracy**: {ps['correct_responses']}/{ps['trials_completed']} ({accuracy:.1f}%)")
    st.info(f"**Speed**: {avg_rt:.0f}ms average RT | {throughput:.1f} trials/sec")
    st.info(f"**Total Trials**: {ps['trials_completed']} in {ps['duration']} minutes")
    
    # Strategy reflection
    st.markdown("### 🤔 Strategy Reflection")
    strategy_rating = st.slider(f"How well did '{ps['strategy']}' work for you?", 1, 5, 3, 
                              help="1=Not helpful, 5=Very helpful")
    
    if st.button("Complete Session & Save Results"):
        # Mark as completed
        mark_completed("processing_speed")
        save_state()
        
        # Strategy feedback
        if strategy_rating >= 4:
            st.success(f"Great! '{ps['strategy']}' is working well for you.")
        elif strategy_rating <= 2:
            st.info(f"'{ps['strategy']}' wasn't very helpful. Try a different strategy next time.")
        
        st.session_state["proc_speed"] = None
        st.rerun()

    # Academic Research References
    st.markdown("---")
    st.caption("**Research Evidence**: Processing speed training improves cognitive efficiency and has been linked to better performance on intelligence tests and real-world tasks (Salthouse, 1996; Kail & Salthouse, 1994). Regular practice enhances perceptual speed and reduces cognitive load.")

# ----- Task Switching with Strategy Training -----
def page_task_switching():
    page_header("Task Switching")
    st.caption("Switch between **Number** (odd/even) and **Letter** (vowel/consonant) categorization tasks based on the cue.")

    # Strategy Training Section
    with st.expander("Strategy Tips (Read First!)", expanded=False):
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

    if "task_switch" not in st.session_state:
        st.session_state["task_switch"] = None

    if st.button("Start Task Switching"):
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

    ts = st.session_state["task_switch"]
    if ts and ts["current"]:
        # Display current strategy
        st.info(f"**Current Strategy**: {ts['strategy']}")
        
        # Task instruction
        task = ts["current"]["task"]
        if task == "NUMBER":
            st.markdown("### NUMBER TASK")
            st.caption("L = Odd number | R = Even number")
        else:
            st.markdown("### 🔤 LETTER TASK") 
            st.caption("L = Vowel | R = Consonant")
        
        # Stimulus display
        stimulus = ts["current"]["stimulus"]
        st.markdown(
            f"<div style='font-size:96px;text-align:center;color:#2E8B57;font-weight:bold;padding:40px;border:3px solid #2E8B57;border-radius:15px;margin:20px 0;'>{stimulus}</div>",
            unsafe_allow_html=True
        )
        
        # Response buttons
        col1, col2 = st.columns(2)
        with col1:
            if task == "NUMBER":
                if st.button("L - ODD", key="ts_left", use_container_width=True):
                    _task_switch_respond("L")
            else:
                if st.button("L - VOWEL", key="ts_left", use_container_width=True):
                    _task_switch_respond("L")
        with col2:
            if task == "NUMBER":
                if st.button("R - EVEN", key="ts_right", use_container_width=True):
                    _task_switch_respond("R")
            else:
                if st.button("R - CONSONANT", key="ts_right", use_container_width=True):
                    _task_switch_respond("R")
        
        # Progress
        st.caption(f"Trial {ts['i']+1}/{len(ts['stimuli'])}")
        
        # Auto-advance on timeout
        if ts["waiting_response"] and ts["start_time"]:
            elapsed = (now_ts() - ts["start_time"]) * 1000
            if elapsed > ts["isi"]:
                _task_switch_timeout()

def _task_switch_next():
    ts = st.session_state["task_switch"]
    if not ts or ts["i"] >= len(ts["stimuli"]):
        if ts:
            _task_switch_finish()
        return
    
    current_item = ts["stimuli"][ts["i"]]
    ts["current"] = current_item
    ts["waiting_response"] = True
    ts["start_time"] = now_ts()
    
    # Determine if this is a switch trial
    if ts["i"] > 0:
        prev_task = ts["stimuli"][ts["i"]-1]["task"]
        if current_item["task"] != prev_task:
            ts["switch_trials"] += 1
        else:
            ts["repeat_trials"] += 1
    else:
        ts["repeat_trials"] += 1  # First trial counts as repeat
    
    st.rerun()

def _task_switch_respond(response):
    ts = st.session_state["task_switch"]
    if not ts or not ts["waiting_response"]:
        return
    
    # Calculate reaction time
    rt = (now_ts() - ts["start_time"]) * 1000
    ts["rt_sum"] += rt
    
    # Check correctness
    correct = response == ts["current"]["correct"]
    if correct:
        ts["correct"] += 1
        st.success(f"✅ Correct! ({rt:.0f}ms)")
        
        # Track switch vs repeat performance
        if ts["i"] == 0 or ts["stimuli"][ts["i"]]["task"] == ts["stimuli"][ts["i"]-1]["task"]:
            ts["repeat_correct"] += 1
        else:
            ts["switch_correct"] += 1
    else:
        st.error(f"❌ Wrong! Correct was {ts['current']['correct']} ({rt:.0f}ms)")
    
    ts["waiting_response"] = False
    ts["i"] += 1
    
    # Brief pause then next trial
    time.sleep(0.5)
    _task_switch_next()

def _task_switch_timeout():
    ts = st.session_state["task_switch"]
    if not ts:
        return
    
    st.warning("⏰ Too slow!")
    ts["waiting_response"] = False
    ts["i"] += 1
    _task_switch_next()

def _task_switch_finish():
    ts = st.session_state["task_switch"]
    if not ts:
        return
    
    # Calculate results
    total_trials = len(ts["stimuli"])
    overall_acc = round((ts["correct"] / total_trials) * 100, 1)
    avg_rt = round(ts["rt_sum"] / max(1, ts["correct"]), 1)  # RT for correct responses only
    
    switch_acc = round((ts["switch_correct"] / max(1, ts["switch_trials"])) * 100, 1) if ts["switch_trials"] > 0 else 0
    repeat_acc = round((ts["repeat_correct"] / max(1, ts["repeat_trials"])) * 100, 1) if ts["repeat_trials"] > 0 else 0
    
    switch_cost = repeat_acc - switch_acc  # Switch cost (should be positive)
    
    st.success(f"**Overall**: {ts['correct']}/{total_trials} correct ({overall_acc}%)")
    st.info(f"**Speed**: {avg_rt}ms average RT")
    st.info(f"**Switch Cost**: {switch_cost:.1f}% (Repeat: {repeat_acc}% - Switch: {switch_acc}%)")
    
    # Strategy reflection
    st.markdown("### 🤔 Strategy Reflection")
    strategy_rating = st.slider(f"How well did '{ts['strategy']}' work for you?", 1, 5, 3, 
                              help="1=Not helpful, 5=Very helpful")
    
    if st.button("Complete Session & Save Results"):
        # Store results - reuse stroop history for now
        S()["stroopHistory"].append({
            "date": today_iso(), "trials": total_trials, "acc": overall_acc, "isi": ts["isi"],
            "switch_acc": switch_acc, "repeat_acc": repeat_acc, "switch_cost": switch_cost,
            "avg_rt": avg_rt, "strategy": ts["strategy"], "strategy_rating": strategy_rating,
            "type": "task_switching"
        })
        
        # Adaptive update
        level_idx = [1500, 1200, 1000, 800].index(ts["isi"]) if ts["isi"] in [1500, 1200, 1000, 800] else 0
        adaptive_update("stroop", level_idx, accuracy=overall_acc/100.0)  # Reuse stroop adaptive
        
        # Mark as completed
        mark_completed("stroop")  # Reuse stroop completion tracking
        save_state()
        
        # Strategy feedback
        if strategy_rating >= 4:
            st.success(f"Great! '{ts['strategy']}' is working well for you.")
        elif strategy_rating <= 2:
            st.info(f"'{ts['strategy']}' wasn't very helpful. Try a different strategy next time.")
        
        st.session_state["task_switch"] = None
        st.rerun()

    # Academic Research References
    st.markdown("---")
    st.caption("**Research Evidence**: Task switching training enhances cognitive flexibility and executive control (Kiesel et al., 2010; Monsell, 2003). Regular practice improves multitasking ability and reduces switch costs in everyday cognitive tasks.")

# ----- Complex Span with Strategy Training -----
def page_complex_span():
    page_header("Complex Span")
    st.caption("Remember letters **in order**, while verifying simple equations between letters (dual task).")

    # Strategy Training Section
    with st.expander("Strategy Tips (Read First!)", expanded=False):
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

    if "cspan" not in st.session_state:
        st.session_state["cspan"] = None

    if st.button("Start Complex Span"):
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
        st.info(f"**Current Strategy**: {cs['strategy']}")
        
        if cs["phase"] == "letters":
            st.markdown(f"### Remember This Letter:")
            st.markdown(
                f"<div style='font-size:72px;text-align:center;color:#2E8B57;font-weight:bold;padding:30px;border:3px solid #2E8B57;border-radius:15px;'>{cs['letters'][cs['i']]}</div>",
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
                f"<div style='font-size:48px;text-align:center;color:#FF6B35;font-weight:bold;padding:20px;border:2px solid #FF6B35;border-radius:10px;'>{a} {op} {b} = {shown}</div>",
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
                
                st.success(f"**Recall**: {correct_positions}/{cs['set_size']} correct ({recall_acc*100:.1f}%)")
                st.success(f"🧮 **Math**: {cs['proc_correct']}/{cs['proc_total']} correct ({proc_acc*100:.1f}%)")
                st.info(f"**Composite Score**: {composite*100:.1f}%")
                st.caption(f"Average processing RT: {avg_proc_rt:.0f}ms")
                
                # Show correct sequence
                st.markdown("**Correct sequence was:** " + " → ".join(cs["letters"]))
                
                # Strategy reflection
                st.markdown("### 🤔 Strategy Reflection")
                strategy_rating = st.slider(f"How well did '{cs['strategy']}' work for you?", 1, 5, 3, 
                                          help="1=Not helpful, 5=Very helpful")
                
                if st.button("Complete Session & Save Results"):
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
                    st.rerun()

    # Academic Research References
    st.markdown("---")
    st.caption("**Research Evidence**: Complex span tasks are reliable measures of working memory capacity and predict academic performance and fluid intelligence (Conway et al., 2005; Unsworth & Engle, 2007). Training improves working memory span and reasoning abilities.")

def _cspan_next():
    # Helper just to kick the first letter display
    pass

# ----- Go/No-Go with Strategy Training -----
def page_gng():
    page_header("Go / No-Go")
    st.caption("Press **GO** for Go stimuli; do **nothing** for No-Go. Measures response inhibition.")

    # Strategy Training Section
    with st.expander("Strategy Tips (Read First!)", expanded=False):
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

    if "gng" not in st.session_state:
        st.session_state["gng"] = None

    if st.button("Start Go/No-Go"):
        seq = []
        for _ in range(trials):
            if random.random() < p_nogo:
                seq.append(("NO_GO", "X"))  # show an 'X' for no-go
            else:
                seq.append(("GO", random.choice("BCDFGHJKLMNPQRSTVWXYZ")))
        st.session_state["gng"] = {
            "seq": seq, "isi": isi, "i": 0, "strategy": chosen_strategy,
            "hits": 0, "misses": 0, "fa": 0, "correct_rejections": 0,
            "reaction_times": [], "last_seen_index": -1, "done": False
        }
        st.rerun()

    g = st.session_state["gng"]
    if g:
        # Display current strategy
        st.info(f"**Current Strategy**: {g['strategy']}")
        
        placeholder = st.empty()
        if not g["done"]:
            if g["i"] < len(g["seq"]):
                stim_type, stim_val = g["seq"][g["i"]]
                with placeholder:
                    color = "#40c463" if stim_type == "GO" else "#ff4d4d"
                    bg_color = "#e8f5e8" if stim_type == "GO" else "#ffe8e8"
                    st.markdown(
                        f"<div style='font-size:96px;text-align:center;color:{color};background-color:{bg_color};padding:40px;border:3px solid {color};border-radius:15px;font-weight:bold;'>{stim_val}</div>",
                        unsafe_allow_html=True
                    )
                
                # Window to respond
                g["last_seen_index"] = g["i"]
                start = now_ts()
                
                # Give user an opportunity to click during ISI (best-effort in Streamlit)
                if stim_type == "GO":
                    btn = st.button("GO", key=f"go_{g['i']}", help="Click for letters only!")
                else:
                    btn = st.button("⛔ (DON'T CLICK)", key=f"go_{g['i']}", disabled=True, help="This is No-Go - don't respond!")
                    btn = False  # Force false for No-Go
                
                if btn:
                    rt = (now_ts() - start) * 1000
                    g["reaction_times"].append(rt)
                    # If current is GO, it's a hit; if NO_GO, it's a false alarm
                    if stim_type == "GO":
                        g["hits"] += 1
                        st.success(f"✅ Hit! ({rt:.0f}ms)")
                    else:
                        g["fa"] += 1
                        st.error("❌ False alarm!")
                
                # Wait till ISI elapses then move on; if GO and user did not press, it's a miss
                time.sleep(g["isi"]/1000.0)
                if stim_type == "GO" and not btn:
                    g["misses"] += 1
                elif stim_type == "NO_GO" and not btn:
                    g["correct_rejections"] += 1
                
                g["i"] += 1
                st.rerun()
            else:
                g["done"] = True

        if g["done"]:
            go_total = sum(1 for t,_ in g["seq"] if t == "GO")
            nogo_total = len(g["seq"]) - go_total
            hit_rate = g["hits"] / max(1, go_total)
            fa_rate  = g["fa"]  / max(1, nogo_total)
            # Balanced accuracy proxy
            composite = (hit_rate + (1.0 - fa_rate)) / 2.0
            avg_rt = sum(g["reaction_times"]) / len(g["reaction_times"]) if g["reaction_times"] else 0
            
            st.success(f"**Go Trials**: {g['hits']}/{go_total} hits, {g['misses']} misses → Hit Rate: {hit_rate*100:.1f}%")
            st.success(f"⛔ **No-Go Trials**: {g['correct_rejections']} correct rejections, {g['fa']} false alarms → Accuracy: {(1-fa_rate)*100:.1f}%")
            st.info(f"**Composite Accuracy**: {composite*100:.1f}%")
            if avg_rt > 0:
                st.caption(f"Average reaction time: {avg_rt:.0f}ms")
            
            # Strategy reflection
            st.markdown("### 🤔 Strategy Reflection")
            strategy_rating = st.slider(f"How well did '{g['strategy']}' work for you?", 1, 5, 3, 
                                      help="1=Not helpful, 5=Very helpful")
            
            if st.button("Complete Session & Save Results"):
                # Adaptive update
                level_idx = GNG_GRID.index(g["isi"])
                adaptive_update("gng", level_idx, accuracy=composite)
                
                # Mark Go/No-Go as completed
                mark_completed("gng")
                save_state()
                
                # Strategy feedback
                if strategy_rating >= 4:
                    st.success(f"Great! '{g['strategy']}' is working well for you.")
                elif strategy_rating <= 2:
                    st.info(f"'{g['strategy']}' wasn't very helpful. Try a different strategy next time.")
                
                st.session_state["gng"] = None
                st.rerun()

    # Academic Research References
    st.markdown("---")
    st.caption("**Research Evidence**: Go/No-Go tasks assess response inhibition, a key component of executive function (Dillon & Pizzagalli, 2007; Simmonds et al., 2008). Training improves impulse control and has applications in ADHD treatment and addiction therapy.")

# ----- Mental Math -----
def page_mm():
    page_header("Mental Math")
    mode = st.selectbox("Mode", ["Percent", "Fraction→Decimal", "Quick Ops", "Fermi"])
    duration_min = st.selectbox("Duration (min)", [2, 3, 5], index=1)
    tol = st.selectbox("Tolerance", ["Exact", "±5%", "±10%"], index=0)

    if "mm" not in st.session_state:
        st.session_state["mm"] = None

    def gen_problem():
        r = random.randint
        if mode == "Percent":
            p, base = r(5, 35), r(40, 900)
            return (f"What is {p}% of {base}?", round(base * p / 100, 2))
        if mode == "Fraction→Decimal":
            n, d = r(1, 9), r(2, 19)
            return (f"Convert {n}/{d} to decimal (4 dp).", round(n / d, 4))
        if mode == "Quick Ops":
            a, b, op = r(12, 99), r(6, 24), random.choice(["×", "÷", "+", "−"])
            ans = a * b if op == "×" else round(a / b, 3) if op == "÷" else a + b if op == "+" else a - b
            return (f"{a} {op} {b} = ?", ans)
        # Fermi
        prompts = [
            ("How many minutes in 6 weeks?", 6*7*24*60),
            ("How many seconds in 3 hours?", 3*3600),
            ("Sheets in 2 cm stack at 0.1 mm/page?", 200),
            ("Liters in 10m×2m×1m pool?", 20000),
        ]
        return random.choice(prompts)

    if st.button("Start"):
        end = now_ts() + duration_min * 60
        st.session_state["mm"] = {"end": end, "score": 0, "total": 0, "cur": gen_problem()}
        st.rerun()

    mm = st.session_state["mm"]
    if mm:
        left = int(mm["end"] - now_ts())
        st.metric("Time", timer_text(left))
        st.write(f"**Problem:** {mm['cur'][0]}")
        ans = st.text_input("Answer", key="mm_ans")
        if st.button("Submit"):
            user = ans.strip()
            if user:
                try:
                    user_val = float(user)
                    correct = False
                    truth = mm["cur"][1]
                    if tol == "Exact":
                        correct = abs(user_val - truth) < 1e-9
                    else:
                        pct = 0.05 if tol == "±5%" else 0.10
                        correct = abs(user_val - truth) <= pct * max(1.0, abs(truth))
                    mm["total"] += 1
                    if correct:
                        mm["score"] += 1
                        st.success("Correct ✓")
                    else:
                        st.error(f"Answer: {truth}")
                except ValueError:
                    st.warning("Enter a number.")
            mm["cur"] = gen_problem()
            st.rerun()

        if left <= 0:
            acc = round((mm["score"] / max(1, mm["total"])) * 100, 1)
            st.success(f"Done. Correct: {mm['score']} / {mm['total']} (Acc {acc}%)")
            S()["mmHistory"].append({"date": today_iso(), "mode": mode, "acc": acc})
            # Mark Mental Math as completed
            mark_completed("mental_math")
            save_state()
            st.session_state["mm"] = None
            if st.button("Restart"):
                st.rerun()

    # Academic Research References
    st.markdown("---")
    st.caption("**Research Evidence**: Mental math training enhances numerical cognition and working memory (Oberauer et al., 2003; Kaufmann et al., 2011). Regular practice improves mathematical fluency and has transfer effects to problem-solving abilities.")

# ----- World-Model Learning -----
def page_world_model():
    page_header("World-Model Learning")
    st.caption("Building a rigorous map of reality through micro-lessons → active recall → spaced repetition")
    
    # Track selection and progress
    wm_state = S()["world_model"]
    current_tracks = wm_state["current_tracks"]
    
    # Track selector
    all_tracks = list(WORLD_MODEL_TRACKS.keys())
    st.markdown("### Select Learning Tracks (2 active)")
    
    track_options = [WORLD_MODEL_TRACKS[t]["name"] for t in all_tracks]
    track_keys = {WORLD_MODEL_TRACKS[t]["name"]: t for t in all_tracks}
    
    selected_names = st.multiselect(
        "Choose 2 tracks to focus on:",
        track_options,
        default=[WORLD_MODEL_TRACKS[t]["name"] for t in current_tracks[:2]],
        max_selections=2
    )
    
    if len(selected_names) == 2:
        new_tracks = [track_keys[name] for name in selected_names]
        if new_tracks != current_tracks:
            wm_state["current_tracks"] = new_tracks
            save_state()
            st.rerun()
    
    # Display current tracks and lessons
    for i, track_key in enumerate(current_tracks[:2]):
        track_info = WORLD_MODEL_TRACKS[track_key]
        lesson = get_current_lesson(track_key)
        progress = wm_state["track_progress"][track_key]
        
        activity_key = f"world_model_{'a' if i == 0 else 'b'}"
        completed_today = is_completed_today(activity_key)
        
        with st.container(border=True):
            status_icon = "Done" if completed_today else "Pending"
            st.markdown(f"### {status_icon} Track {'A' if i == 0 else 'B'}: {track_info['name']}")
            st.markdown(f"**Lesson {progress['lesson'] + 1}: {lesson['title']}**")
            
            if not completed_today:
                # Show lesson content
                st.markdown("#### Key Concepts")
                st.info(lesson['content'])
                
                st.markdown("#### Worked Example")
                st.success(lesson['example'])
                
                # Active recall questions
                st.markdown("#### Active Recall (answer mentally, then check)")
                for j, question in enumerate(lesson['questions']):
                    with st.expander(f"Q{j+1}: {question}"):
                        # This would contain the answer/explanation
                        if j == 0:
                            st.write("Think through this before expanding...")
                        else:
                            st.write("Consider the concepts above...")
                
                # Transfer question
                st.markdown("#### Transfer Challenge")
                st.warning(f"Transfer Challenge: {lesson['transfer']}")
                
                # Completion button
                if st.button(f"Complete Track {'A' if i == 0 else 'B'} Lesson", key=f"complete_{track_key}"):
                    mark_completed(activity_key)
                    advance_lesson(track_key)
                    
                    # Generate spaced repetition cards from this lesson
                    for question in lesson['questions']:
                        new_card = {
                            "id": new_id(),
                            "front": question,
                            "back": f"From {lesson['title']}: {lesson['content'][:200]}...",
                            "tags": ["world-model", track_key.replace("_", "-")],
                            "ef": 2.5,
                            "reps": 0,
                            "interval": 0,
                            "due": today_iso(),
                            "history": [],
                            "new": True
                        }
                        S()["cards"].append(new_card)
                    
                    # Log lesson completion
                    wm_state["lesson_history"].append({
                        "date": today_iso(),
                        "track": track_key,
                        "lesson": lesson['title'],
                        "lesson_number": progress['lesson']
                    })
                    
                    save_state()
                    st.success(f"Lesson completed! Added {len(lesson['questions'])} cards to spaced repetition.")
                    st.rerun()
            else:
                st.success(f"✅ Today's lesson completed: {lesson['title']}")
                # Show brief summary for completed lessons
                st.caption(lesson['content'][:150] + "...")
    
    # Progress tracking
    st.markdown("### Learning Progress")
    total_lessons = sum(len(WORLD_MODEL_TRACKS[track]["lessons"]) for track in all_tracks)
    completed_lessons = sum(len(wm_state["track_progress"][track]["completed"]) for track in all_tracks)
    
    progress_pct = (completed_lessons / max(1, total_lessons)) * 100
    st.progress(progress_pct / 100.0)
    st.write(f"**{completed_lessons}/{total_lessons}** lessons completed across all tracks ({progress_pct:.1f}%)")
    
    # Recent lesson history
    if wm_state["lesson_history"]:
        st.markdown("### Recent Lessons")
        for session in wm_state["lesson_history"][-5:]:
            track_name = WORLD_MODEL_TRACKS[session["track"]]["name"]
            st.write(f"• {session['date']}: **{session['lesson']}** ({track_name})")

    # Academic Research References
    st.markdown("---")
    st.caption("**Research Evidence**: Spaced repetition and active recall significantly improve long-term retention and learning efficiency (Bjork, 1994; Roediger & Karpicke, 2006). Structured knowledge building enhances expert-level understanding and transfer.")

# ----- Enhanced Writing with World-Model Integration -----
def page_writing():
    page_header("Writing Sprint (12 min)")
    if "w" not in st.session_state:
        st.session_state["w"] = None

    # Enhanced prompts using concepts from your card deck
    prompts = [
        "Explain Bayes' theorem using a medical test example and why base rates matter.",
        "Describe a multipolar trap and provide a real-world example (climate change, arms race, etc.).",
        "What is Moloch? How does it manifest in modern society and what can be done about it?",
        "Explain the concept of feedback loops using examples from both positive and negative feedback.",
        "What is Goodhart's Law and how does it apply to modern metrics and KPIs?",
        "Describe the difference between System 1 and System 2 thinking with practical examples.",
        "What is emergence and how does it apply to complex systems?",
        "Explain instrumental convergence and why it matters for AI safety.",
        "What makes a good explanation according to David Deutsch's epistemology?",
        "How do leverage points in systems thinking help us create change effectively?",
    ]
    
    colA, colB = st.columns([1,2])
    with colA:
        ptxt = st.text_area("Prompt", value=random.choice(prompts), height=100)
        if st.button("Start 12-min"):
            st.session_state["w"] = {"end": now_ts() + 12*60, "prompt": ptxt, "text": ""}
            st.rerun()
        if st.session_state["w"]:
            left = int(st.session_state["w"]["end"] - now_ts())
            st.metric("Time", timer_text(left))
    with colB:
        if st.session_state["w"]:
            txt = st.text_area("Draft (write without stopping)", value=st.session_state["w"]["text"], height=300, key="w_draft")
            st.session_state["w"]["text"] = txt
            if now_ts() >= st.session_state["w"]["end"]:
                st.success("Time! Review your draft.")
                if st.button("Save session"):
                    S()["writingSessions"].append({
                        "date": today_iso(),
                        "prompt": st.session_state["w"]["prompt"],
                        "text": st.session_state["w"]["text"]
                    })
                    # Mark Writing as completed
                    mark_completed("writing")
                    save_state()
                    st.session_state["w"] = None
                    st.rerun()

    # Academic Research References
    st.markdown("---")
    st.caption("**Research Evidence**: Regular writing practice improves cognitive clarity, working memory, and executive function (Klein & Boals, 2001; Pennebaker & Chung, 2011). Expressive writing has therapeutic benefits and enhances learning consolidation.")

# ----- Forecasts -----
def page_forecasts():
    page_header("Forecast Journal")
    q = st.text_input("Question", placeholder="Will X happen by YYYY-MM-DD?")
    p = st.number_input("Probability (%)", min_value=1, max_value=99, value=60)
    due = st.date_input("Due date")
    notes = st.text_area("Notes", height=80)
    if st.button("Add"):
        if q.strip():
            S()["forecasts"].append({
                "id": new_id(), "q": q.strip(), "p": int(p),
                "due": due.isoformat(), "notes": notes.strip(),
                "created": today_iso(), "resolved": None, "outcome": None
            })
            # Mark Forecasts as completed when a forecast is added
            mark_completed("forecasts")
            save_state()
            st.success("Added.")
            st.rerun()

    items = sorted(S()["forecasts"], key=lambda x: x["due"])
    for f in items:
        with st.container(border=True):
            st.write(f"**{f['q']}**")
            st.caption(f"p={f['p']}% • due {f['due']} • created {f['created']}")
            if f.get("notes"):
                st.write(f"_{f['notes']}_")
            c1, c2, c3 = st.columns(3)
            if not f["resolved"]:
                if c1.button("Resolve TRUE", key=f"t_{f['id']}"):
                    f["resolved"] = today_iso(); f["outcome"] = 1; save_state(); st.rerun()
                if c2.button("Resolve FALSE", key=f"f_{f['id']}"):
                    f["resolved"] = today_iso(); f["outcome"] = 0; save_state(); st.rerun()
            if c3.button("Delete", key=f"d_{f['id']}"):
                S()["forecasts"] = [x for x in S()["forecasts"] if x["id"] != f["id"]]
                save_state(); st.rerun()

    resolved = [f for f in S()["forecasts"] if f["resolved"] is not None]
    if resolved:
        st.markdown("### Calibration & Brier decomposition")
        if st.button("Show calibration"):
            reliability_curve(resolved)
            brier, rel, res, unc = brier_decomposition(resolved)
            st.write(f"**Brier**: {brier:.3f}  •  Reliability: {rel:.3f}  •  Resolution: {res:.3f}  •  Uncertainty: {unc:.3f}")
            st.caption("Lower Brier is better. Reliability↓ (calibration) and Resolution↑ (discrimination) are desirable; Uncertainty depends on base rate.")

def reliability_curve(resolved: List[Dict[str, Any]]):
    # Bin into deciles by forecast p
    bins = [(i/10.0, (i+1)/10.0) for i in range(10)]
    pts = []
    for lo, hi in bins:
        xs = [f for f in resolved if lo <= f["p"]/100.0 < hi] if hi < 1 else [f for f in resolved if lo <= f["p"]/100.0 <= hi]
        if xs:
            mean_p = sum(f["p"]/100.0 for f in xs)/len(xs)
            freq = sum(f["outcome"] for f in xs)/len(xs)
            pts.append((mean_p, freq, len(xs)))
    fig, ax = plt.subplots()
    ax.plot([0,1],[0,1], linestyle="--")
    if pts:
        ax.scatter([x for x,_,_ in pts],[y for _,y,_ in pts])
        for x,y,n in pts:
            ax.text(x, y, f"{n}", fontsize=8, ha="center", va="bottom")
    ax.set_xlabel("Forecast probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_title("Forecast calibration")
    st.pyplot(fig)

    # Academic Research References
    st.markdown("---")
    st.caption("**Research Evidence**: Forecasting and prediction practice improves metacognitive accuracy and calibration (Tetlock & Gardner, 2015; Mellers et al., 2014). Regular forecasting enhances critical thinking and decision-making under uncertainty.")

def brier_decomposition(resolved: List[Dict[str, Any]]) -> Tuple[float,float,float,float]:
    """Murphy (1973) decomposition: Brier = Reliability - Resolution + Uncertainty"""
    y = [f["outcome"] for f in resolved]
    p = [f["p"]/100.0 for f in resolved]
    n = len(y)
    # Overall Brier
    brier = sum((pi - yi)**2 for pi, yi in zip(p,y))/n
    # Group by deciles
    buckets: Dict[int, List[int]] = {}
    for i,(pi,yi) in enumerate(zip(p,y)):
        b = min(9, int(pi*10))
        buckets.setdefault(b, []).append(i)
    rel = 0.0
    res = 0.0
    ybar = sum(y)/n
    for idxs in buckets.values():
        ni = len(idxs)
        pbar_i = sum(p[i] for i in idxs)/ni
        ybar_i = sum(y[i] for i in idxs)/ni
        rel += ni/n * (pbar_i - ybar_i)**2
        res += ni/n * (ybar_i - ybar)**2
    unc = ybar * (1 - ybar)
    # Brier ≈ rel - res + unc (floating error possible)
    return brier, rel, res, unc

# ----- Enhanced Argument Map -----
def page_argmap():
    page_header("Argument Map")
    
    # Suggest concepts from your card deck for argument practice
    suggested_topics = [
        "Interleaving beats blocking for durable learning",
        "Utilitarianism is the best moral framework",
        "AI alignment is solvable with current approaches", 
        "Systems thinking is superior to reductionist thinking",
        "Bayesian reasoning should be taught in schools",
        "Moloch problems require coordination solutions",
        "Emergent properties can't be predicted from components",
        "Fallibilism is the most rational epistemic stance"
    ]
    
    thesis = st.selectbox("Choose a thesis (or write your own below):", 
                         [""] + suggested_topics, 
                         index=0)
    if not thesis:
        thesis = st.text_input("Custom thesis", value="Interleaving beats blocking for durable learning.")
    
    pros = st.text_area("Reasons (one per line)").strip().splitlines()
    cons = st.text_area("Objections (one per line)").strip().splitlines()
    rebs = st.text_area("Rebuttals (one per line)").strip().splitlines()
    
    if st.button("Render Argument Map"):
        dot = Digraph(engine="dot")
        dot.attr(rankdir="TB", bgcolor="transparent")
        dot.node("T", thesis, shape="box", style="rounded,filled", fillcolor="#1e2433", fontcolor="white")
        
        for i, r in enumerate([p for p in pros if p.strip()]):
            nid = f"P{i}"
            dot.node(nid, r, shape="box", style="rounded,filled", fillcolor="#1b2a1d", fontcolor="white")
            dot.edge(nid, "T", color="green")
            
        for i, c in enumerate([c for c in cons if c.strip()]):
            nid = f"C{i}"
            dot.node(nid, c, shape="box", style="rounded,filled", fillcolor="#331e1e", fontcolor="white")
            dot.edge("T", nid, color="red")
            
        for i, rb in enumerate([r for r in rebs if r.strip()]):
            nid = f"R{i}"
            dot.node(nid, rb, shape="box", style="rounded,filled", fillcolor="#262233", fontcolor="white")
            if cons:
                dot.edge(nid, "C0", color="blue")
            else:
                dot.edge(nid, "T", color="blue")
                
        st.graphviz_chart(dot)

# ----- Settings -----
def page_settings():
    page_header("Settings & Backup")
    s = S()["settings"]
    
    # Theme Settings
    st.markdown("### Appearance")
    
    # Theme mode selection
    theme_mode = "Blackout" if s.get("blackoutMode", False) else ("Dark" if s.get("darkMode", False) else "Light")
    new_theme = st.selectbox(
        "� Theme Mode", 
        ["Light", "Dark", "Blackout"],
        index=["Light", "Dark", "Blackout"].index(theme_mode),
        help="Choose your preferred visual theme"
    )
    
    # Update theme settings
    if new_theme != theme_mode:
        s["darkMode"] = new_theme == "Dark"
        s["blackoutMode"] = new_theme == "Blackout" 
        save_state()
        st.success(f"Switched to {new_theme} mode! Refreshing...")
        st.rerun()
    
    # Theme preview
    col1, col2 = st.columns([3, 1])
    with col2:
        if new_theme == "Blackout":
            preview = "●"
        elif new_theme == "Dark":
            preview = "●"
        else:
            preview = "●"
        st.markdown(f"<div style='font-size: 3rem; text-align: center;'>{preview}</div>", unsafe_allow_html=True)
    
    with col1:
        if new_theme == "Blackout":
            st.caption("Pure black aesthetic for focused study sessions")
        elif new_theme == "Dark":
            st.caption("Purple-grey dark mode for comfortable evening use")
        else:
            st.caption("Clean Apple-inspired light mode for daytime use")
    
    st.markdown("---")
    
    # Card Settings
    st.markdown("### Spaced Repetition")
    nl = st.number_input("Daily new cards", min_value=0, max_value=50, value=s["newLimit"])
    rl = st.number_input("Review limit", min_value=10, max_value=500, value=s["reviewLimit"])
    if st.button("Save Settings"):
        s["newLimit"] = int(nl); s["reviewLimit"] = int(rl); save_state(); st.success("Saved.")

    st.markdown("### Export / Import")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Export Data", 
            data=export_json(), 
            file_name="max_mind_trainer_backup.json",
            help="Download your complete training data"
        )
    with col2:
        up = st.file_uploader("Import Data", type=["json"])
        if up and st.button("Import Now"):
            import_json(up.read().decode("utf-8"))

    st.markdown("### Statistics")
    cards = S()["cards"]
    total_cards = len(cards)
    learned_cards = len([c for c in cards if not c.get("new")])
    
    # Statistics in a nice layout
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Cards", total_cards)
    with col2:
        st.metric("Learned", learned_cards)
    with col3:
        st.metric("New", total_cards - learned_cards)
    
    # Tag statistics
    tag_counts = {}
    for card in cards:
        for tag in card.get("tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    if tag_counts:
        with st.expander("Cards by Tag"):
            for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
                st.write(f"**{tag}**: {count} cards")

    st.markdown("### Reset Options")
    with st.expander("Reset to Default Cards", expanded=False):
        st.warning("This will replace all current cards with the default seed. Your progress will be lost!")
        if st.button("Reset to Default Cards"):
            S()["cards"] = [asdict(c) for c in make_cards(DEFAULT_SEED)]
            save_state()
            st.success("Cards reset to default seed.")
            st.rerun()

# ----- World Model Integration -----
def integrate_mastered_topics():
    """Convert mastered AI topic suggestions into spaced repetition cards"""
    topic_state = S()["topic_suggestions"]
    
    # Handle missing keys gracefully
    completed_topics = topic_state.get("completed", [])
    mastered_topics = [t for t in completed_topics if t.get("mastered", False)]
    
    for topic in mastered_topics:
        if not topic.get("integrated", False):
            # Create cards from the topic content
            topic_data = TOPIC_KNOWLEDGE_BASE[topic["topic"]]
            
            # Main concept card
            main_card = {
                "id": new_id(),
                "front": f"What is {topic_data['title']}?",
                "back": topic_data["content"],
                "tags": ["ai-topic", "mastered", topic["topic"].replace("_", "-")],
                "ef": 2.5,
                "reps": 0,
                "interval": 0,
                "due": today_iso(),
                "history": [],
                "new": True
            }
            S()["cards"].append(main_card)
            
            # Application cards from examples
            for i, example in enumerate(topic_data["examples"]):
                app_card = {
                    "id": new_id(),
                    "front": f"Apply {topic_data['title']}: {example[:100]}...",
                    "back": example,
                    "tags": ["ai-topic", "application", topic["topic"].replace("_", "-")],
                    "ef": 2.5,
                    "reps": 0,
                    "interval": 0,
                    "due": today_iso(),
                    "history": [],
                    "new": True
                }
                S()["cards"].append(app_card)
            
            # Mark as integrated
            topic["integrated"] = True
    
    save_state()
    return len(mastered_topics)

# ========== Router ==========
# ========== Navigation & Pages ==========
PAGES = [
    "Dashboard",
    "SPACED LEARNING",
    "Spaced Review",
    "World Model", 
    "Topic Study",
    "Card Management",
    "COGNITIVE DRILLS",
    "N-Back",
    "Task Switching",
    "Complex Span",
    "Go/No-Go",
    "Processing Speed",
    "ADDITIONAL TRAINING",
    "Mental Math",
    "Writing",
    "Forecasts",
    "Argument Map",
    "SETTINGS",
    "Settings",
]

st.set_page_config(
    page_title="Max Mind Trainer", 
    page_icon="M", 
    layout="centered",
    initial_sidebar_state="auto"
)

def apply_theme_styling():
    """Apply theme-based CSS styling"""
    dark_mode = S().get("settings", {}).get("darkMode", False)
    blackout_mode = S().get("settings", {}).get("blackoutMode", False)
    
    if blackout_mode:
        # Blackout mode styling - pure black aesthetic
        st.markdown("""
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <style>
            /* Global mobile optimizations */
            * {
                box-sizing: border-box;
            }
            
            /* Prevent horizontal scroll on mobile */
            .main .block-container {
                overflow-x: hidden;
            }
            
            /* Blackout mode main app styling */
            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 900px;
                background-color: #000000;
                color: #ffffff;
            }
            
            /* Blackout sidebar styling */
            .css-1d391kg {
                background: linear-gradient(180deg, #111111 0%, #000000 100%);
                border-right: 1px solid #333333;
            }
            
            /* Blackout typography */
            .main h1, .main h2, .main h3 {
                color: #ffffff;
                font-weight: 600;
                letter-spacing: -0.025em;
            }
            
            /* Blackout buttons */
            .stButton > button {
                background: linear-gradient(145deg, #1a1a1a, #0a0a0a) !important;
                color: #ffffff !important;
                border: 1px solid #333333 !important;
                border-radius: 12px !important;
                padding: 0.75rem 1.5rem !important;
                font-weight: 600 !important;
                font-size: 1rem !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 2px 10px rgba(255, 255, 255, 0.1) !important;
            }
            
            .stButton > button:hover {
                background: linear-gradient(145deg, #2a2a2a, #1a1a1a) !important;
                border-color: #444444 !important;
                transform: translateY(-2px) !important;
                box-shadow: 0 8px 20px rgba(255, 255, 255, 0.15) !important;
            }
            
            /* Override Streamlit's default background */
            .stApp {
                background-color: #000000;
            }
            
            /* Other blackout styles */
            [data-testid="metric-container"] {
                background: #1a1a1a;
                border: 1px solid #333333;
                padding: 1.5rem;
                border-radius: 16px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
            }
            
            /* Mobile Responsive Design for Blackout Mode */
            @media screen and (max-width: 768px) {
                .main .block-container {
                    padding-top: 1rem;
                    padding-bottom: 1rem;
                    padding-left: 0.5rem;
                    padding-right: 0.5rem;
                    max-width: 100%;
                }
                
                .stButton > button {
                    padding: 0.5rem 1rem !important;
                    font-size: 0.9rem !important;
                    width: 100% !important;
                    margin-bottom: 0.5rem !important;
                    min-height: 44px !important; /* Minimum touch target size */
                    touch-action: manipulation !important; /* Improve touch responsiveness */
                }
                
                .main h1 {
                    font-size: 1.8rem !important;
                    margin-bottom: 0.5rem !important;
                    line-height: 1.2 !important;
                }
                
                .main h2 {
                    font-size: 1.4rem !important;
                    margin-bottom: 0.75rem !important;
                    line-height: 1.3 !important;
                }
                
                .main h3 {
                    font-size: 1.2rem !important;
                    margin-bottom: 0.5rem !important;
                    line-height: 1.3 !important;
                }
                
                [data-testid="metric-container"] {
                    padding: 1rem !important;
                    margin-bottom: 0.5rem !important;
                }
                
                .css-1d391kg {
                    width: 100% !important;
                    min-width: unset !important;
                }
                
                /* Compact sidebar for mobile */
                .stRadio label {
                    padding: 0.25rem 0.5rem !important;
                    font-size: 0.9rem !important;
                }
                
                /* Compact input fields */
                .stTextInput > div > div > input,
                .stTextArea > div > div > textarea,
                .stSelectbox > div > div > select {
                    padding: 0.5rem !important;
                    font-size: 0.9rem !important;
                }
                
                /* Compact tabs */
                .stTabs [data-baseweb="tab"] {
                    padding: 0.25rem 0.5rem !important;
                    font-size: 0.9rem !important;
                }
                
                /* Hide or compact less essential elements */
                .element-container {
                    margin-bottom: 0.5rem !important;
                }
                
                /* Mobile-friendly columns */
                [data-testid="column"] {
                    min-width: unset !important;
                    flex: 1 !important;
                }
                
                /* Adjust progress indicators for mobile */
                .stProgress {
                    margin: 0.25rem 0 !important;
                }
                
                /* Compact expanders */
                .streamlit-expanderHeader {
                    padding: 0.5rem !important;
                    font-size: 0.9rem !important;
                }
            }
        </style>
        """, unsafe_allow_html=True)
    elif dark_mode:
        # Dark mode styling
        st.markdown("""
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <style>
            /* Global mobile optimizations */
            * {
                box-sizing: border-box;
            }
            
            /* Prevent horizontal scroll on mobile */
            .main .block-container {
                overflow-x: hidden;
            }
            
            /* Dark mode main app styling */
            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 900px;
                background-color: #0d1117;
                color: #f0f6fc;
            }
            
            /* Dark sidebar styling */
            .css-1d391kg {
                background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
                border-right: 1px solid #30363d;
            }
            
            /* Dark typography */
            .main h1, .main h2, .main h3 {
                color: #f0f6fc;
                font-weight: 600;
                letter-spacing: -0.025em;
            }
            
            /* Dark buttons */
            .stButton > button {
                background: linear-gradient(145deg, #4c1d95, #312e81) !important;
                color: white !important;
                border: 1px solid #6366f1 !important;
                border-radius: 12px !important;
                padding: 0.75rem 1.5rem !important;
                font-weight: 600 !important;
                font-size: 1rem !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3) !important;
            }
            
            .stButton > button:hover {
                background: linear-gradient(145deg, #7c3aed, #6366f1) !important;
                transform: translateY(-2px) !important;
                box-shadow: 0 8px 25px rgba(124, 58, 237, 0.4) !important;
            }
            
            .stButton > button:active {
                transform: translateY(0px);
                box-shadow: 0 2px 4px rgba(35, 134, 54, 0.2);
            }
            
            /* Dark radio buttons */
            .stRadio label {
                background: transparent;
                color: #8b949e;
                font-weight: 500;
                padding: 0.5rem 1rem;
                border-radius: 8px;
                transition: all 0.2s ease;
            }
            
            .stRadio label:hover {
                background: rgba(35, 134, 54, 0.1);
                color: #58a6ff;
            }
            
            /* Dark metrics */
            [data-testid="metric-container"] {
                background: #161b22;
                border: 1px solid #30363d;
                padding: 1.5rem;
                border-radius: 16px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
            }
            
            /* Dark progress bars */
            .stProgress > div > div {
                background: linear-gradient(90deg, #58a6ff 0%, #238636 100%);
                border-radius: 8px;
            }
            
            /* Dark alert boxes */
            .stAlert {
                border-radius: 12px;
                border: none;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
                background: #161b22;
                border: 1px solid #30363d;
            }
            
            /* Dark expanders */
            .streamlit-expanderHeader {
                background: #161b22;
                border-radius: 12px;
                border: 1px solid #30363d;
                font-weight: 500;
                color: #f0f6fc;
            }
            
            /* Dark input fields */
            .stTextInput > div > div > input,
            .stTextArea > div > div > textarea,
            .stSelectbox > div > div > select {
                background: #0d1117;
                border-radius: 12px;
                border: 1.5px solid #30363d;
                padding: 0.75rem 1rem;
                font-size: 1rem;
                color: #f0f6fc;
                transition: all 0.2s ease;
            }
            
            .stTextInput > div > div > input:focus,
            .stTextArea > div > div > textarea:focus,
            .stSelectbox > div > div > select:focus {
                border-color: #58a6ff;
                box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.1);
                outline: none;
            }
            
            /* Dark tabs */
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
                background: #161b22;
                border-radius: 12px;
                padding: 4px;
            }
            
            .stTabs [data-baseweb="tab"] {
                background: transparent;
                border-radius: 8px;
                color: #8b949e;
                font-weight: 500;
                padding: 0.5rem 1rem;
                transition: all 0.2s ease;
            }
            
            .stTabs [aria-selected="true"] {
                background: #0d1117;
                color: #58a6ff;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
            }
            
            /* Custom dark spacing */
            .element-container {
                margin-bottom: 1rem;
            }
            
            /* Override Streamlit's default background */
            .stApp {
                background-color: #0d1117;
            }
            
            /* Mobile Responsive Design for Dark Mode */
            @media screen and (max-width: 768px) {
                .main .block-container {
                    padding-top: 1rem;
                    padding-bottom: 1rem;
                    padding-left: 0.5rem;
                    padding-right: 0.5rem;
                    max-width: 100%;
                }
                
                .stButton > button {
                    padding: 0.5rem 1rem !important;
                    font-size: 0.9rem !important;
                    width: 100% !important;
                    margin-bottom: 0.5rem !important;
                    min-height: 44px !important; /* Minimum touch target size */
                    touch-action: manipulation !important; /* Improve touch responsiveness */
                }
                
                .main h1 {
                    font-size: 1.8rem !important;
                    margin-bottom: 0.5rem !important;
                    line-height: 1.2 !important;
                }
                
                .main h2 {
                    font-size: 1.4rem !important;
                    margin-bottom: 0.75rem !important;
                    line-height: 1.3 !important;
                }
                
                .main h3 {
                    font-size: 1.2rem !important;
                    margin-bottom: 0.5rem !important;
                    line-height: 1.3 !important;
                }
                
                [data-testid="metric-container"] {
                    padding: 1rem !important;
                    margin-bottom: 0.5rem !important;
                }
                
                .css-1d391kg {
                    width: 100% !important;
                    min-width: unset !important;
                }
                
                /* Compact sidebar for mobile */
                .stRadio label {
                    padding: 0.25rem 0.5rem !important;
                    font-size: 0.9rem !important;
                }
                
                /* Compact input fields */
                .stTextInput > div > div > input,
                .stTextArea > div > div > textarea,
                .stSelectbox > div > div > select {
                    padding: 0.5rem !important;
                    font-size: 0.9rem !important;
                }
                
                /* Compact tabs */
                .stTabs [data-baseweb="tab"] {
                    padding: 0.25rem 0.5rem !important;
                    font-size: 0.9rem !important;
                }
                
                /* Hide or compact less essential elements */
                .element-container {
                    margin-bottom: 0.5rem !important;
                }
                
                /* Mobile-friendly columns */
                [data-testid="column"] {
                    min-width: unset !important;
                    flex: 1 !important;
                }
                
                /* Adjust progress indicators for mobile */
                .stProgress {
                    margin: 0.25rem 0 !important;
                }
                
                /* Compact expanders */
                .streamlit-expanderHeader {
                    padding: 0.5rem !important;
                    font-size: 0.9rem !important;
                }
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        # Light mode styling (existing)
        st.markdown("""
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <style>
            /* Global mobile optimizations */
            * {
                box-sizing: border-box;
            }
            
            /* Prevent horizontal scroll on mobile */
            .main .block-container {
                overflow-x: hidden;
            }
            
            /* Main app styling */
            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 900px;
            }
            
            /* Sidebar styling */
            .css-1d391kg {
                background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
                border-right: 1px solid #e2e8f0;
            }
            
            /* Typography */
            .main h1, .main h2, .main h3 {
                color: #1e293b;
                font-weight: 600;
                letter-spacing: -0.025em;
            }
            
            .main h1 {
                font-size: 2.5rem;
                margin-bottom: 0.5rem;
            }
            
            .main h2 {
                font-size: 1.875rem;
                margin-bottom: 1rem;
            }
            
            .main h3 {
                font-size: 1.5rem;
                margin-bottom: 0.75rem;
            }
            
            /* Buttons - Apple style */
            .stButton > button {
                background: linear-gradient(145deg, #3b82f6, #2563eb) !important;
                color: white !important;
                border: none !important;
                border-radius: 12px !important;
                padding: 0.75rem 1.5rem !important;
                font-weight: 600 !important;
                font-size: 1rem !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
            }
            
            .stButton > button:hover {
                background: linear-gradient(145deg, #2563eb, #1d4ed8) !important;
                transform: translateY(-2px) !important;
                box-shadow: 0 8px 25px rgba(37, 99, 235, 0.4) !important;
            }
            
            .stButton > button:active {
                transform: translateY(0px);
                box-shadow: 0 2px 4px rgba(0, 122, 255, 0.2);
            }
            
            /* Radio buttons - sidebar navigation */
            .stRadio > div {
                background: transparent;
            }
            
            .stRadio label {
                background: transparent;
                color: #475569;
                font-weight: 500;
                padding: 0.5rem 1rem;
                border-radius: 8px;
                transition: all 0.2s ease;
            }
            
            .stRadio label:hover {
                background: rgba(0, 122, 255, 0.1);
                color: #007aff;
            }
            
            /* Metrics */
            [data-testid="metric-container"] {
                background: white;
                border: 1px solid #e2e8f0;
                padding: 1.5rem;
                border-radius: 16px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            }
            
            /* Cards/Containers */
            .stContainer > div {
                background: white;
                border-radius: 16px;
                border: 1px solid #e2e8f0;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
                padding: 1.5rem;
            }
            
            /* Progress bars */
            .stProgress > div > div {
                background: linear-gradient(90deg, #007aff 0%, #00d4ff 100%);
                border-radius: 8px;
            }
            
            /* Info/Success/Warning boxes */
            .stAlert {
                border-radius: 12px;
                border: none;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            }
            
            /* Expanders */
            .streamlit-expanderHeader {
                background: #f8fafc;
                border-radius: 12px;
                border: 1px solid #e2e8f0;
                font-weight: 500;
                color: #1e293b;
            }
            
            /* Input fields */
            .stTextInput > div > div > input,
            .stTextArea > div > div > textarea,
            .stSelectbox > div > div > select {
                border-radius: 12px;
                border: 1.5px solid #e2e8f0;
                padding: 0.75rem 1rem;
                font-size: 1rem;
                transition: all 0.2s ease;
            }
            
            .stTextInput > div > div > input:focus,
            .stTextArea > div > div > textarea:focus,
            .stSelectbox > div > div > select:focus {
                border-color: #007aff;
                box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.1);
                outline: none;
            }
            
            /* Slider */
            .stSlider > div > div {
                border-radius: 8px;
            }
            
            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
                background: #f8fafc;
                border-radius: 12px;
                padding: 4px;
            }
            
            .stTabs [data-baseweb="tab"] {
                background: transparent;
                border-radius: 8px;
                color: #64748b;
                font-weight: 500;
                padding: 0.5rem 1rem;
                transition: all 0.2s ease;
            }
            
            .stTabs [aria-selected="true"] {
                background: white;
                color: #007aff;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            }
            
            /* Remove default streamlit styling */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            
            /* Custom spacing */
            .element-container {
                margin-bottom: 1rem;
            }
            
            /* Mobile Responsive Design for Light Mode */
            @media screen and (max-width: 768px) {
                .main .block-container {
                    padding-top: 1rem;
                    padding-bottom: 1rem;
                    padding-left: 0.5rem;
                    padding-right: 0.5rem;
                    max-width: 100%;
                }
                
                .stButton > button {
                    padding: 0.5rem 1rem !important;
                    font-size: 0.9rem !important;
                    width: 100% !important;
                    margin-bottom: 0.5rem !important;
                    min-height: 44px !important; /* Minimum touch target size */
                    touch-action: manipulation !important; /* Improve touch responsiveness */
                }
                
                .main h1 {
                    font-size: 1.8rem !important;
                    margin-bottom: 0.5rem !important;
                    line-height: 1.2 !important;
                }
                
                .main h2 {
                    font-size: 1.4rem !important;
                    margin-bottom: 0.75rem !important;
                    line-height: 1.3 !important;
                }
                
                .main h3 {
                    font-size: 1.2rem !important;
                    margin-bottom: 0.5rem !important;
                    line-height: 1.3 !important;
                }
                
                [data-testid="metric-container"] {
                    padding: 1rem !important;
                    margin-bottom: 0.5rem !important;
                }
                
                .css-1d391kg {
                    width: 100% !important;
                    min-width: unset !important;
                }
                
                /* Compact sidebar for mobile */
                .stRadio label {
                    padding: 0.25rem 0.5rem !important;
                    font-size: 0.9rem !important;
                }
                
                /* Compact input fields */
                .stTextInput > div > div > input,
                .stTextArea > div > div > textarea,
                .stSelectbox > div > div > select {
                    padding: 0.5rem !important;
                    font-size: 0.9rem !important;
                }
                
                /* Compact tabs */
                .stTabs [data-baseweb="tab"] {
                    padding: 0.25rem 0.5rem !important;
                    font-size: 0.9rem !important;
                }
                
                /* Hide or compact less essential elements */
                .element-container {
                    margin-bottom: 0.5rem !important;
                }
                
                /* Mobile-friendly columns */
                [data-testid="column"] {
                    min-width: unset !important;
                    flex: 1 !important;
                }
                
                /* Adjust progress indicators for mobile */
                .stProgress {
                    margin: 0.25rem 0 !important;
                }
                
                /* Compact expanders */
                .streamlit-expanderHeader {
                    padding: 0.5rem !important;
                    font-size: 0.9rem !important;
                }
                
                /* Mobile-specific card styling */
                .stContainer > div {
                    padding: 1rem !important;
                    margin-bottom: 0.5rem !important;
                }
                
                /* Compact alert boxes for mobile */
                .stAlert {
                    padding: 0.75rem !important;
                    margin-bottom: 0.5rem !important;
                }
            }
        </style>
        """, unsafe_allow_html=True)

# Apply theme styling
apply_theme_styling()

def get_card_styles():
    """Get theme-appropriate card styling"""
    dark_mode = S().get("settings", {}).get("darkMode", False)
    blackout_mode = S().get("settings", {}).get("blackoutMode", False)
    
    if blackout_mode:
        return {
            "background": "#111111",
            "border": "1px solid #333333", 
            "text_color": "#ffffff",
            "accent_color": "#cccccc",
            "muted_color": "#888888",
            "shadow": "0 4px 15px rgba(255, 255, 255, 0.1)"
        }
    elif dark_mode:
        return {
            "background": "#1f2937",
            "border": "1px solid #4b5563",
            "text_color": "#f1f5f9",
            "accent_color": "#8b5cf6",
            "muted_color": "#9ca3af",
            "shadow": "0 4px 15px rgba(139, 92, 246, 0.2)"
        }
    else:
        return {
            "background": "white",
            "border": "1px solid #e2e8f0",
            "text_color": "#1e293b",
            "accent_color": "#3b82f6",
            "muted_color": "#64748b",
            "shadow": "0 2px 8px rgba(0, 0, 0, 0.06)"
        }

with st.sidebar:
    st.markdown("# MaxMind")
    st.markdown("*Enhanced Cognitive Training*")
    st.markdown("---")
    st.session_state.setdefault("page", "Dashboard")
    
    # Custom navigation with sections and completion indicators
    current_page = st.session_state["page"]
    completion_status = get_completion_status()
    
    # Get theme settings for styling
    dark_mode = S().get("settings", {}).get("darkMode", False)
    blackout_mode = S().get("settings", {}).get("blackoutMode", False)
    
    # Check if all activities completed for dashboard indicator
    total_activities = len(completion_status)
    completed_count = sum(completion_status.values())
    all_completed = completed_count == total_activities
    
    for page in PAGES:
        if page.isupper() and not any(c.islower() for c in page):
            # Section header (all caps)
            st.markdown(f"**{page}**")
        else:
            # Regular page button with custom styling
            
            # Determine completion status for this page
            page_completed = False
            if "Spaced Review" in page and completion_status.get("review", False):
                page_completed = True
            elif "N-Back" in page and completion_status.get("nback", False):
                page_completed = True
            elif "Task Switching" in page and completion_status.get("task_switching", False):
                page_completed = True
            elif "Complex Span" in page and completion_status.get("complex_span", False):
                page_completed = True
            elif "Go/No-Go" in page and completion_status.get("gng", False):
                page_completed = True
            elif "Processing Speed" in page and completion_status.get("processing_speed", False):
                page_completed = True
            elif "Mental Math" in page and completion_status.get("mental_math", False):
                page_completed = True
            elif "Writing" in page and completion_status.get("writing", False):
                page_completed = True
            elif "Forecasts" in page and completion_status.get("forecasts", False):
                page_completed = True
            elif "World Model" in page and (completion_status.get("world_model_a", False) or completion_status.get("world_model_b", False)):
                page_completed = True
            elif "Topic Study" in page and completion_status.get("topic_study", False):
                page_completed = True
            elif "Dashboard" in page and all_completed:
                page_completed = True
            
            # Determine button styling
            is_current = page == current_page
            
            # Create custom styled button using HTML
            if blackout_mode:
                if is_current:
                    button_bg = "#333333"
                    button_color = "#ffffff"
                    button_border = "#555555"
                elif page_completed:
                    button_bg = "rgba(0, 255, 0, 0.2)"
                    button_color = "#ffffff"
                    button_border = "#00ff00"
                else:
                    button_bg = "#1a1a1a"
                    button_color = "#cccccc"
                    button_border = "#333333"
                hover_bg = "#444444"
            elif dark_mode:
                if is_current:
                    button_bg = "#4c1d95"
                    button_color = "#ffffff"
                    button_border = "#6366f1"
                elif page_completed:
                    button_bg = "rgba(34, 197, 94, 0.4)"
                    button_color = "#ffffff"
                    button_border = "#22c55e"
                else:
                    button_bg = "#374151"
                    button_color = "#e2e8f0"
                    button_border = "#4b5563"
                hover_bg = "#6366f1"
            else:
                if is_current:
                    button_bg = "#2563eb"
                    button_color = "#ffffff"
                    button_border = "#3b82f6"
                elif page_completed:
                    button_bg = "rgba(34, 197, 94, 0.4)"
                    button_color = "#ffffff"
                    button_border = "#22c55e"
                else:
                    button_bg = "#f8fafc"
                    button_color = "#374151"
                    button_border = "#d1d5db"
                hover_bg = "#3b82f6"
            
            # Use st.button with unique key
            button_key = f"nav_{page.replace(' ', '_').replace('/', '_').replace('─', '_')}"
            
            # Create container for custom styling
            nav_container = st.container()
            with nav_container:
                if st.button(page, key=button_key, use_container_width=True):
                    st.session_state["page"] = page
                    st.rerun()
            
            # Apply inline CSS immediately after button
            st.markdown(f"""
            <style>
            button[data-testid="baseButton-secondary"][title="{page}"] {{
                background: {button_bg} !important;
                color: {button_color} !important;
                border: 1px solid {button_border} !important;
                border-radius: 8px !important;
                font-weight: {"700" if is_current else "500"} !important;
                font-size: 0.9rem !important;
                padding: 0.5rem 1rem !important;
                margin-bottom: 0.25rem !important;
                transition: all 0.2s ease !important;
                width: 100% !important;
            }}
            button[data-testid="baseButton-secondary"][title="{page}"]:hover {{
                background: {hover_bg} !important;
                color: white !important;
                transform: translateX(2px) !important;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
            }}
            </style>
            """, unsafe_allow_html=True)

page = st.session_state["page"]
if page == "Dashboard": page_dashboard()
elif page == "Spaced Review": page_review()
elif page == "Topic Study": page_topic_study()
elif page == "Card Management": page_card_management()
elif page == "World Model": page_world_model()
elif page == "N-Back": page_nback()
elif page == "Task Switching": page_task_switching()
elif page == "Complex Span": page_complex_span()
elif page == "Go/No-Go": page_gng()
elif page == "Processing Speed": page_processing_speed()
elif page == "Mental Math": page_mm()
elif page == "Writing": page_writing()
elif page == "Forecasts": page_forecasts()
elif page == "Argument Map": page_argmap()
elif page == "Settings": page_settings()

# Auto-save state after each page load
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
    const viewport = document.createElement('meta');
    viewport.name = 'viewport';
    viewport.content = 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no';
    head.appendChild(viewport);
    
    // Theme color
    const themeColor = document.createElement('meta');
    themeColor.name = 'theme-color';
    themeColor.content = '#000000';
    head.appendChild(themeColor);
}
</script>
""", unsafe_allow_html=True)

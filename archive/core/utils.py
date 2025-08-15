"""
Core utilities and data models for MaxMind
"""
import json
import math
import random
import time
import uuid
from dataclasses import dataclass, asdict, field
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st

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

@st.cache_data
def load_default_cards():
    """Load default cards from JSON file"""
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
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as e:
        # Fallback to hardcoded cards if file issues
        return get_fallback_cards()

def get_fallback_cards():
    """Fallback cards if JSON loading fails"""
    fallback_data = [
        {"front":"Expected value (EV)?","back":"Sum of outcomes weighted by probabilities; choose the higher EV if risk-neutral.","tags":["decision"]},
        {"front":"Base rate — why it matters","back":"It's the prior prevalence; ignoring it leads to base-rate neglect and miscalibration.","tags":["probability"]},
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

def detect_mobile():
    """Detect if user is on mobile device"""
    # Simple heuristic based on user agent (basic implementation)
    user_agent = st.get_option("browser.gatherUsageStats")
    # For now, we'll use a session state flag that can be toggled
    if "mobile_view" not in st.session_state:
        st.session_state.mobile_view = False
    return st.session_state.mobile_view

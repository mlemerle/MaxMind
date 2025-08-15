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
    except FileNotFoundError:
        # Fallback to empty list if file not found
        return []

def detect_mobile():
    """Detect if user is on mobile device"""
    # Simple heuristic based on user agent (basic implementation)
    user_agent = st.get_option("browser.gatherUsageStats")
    # For now, we'll use a session state flag that can be toggled
    if "mobile_view" not in st.session_state:
        st.session_state.mobile_view = False
    return st.session_state.mobile_view

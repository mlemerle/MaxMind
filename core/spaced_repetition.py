"""
Spaced repetition system using SM-2 algorithm
"""
import random
from datetime import date, timedelta
from core.state_management import get_state, save_state
from core.utils import today_iso, new_id

def schedule(card, q: int):
    """Schedule card using SM-2 algorithm"""
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

def due_cards():
    """Get cards due for review"""
    state = get_state()
    today = date.today().isoformat()
    cards = state["cards"]
    
    due = [c for c in cards if (not c.get("new")) and c.get("due", today) <= today]
    newbies = [c for c in cards if c.get("new")]
    
    # Randomize the order but maintain separation between due and new
    random.shuffle(due)
    random.shuffle(newbies)
    
    # Apply limits
    due = due[:state["settings"]["reviewLimit"]]
    newbies = newbies[:state["settings"]["newLimit"]]
    return due + newbies

def add_card(front: str, back: str, tags=None):
    """Add a new card to the deck"""
    if tags is None:
        tags = []
    
    from core.utils import Card
    from dataclasses import asdict
    
    new_card = Card(
        id=new_id(),
        front=front.strip(),
        back=back.strip(), 
        tags=tags,
        new=True
    )
    
    get_state()["cards"].append(asdict(new_card))
    save_state()

def remove_card(card_id: str) -> bool:
    """Remove a card by ID, returns True if found and removed"""
    cards = get_state()["cards"]
    for i, card in enumerate(cards):
        if card["id"] == card_id:
            cards.pop(i)
            save_state()
            return True
    return False

def search_cards(query: str):
    """Search cards by front, back, or tags"""
    query = query.lower().strip()
    if not query:
        return get_state()["cards"]
    
    results = []
    for card in get_state()["cards"]:
        if (query in card["front"].lower() or 
            query in card["back"].lower() or 
            any(query in tag.lower() for tag in card.get("tags", []))):
            results.append(card)
    
    return results

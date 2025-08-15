#!/usr/bin/env python3
"""
Script to parse the All Decks.txt file and generate Python card data
"""
import re
import html

def clean_html(text):
    """Remove HTML tags and decode entities"""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Decode HTML entities
    text = html.unescape(text)
    # Fix encoding issues
    text = text.replace('â€"', '—').replace('â€™', "'").replace('â€œ', '"').replace('â€', '"')
    return text.strip()

def parse_deck_file(filename):
    """Parse the tab-separated deck file"""
    cards = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        parts = line.split('\t')
        if len(parts) >= 2:
            front = clean_html(parts[0])
            back = clean_html(parts[1])
            
            # Determine tags based on content
            tags = []
            if any(word in front.lower() + back.lower() for word in ['moloch', 'multipolar', 'goodhart', 'chesterton']):
                tags.append('rationalism')
            if any(word in front.lower() + back.lower() for word in ['system', 'feedback', 'flow', 'stock']):
                tags.append('systems')
            if any(word in front.lower() + back.lower() for word in ['bayesian', 'probability', 'forecast']):
                tags.append('probability')
            if any(word in front.lower() + back.lower() for word in ['economics', 'economic', 'market', 'trap']):
                tags.append('economics')
            if any(word in front.lower() + back.lower() for word in ['philosophy', 'moral', 'ethics', 'utilitarian']):
                tags.append('philosophy')
            if any(word in front.lower() + back.lower() for word in ['ai', 'intelligence', 'paperclip', 'alignment']):
                tags.append('ai-safety')
            if 'quote' in front.lower() or front.startswith('"') or front.startswith("'"):
                tags.append('quotes')
            if any(word in front.lower() + back.lower() for word in ['chaos', 'fractal', 'attractor', 'lyapunov']):
                tags.append('chaos-theory')
            if any(word in front.lower() + back.lower() for word in ['game', 'nash', 'pareto', 'zero-sum']):
                tags.append('game-theory')
            
            if not tags:
                tags = ['general']
            
            cards.append({
                'front': front,
                'back': back,
                'tags': tags
            })
    
    return cards

def generate_python_cards(cards):
    """Generate Python code for the cards"""
    output = []
    
    for card in cards:
        front_escaped = card['front'].replace('"', '\\"').replace('\n', '\\n')
        back_escaped = card['back'].replace('"', '\\"').replace('\n', '\\n')
        tags_str = '","'.join(card['tags'])
        
        output.append(f'    {{"front":"{front_escaped}","back":"{back_escaped}","tags":["{tags_str}"]}},')
    
    return '\n'.join(output)

if __name__ == "__main__":
    cards = parse_deck_file("c:\\Users\\maxle\\OneDrive\\Desktop\\All Decks.txt")
    print(f"# Parsed {len(cards)} cards")
    print("EXTENDED_SEED = [")
    print(generate_python_cards(cards))
    print("]")

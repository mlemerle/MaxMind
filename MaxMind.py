# MaxMind.py
# Run: streamlit run MaxMind.py
# Deps: pip install streamlit graphviz matplotlib

from __future__ import annotations
import json, math, random, time, uuid
from dataclasses import dataclass, asdict, field
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import numpy as np
from graphviz import Digraph
import matplotlib.pyplot as plt

# AI Integration for Streamlit Cloud
def get_ai_api_key():
    """Get AI API key from Streamlit secrets or user input"""
    try:
        # First, try to get from Streamlit secrets (for admin-controlled deployment)
        if "openai" in st.secrets:
            return st.secrets["openai"]["api_key"]
        elif "ai" in st.secrets:
            return st.secrets["ai"]["openai_api_key"]
    except:
        pass
    
    # If no secrets, get from user input (for user-controlled deployment)
    return st.session_state.get("user_api_key", None)

def setup_ai_configuration():
    """Setup AI configuration in sidebar with persistent browser storage"""
    with st.sidebar:
        with st.expander("AI Configuration", expanded=False):
            st.markdown("### AI Content Generation")
            
            # Always show user input option
            st.markdown("**Enter your OpenAI API key:**")
            st.caption("Stored securely in your browser only")
            
            api_key = st.text_input(
                "API Key", 
                value=st.session_state.get("user_api_key", ""),
                type="password",
                placeholder="sk-...",
                help="Your key is saved in browser storage - no need to re-enter each visit"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save API Key"):
                    if api_key.startswith("sk-"):
                        st.session_state["user_api_key"] = api_key
                        # Save to browser localStorage
                        st.markdown(f"""
                        <script>
                        localStorage.setItem('maxmind_openai_key', '{api_key}');
                        </script>
                        """, unsafe_allow_html=True)
                        st.success("API key saved!")
                        st.rerun()
                    else:
                        st.error("Please enter a valid OpenAI API key")
            
            with col2:
                # Clear API key option
                if st.session_state.get("user_api_key"):
                    if st.button("Clear Key"):
                        st.session_state.pop("user_api_key", None)
                        st.markdown("""
                        <script>
                        localStorage.removeItem('maxmind_openai_key');
                        </script>
                        """, unsafe_allow_html=True)
                        st.warning("API key cleared")
                        st.rerun()
            
            # Show status
            if get_ai_api_key():
                st.success("AI content generation is enabled!")
                st.info("Ready to generate Topic Study content")
                return True
            else:
                st.warning("Enter API key to enable AI features")
                st.info("Get your key from https://platform.openai.com/api-keys")
                return False

def generate_ai_content(prompt: str):
    """Generate AI content using OpenAI with error handling"""
    api_key = get_ai_api_key()
    if not api_key:
        return "AI content requires API key configuration."
    
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an educational content generator. Create engaging, informative content suitable for adult learners."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "insufficient" in error_msg.lower():
            return f"""**API Quota Exceeded**

Unfortunately, the OpenAI API quota has been exceeded. Here's a fallback topic for today:

**Topic**: {generate_fallback_topic_content()}

*Note: Add your own OpenAI API key in the sidebar to enable AI-generated content, or continue with the manual topic selection.*"""
        else:
            return f"AI generation error: {error_msg}"

def generate_fallback_topic_content():
    """Generate fallback content when AI is unavailable"""
    fallback_topics = [
        "The Philosophy of Scientific Progress - How do we determine when one scientific theory is better than another?",
        "Cognitive Biases in Decision Making - Why do intelligent people make predictably irrational choices?", 
        "The Network Effects of Technology - How do platforms become dominant and what are the implications?",
        "Emergence in Complex Systems - How do simple rules create complex behaviors in nature and society?",
        "The Psychology of Expertise - What separates true experts from novices in any field?",
        "Economic Incentives and Human Behavior - How do market structures shape individual and collective actions?",
        "The History of Ideas - How do intellectual movements spread and transform societies?",
        "Metacognition and Learning - How can we improve our ability to learn and think about thinking?",
        "Political Economy and Institutional Design - Why do some institutions succeed while others fail?",
        "The Epistemology of Knowledge - How do we know what we know, and what are the limits of human understanding?"
    ]
    
    return random.choice(fallback_topics)

def generate_category_topic(category: str):
    """Generate AI-powered topic for Topic Study page"""
    api_key = get_ai_api_key()
    if not api_key:
        # Fallback to static topic
        return {
            'title': f"Introduction to {category}",
            'category': category,
            'difficulty': 'medium',
            'description': f"A foundational overview of key concepts in {category}.",
            'content': f"Study the fundamental principles and major concepts that define {category}. This topic will help you build a solid foundation for more advanced topics.",
            'questions': [
                f"What are the core principles of {category}?",
                f"How has {category} evolved over time?",
                f"What are the practical applications of {category}?"
            ],
            'applications': [f"Understanding {category} enhances critical thinking and problem-solving abilities."]
        }
    
    prompt = f"""Generate a detailed study topic for {category}. Return a JSON structure with:
    - title: An engaging, specific topic name
    - category: {category}
    - difficulty: easy/medium/hard
    - description: 2-sentence overview
    - content: 3-4 paragraph study material (use markdown)
    - questions: 3 thought-provoking questions
    - applications: 2-3 real-world applications

    Make it intellectually rigorous but accessible. Focus on concepts that build critical thinking."""
    
    try:
        import json
        response = generate_ai_content(prompt)
        # Try to parse as JSON, fallback to structured text
        if response.startswith('{'):
            return json.loads(response)
        else:
            return {
                'title': f"AI-Generated {category} Topic",
                'category': category,
                'difficulty': 'medium',
                'description': "An AI-curated topic for deep learning.",
                'content': response,
                'questions': [
                    f"What are the key insights from this {category} topic?",
                    "How can you apply this knowledge?",
                    "What questions does this raise for further study?"
                ],
                'applications': ["Enhanced critical thinking", "Improved problem-solving"]
            }
    except:
        # Fallback if AI fails
        return {
            'title': f"Exploring {category} Fundamentals",
            'category': category,
            'difficulty': 'medium',
            'description': f"A comprehensive introduction to {category} concepts.",
            'content': f"This topic explores the foundational elements of {category}, examining key theories, methodologies, and applications that shape our understanding of this field.",
            'questions': [
                f"What defines the core of {category}?",
                f"How do {category} concepts apply to daily life?",
                f"What are the latest developments in {category}?"
            ],
            'applications': ["Critical thinking development", "Analytical reasoning enhancement"]
        }

def generate_intelligent_topic(domain=None):
    """Generate AI-powered topic for World Model page"""
    api_key = get_ai_api_key()
    
    domains = ["philosophy", "neuroscience", "psychology", "economics", "physics", "mathematics", "biology", "history"]
    if not domain:
        import random
        domain = random.choice(domains)
    
    if not api_key:
        # Fallback to static topic
        return {
            'topic': f"Fundamentals of {domain.title()}",
            'domain': domain,
            'level': 1,
            'description': f"Build foundational knowledge in {domain}",
            'suggested_duration': "15-20 minutes",
            'learning_objective': f"Understand basic {domain} concepts"
        }
    
    prompt = f"""Generate a micro-learning topic for {domain}. Return JSON with:
    - topic: Specific, intriguing topic name (not generic)
    - domain: {domain}
    - level: 1-5 (1=beginner, 5=expert)
    - description: 1 sentence explaining why this matters
    - suggested_duration: time estimate (e.g., "10-15 minutes")
    - learning_objective: clear goal statement

    Focus on topics that build mental models and connect to other domains. Make it intellectually stimulating."""
    
    try:
        import json
        import random
        response = generate_ai_content(prompt)
        if response.startswith('{'):
            return json.loads(response)
        else:
            # Fallback with better topics
            topics_by_domain = {
                'philosophy': ['Epistemic Humility', 'The Frame Problem', 'Moral Uncertainty'],
                'neuroscience': ['Predictive Processing', 'Neural Plasticity', 'Default Mode Network'],
                'psychology': ['Cognitive Biases', 'System 1 vs System 2', 'Flow States'],
                'economics': ['Nash Equilibrium', 'Opportunity Cost', 'Market Failures'],
                'physics': ['Entropy and Information', 'Quantum Superposition', 'Conservation Laws'],
                'mathematics': ['Proof by Contradiction', 'Exponential Growth', 'Fractals'],
                'biology': ['Evolutionary Pressure', 'Emergent Properties', 'Homeostasis'],
                'history': ['Historical Contingency', 'Cultural Evolution', 'Technology Adoption']
            }
            topic_name = random.choice(topics_by_domain.get(domain, [f"{domain.title()} Concepts"]))
            return {
                'topic': topic_name,
                'domain': domain,
                'level': random.randint(1, 3),
                'description': f"Explore {topic_name} and its implications for understanding reality",
                'suggested_duration': f"{random.choice([10, 15, 20])}-{random.choice([15, 20, 25])} minutes",
                'learning_objective': f"Develop intuition for {topic_name} principles"
            }
    except:
        return {
            'topic': f"Core Concepts in {domain.title()}",
            'domain': domain,
            'level': 1,
            'description': f"Build foundational understanding of {domain}",
            'suggested_duration': "15-20 minutes",
            'learning_objective': f"Grasp essential {domain} principles"
        }

# Import persistent storage
try:
    from storage import storage
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False

# ========== Utilities ==========
def parse_number_flexible(text):
    """Parse various number formats flexibly"""
    if not text:
        return None
    
    # Clean the text
    text = str(text).strip()
    
    # Remove common formatting
    text = text.replace(",", "")  # Remove commas
    text = text.replace("$", "")  # Remove dollar signs
    text = text.replace("%", "")  # Remove percentage signs
    text = text.replace(" ", "")  # Remove spaces
    
    # Handle billion, million, thousand suffixes
    multipliers = {
        'billion': 1_000_000_000, 'b': 1_000_000_000,
        'million': 1_000_000, 'm': 1_000_000,
        'thousand': 1_000, 'k': 1_000,
        'hundred': 100, 'h': 100
    }
    
    text_lower = text.lower()
    multiplier = 1
    
    for suffix, mult in multipliers.items():
        if text_lower.endswith(suffix):
            text = text_lower[:-len(suffix)]
            multiplier = mult
            break
    
    try:
        # Try to parse as float
        number = float(text) * multiplier
        return number
    except ValueError:
        return None

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

def evaluate_writing_with_ai(prompt, text):
    """Evaluate writing using OpenAI API and provide structured feedback"""
    api_key = get_ai_api_key()
    if not api_key:
        return {
            "error": "OpenAI API not available",
            "fallback_rating": "N/A",
            "fallback_feedback": "AI evaluation requires OpenAI API key. Manual self-reflection: Consider clarity, structure, evidence use, and logical flow."
        }
    
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        evaluation_prompt = f"""
        Please evaluate this writing piece on the following criteria and provide a structured response:

        **Prompt**: {prompt}

        **Written Response**: {text}

        Please provide:
        1. Overall Score (1-10): Rate the overall quality and effectiveness
        2. Clarity & Structure (1-10): How well organized and clear is the writing?
        3. Depth of Thinking (1-10): How deeply does the writer engage with the topic?
        4. Evidence & Examples (1-10): How well does the writer use examples and evidence?
        5. Areas for Improvement: 3 specific, actionable suggestions
        6. Strengths: 2-3 things the writer did well

        Format your response as:
        OVERALL_SCORE: [1-10]
        CLARITY_SCORE: [1-10] 
        DEPTH_SCORE: [1-10]
        EVIDENCE_SCORE: [1-10]
        
        STRENGTHS:
        - [strength 1]
        - [strength 2]
        
        IMPROVEMENTS:
        - [improvement 1]
        - [improvement 2] 
        - [improvement 3]
        
        DETAILED_FEEDBACK:
        [2-3 sentences of specific feedback about the content and structure]
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert writing coach focused on clear thinking and structured communication. Provide constructive, specific feedback."},
                {"role": "user", "content": evaluation_prompt}
            ],
            max_tokens=800,
            temperature=0.3
        )
        
        evaluation_text = response.choices[0].message.content
        
        # Parse the structured response
        lines = evaluation_text.split('\n')
        result = {
            "overall_score": 0,
            "clarity_score": 0, 
            "depth_score": 0,
            "evidence_score": 0,
            "strengths": [],
            "improvements": [],
            "detailed_feedback": "",
            "raw_response": evaluation_text
        }
        
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith("OVERALL_SCORE:"):
                try:
                    result["overall_score"] = int(line.split(":")[1].strip())
                except:
                    pass
            elif line.startswith("CLARITY_SCORE:"):
                try:
                    result["clarity_score"] = int(line.split(":")[1].strip())
                except:
                    pass
            elif line.startswith("DEPTH_SCORE:"):
                try:
                    result["depth_score"] = int(line.split(":")[1].strip())
                except:
                    pass
            elif line.startswith("EVIDENCE_SCORE:"):
                try:
                    result["evidence_score"] = int(line.split(":")[1].strip())
                except:
                    pass
            elif line.startswith("STRENGTHS:"):
                current_section = "strengths"
            elif line.startswith("IMPROVEMENTS:"):
                current_section = "improvements"
            elif line.startswith("DETAILED_FEEDBACK:"):
                current_section = "feedback"
            elif line.startswith("- ") and current_section == "strengths":
                result["strengths"].append(line[2:])
            elif line.startswith("- ") and current_section == "improvements":
                result["improvements"].append(line[2:])
            elif current_section == "feedback" and line:
                result["detailed_feedback"] += line + " "
        
        return result
        
    except Exception as e:
        return {
            "error": f"AI evaluation failed: {str(e)}",
            "fallback_rating": "Self-Assessment Needed",
            "fallback_feedback": "Consider: Was your argument clear? Did you use good examples? How could you structure your thoughts better?"
        }

# ========== Adaptive Difficulty System ==========
def update_difficulty(activity: str, accuracy: float, time_taken: float = None):
    """Update difficulty level based on performance (1-100 scale)"""
    if "difficulty" not in S():
        S()["difficulty"] = {}
    
    if activity not in S()["difficulty"]:
        S()["difficulty"][activity] = {"level": 50, "history": []}
    
    current_level = S()["difficulty"][activity]["level"]
    
    # Record performance
    performance_record = {
        "date": today_iso(),
        "accuracy": accuracy,
        "time_taken": time_taken,
        "level": current_level
    }
    S()["difficulty"][activity]["history"].append(performance_record)
    
    # Keep only last 10 records
    if len(S()["difficulty"][activity]["history"]) > 10:
        S()["difficulty"][activity]["history"] = S()["difficulty"][activity]["history"][-10:]
    
    # Adaptive logic: target 80-85% accuracy
    if accuracy > 0.85:
        # Too easy, increase difficulty
        new_level = min(100, current_level + 5)
        adjustment = "increased"
    elif accuracy < 0.80:
        # Too hard, decrease difficulty  
        new_level = max(1, current_level - 5)
        adjustment = "decreased"
    else:
        # In target range, no change
        new_level = current_level
        adjustment = "maintained"
    
    S()["difficulty"][activity]["level"] = new_level
    
    return {
        "old_level": current_level,
        "new_level": new_level,
        "adjustment": adjustment,
        "accuracy": accuracy
    }

def get_difficulty_level(activity: str) -> int:
    """Get current difficulty level for an activity (1-100)"""
    if "difficulty" not in S():
        return 50
    return S()["difficulty"].get(activity, {"level": 50})["level"]

def difficulty_to_scale(level: int, min_val: float, max_val: float) -> float:
    """Convert difficulty level (1-100) to a parameter scale"""
    # Level 1 = easiest (min_val), Level 100 = hardest (max_val)
    return min_val + (max_val - min_val) * (level - 1) / 99

def generate_ai_math_problem(difficulty_level: int, mode: str):
    """Generate AI-powered math problems based on difficulty"""
    # Fallback local generation if AI fails
    def fallback_generation():
        r = random.randint
        if mode == "Percent":
            if difficulty_level < 30:
                p, base = r(5, 20), r(50, 200)
            elif difficulty_level < 70:
                p, base = r(15, 40), r(100, 500)
            else:
                p, base = r(25, 85), r(200, 1500)
            return (f"What is {p}% of {base}?", round(base * p / 100, 2))
        
        elif mode == "Fraction→Decimal":
            if difficulty_level < 30:
                n, d = r(1, 5), r(2, 10)
            elif difficulty_level < 70:
                n, d = r(1, 12), r(3, 20)
            else:
                n, d = r(1, 25), r(5, 50)
            return (f"Convert {n}/{d} to decimal (4 dp).", round(n / d, 4))
        
        elif mode == "Quick Ops":
            if difficulty_level < 30:
                a, b = r(5, 25), r(2, 12)
                op = random.choice(["+", "−", "×"])
            elif difficulty_level < 70:
                a, b = r(12, 99), r(6, 24)
                op = random.choice(["×", "÷", "+", "−"])
            else:
                a, b = r(50, 999), r(12, 99)
                op = random.choice(["×", "÷"])
            
            if op == "×":
                ans = a * b
            elif op == "÷":
                ans = round(a / b, 3)
            elif op == "+":
                ans = a + b
            else:  # "−"
                ans = a - b
            return (f"{a} {op} {b} = ?", ans)
        
        else:  # Fermi
            fermi_problems = [
                # Easy (level < 30)
                [("How many minutes in 3 days?", 3*24*60),
                 ("How many hours in 2 weeks?", 2*7*24),
                 ("How many seconds in 1 hour?", 3600)],
                # Medium (level 30-70)
                [("How many minutes in 6 weeks?", 6*7*24*60),
                 ("Sheets in 2 cm stack at 0.1 mm/page?", 200),
                 ("Liters in 5m×3m×1m pool?", 15000)],
                # Hard (level > 70)
                [("Grains of sand in 1 cubic meter beach?", 1000000000),
                 ("Heartbeats in average human lifetime?", 2500000000),
                 ("Seconds in a century?", 100*365*24*3600)]
            ]
            
            if difficulty_level < 30:
                problems = fermi_problems[0]
            elif difficulty_level < 70:
                problems = fermi_problems[1]
            else:
                problems = fermi_problems[2]
            
            return random.choice(problems)
    
    # For now, use enhanced local generation
    # TODO: Add OpenAI API integration for truly dynamic problems
    return fallback_generation()

def record_session_performance(activity: str, score: int, total: int, time_seconds: float = None, **kwargs):
    """Record performance and update difficulty"""
    accuracy = score / max(1, total)
    
    # Update difficulty
    adjustment = update_difficulty(activity, accuracy, time_seconds)
    
    # Store session data
    session_data = {
        "date": today_iso(),
        "score": score,
        "total": total,
        "accuracy": accuracy,
        "time_seconds": time_seconds,
        "difficulty_level": adjustment["old_level"],
        "new_difficulty": adjustment["new_level"],
        "adjustment": adjustment["adjustment"],
        **kwargs
    }
    
    # Store in appropriate history
    if activity == "mental_math":
        S()["mmHistory"].append(session_data)
    # Add other activities as needed
    
    return adjustment

# ========== AI Topic Generation System ==========
def generate_intelligent_topic(domain: str = None, difficulty_level: int = 1):
    """Generate educational topics based on domain and difficulty"""
    
    # Knowledge domain topic pools organized by difficulty
    topic_pools = {
        "philosophy": {
            1: ["Socratic Method", "Stoicism basics", "Plato's Cave Allegory", "Aristotelian Ethics", "Free Will vs Determinism"],
            2: ["Existentialism", "Phenomenology", "Virtue Ethics", "Deontological Ethics", "Philosophy of Mind"],
            3: ["Logical Positivism", "Post-Structuralism", "Modal Logic", "Philosophy of Language", "Metaphysics of Time"]
        },
        "history": {
            1: ["Industrial Revolution", "Enlightenment Era", "Renaissance", "Ancient Greek Democracy", "Roman Empire"],
            2: ["Scientific Revolution", "Age of Exploration", "French Revolution", "American Civil War", "World War I"],
            3: ["Weimar Republic", "Decolonization", "Cold War Dynamics", "Medieval Scholasticism", "Byzantine Empire"]
        },
        "neuroscience": {
            1: ["Neuroplasticity", "Dopamine and Motivation", "Memory Formation", "Sleep and Brain Health", "Stress Response"],
            2: ["Prefrontal Cortex Function", "Neurotransmitter Systems", "Cognitive Load Theory", "Working Memory", "Attention Networks"],
            3: ["Default Mode Network", "Hemispheric Specialization", "Synaptic Plasticity", "Neurogenesis", "Glial Cell Function"]
        },
        "psychology": {
            1: ["Cognitive Biases", "Classical Conditioning", "Growth Mindset", "Maslow's Hierarchy", "Social Psychology"],
            2: ["Dual Process Theory", "Flow State", "Intrinsic Motivation", "Attachment Theory", "Cognitive Dissonance"],
            3: ["Terror Management Theory", "Social Identity Theory", "Prospect Theory", "Embodied Cognition", "Theory of Mind"]
        },
        "economics": {
            1: ["Supply and Demand", "Opportunity Cost", "Market Failures", "Behavioral Economics", "Public Goods"],
            2: ["Game Theory", "Information Asymmetry", "Network Effects", "Prisoner's Dilemma", "Tragedy of Commons"],
            3: ["Mechanism Design", "Principal-Agent Problems", "Auction Theory", "Monetary Policy", "Economic Cycles"]
        },
        "political_science": {
            1: ["Democracy vs Autocracy", "Separation of Powers", "Civil Liberties", "Political Participation", "Federalism"],
            2: ["Public Choice Theory", "Electoral Systems", "Interest Groups", "Political Culture", "Comparative Politics"],
            3: ["Institutional Design", "Democratic Transitions", "International Relations", "Political Economy", "Governance Theory"]
        },
        "cognitive_science": {
            1: ["System 1 vs System 2", "Heuristics and Biases", "Mental Models", "Cognitive Load", "Pattern Recognition"],
            2: ["Metacognition", "Transfer Learning", "Expertise Development", "Analogical Reasoning", "Conceptual Change"],
            3: ["Embodied Cognition", "Extended Mind", "Cognitive Architectures", "Computational Models", "Consciousness"]
        },
        "systems_thinking": {
            1: ["Feedback Loops", "Emergence", "Leverage Points", "Systems Archetypes", "Stocks and Flows"],
            2: ["Complex Adaptive Systems", "Network Effects", "Phase Transitions", "Resilience", "Path Dependence"],
            3: ["Autopoiesis", "Cybernetics", "Chaos Theory", "Self-Organization", "Complexity Science"]
        },
        "decision_theory": {
            1: ["Expected Value", "Sunk Cost Fallacy", "Base Rate Neglect", "Anchoring Bias", "Confirmation Bias"],
            2: ["Bayesian Reasoning", "Multi-Criteria Decision", "Risk Assessment", "Utility Theory", "Decision Trees"],
            3: ["Prospect Theory", "Ambiguity Aversion", "Dynamic Inconsistency", "Social Choice Theory", "Mechanism Design"]
        },
        "epistemology": {
            1: ["Scientific Method", "Falsifiability", "Induction vs Deduction", "Knowledge vs Belief", "Empiricism vs Rationalism"],
            2: ["Paradigm Shifts", "Theory-Ladenness", "Underdetermination", "Realism vs Anti-realism", "Social Construction"],
            3: ["Gettier Problems", "Externalism", "Reliabilism", "Foundationalism", "Coherentism"]
        }
    }
    
    # Select domain
    if not domain:
        domain = random.choice(list(topic_pools.keys()))
    
    # Get user's current level in this domain
    user_level = S().get("topic_suggestions", {}).get("knowledge_domains", {}).get(domain, {}).get("level", 1)
    effective_level = min(3, max(1, user_level))
    
    # Get topic pool for this level
    available_topics = topic_pools.get(domain, {}).get(effective_level, ["General Knowledge"])
    
    # Avoid recent topics
    recent_topics = S().get("topic_suggestions", {}).get("knowledge_domains", {}).get(domain, {}).get("recent_topics", [])
    fresh_topics = [t for t in available_topics if t not in recent_topics]
    
    if not fresh_topics:
        fresh_topics = available_topics  # Reset if all topics used
    
    selected_topic = random.choice(fresh_topics)
    
    return {
        "topic": selected_topic,
        "domain": domain,
        "level": effective_level,
        "description": f"Explore {selected_topic} in the context of {domain.replace('_', ' ').title()}",
        "suggested_duration": "15-20 minutes",
        "learning_objective": f"Understand key concepts and applications of {selected_topic}"
    }

def update_topic_progress(domain: str, topic: str, mastered: bool = False):
    """Update progress in knowledge domains"""
    if "topic_suggestions" not in S():
        S()["topic_suggestions"] = {"knowledge_domains": {}}
    
    if "knowledge_domains" not in S()["topic_suggestions"]:
        S()["topic_suggestions"]["knowledge_domains"] = {}
    
    if domain not in S()["topic_suggestions"]["knowledge_domains"]:
        S()["topic_suggestions"]["knowledge_domains"][domain] = {"level": 1, "recent_topics": []}
    
    domain_data = S()["topic_suggestions"]["knowledge_domains"][domain]
    
    # Add to recent topics
    if topic not in domain_data["recent_topics"]:
        domain_data["recent_topics"].append(topic)
    
    # Keep only last 5 recent topics per domain
    if len(domain_data["recent_topics"]) > 5:
        domain_data["recent_topics"] = domain_data["recent_topics"][-5:]
    
    # Level up if mastered (every 3 mastered topics in a domain)
    if mastered:
        if "mastered_count" not in domain_data:
            domain_data["mastered_count"] = 0
        domain_data["mastered_count"] += 1
        
        if domain_data["mastered_count"] % 3 == 0 and domain_data["level"] < 3:
            domain_data["level"] += 1
            return f"🎓 Leveled up in {domain.replace('_', ' ').title()}! Now at Level {domain_data['level']}"
    
    return None

# ========== Healthy Baseline System ==========
def get_healthy_baseline_status():
    """Get completion status for healthy baseline activities"""
    if "healthy_baseline" not in S():
        return {}
    
    # Reset if new day
    if S()["healthy_baseline"]["last_reset"] != today_iso():
        reset_healthy_baseline()
    
    return S()["healthy_baseline"]["completed"]

def mark_healthy_baseline_completed(activity: str):
    """Mark a healthy baseline activity as completed"""
    if "healthy_baseline" not in S():
        S()["healthy_baseline"] = {
            "last_reset": today_iso(),
            "completed": {},
            "streaks": {},
            "daily_history": {}
        }
    
    S()["healthy_baseline"]["completed"][activity] = True
    
    # Update streak
    if activity not in S()["healthy_baseline"]["streaks"]:
        S()["healthy_baseline"]["streaks"][activity] = 0
    S()["healthy_baseline"]["streaks"][activity] += 1
    
    save_state()

def reset_healthy_baseline():
    """Reset healthy baseline for new day"""
    if "healthy_baseline" not in S():
        return
    
    # Save yesterday's completion to history
    yesterday_completed = S()["healthy_baseline"]["completed"].copy()
    yesterday_date = S()["healthy_baseline"]["last_reset"]
    
    S()["healthy_baseline"]["daily_history"][yesterday_date] = yesterday_completed
    
    # Reset all completions for today
    for activity in S()["healthy_baseline"]["completed"]:
        if not S()["healthy_baseline"]["completed"][activity]:
            # Streak broken
            S()["healthy_baseline"]["streaks"][activity] = 0
    
    # Reset completions
    S()["healthy_baseline"]["completed"] = {
        "meditation": False,
        "sleep_quality": False,
        "nutrition": False,
        "exercise": False,
        "social_engagement": False,
        "hydration": False,
        "sunlight": False,
        "reading": False
    }
    
    S()["healthy_baseline"]["last_reset"] = today_iso()
    save_state()

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
        "mental_math": {"skill": 1200.0, "last_level": None, "recent_scores": [], "sessions_today": 0},
        "crt": {"skill": 1200.0, "last_level": None, "recent_scores": [], "sessions_today": 0},
        "base_rate": {"skill": 1200.0, "last_level": None, "recent_scores": [], "sessions_today": 0},
        "anchoring": {"skill": 1200.0, "last_level": None, "recent_scores": [], "sessions_today": 0},
        "K": 32.0,        # Elo K-factor
        "target_min": 0.80,  # target range 80-85%
        "target_max": 0.85,
        "base": 1100.0,   # base rating for easiest level
        "step": 50.0,     # rating step per level increment
        "window_size": 5  # sessions to consider for auto-adjustment
    },
    # NEW: Enhanced difficulty system (1-100 scale)
    "difficulty": {
        "mental_math": {"level": 50, "history": []},
        "crt": {"level": 50, "history": []},
        "base_rate": {"level": 50, "history": []},
        "anchoring": {"level": 50, "history": []},
        "nback": {"level": 50, "history": []},
        "task_switching": {"level": 50, "history": []},
        "complex_span": {"level": 50, "history": []},
        "gng": {"level": 50, "history": []},
        "processing_speed": {"level": 50, "history": []},
        "writing": {"level": 50, "history": []},
        "forecasts": {"level": 50, "history": []}
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
            "crt": False,
            "base_rate": False,
            "anchoring": False,
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
        "last_suggestion_date": None,
        "knowledge_domains": {
            "philosophy": {"level": 1, "recent_topics": []},
            "history": {"level": 1, "recent_topics": []},
            "neuroscience": {"level": 1, "recent_topics": []},
            "psychology": {"level": 1, "recent_topics": []},
            "economics": {"level": 1, "recent_topics": []},
            "political_science": {"level": 1, "recent_topics": []},
            "cognitive_science": {"level": 1, "recent_topics": []},
            "systems_thinking": {"level": 1, "recent_topics": []},
            "decision_theory": {"level": 1, "recent_topics": []},
            "epistemology": {"level": 1, "recent_topics": []}
        }
    },
    # NEW: Healthy Baseline Tracking
    "healthy_baseline": {
        "last_reset": today_iso(),
        "completed": {
            "meditation": False,
            "sleep_quality": False,
            "nutrition": False,
            "exercise": False,
            "social_engagement": False,
            "hydration": False,
            "sunlight": False,
            "reading": False
        },
        "streaks": {
            "meditation": 0,
            "sleep_quality": 0,
            "nutrition": 0,
            "exercise": 0,
            "social_engagement": 0,
            "hydration": 0,
            "sunlight": 0,
            "reading": 0
        },
        "daily_history": {}
    },
    # NEW: World-Model Learning system
    "world_model": {
        "current_tracks": ["probabilistic_reasoning", "systems_complexity"],
        "track_progress": {
            "probabilistic_reasoning": {"lesson": 0, "completed": []},
            "systems_complexity": {"lesson": 0, "completed": []},
            "decision_science": {"lesson": 0, "completed": []},
            "cognitive_science": {"lesson": 0, "completed": []},
            "evolutionary_thinking": {"lesson": 0, "completed": []},
            "information_theory": {"lesson": 0, "completed": []},
            "game_theory": {"lesson": 0, "completed": []},
            "network_effects": {"lesson": 0, "completed": []},
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

# ========== Progress Tracking & Milestones ==========
def get_weekly_progress():
    """Get current week progress and milestone data"""
    state = S()
    
    # Initialize progress tracking if not exists
    if "progress_tracking" not in state:
        state["progress_tracking"] = {
            "start_date": today_iso(),
            "weekly_milestones": {},
            "achievements": []
        }
        save_state()
    
    # Calculate current week
    from datetime import datetime
    start_date = datetime.fromisoformat(state["progress_tracking"]["start_date"])
    current_date = datetime.now()
    days_elapsed = (current_date - start_date).days
    current_week = (days_elapsed // 7) + 1
    
    # Get this week's milestones
    week_key = f"week_{current_week}"
    weekly_data = state["progress_tracking"]["weekly_milestones"].get(week_key, {})
    
    # Count milestones hit this week
    cognitive_domains = ["nback", "stroop", "complex_span", "gng", "processing_speed", "review", "topic_study"]
    milestones_hit = 0
    
    for domain in cognitive_domains:
        if weekly_data.get(f"{domain}_milestone", False):
            milestones_hit += 1
    
    return {
        "current_week": current_week,
        "milestones_this_week": milestones_hit,
        "total_milestones": 7,
        "weekly_data": weekly_data
    }

def check_and_award_milestone(drill: str, level_achieved: int, consistency_days: int):
    """Check if user hit a milestone and award if so"""
    state = S()
    progress = state["progress_tracking"]
    current_week = get_weekly_progress()["current_week"]
    week_key = f"week_{current_week}"
    
    if week_key not in progress["weekly_milestones"]:
        progress["weekly_milestones"][week_key] = {}
    
    milestone_key = f"{drill}_milestone"
    
    # Milestone criteria: 3+ sessions this week AND level progression
    if consistency_days >= 3 and not progress["weekly_milestones"][week_key].get(milestone_key, False):
        # Check if they've improved from last week
        last_week_key = f"week_{current_week - 1}" if current_week > 1 else None
        last_week_level = 0
        
        if last_week_key and last_week_key in progress["weekly_milestones"]:
            last_week_level = progress["weekly_milestones"][last_week_key].get(f"{drill}_best_level", 0)
        
        if level_achieved > last_week_level:
            # Award milestone!
            progress["weekly_milestones"][week_key][milestone_key] = True
            progress["weekly_milestones"][week_key][f"{drill}_best_level"] = level_achieved
            
            # Add achievement
            achievement = {
                "date": today_iso(),
                "type": "milestone",
                "drill": drill,
                "week": current_week,
                "level": level_achieved
            }
            progress["achievements"].append(achievement)
            save_state()
            return True
    
    return False

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
    },
    "cognitive_science": {
        "name": "Cognitive Science & Mental Models",
        "lessons": [
            {
                "title": "Dual Process Theory (System 1 & 2)",
                "content": "System 1: Fast, automatic, intuitive thinking. System 2: Slow, deliberate, analytical thinking. Most cognitive biases emerge from System 1's shortcuts. Knowing when to engage System 2 is crucial.",
                "example": "Stroop effect: naming colors when word and color don't match (RED written in blue). System 1 reads the word automatically, System 2 must override to name the color.",
                "questions": [
                    "When should you trust System 1 vs System 2?",
                    "How can you recognize when biases are active?",
                    "What triggers analytical thinking?"
                ],
                "transfer": "Apply to: financial decisions, hiring judgments, or any high-stakes choices."
            },
            {
                "title": "Mental Models & Analogical Reasoning",
                "content": "Mental models are simplified representations of how things work. Good models capture essential structure while ignoring irrelevant details. Cross-domain analogies transfer insights between fields.",
                "example": "Atom model evolved: plum pudding → planetary → quantum orbital. Each model useful for different purposes. Flow dynamics apply to traffic, rivers, and electric current.",
                "questions": [
                    "What makes a mental model useful?",
                    "How do analogies facilitate learning?",
                    "When do models break down?"
                ],
                "transfer": "Apply to: learning new domains, problem-solving, or explaining complex ideas."
            }
        ]
    },
    "evolutionary_thinking": {
        "name": "Evolutionary & Adaptive Systems",
        "lessons": [
            {
                "title": "Selection Pressures & Fitness Landscapes",
                "content": "Evolution optimizes for reproductive fitness, not happiness or truth. Selection pressures shape all adaptive systems: biological, cultural, technological. Fitness landscapes have peaks (good solutions) and valleys (poor solutions).",
                "example": "Peacock's tail: sexually selected despite survival cost. Business strategy: companies evolve features that attract customers, even if inefficient. Local vs global optima problem.",
                "questions": [
                    "What selection pressures shape human behavior?",
                    "How do systems get trapped in local optima?",
                    "What drives cultural vs genetic evolution?"
                ],
                "transfer": "Apply to: organizational design, product development, or understanding social norms."
            },
            {
                "title": "Red Queen Dynamics & Arms Races",
                "content": "Red Queen hypothesis: constant evolution needed just to maintain relative fitness. Arms races: competing systems drive each other to greater complexity. Zero-sum competition vs positive-sum cooperation.",
                "example": "Predator-prey coevolution: cheetahs get faster, antelopes get faster. Technology: virus vs antivirus software. Markets: competitors constantly innovate to maintain advantage.",
                "questions": [
                    "When do you see Red Queen dynamics?",
                    "How can arms races become wasteful?",
                    "When does cooperation beat competition?"
                ],
                "transfer": "Apply to: competitive strategy, technological development, or geopolitical analysis."
            }
        ]
    },
    "information_theory": {
        "name": "Information Theory & Communication",
        "lessons": [
            {
                "title": "Entropy, Surprise, and Information Content",
                "content": "Information = reduction in uncertainty. Entropy measures average surprise. High-probability events carry little information. Compression exploits redundancy patterns.",
                "example": "Weather report: 'sunny in Phoenix' (low info) vs 'snow in Phoenix' (high info). Language: 'the' appears frequently, 'sesquipedalian' rarely. Zip files work by finding patterns.",
                "questions": [
                    "What makes a message informative?",
                    "How does redundancy relate to compression?",
                    "Why do rare events feel more significant?"
                ],
                "transfer": "Apply to: data analysis, communication strategy, or understanding media attention."
            },
            {
                "title": "Signal vs Noise & Channel Capacity",
                "content": "Signal = meaningful information. Noise = random interference. Signal-to-noise ratio determines communication quality. Channel capacity limits maximum information transmission rate.",
                "example": "Conversation in noisy restaurant: must speak louder (amplify signal) or find quieter spot (reduce noise). Email: subject line = high signal, spam = noise.",
                "questions": [
                    "How do you improve signal-to-noise ratio?",
                    "What limits information transmission?",
                    "How do systems adapt to noisy channels?"
                ],
                "transfer": "Apply to: data interpretation, effective communication, or filtering information."
            }
        ]
    },
    "game_theory": {
        "name": "Game Theory & Strategic Thinking",
        "lessons": [
            {
                "title": "Nash Equilibrium & Strategic Stability",
                "content": "Nash equilibrium: no player can improve by unilaterally changing strategy. Represents stable outcome where everyone's strategy is optimal given others' strategies. Multiple equilibria possible.",
                "example": "Prisoner's dilemma: mutual defection is Nash equilibrium but mutual cooperation gives better outcomes. Traffic conventions: driving on right vs left both stable equilibria.",
                "questions": [
                    "What makes a strategy equilibrium stable?",
                    "Can Nash equilibria be inefficient?",
                    "How do you find equilibrium points?"
                ],
                "transfer": "Apply to: business strategy, negotiations, or understanding social conventions."
            },
            {
                "title": "Coordination vs Competition Games",
                "content": "Coordination games: players benefit from matching strategies. Competition games: players benefit from different strategies. Mixed-motive games combine both elements.",
                "example": "Coordination: choosing meeting location, technology standards. Competition: market positioning, resource allocation. Mixed: negotiation has both cooperative and competitive elements.",
                "questions": [
                    "When do coordination problems arise?",
                    "How do you solve coordination failures?",
                    "What strategies work in mixed-motive games?"
                ],
                "transfer": "Apply to: team coordination, industry standards, or international relations."
            }
        ]
    },
    "network_effects": {
        "name": "Network Effects & Emergence",
        "lessons": [
            {
                "title": "Scale-Free Networks & Power Laws",
                "content": "Scale-free networks: few highly connected hubs, many poorly connected nodes. Power law degree distribution. Preferential attachment: rich get richer. Small-world property: short paths between nodes.",
                "example": "Internet, social networks, citation networks. 80/20 rule emerges naturally. Six degrees of separation. Viral spread depends on network structure, not just transmission rate.",
                "questions": [
                    "What creates hub-and-spoke structures?",
                    "How do power laws emerge naturally?",
                    "Why are networks robust yet fragile?"
                ],
                "transfer": "Apply to: social influence, information spread, or understanding market concentration."
            },
            {
                "title": "Emergence & Phase Transitions",
                "content": "Emergence: system-level properties not present in components. Phase transitions: sudden qualitative changes from quantitative changes. Critical thresholds and tipping points.",
                "example": "Consciousness from neurons, traffic jams from individual cars, market crashes from individual trades. Water to ice at 32°F. Social movements reaching critical mass.",
                "questions": [
                    "How do simple rules create complex behavior?",
                    "What triggers sudden system changes?",
                    "How do you predict phase transitions?"
                ],
                "transfer": "Apply to: organizational change, social movements, or technology adoption."
            }
        ]
    }
}

def get_ai_enhanced_tracks():
    """Get available tracks, potentially enhanced with AI-generated content"""
    base_tracks = WORLD_MODEL_TRACKS.copy()
    
    # If AI is available, could potentially generate new tracks here
    api_key = get_ai_api_key()
    if api_key:
        # For now, return the expanded static tracks
        # Future: could generate personalized tracks based on user interests
        pass
    
    return base_tracks

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

def generate_category_topic(category: str) -> Dict[str, Any]:
    """Generate a topic suggestion based on selected category"""
    import random
    
    # Map display categories to internal topic pools
    category_mapping = {
        "Psychology": "psychology",
        "Neuroscience": "neuroscience", 
        "Philosophy": "philosophy",
        "History": "history",
        "Mathematics": "probability",  # Use existing math content
        "Physics": "neuroscience",    # Map to related content
        "Biology": "neuroscience",    # Map to neuroscience
        "Computer Science": "cognitive_science",
        "Economics": "economics",
        "Literature": "philosophy"    # Map to philosophy
    }
    
    topic_key = category_mapping.get(category, "psychology")
    
    # Get topic pools (these were defined earlier in the file)
    topic_pools = {
        "philosophy": {
            1: ["Socratic Method", "Stoicism basics", "Plato's Cave Allegory", "Aristotelian Ethics", "Free Will vs Determinism"],
            2: ["Existentialism", "Phenomenology", "Virtue Ethics", "Deontological Ethics", "Philosophy of Mind"],
            3: ["Logical Positivism", "Post-Structuralism", "Modal Logic", "Philosophy of Language", "Metaphysics of Time"]
        },
        "psychology": {
            1: ["Cognitive Biases", "Classical Conditioning", "Growth Mindset", "Maslow's Hierarchy", "Social Psychology"],
            2: ["Dual Process Theory", "Flow State", "Intrinsic Motivation", "Attachment Theory", "Cognitive Dissonance"],
            3: ["Terror Management Theory", "Social Identity Theory", "Prospect Theory", "Embodied Cognition", "Theory of Mind"]
        },
        "neuroscience": {
            1: ["Neuroplasticity", "Dopamine and Motivation", "Memory Formation", "Sleep and Brain Health", "Stress Response"],
            2: ["Prefrontal Cortex Function", "Neurotransmitter Systems", "Cognitive Load Theory", "Working Memory", "Attention Networks"],
            3: ["Default Mode Network", "Hemispheric Specialization", "Synaptic Plasticity", "Neurogenesis", "Glial Cell Function"]
        },
        "history": {
            1: ["Industrial Revolution", "Enlightenment Era", "Renaissance", "Ancient Greek Democracy", "Roman Empire"],
            2: ["Scientific Revolution", "Age of Exploration", "French Revolution", "American Civil War", "World War I"],
            3: ["Weimar Republic", "Decolonization", "Cold War Dynamics", "Medieval Scholasticism", "Byzantine Empire"]
        },
        "economics": {
            1: ["Supply and Demand", "Opportunity Cost", "Market Failures", "Behavioral Economics", "Public Goods"],
            2: ["Game Theory", "Information Asymmetry", "Network Effects", "Prisoner's Dilemma", "Tragedy of Commons"],
            3: ["Mechanism Design", "Principal-Agent Problems", "Auction Theory", "Monetary Policy", "Economic Cycles"]
        },
        "cognitive_science": {
            1: ["System 1 vs System 2", "Heuristics and Biases", "Mental Models", "Cognitive Load", "Pattern Recognition"],
            2: ["Metacognition", "Transfer Learning", "Expertise Development", "Analogical Reasoning", "Conceptual Change"],
            3: ["Embodied Cognition", "Extended Mind", "Cognitive Architectures", "Computational Models", "Consciousness"]
        },
        "probability": {  # For Mathematics category
            1: ["Bayes' Theorem", "Expected Value", "Law of Large Numbers", "Central Limit Theorem", "Probability Trees"],
            2: ["Conditional Probability", "Markov Chains", "Statistical Inference", "Confidence Intervals", "Hypothesis Testing"],
            3: ["Bayesian Networks", "Monte Carlo Methods", "Information Theory", "Decision Theory", "Stochastic Processes"]
        }
    }
    
    # Determine difficulty level based on user progress (simplified)
    state = S()
    if "topic_suggestions" not in state:
        state["topic_suggestions"] = {"study_history": []}
    
    study_count = len(state["topic_suggestions"]["study_history"])
    if study_count < 5:
        difficulty = 1
    elif study_count < 15:
        difficulty = 2
    else:
        difficulty = 3
    
    # Get topic from appropriate category and difficulty
    if topic_key in topic_pools and difficulty in topic_pools[topic_key]:
        topics = topic_pools[topic_key][difficulty]
        selected_topic = random.choice(topics)
        
        return {
            "title": selected_topic,
            "description": f"A {['beginner', 'intermediate', 'advanced'][difficulty-1]} topic in {category.lower()}",
            "category": category,
            "difficulty": difficulty,
            "key": f"{topic_key}_{selected_topic.lower().replace(' ', '_')}"
        }
    
    # Fallback to existing knowledge base
    fallback_topics = [k for k, v in TOPIC_KNOWLEDGE_BASE.items() if v.get("category", "").startswith(topic_key[:4])]
    if fallback_topics:
        topic_key = random.choice(fallback_topics)
        return {
            "key": topic_key,
            **TOPIC_KNOWLEDGE_BASE[topic_key]
        }
    
    # Final fallback
    return {
        "title": f"{category} Study Topic",
        "description": f"Explore a topic in {category.lower()}",
        "category": category,
        "difficulty": 1,
        "key": f"{category.lower()}_general"
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

def generate_key_terms_from_topic(topic: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate key terms and definitions from a topic for flashcard creation"""
    terms = []
    topic_title = topic.get('title', '')
    category = topic.get('category', 'general')
    content = topic.get('content', '')
    
    # Define category-specific key terms
    category_terms = {
        "psychology": [
            {"term": "Cognitive Bias", "definition": "Systematic errors in thinking that affect decisions and judgments"},
            {"term": "Classical Conditioning", "definition": "Learning process where a neutral stimulus becomes associated with a response"},
            {"term": "Growth Mindset", "definition": "Belief that abilities can be developed through dedication and hard work"},
            {"term": "Flow State", "definition": "Mental state of complete immersion and focused motivation in an activity"},
            {"term": "Cognitive Dissonance", "definition": "Mental discomfort from holding contradictory beliefs or values simultaneously"}
        ],
        "neuroscience": [
            {"term": "Neuroplasticity", "definition": "Brain's ability to reorganize and form new neural connections throughout life"},
            {"term": "Dopamine", "definition": "Neurotransmitter associated with reward, motivation, and pleasure"},
            {"term": "Working Memory", "definition": "System for temporarily holding and processing information during cognitive tasks"},
            {"term": "Prefrontal Cortex", "definition": "Brain region responsible for executive functions, decision-making, and planning"},
            {"term": "Synaptic Plasticity", "definition": "Ability of synapses to strengthen or weaken over time based on activity"}
        ],
        "philosophy": [
            {"term": "Socratic Method", "definition": "Form of inquiry and discussion using questions to examine ideas and beliefs"},
            {"term": "Stoicism", "definition": "Philosophy emphasizing virtue, tolerance of pain, and indifference to external circumstances"},
            {"term": "Existentialism", "definition": "Philosophy focusing on individual existence, freedom, and choice"},
            {"term": "Virtue Ethics", "definition": "Ethical theory emphasizing character traits rather than actions or consequences"},
            {"term": "Free Will", "definition": "Ability to make choices unconstrained by prior causes or divine decree"}
        ],
        "probability": [
            {"term": "Bayes' Theorem", "definition": "Formula for updating probability estimates based on new evidence: P(A|B) = P(B|A) × P(A) / P(B)"},
            {"term": "Base Rate", "definition": "Prior probability of an event before considering new evidence"},
            {"term": "Expected Value", "definition": "Average outcome of a random variable, calculated as sum of probability × outcome"},
            {"term": "Monte Carlo", "definition": "Method using random sampling to solve mathematical problems"},
            {"term": "Conditional Probability", "definition": "Probability of an event occurring given that another event has occurred"}
        ],
        "economics": [
            {"term": "Opportunity Cost", "definition": "Value of the best alternative forgone when making a choice"},
            {"term": "Network Effects", "definition": "Phenomenon where a product becomes more valuable as more people use it"},
            {"term": "Game Theory", "definition": "Mathematical framework for analyzing strategic interactions between rational decision-makers"},
            {"term": "Nash Equilibrium", "definition": "Solution where no player can benefit by changing strategy while others keep theirs unchanged"},
            {"term": "Behavioral Economics", "definition": "Field combining psychological insights with economic theory"}
        ],
        "history": [
            {"term": "Industrial Revolution", "definition": "Period of major industrialization and technological advancement (late 18th-19th century)"},
            {"term": "Enlightenment", "definition": "Intellectual movement emphasizing reason, science, and individual rights (17th-18th century)"},
            {"term": "Renaissance", "definition": "Cultural movement marking transition from medieval to modern Europe (14th-17th century)"},
            {"term": "Scientific Revolution", "definition": "Period of major advances in scientific thought and methodology (16th-17th century)"},
            {"term": "Cold War", "definition": "Period of geopolitical tension between US and Soviet Union (1947-1991)"}
        ],
        "cognitive_science": [
            {"term": "System 1 vs System 2", "definition": "Fast, automatic thinking vs. slow, deliberate reasoning (Kahneman's dual-process theory)"},
            {"term": "Heuristics", "definition": "Mental shortcuts that enable quick decision-making and problem-solving"},
            {"term": "Metacognition", "definition": "Awareness and understanding of one's own thought processes"},
            {"term": "Cognitive Load", "definition": "Amount of mental effort used in working memory during learning or problem-solving"},
            {"term": "Transfer Learning", "definition": "Application of knowledge and skills learned in one context to new situations"}
        ]
    }
    
    # Get terms for this category
    if category in category_terms:
        relevant_terms = category_terms[category][:3]  # Take first 3 most relevant
        terms.extend(relevant_terms)
    
    # Add topic-specific term based on title
    topic_specific = {
        "term": topic_title,
        "definition": topic.get('description', f"Key concept in {category}: {topic_title}")
    }
    terms.insert(0, topic_specific)  # Add at beginning
    
    # If content mentions specific terms, try to extract them
    if 'Bayes' in content or 'bayes' in content.lower():
        if not any(t['term'] == "Bayes' Theorem" for t in terms):
            terms.append({"term": "Bayes' Theorem", "definition": "P(A|B) = P(B|A) × P(A) / P(B) - updates probability based on new evidence"})
    
    if 'network effect' in content.lower():
        if not any('Network' in t['term'] for t in terms):
            terms.append({"term": "Network Effects", "definition": "Product becomes more valuable as more people use it"})
    
    if 'cognitive load' in content.lower():
        if not any('Cognitive Load' in t['term'] for t in terms):
            terms.append({"term": "Cognitive Load", "definition": "Mental effort used in working memory during learning or problem-solving"})
    
    return terms[:5]  # Return max 5 terms

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
    """Get completion status for all activities including healthy baseline"""
    check_daily_reset()
    
    # Get cognitive training completion status
    cognitive_completed = S()["daily"]["completed"].copy()
    
    # Get healthy baseline completion status
    baseline_completed = get_healthy_baseline_status()
    
    # Combine both for total daily progress
    all_completed = {**cognitive_completed, **baseline_completed}
    
    return all_completed

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
                            emoji = "🏆"
                            color = "#FFD700"  # Gold
                            border_color = "#FFA500"
                            text_color = "#000000"
                        elif percentage >= 80:
                            emoji = "●"
                            color = "#22c55e"  # Green
                            border_color = "#16a34a"
                            text_color = "#ffffff"
                        elif percentage >= 50:
                            emoji = "●"
                            color = "#f59e0b"  # Orange
                            border_color = "#d97706"
                            text_color = "#ffffff"
                        else:
                            emoji = "●"
                            color = "#ef4444"  # Red
                            border_color = "#dc2626"
                            text_color = "#ffffff"
                        display_text = f"{emoji}<br>{day_name} {day_num}<br><b>{percentage}%</b>"
                    elif day["is_today"]:
                        emoji = "●"
                        color = "#3b82f6"  # Blue
                        border_color = "#2563eb"
                        text_color = "#ffffff"
                        display_text = f"{emoji}<br>{day_name} {day_num}<br><b>TODAY</b>"
                    elif day["is_future"]:
                        emoji = "○"
                        color = "#6b7280"  # Gray
                        border_color = "#4b5563"
                        text_color = "#ffffff"
                        display_text = f"{emoji}<br>{day_name} {day_num}<br>—"
                    else:
                        emoji = "○"
                        color = "#6b7280"  # Gray
                        border_color = "#4b5563"
                        text_color = "#ffffff"
                        display_text = f"{emoji}<br>{day_name} {day_num}<br>Missed"
                    
                    # Create a button-like display with gold styling for perfect days
                    shadow = "0 0 15px rgba(255, 215, 0, 0.6)" if (day["completion"] and day["completion"]["percentage"] == 100) else "0 2px 4px rgba(0,0,0,0.1)"
                    
                    st.markdown(
                        f"""<div style='
                            text-align: center; 
                            padding: 8px; 
                            margin: 2px; 
                            background: {color}; 
                            border: 2px solid {border_color}; 
                            border-radius: 8px; 
                            font-size: 10px; 
                            height: 65px; 
                            display: flex; 
                            flex-direction: column; 
                            justify-content: center;
                            color: {text_color};
                            font-weight: bold;
                            box-shadow: {shadow};
                            transition: all 0.3s ease;
                        '>
                        {display_text}
                        </div>""",
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
    
    # Progress bar styling
    if progress_pct == 100:
        progress_gradient = "linear-gradient(90deg, #FFD700 0%, #FFA500 50%, #FFD700 100%)"
        progress_text_color = "#000000"
        progress_celebration = "PERFECT DAY"
    else:
        progress_gradient = "linear-gradient(90deg, #58a6ff 0%, #238636 100%)" if S().get("settings", {}).get("darkMode", False) else "linear-gradient(90deg, #007aff 0%, #00d4ff 100%)"
        progress_text_color = styles['text_color']
        progress_celebration = "Today's Progress"
    
    background_bar = "#21262d" if S().get("settings", {}).get("darkMode", False) else "#f1f5f9"
    
    st.markdown(f"""
    <div style="
        background: {styles['background']};
        padding: 1.5rem;
        border-radius: 12px;
        border: {styles['border']};
        box-shadow: {styles['shadow']};
        margin-bottom: 1.5rem;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
            <div style="font-weight: 600; color: {progress_text_color}; font-size: 1.1rem;">{progress_celebration}</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: {styles['accent_color']};">{completed_count}/{total_activities}</div>
        </div>
        <div style="
            background: {background_bar};
            border-radius: 8px;
            height: 8px;
            overflow: hidden;
            margin-bottom: 0.5rem;
        ">
            <div style="
                background: {progress_gradient};
                height: 100%;
                width: {progress_pct}%;
                border-radius: 8px;
                transition: width 0.3s ease;
                box-shadow: {'0 0 8px rgba(255, 215, 0, 0.4)' if progress_pct == 100 else 'none'};
            "></div>
        </div>
        <div style="color: {progress_text_color}; font-size: 0.85rem; text-align: center; font-weight: {'bold' if progress_pct == 100 else 'normal'};">{progress_pct}% Complete</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Weekly Progress Section (compact, as requested)
    progress_data = get_weekly_progress()
    current_week = progress_data["current_week"]
    milestones_hit = progress_data["milestones_this_week"]
    completion_pct = int((milestones_hit/7)*100)
    
    st.markdown(f"""
    <div style="
        background: {styles['background']};
        border: {styles['border']};
        padding: 0.75rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    ">
        <div style="
            color: {styles['text_color']};
            font-size: 0.9rem;
        ">
            <span style="font-weight: 600;">Week {current_week}:</span> {milestones_hit}/7 milestones • {completion_pct}% complete
        </div>
        <div style="
            background: {'#22c55e' if completion_pct > 50 else '#64748b'};
            color: white;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
        ">
            {'On Track' if completion_pct > 50 else 'Keep Going'}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Four main sections with clean dropdown functionality
    col1, col2, col3, col4 = st.columns(4)
    
    # Section 1: Healthy Baseline
    with col1:
        baseline_activities = ["reading", "meditation", "exercise", "sleep_quality", "hydration", "social_engagement", "nutrition", "sunlight"]
        baseline_completed = sum(1 for activity in baseline_activities if completed.get(activity, False))
        all_baseline_complete = baseline_completed == len(baseline_activities)
        
        st.markdown(f"""
        <div style="
            background: {styles['background']};
            padding: 1.25rem;
            border-radius: 12px;
            border: {styles['border']};
            box-shadow: {styles['shadow']};
            text-align: center;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            margin-bottom: 0.75rem;
            min-height: 140px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
            <div style="font-size: 1.25rem; margin-bottom: 0.5rem; color: {'#22c55e' if all_baseline_complete else styles['muted_color']};">{'✓' if all_baseline_complete else '○'}</div>
            <div style="font-weight: 600; color: {styles['text_color']}; margin-bottom: 0.25rem; font-size: 1rem;">Healthy Baseline</div>
            <div style="font-size: 1.75rem; font-weight: 700; color: {styles['accent_color']}; margin-bottom: 0.5rem;">{baseline_completed}/{len(baseline_activities)}</div>
            <div style="color: {styles['muted_color']}; font-size: 0.8rem;">activities done</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("View Activities", key="toggle_baseline", use_container_width=True):
            st.session_state.show_baseline_details = not st.session_state.get("show_baseline_details", False)
            st.rerun()
        
        if st.session_state.get("show_baseline_details", False):
            if st.button("Go to Healthy Baseline", key="baseline_page", use_container_width=True):
                st.session_state["page"] = "Healthy Baseline"
                st.rerun()

    # Section 2: Spaced Learning (includes Review, Topic Study, World Model)
    with col2:
        # Calculate spaced learning completion
        spaced_checks = {
            "review": completed.get("review", False),
            "topic_study": completed.get("topic_study", False),
            "world_model_a": completed.get("world_model_a", False),
            "world_model_b": completed.get("world_model_b", False)
        }
        spaced_completed = sum(1 for v in spaced_checks.values() if v)
        all_spaced_complete = spaced_completed == 4
        
        st.markdown(f"""
        <div style="
            background: {styles['background']};
            padding: 1.25rem;
            border-radius: 12px;
            border: {styles['border']};
            box-shadow: {styles['shadow']};
            text-align: center;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            margin-bottom: 0.75rem;
            min-height: 140px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
            <div style="font-size: 1.25rem; margin-bottom: 0.5rem; color: {'#22c55e' if all_spaced_complete else styles['muted_color']};">{'✓' if all_spaced_complete else '○'}</div>
            <div style="font-weight: 600; color: {styles['text_color']}; margin-bottom: 0.25rem; font-size: 1rem;">Spaced Learning</div>
            <div style="font-size: 1.75rem; font-weight: 700; color: {styles['accent_color']}; margin-bottom: 0.5rem;">{spaced_completed}/4</div>
            <div style="color: {styles['muted_color']}; font-size: 0.8rem;">activities done</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Toggle dropdown
        if st.button("View Activities", key="toggle_spaced", use_container_width=True):
            st.session_state.show_spaced_details = not st.session_state.get("show_spaced_details", False)
            st.rerun()
        
        # Dropdown content
        if st.session_state.get("show_spaced_details", False):
            
            # Review (with cards due count)
            dc = len(due_cards(S()))
            if st.button(f"Spaced Review ({dc} cards)", key="spaced_review", use_container_width=True):
                st.session_state["page"] = "Spaced Review"
                st.rerun()
            
            # Topic Study
            if st.button("Topic Study", key="spaced_topic", use_container_width=True):
                st.session_state["page"] = "Topic Study"
                st.rerun()
            
            # World Model A & B
            if st.button("World Model A", key="spaced_wm_a", use_container_width=True):
                st.session_state["page"] = "World Model"
                st.rerun()
            
            if st.button("World Model B", key="spaced_wm_b", use_container_width=True):
                st.session_state["page"] = "World Model"
                st.rerun()

    # Section 3: Cognitive Drills
    with col3:
        drill_checks = {
            "nback": completed.get("nback", False),
            "task_switching": completed.get("task_switching", False), 
            "complex_span": completed.get("complex_span", False),
            "gng": completed.get("gng", False),
            "processing_speed": completed.get("processing_speed", False)
        }
        completed_drills = sum(1 for v in drill_checks.values() if v)
        
        st.markdown(f"""
        <div style="
            background: {styles['background']};
            padding: 1.25rem;
            border-radius: 12px;
            border: {styles['border']};
            box-shadow: {styles['shadow']};
            text-align: center;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            margin-bottom: 0.75rem;
            min-height: 140px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
            <div style="font-size: 1.25rem; margin-bottom: 0.5rem; color: {'#22c55e' if completed_drills == 5 else styles['muted_color']};">{'✓' if completed_drills == 5 else '○'}</div>
            <div style="font-weight: 600; color: {styles['text_color']}; margin-bottom: 0.25rem; font-size: 1rem;">Cognitive Drills</div>
            <div style="font-size: 1.75rem; font-weight: 700; color: {styles['accent_color']}; margin-bottom: 0.5rem;">{completed_drills}/5</div>
            <div style="color: {styles['muted_color']}; font-size: 0.8rem;">completed today</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("View Drills", key="toggle_drills", use_container_width=True):
            st.session_state.show_drills_details = not st.session_state.get("show_drills_details", False)
            st.rerun()
        
        if st.session_state.get("show_drills_details", False):
            
            drill_activities = [
                ("Visual N-Back", "N-Back", "nback"),
                ("Task Switching", "Task Switching", "task_switching"),
                ("Complex Span", "Complex Span", "complex_span"),
                ("Go/No-Go", "Go/No-Go", "gng"),
                ("Processing Speed", "Processing Speed", "processing_speed")
            ]
            
            for name, page, key in drill_activities:
                if st.button(f"{name}", key=f"drill_{key}", use_container_width=True):
                    st.session_state["page"] = page
                    st.rerun()

    # Section 4: Learning Plus (includes Mental Math, Writing, Forecasts, Base Rate, Anchoring)
    with col4:
        additional_checks = {
            "writing": completed.get("writing", False),
            "forecasts": completed.get("forecasts", False),
            "mental_math": completed.get("mental_math", False),
            "base_rate": completed.get("base_rate", False),
            "anchoring": completed.get("anchoring", False)
        }
        additional_completed = sum(1 for v in additional_checks.values() if v)
        all_additional_complete = additional_completed == 5
        
        st.markdown(f"""
        <div style="
            background: {styles['background']};
            padding: 1.25rem;
            border-radius: 12px;
            border: {styles['border']};
            box-shadow: {styles['shadow']};
            text-align: center;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            margin-bottom: 0.75rem;
            min-height: 140px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
            <div style="font-size: 1.25rem; margin-bottom: 0.5rem; color: {'#22c55e' if all_additional_complete else styles['muted_color']};">{'✓' if all_additional_complete else '○'}</div>
            <div style="font-weight: 600; color: {styles['text_color']}; margin-bottom: 0.25rem; font-size: 1rem;">Learning Plus</div>
            <div style="font-size: 1.75rem; font-weight: 700; color: {styles['accent_color']}; margin-bottom: 0.5rem;">{additional_completed}/5</div>
            <div style="color: {styles['muted_color']}; font-size: 0.8rem;">activities done</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("View Activities", key="toggle_additional", use_container_width=True):
            st.session_state.show_additional_details = not st.session_state.get("show_additional_details", False)
            st.rerun()
        
        if st.session_state.get("show_additional_details", False):
            
            additional_activities = [
                ("Writing Exercise", "Writing", "writing"),
                ("Forecasting", "Forecasts", "forecasts"),
                ("Mental Math", "Mental Math", "mental_math"),
                ("Base Rate Training", "Base Rate", "base_rate"),
                ("Anchoring Resistance", "Anchoring", "anchoring")
            ]
            
            for name, page, key in additional_activities:
                if st.button(f"{name}", key=f"additional_{key}", use_container_width=True):
                    st.session_state["page"] = page
                    st.rerun()

    # Progress Management Section (as requested in red box area)
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("Progress", key="view_progress", use_container_width=True):
            st.session_state["page"] = "Adaptive Progression"
            st.rerun()
    with col2:
        if st.button("Reset Daily Progress", key="reset_progress", use_container_width=True):
            for key in S()["daily"]["completed"]:
                S()["daily"]["completed"][key] = False
            save_state()
            st.success("Daily progress reset!")
            st.rerun()
    with col3:
        # Show weekly milestone status
        milestones_this_week = get_weekly_progress()["milestones_this_week"]
        if st.button(f"{milestones_this_week}/7", key="milestone_status", use_container_width=True, help="Weekly milestones earned"):
            st.session_state["page"] = "Adaptive Progression"
            st.rerun()

    # Adaptive Suggestions
    st.markdown("### Adaptive Suggestions")
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
            
        st.write(f"**Complex Span** → Set size={CSPAN_GRID[cspan_idx]}")
        cs_feedback = get_performance_feedback("complex_span")
        if cs_feedback:
            st.caption(cs_feedback)
    
    with col_b:
        st.write(f"**Task Switching** → ISI={STROOP_GRID[ts_idx]}ms")
        ts_feedback = get_performance_feedback("stroop")
        if ts_feedback:
            st.caption(ts_feedback)
            
        st.write(f"**Go/No-Go** → %NoGo={GNG_GRID[gng_idx]}")
        gng_feedback = get_performance_feedback("gng")
        if gng_feedback:
            st.caption(gng_feedback)

    # Enhanced Topic Selection
    if not is_completed_today("topic_study"):
        st.markdown("### Today's Suggested Topic")
        
        # Category selection for topic study
        topic_categories = [
            "Psychology", "Neuroscience", "Philosophy", "History", 
            "Mathematics", "Physics", "Biology", "Computer Science",
            "Economics", "Literature"
        ]
        
        selected_category = st.selectbox(
            "Choose a category for today's study topic:",
            topic_categories,
            index=0,
            key="topic_category_select"
        )
        
        if st.button("Generate Topic for Selected Category", key="generate_topic"):
            # Generate topic based on selected category
            topic = generate_category_topic(selected_category)
            st.session_state["selected_topic"] = topic
            st.session_state["selected_category"] = selected_category
            st.rerun()
        
        # Display selected or default topic
        if "selected_topic" in st.session_state:
            topic = st.session_state["selected_topic"]
        else:
            topic = get_daily_topic_suggestion()
        
        with st.container():
            col_topic, col_button = st.columns([3, 1])
            with col_topic:
                st.markdown(f"**{topic['title']}**")
                st.caption(topic['description'])
            with col_button:
                if st.button("Start Study", key="start_study_btn"):
                    # Store the current category selection
                    st.session_state["selected_category"] = selected_category
                    st.session_state["page"] = "Topic Study"
                    st.rerun()
    
    # 60-Day Calendar
    render_calendar_grid()

def page_progress_dashboard():
    """Comprehensive progress tracking and milestone visualization"""
    page_header("📈 Adaptive Progression")
    
    progress_data = get_weekly_progress()
    current_week = progress_data["current_week"]
    
    # Weekly Progress Overview
    st.markdown("### Weekly Progress Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Week", current_week)
    with col2:
        st.metric("Milestones This Week", f"{progress_data['milestones_this_week']}/7")
    with col3:
        completion_rate = (progress_data['milestones_this_week'] / 7) * 100
        st.metric("Completion Rate", f"{completion_rate:.0f}%")
    
    # Weekly Progress Chart
    st.markdown("### Weekly Milestone Progress")
    
    # Get milestone data for chart
    state = S()
    weekly_milestones = state["progress_tracking"]["weekly_milestones"]
    
    weeks = []
    milestone_counts = []
    
    for week_num in range(1, current_week + 1):
        week_key = f"week_{week_num}"
        week_data = weekly_milestones.get(week_key, {})
        
        # Count milestones for this week
        count = 0
        domains = ["nback", "stroop", "complex_span", "gng", "processing_speed", "review", "topic_study"]
        for domain in domains:
            if week_data.get(f"{domain}_milestone", False):
                count += 1
        
        weeks.append(f"Week {week_num}")
        milestone_counts.append(count)
    
    # Simple bar chart using Streamlit
    if weeks:
        chart_data = {"Week": weeks, "Milestones": milestone_counts}
        st.bar_chart(chart_data, x="Week", y="Milestones")
    
    # Current Week Breakdown
    st.markdown("### This Week's Milestone Status")
    
    current_week_data = weekly_milestones.get(f"week_{current_week}", {})
    
    domains = [
        ("N-Back Training", "nback", "🧠"),
        ("Stroop/Task Switch", "stroop", "🔄"), 
        ("Complex Span", "complex_span", "📊"),
        ("Go/No-Go", "gng", "⚡"),
        ("Processing Speed", "processing_speed", "⚡"),
        ("Spaced Review", "review", "📚"),
        ("Topic Study", "topic_study", "🎓")
    ]
    
    for name, key, emoji in domains:
        milestone_achieved = current_week_data.get(f"{key}_milestone", False)
        best_level = current_week_data.get(f"{key}_best_level", 0)
        
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.write(f"{emoji} **{name}**")
        with col2:
            if milestone_achieved:
                st.success("✅ Milestone")
            else:
                st.info("⏳ In Progress")
        with col3:
            if best_level > 0:
                st.write(f"Level {best_level}")
    
    # Recent Achievements
    st.markdown("### Recent Achievements")
    
    achievements = state["progress_tracking"]["achievements"]
    recent_achievements = sorted(achievements, key=lambda x: x["date"], reverse=True)[:5]
    
    if recent_achievements:
        for achievement in recent_achievements:
            st.success(f"🏆 Week {achievement['week']}: {achievement['drill'].title()} Level {achievement['level']} milestone!")
    else:
        st.info("Complete 3+ sessions in a cognitive domain this week to earn your first milestone!")
    
    # Progress Tips
    st.markdown("### Progress Tips")
    st.info("""
    **How to earn milestones:**
    - Complete 3+ sessions in a cognitive domain this week
    - Show improvement from your previous week's best level
    - Milestones track your consistent progress and skill development
    
    **Why this matters:**
    - Progressive overload is essential for cognitive improvement
    - Consistent practice builds neural pathways
    - Week-over-week progress indicates real cognitive gains
    """)
    
    if st.button("← Back to Dashboard", use_container_width=True):
        st.session_state["page"] = "Dashboard"
        st.rerun()

def page_review():
    page_header("Spaced Repetition")
    st.caption("Flip → grade: Again/Hard/Good/Easy (SM-2).")

    # Always get fresh due cards and randomize them each time
    if "review_queue" not in st.session_state or st.button("🔄 Refresh Cards", help="Get fresh randomized cards"):
        fresh_due_cards = due_cards(S())
        # Double randomization to ensure proper shuffling
        random.shuffle(fresh_due_cards)
        st.session_state["review_queue"] = fresh_due_cards
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
    st.caption("Build your knowledge base with AI-generated topics for deep domain expertise.")
    
    # Check if already completed today
    if is_completed_today("topic_study"):
        st.success("Topic study completed for today!")
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

    # Domain selection and today's topic assignment
    st.markdown("### Today's Learning Assignment")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Check if user has a manually selected domain preference for today
        today_key = f"domain_preference_{today_iso()}"
        selected_domain = st.session_state.get(today_key, "Random")
        
        # Get or generate today's topic
        daily_topic_key = f"daily_topic_{today_iso()}"
        if daily_topic_key not in st.session_state:
            # Generate today's topic based on domain preference
            if selected_domain == "Random":
                topic = generate_intelligent_topic()
            else:
                domain_key = selected_domain.lower().replace(' ', '_')
                topic = generate_intelligent_topic(domain_key)
            st.session_state[daily_topic_key] = topic
        else:
            topic = st.session_state[daily_topic_key]
        
        # Display today's assigned topic
        with st.container(border=True):
            st.markdown(f"**Today's Topic: {topic['topic']}**")
            st.markdown(f"*Domain: {topic['domain'].replace('_', ' ').title()}* | *Level {topic['level']}*")
            
            # Handle AI-generated vs fallback content
            if isinstance(topic['description'], str) and topic['description'].startswith("**API Quota Exceeded**"):
                st.warning(topic['description'])
            else:
                st.info(topic['description'])
                
            st.caption(f"Duration: {topic['suggested_duration']} | Objective: {topic['learning_objective']}")
            
            # Action buttons
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Start Studying", key="start_study", use_container_width=True):
                    update_topic_progress(topic['domain'], topic['topic'])
                    st.success(f"Started studying: {topic['topic']}")
                    st.rerun()
            
            with col_b:
                if st.button("Mark as Mastered", key="master_topic", use_container_width=True):
                    level_up_msg = update_topic_progress(topic['domain'], topic['topic'], mastered=True)
                    if level_up_msg:
                        st.success(level_up_msg)
                    else:
                        st.success("Topic mastered!")
                    st.rerun()
    
    with col2:
        st.markdown("**Available Domains:**")
        
        # Comprehensive domain options with descriptions
        domain_descriptions = {
            "philosophy": "Fundamental questions about existence, knowledge, values",
            "history": "Human civilizations, events, and cultural evolution", 
            "neuroscience": "Brain structure, function, and cognitive processes",
            "psychology": "Human behavior, mind, and mental processes",
            "economics": "Markets, trade, resource allocation, and decision-making",
            "political_science": "Government, politics, and power structures",
            "cognitive_science": "How minds process information and learn",
            "systems_thinking": "Complex systems, feedback loops, and emergence",
            "decision_theory": "Rational choice, uncertainty, and optimization",
            "epistemology": "Nature of knowledge, truth, and justified belief"
        }
        
        # Show current domain preference
        today_key = f"domain_preference_{today_iso()}"
        current_selection = st.session_state.get(today_key, "Random")
        
        domain_options = ["Random"] + [key.replace("_", " ").title() for key in domain_descriptions.keys()]
        domain_keys = ["Random"] + list(domain_descriptions.keys())
        
        # Find current index
        try:
            current_idx = domain_keys.index(current_selection) if current_selection in domain_keys else 0
        except:
            current_idx = 0
            
        selected_idx = st.selectbox(
            "Choose learning focus:", 
            range(len(domain_options)),
            format_func=lambda x: domain_options[x],
            index=current_idx
        )
        
        selected_domain = domain_keys[selected_idx]
        
        if st.button("Get New Topic", key="new_daily_topic", use_container_width=True):
            # Update domain preference and generate new topic
            st.session_state[today_key] = selected_domain
            
            if selected_domain == "Random":
                new_topic = generate_intelligent_topic()
            else:
                new_topic = generate_intelligent_topic(selected_domain)
            
            st.session_state[daily_topic_key] = new_topic
            st.success(f"Generated new topic in {new_topic['domain'].replace('_', ' ').title()}!")
            st.rerun()
    
    # Domain Progress Overview
    st.markdown("### Domain Progress")
    domains = S().get("topic_suggestions", {}).get("knowledge_domains", {})
    if domains:
        cols = st.columns(min(len(domains), 4))
        for i, (domain, data) in enumerate(domains.items()):
            level = data.get("level", 1)
            mastered_count = len(data.get("recent_topics", []))
            
            with cols[i % len(cols)]:
                progress_stars = "★" * min(level, 5) + "☆" * max(0, 5 - level)
                st.metric(
                    domain.replace('_', ' ').title(),
                    f"Level {level}",
                    f"{mastered_count} mastered"
                )
                st.caption(progress_stars)
    
    st.markdown("---")

    # Study Material for today's topic
    st.markdown(f"### Study: {topic['topic']}")
    
    # Topic metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Domain**: {topic['domain'].replace('_', ' ').title()}")
    with col2:
        st.write(f"**Level**: {topic['level']}")
    with col3:
        st.write(f"**Duration**: {topic['suggested_duration']}")
    
    st.markdown(f"**Learning Objective**: {topic['learning_objective']}")
    
    # External References Section
    st.markdown("### 📚 External References")
    with st.expander("Research Papers, Videos & Resources", expanded=True):
        if st.button("Find External References", key="generate_references"):
            with st.spinner("Finding top papers, videos, and resources..."):
                references = generate_ai_content(
                    f"Find the most authoritative and helpful external references for learning about: {topic['topic']} in {topic['domain']}. "
                    f"Provide: 1) 3-5 top research papers or academic articles (with specific titles and authors if possible), "
                    f"2) 2-3 excellent YouTube videos or educational channels, "
                    f"3) 2-3 high-quality websites or online resources, "
                    f"4) 1-2 recommended books for deeper study. "
                    f"Format clearly with sections and include brief descriptions of why each resource is valuable."
                )
                st.session_state[f"references_{daily_topic_key}"] = references
                st.rerun()
        
        references_key = f"references_{daily_topic_key}"
        if references_key in st.session_state:
            st.markdown(st.session_state[references_key])
            st.info("💡 **Tip**: Use these references to deepen your understanding after completing the basic study material.")
        else:
            # Default references while AI content loads
            st.markdown("""
            **Academic Sources:**
            • Search Google Scholar for peer-reviewed papers on this topic
            • Check your local university library for academic journals
            
            **Video Resources:**
            • YouTube educational channels like Khan Academy, Coursera, TED-Ed
            • MIT OpenCourseWare and Stanford Online lectures
            
            **Web Resources:**
            • Wikipedia for basic overviews and further reading links
            • Professional association websites in the relevant field
            """)
    
    # Generate detailed content using AI
    if st.button("Generate Study Material", key="generate_content"):
        with st.spinner("Generating comprehensive study material..."):
            detailed_content = generate_ai_content(
                f"Create comprehensive study material for the topic: {topic['topic']} in the domain of {topic['domain']}. "
                f"Include: 1) Core concepts and definitions, 2) Key principles, 3) Examples and applications, "
                f"4) Common misconceptions, 5) Connections to related topics. "
                f"Make it suitable for someone at level {topic['level']} understanding."
            )
            st.session_state[f"content_{daily_topic_key}"] = detailed_content
            st.rerun()
    
    # Display generated content
    content_key = f"content_{daily_topic_key}"
    if content_key in st.session_state:
        with st.expander("Study Material", expanded=True):
            st.markdown(st.session_state[content_key])
        
        # Self-assessment questions
        with st.expander("Check Your Understanding"):
            if st.button("Generate Practice Questions", key="generate_questions"):
                questions = generate_ai_content(
                    f"Create 5 thoughtful questions to test understanding of: {topic['topic']}. "
                    f"Include a mix of conceptual, application, and analytical questions. "
                    f"Format as numbered list with clear, specific questions."
                )
                st.session_state[f"questions_{daily_topic_key}"] = questions
                st.rerun()
            
            questions_key = f"questions_{daily_topic_key}"
            if questions_key in st.session_state:
                st.markdown(st.session_state[questions_key])
                st.info("**Tip**: Try to answer these questions mentally before moving on.")
        
        # Real-World Applications
        with st.expander("Real-World Applications"):
            if st.button("Generate Applications", key="generate_applications"):
                applications = generate_ai_content(
                    f"List 5-7 practical, real-world applications of: {topic['topic']}. "
                    f"Include specific examples from daily life, professional contexts, and decision-making scenarios. "
                    f"Format as bullet points with clear, actionable examples."
                )
                st.session_state[f"applications_{daily_topic_key}"] = applications
                st.rerun()
            
            applications_key = f"applications_{daily_topic_key}"
            if applications_key in st.session_state:
                st.markdown(st.session_state[applications_key])
            else:
                # Default applications while AI generates custom ones
                st.markdown("""
                • **Critical thinking and problem solving** in complex situations
                • **Decision making** in personal and professional contexts  
                • **Understanding systems and relationships** in your environment
                • **Improved communication** when discussing related concepts
                • **Enhanced learning** in related academic or professional areas
                """)
        
        # Enhanced Flashcard Integration
        with st.expander("Create Flashcards from This Topic"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if st.button("Auto-Extract Key Terms", key="auto_extract_terms"):
                    with st.spinner("Extracting key terms and definitions..."):
                        terms_content = generate_ai_content(
                            f"Extract 6-8 key terms and concepts from the topic: {topic['topic']}. "
                            f"For each term, provide a clear, concise definition suitable for flashcards. "
                            f"Format exactly as: TERM | DEFINITION (one per line, using | as separator)"
                        )
                        
                        # Parse the AI response and create individual flashcards
                        if terms_content:
                            lines = terms_content.strip().split('\n')
                            created_count = 0
                            
                            for line in lines:
                                if '|' in line:
                                    parts = line.split('|', 1)
                                    if len(parts) == 2:
                                        term = parts[0].strip()
                                        definition = parts[1].strip()
                                        
                                        # Clean up formatting
                                        term = term.replace('**', '').replace('*', '').strip()
                                        definition = definition.replace('**', '').replace('*', '').strip()
                                        
                                        if term and definition:
                                            add_card(term, definition, [topic['domain'], 'topic-study', 'ai-generated'])
                                            created_count += 1
                            
                            if created_count > 0:
                                st.success(f"✅ Created {created_count} flashcards from key terms!")
                                st.info("Visit the Spaced Repetition page to study these cards.")
                            else:
                                st.warning("Could not extract terms in the expected format. Try manual entry below.")
                        st.rerun()
                
                # Manual flashcard creation
                st.markdown("**Manual Flashcard Creation:**")
                with st.form("manual_flashcard"):
                    term = st.text_input("Term/Concept", placeholder="e.g., Cognitive Bias")
                    definition = st.text_area("Definition/Explanation", 
                                            placeholder="e.g., Systematic errors in thinking that affect decisions and judgments",
                                            height=80)
                    
                    if st.form_submit_button("Add to Flashcards"):
                        if term.strip() and definition.strip():
                            add_card(term.strip(), definition.strip(), [topic['domain'], 'topic-study', 'manual'])
                            st.success(f"✅ Added '{term}' to your flashcards!")
                            st.rerun()
                        else:
                            st.error("Please fill in both term and definition")
            
            with col2:
                st.markdown("**Quick Actions:**")
                
                # Study existing related cards
                if st.button("Study Related Cards", key="study_related"):
                    domain_cards = [card for card in S()["cards"] 
                                  if topic['domain'] in card.get('tags', [])]
                    
                    if domain_cards:
                        st.info(f"Found {len(domain_cards)} related cards in {topic['domain']} domain")
                        st.caption("Go to Spaced Repetition to study them!")
                    else:
                        st.info("No existing cards found. Create some above!")
                
                # Export topic summary
                if st.button("Export Topic Summary", key="export_summary"):
                    summary = f"""# {topic['topic']}
                    
**Domain**: {topic['domain'].replace('_', ' ').title()}
**Level**: {topic['level']}
**Learning Objective**: {topic['learning_objective']}

## Study Material
{st.session_state.get(content_key, 'Generate study material first')}

## Practice Questions  
{st.session_state.get(questions_key, 'Generate questions first')}

## Applications
{st.session_state.get(applications_key, 'Generate applications first')}

---
*Generated by MaxMind Training on {today_iso()}*
"""
                    st.download_button(
                        "📄 Download Summary",
                        summary,
                        f"{topic['topic'].replace(' ', '_')}_summary.md",
                        "text/markdown"
                    )
                applications = generate_ai_content(
                    f"Identify 5-7 practical, real-world applications of: {topic['topic']}. "
                    f"Include specific examples from different domains like business, technology, "
                    f"personal development, society, and daily life. Format as bullet points."
                )
                st.session_state[f"applications_{daily_topic_key}"] = applications
                st.rerun()
            
            applications_key = f"applications_{daily_topic_key}"
            if applications_key in st.session_state:
                st.markdown(st.session_state[applications_key])
            else:
                # Default applications while AI content loads
                st.write("• Critical thinking and problem solving")
                st.write("• Decision making in personal and professional contexts")
                st.write("• Understanding complex systems and relationships")
        
        # Key Terms for Flashcards with proper integration
        with st.expander("Add Key Terms to Flashcards"):
            if st.button("Extract Key Terms", key="extract_terms"):
                terms_content = generate_ai_content(
                    f"Extract 5-7 key terms and concepts from the topic: {topic['topic']}. "
                    f"For each term, provide a clear, concise definition suitable for flashcards. "
                    f"Return as JSON format: [{{'term': 'Term Name', 'definition': 'Clear definition'}}]"
                )
                st.session_state[f"terms_{daily_topic_key}"] = terms_content
                st.rerun()
            
            terms_key = f"terms_{daily_topic_key}"
            if terms_key in st.session_state:
                terms_text = st.session_state[terms_key]
                
                # Try to parse as JSON for better integration
                try:
                    import json
                    # Clean the text to extract JSON
                    if "```json" in terms_text:
                        json_start = terms_text.find("[")
                        json_end = terms_text.rfind("]") + 1
                        if json_start != -1 and json_end > json_start:
                            terms_json = terms_text[json_start:json_end]
                            suggested_terms = json.loads(terms_json)
                            
                            st.markdown("**Suggested Terms:**")
                            selected_terms = []
                            
                            for term_data in suggested_terms:
                                term = term_data.get('term', '')
                                definition = term_data.get('definition', '')
                                if st.checkbox(f"**{term}**", key=f"term_{term}"):
                                    selected_terms.append(term_data)
                                    st.caption(f"*{definition}*")
                            
                            # Add selected terms to flashcards
                            if selected_terms:
                                if st.button(f"Add {len(selected_terms)} Selected Terms to Flashcards", key="add_selected_terms"):
                                    for term_data in selected_terms:
                                        add_card(
                                            term_data['term'], 
                                            term_data['definition'], 
                                            [topic['domain'].lower(), 'key-terms']
                                        )
                                    st.success(f"Added {len(selected_terms)} terms to your flashcards!")
                                    st.rerun()
                        else:
                            raise ValueError("No JSON found")
                    else:
                        raise ValueError("No JSON format")
                        
                except (json.JSONDecodeError, ValueError, KeyError):
                    # Fallback to text display
                    st.markdown("**Generated Terms:**")
                    st.markdown(terms_text)
                    st.caption("Copy important terms to create your own flashcards in the Spaced Repetition section.")
            
            # Manual term addition
            st.markdown("**Add Custom Term:**")
            with st.form("add_custom_term"):
                custom_term = st.text_input("Term/Concept", placeholder="e.g., Bayes' Theorem")
                custom_definition = st.text_area("Definition", placeholder="P(A|B) = P(B|A) × P(A) / P(B)", height=100)
                
                if st.form_submit_button("Add Custom Term"):
                    if custom_term and custom_definition:
                        add_card(custom_term, custom_definition, [topic['domain'].lower(), 'study'])
                        st.success(f"Added '{custom_term}' to your flashcards!")
                        st.rerun()

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
            complete_topic_study(topic['topic'], understanding, notes)
            st.success("Topic study completed! Great work!")
            # Clean up session state for tomorrow
            keys_to_remove = [k for k in st.session_state.keys() if daily_topic_key in k]
            for key in keys_to_remove:
                del st.session_state[key]
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

def handle_grade(card: Dict[str, Any], q: int):
    true_card = next((x for x in S()["cards"] if x["id"] == card["id"]), None)
    if true_card:
        schedule(true_card, q)
        true_card.setdefault("history", []).append({"date": today_iso(), "q": q})
        save_state()
    st.session_state["current_card"] = None
    st.session_state["show_back"] = False
    st.rerun()

# ----- Professional Dual N-Back Implementation -----
def page_nback():
    page_header("Visual N-Back (Simple & Working)")
    st.caption("Track **visual positions** in a 3x3 grid. Click **Visual Match** when current position matches N steps back.")

    # Instructions
    with st.expander("How to Play Visual N-Back", expanded=False):
        st.markdown("""
        **Visual N-Back trains working memory through pattern recognition:**
        
        **Visual Task**: Watch for squares flashing in the 3×3 grid  
        
        **Your Goal**: For each trial, determine if:
        - The **current visual position** matches the position from N trials ago
        
        **Controls**:
        - Click **Visual Match** when the current position matches the position from N steps back
        - Do nothing if it doesn't match
        
        **Tips**:
        - Start with N=2 and increase difficulty gradually
        - Use mental rehearsal: remember the sequence of positions
        - Accuracy is more important than speed
        - Focus on the pattern of positions over time
        """)

    # Adaptive Suggestions
    adaptive_idx = adaptive_suggest_index("nback")
    suggested_n, suggested_interval_ms = NBACK_GRID[adaptive_idx]
    suggested_speed = "Expert (1.5s)" if suggested_interval_ms <= 900 else "Fast (2s)" if suggested_interval_ms <= 1200 else "Medium (2.5s)" if suggested_interval_ms <= 1500 else "Slow (3s)"
    
    feedback = get_performance_feedback("nback")
    if feedback:
        st.info(f"🎯 **Adaptive Suggestion**: N={suggested_n}, Speed={suggested_speed} | {feedback}")

    # Settings
    col1, col2, col3 = st.columns(3)
    with col1:
        n = st.selectbox("N-Back Level", [1, 2, 3, 4], index=1, help="Memory span - how many steps back to remember")
    with col2:
        trials = st.selectbox("Number of Trials", [15, 20, 25, 30], index=1)
    with col3:
        speed = st.selectbox("Speed", ["Slow (3s)", "Medium (2.5s)", "Fast (2s)", "Expert (1.5s)"], index=1)
    
    # Convert speed to interval
    speed_map = {"Slow (3s)": 3.0, "Medium (2.5s)": 2.5, "Fast (2s)": 2.0, "Expert (1.5s)": 1.5}
    interval = speed_map[speed]

    # Initialize session state
    if "dual_nback" not in st.session_state:
        st.session_state["dual_nback"] = None

    # Start button
    if st.button("Start Visual N-Back Session", type="primary", use_container_width=True):
        # Generate sequences
        positions = [random.randint(0, 8) for _ in range(trials)]  # 3x3 grid positions (0-8)
        
        # Determine target trials (visual only)
        visual_targets = set()
        
        for i in range(n, trials):
            if positions[i] == positions[i - n]:
                visual_targets.add(i)
        
        # Initialize session
        st.session_state["dual_nback"] = {
            "n": n, "trials": trials, "interval": interval,
            "positions": positions,
            "visual_targets": visual_targets,
            "current_trial": 0, "visual_responses": set(),
            "session_start": time.time(), "trial_start": None,
            "status": "running", "results": None
        }
        st.rerun()

    # Main game interface
    dnb = st.session_state.get("dual_nback")
    if dnb and dnb["status"] == "running":
        # Progress bar
        progress = dnb["current_trial"] / dnb["trials"]
        st.progress(progress, text=f"Trial {dnb['current_trial'] + 1} of {dnb['trials']} • N={dnb['n']} • {speed}")
        
        # Current trial display
        if dnb["current_trial"] < dnb["trials"]:
            current_pos = dnb["positions"][dnb["current_trial"]]
            
            # Visual display
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("### Visual Grid")
                _render_dual_nback_grid(current_pos)
                
            with col2:
                st.markdown("### Response Control")
                st.info("Click when the current position matches the position from N trials ago")
                
                if st.button("Visual Match", key="visual_btn", use_container_width=True, type="primary"):
                    dnb["visual_responses"].add(dnb["current_trial"])
                    st.success("Match recorded!")
                    time.sleep(0.3)
            
            # Trial info and timing
            if dnb["current_trial"] >= dnb["n"]:
                with st.expander("N-Back Reference (for learning)", expanded=False):
                    past_pos = dnb["positions"][dnb["current_trial"] - dnb["n"]] + 1
                    is_visual_target = dnb["current_trial"] in dnb["visual_targets"]
                    status = "TARGET" if is_visual_target else "No match"
                    st.caption(f"Visual N-back: Position {past_pos} {status}")
            
            # Simplified timing display
            is_paused = dnb.get("paused", False)
            
            if not is_paused:
                if dnb["trial_start"] is None:
                    dnb["trial_start"] = time.time()
                
                elapsed = time.time() - dnb["trial_start"]
                remaining = max(0, dnb["interval"] - elapsed)
                
                # Progress bar for trial timing
                progress_pct = min(1.0, elapsed / dnb["interval"])
                st.progress(progress_pct, text=f"Trial {dnb['current_trial'] + 1} - Next in: {remaining:.1f}s")
                
                # Auto-advance when time is up
                if remaining <= 0:
                    dnb["current_trial"] += 1
                    dnb["trial_start"] = None
                    
                    if dnb["current_trial"] >= dnb["trials"]:
                        dnb["status"] = "completed"
                        dnb["results"] = _calculate_dual_nback_results(dnb)
                    
                    st.rerun()
                
                # Auto-refresh every second for smooth timer
                if remaining > 0:
                    time.sleep(1.0)
                    st.rerun()
            else:
                # Show paused state
                st.progress(0.0, text="Session Paused - Click Resume to continue")
        
        # Enhanced session controls
        st.markdown("### Session Controls")
        control_col1, control_col2, control_col3 = st.columns(3)
        
        # Check if session is paused
        is_paused = dnb.get("paused", False)
        
        with control_col1:
            if is_paused:
                if st.button("Resume", key="resume_session", help="Resume the paused session", use_container_width=True, type="primary"):
                    dnb["paused"] = False
                    dnb["trial_start"] = time.time()  # Reset trial timer
                    st.rerun()
            else:
                if st.button("Pause", key="pause_session", help="Pause the current session", use_container_width=True):
                    dnb["paused"] = True
                    dnb["trial_start"] = None
                    st.rerun()
        
        with control_col2:
            if st.button("Skip Trial", key="manual_next", help="Skip the timer and go to next trial", use_container_width=True):
                dnb["current_trial"] += 1
                dnb["trial_start"] = None
                dnb["paused"] = False  # Unpause if skipping
                if dnb["current_trial"] >= dnb["trials"]:
                    dnb["status"] = "completed"
                    dnb["results"] = _calculate_dual_nback_results(dnb)
                st.rerun()
        
        with control_col3:
            if st.button("Restart", key="restart_session", help="Start a new session", use_container_width=True):
                st.session_state["dual_nback"] = None
                st.rerun()
        
        # Show pause status
        if is_paused:
            st.warning("Session is paused. Click 'Resume' to continue or 'Skip Trial' to advance.")

    # Results display
    elif dnb and dnb["status"] == "completed":
        _display_dual_nback_results(dnb)

def _render_dual_nback_grid(highlighted_pos):
    """Render 3x3 grid with highlighted position using CSS styling for visual clarity"""
    
    # CSS for proper grid styling
    grid_css = """
    <style>
    .nback-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        grid-template-rows: repeat(3, 1fr);
        gap: 8px;
        width: 300px;
        height: 300px;
        margin: 20px auto;
        background-color: #2e2e2e;
        padding: 20px;
        border-radius: 10px;
    }
    .grid-cell {
        background-color: #1e1e1e;
        border: 2px solid #444;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .grid-cell.highlighted {
        background-color: #00ff88;
        border-color: #00cc66;
        color: #000;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
    }
    .grid-cell.normal {
        background-color: #333;
        border-color: #666;
        color: #888;
    }
    </style>
    """
    
    # Generate grid HTML
    grid_html = grid_css + '<div class="nback-grid">'
    
    for pos in range(9):
        if pos == highlighted_pos:
            grid_html += f'<div class="grid-cell highlighted">●</div>'
        else:
            grid_html += f'<div class="grid-cell normal"></div>'
    
    grid_html += '</div>'
    
    st.markdown(grid_html, unsafe_allow_html=True)
    st.caption(f"Position {highlighted_pos + 1} is highlighted")

def _calculate_dual_nback_results(dnb):
    """Calculate performance metrics for visual n-back session"""
    # Visual performance only
    visual_hits = len(dnb["visual_responses"] & dnb["visual_targets"])
    visual_misses = len(dnb["visual_targets"] - dnb["visual_responses"])
    visual_false_alarms = len(dnb["visual_responses"] - dnb["visual_targets"])
    visual_correct_rejections = dnb["trials"] - len(dnb["visual_targets"]) - visual_false_alarms
    
    # Calculate accuracy
    visual_accuracy = (visual_hits + visual_correct_rejections) / dnb["trials"] * 100
    
    return {
        "visual_hits": visual_hits,
        "visual_misses": visual_misses,
        "visual_false_alarms": visual_false_alarms,
        "visual_correct_rejections": visual_correct_rejections,
        "visual_accuracy": visual_accuracy,
        "overall_accuracy": visual_accuracy,
        "total_targets": len(dnb["visual_targets"]),
        "duration": (time.time() - dnb["session_start"]) / 60
    }

def _display_dual_nback_results(dnb):
    """Display session results and performance analysis"""
    st.success("Visual N-Back Session Complete!")
    
    results = dnb["results"]
    
    # Performance overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Score", f"{results['overall_accuracy']:.1f}%")
    with col2:
        st.metric("Correct Hits", results['visual_hits'])
    with col3:
        st.metric("False Alarms", results['visual_false_alarms'])
    
    # Detailed breakdown
    st.markdown("### Performance Details")
    
    # Visual performance only
    st.markdown("**Visual Performance:**")
    visual_col1, visual_col2, visual_col3, visual_col4 = st.columns(4)
    
    with visual_col1:
        st.metric("Hits", results['visual_hits'])
    with visual_col2:
        st.metric("Misses", results['visual_misses'])
    with visual_col3:
        st.metric("False Alarms", results['visual_false_alarms'])
    with visual_col4:
        st.metric("Accuracy", f"{results['visual_accuracy']:.1f}%")
    
    # Performance feedback
    if results['overall_accuracy'] >= 80:
        st.success("Excellent performance! Consider increasing difficulty.")
    elif results['overall_accuracy'] >= 60:
        st.info("Good performance! Keep practicing at this level.")
    else:
        st.error("Consider reducing difficulty")
    
    # Session info
    st.markdown("### Session Summary")
    st.write(f"**N-Level:** {dnb['n']}")
    st.write(f"**Trials:** {dnb['trials']}")
    st.write(f"**Duration:** {results['duration']:.1f} minutes")
    st.write(f"**Total Targets:** {results['total_targets']}")
    
    # Save results option
    if st.button("Save Results & Start New Session", type="primary"):
        # Simple save to session state for now
        if "nback_history" not in st.session_state:
            st.session_state["nback_history"] = []
        
        session_data = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "n_level": dnb['n'],
            "trials": dnb['trials'],
            "accuracy": results['overall_accuracy'],
            "hits": results['visual_hits'],
            "false_alarms": results['visual_false_alarms']
        }
        
        st.session_state["nback_history"].append(session_data)
        st.success("Results saved!")
        st.session_state["dual_nback"] = None
        st.rerun()
    
    # Strategy reflection
    st.markdown("### Strategy Reflection")
    strategy_rating = st.slider(f"How well did your strategy work?", 1, 5, 3, 
                              help="1=Not helpful, 5=Very helpful")
    
    with col1:
        st.metric("👁️ Visual Accuracy", f"{results['visual']['accuracy']:.1f}%")
        st.caption(f"Hits: {results['visual']['hits']} • False Alarms: {results['visual']['false_alarms']}")
    
    with col2:
        st.metric("🔊 Audio Accuracy", f"{results['audio']['accuracy']:.1f}%")
        st.caption(f"Hits: {results['audio']['hits']} • False Alarms: {results['audio']['false_alarms']}")
    
    with col3:
        st.metric("Overall Score", f"{results['overall_accuracy']:.1f}%")
        
        # Performance feedback
        if results['overall_accuracy'] >= 85:
            st.success("🌟 Excellent! Try increasing N-level")
        elif results['overall_accuracy'] >= 70:
            st.info("👍 Good performance!")
        elif results['overall_accuracy'] >= 55:
            st.warning("💪 Keep practicing at this level")
        else:
            st.error("Consider reducing difficulty")
    
    # Detailed breakdown
    with st.expander("Detailed Performance Analysis"):
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**Visual Performance:**")
            st.write(f"• Hits: {results['visual']['hits']}")
            st.write(f"• Misses: {results['visual']['misses']}")
            st.write(f"• False Alarms: {results['visual']['false_alarms']}")
            st.write(f"• Correct Rejections: {results['visual']['correct_rejections']}")
        
        with col_b:
            st.markdown("**Audio Performance:**")
            st.write(f"• Hits: {results['audio']['hits']}")
            st.write(f"• Misses: {results['audio']['misses']}")
            st.write(f"• False Alarms: {results['audio']['false_alarms']}")
            st.write(f"• Correct Rejections: {results['audio']['correct_rejections']}")
    
    # Save results and mark completion
    if st.button("💾 Save Results & Start New Session", type="primary"):
        _save_dual_nback_results(dnb, results)
        st.session_state["dual_nback"] = None
        st.success("Results saved!")
        st.rerun()

def _save_dual_nback_results(dnb, results):
    """Save dual n-back session results to user profile"""
    if "nback_history" not in S():
        S()["nback_history"] = []
    
    session_data = {
        "date": today_iso(),
        "type": "dual_nback",
        "n_level": dnb["n"],
        "trials": dnb["trials"],
        "interval": dnb["interval"],
        "visual_accuracy": results["visual"]["accuracy"],
        "audio_accuracy": results["audio"]["accuracy"], 
        "overall_accuracy": results["overall_accuracy"],
        "duration": results["duration"],
        "timestamp": time.time()
    }
    
    S()["nback_history"].append(session_data)
    mark_completed("nback")
    
    # Adaptive system update
    level_achieved = NBACK_GRID.index((dnb["n"], dnb["interval"])) if (dnb["n"], dnb["interval"]) in NBACK_GRID else 0
    adaptive_update("nback", level_achieved, results["overall_accuracy"])
    
    # Check and award milestone for progressive improvement
    sessions_this_week = len([s for s in S()["nback_history"] if (time.time() - s["timestamp"]) < 604800])  # 7 days
    
    milestone_awarded = check_and_award_milestone("nback", level_achieved, sessions_this_week)
    if milestone_awarded:
        st.balloons()
        st.success("🏆 Milestone achieved! You've shown consistent improvement in N-Back training this week!")
    
    save_state()

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
    st.markdown("### Strategy Reflection")
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
    
    # Get current difficulty level
    current_difficulty = get_difficulty_level("mental_math")
    
    st.info(f"🎯 Current Difficulty Level: {current_difficulty}/100 (Target: 80-85% accuracy)")
    
    mode = st.selectbox("Mode", ["Percent", "Fraction→Decimal", "Quick Ops", "Fermi"])
    duration_min = st.selectbox("Duration (min)", [2, 3, 5], index=1)
    tol = st.selectbox("Tolerance", ["Exact", "±5%", "±10%"], index=0)

    if "mm" not in st.session_state:
        st.session_state["mm"] = None

    def gen_problem():
        return generate_ai_math_problem(current_difficulty, mode)

    if st.button("Start"):
        end = now_ts() + duration_min * 60
        st.session_state["mm"] = {
            "end": end, 
            "score": 0, 
            "total": 0, 
            "cur": gen_problem(),
            "start_time": now_ts(),
            "problems_attempted": []
        }
        st.rerun()

    mm = st.session_state["mm"]
    if mm:
        left = int(mm["end"] - now_ts())
        st.metric("Time", timer_text(left))
        
        # Show current score
        if mm["total"] > 0:
            current_accuracy = mm["score"] / mm["total"]
            st.metric("Current Accuracy", f"{current_accuracy:.1%}")
        
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
                    
                    # Record problem attempt
                    mm["problems_attempted"].append({
                        "question": mm["cur"][0],
                        "user_answer": user_val,
                        "correct_answer": truth,
                        "correct": correct
                    })
                    
                except ValueError:
                    st.warning("Enter a number.")
            mm["cur"] = gen_problem()
            st.rerun()

        if left <= 0:
            # Session completed
            total_time = now_ts() - mm["start_time"]
            accuracy = mm["score"] / max(1, mm["total"])
            
            # Record performance and get difficulty adjustment
            adjustment = record_session_performance(
                "mental_math", 
                mm["score"], 
                mm["total"], 
                total_time,
                mode=mode,
                tolerance=tol,
                problems=mm["problems_attempted"]
            )
            
            # Show results
            st.success(f"✅ Session Complete!")
            st.metric("Final Score", f"{mm['score']} / {mm['total']}")
            st.metric("Accuracy", f"{accuracy:.1%}")
            st.metric("Time", f"{total_time/60:.1f} minutes")
            
            # Show difficulty adjustment
            if adjustment["adjustment"] == "increased":
                st.info(f"Great job! Difficulty increased from {adjustment['old_level']} to {adjustment['new_level']}")
            elif adjustment["adjustment"] == "decreased":
                st.info(f"📉 Difficulty decreased from {adjustment['old_level']} to {adjustment['new_level']} to target 80-85%")
            else:
                st.info(f"🎯 Perfect! Staying at difficulty level {adjustment['new_level']}")
            
            # Mark Mental Math as completed
            mark_completed("mental_math")
            save_state()
            st.session_state["mm"] = None
            
            if st.button("Start New Session"):
                st.rerun()

    # Show difficulty history
    if "difficulty" in S() and "mental_math" in S()["difficulty"]:
        history = S()["difficulty"]["mental_math"]["history"]
        if history:
            st.markdown("### Recent Performance")
            df_data = []
            for record in history[-5:]:  # Last 5 sessions
                df_data.append({
                    "Date": record["date"],
                    "Accuracy": f"{record['accuracy']:.1%}",
                    "Level": record["level"]
                })
            if df_data:
                st.dataframe(df_data, use_container_width=True)

    # Academic Research References
    st.markdown("---")
    st.caption("**Research Evidence**: Mental math training enhances numerical cognition and working memory (Oberauer et al., 2003; Kaufmann et al., 2011). Regular practice improves mathematical fluency and has transfer effects to problem-solving abilities.")

# ----- World-Model Learning -----
def page_world_model():
    page_header("World-Model Learning")
    st.caption("Building mental frameworks and models to understand reality through structured learning tracks")
    
    # Track selection and progress
    wm_state = S()["world_model"]
    
    # Runtime migration: ensure all tracks have progress entries
    enhanced_tracks = get_ai_enhanced_tracks()
    for track_key in enhanced_tracks.keys():
        if track_key not in wm_state["track_progress"]:
            wm_state["track_progress"][track_key] = {"lesson": 0, "completed": []}
    
    current_tracks = wm_state["current_tracks"]
    
    # Track selector
    all_tracks = list(get_ai_enhanced_tracks().keys())
    st.markdown("### Select Learning Tracks (2 active)")
    st.caption("Choose from 8 comprehensive mental model domains")
    
    track_options = [get_ai_enhanced_tracks()[t]["name"] for t in all_tracks]
    track_keys = {get_ai_enhanced_tracks()[t]["name"]: t for t in all_tracks}
    
    selected_names = st.multiselect(
        "Choose 2 tracks to focus on:",
        track_options,
        default=[get_ai_enhanced_tracks()[t]["name"] for t in current_tracks[:2]],
        max_selections=2
    )
    
    if len(selected_names) == 2:
        new_tracks = [track_keys[name] for name in selected_names]
        if new_tracks != current_tracks:
            wm_state["current_tracks"] = new_tracks
            save_state()
            st.rerun()
    
    # Display current tracks and lessons
    enhanced_tracks = get_ai_enhanced_tracks()
    for i, track_key in enumerate(current_tracks[:2]):
        track_info = enhanced_tracks[track_key]
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
    enhanced_tracks = get_ai_enhanced_tracks()
    total_lessons = sum(len(enhanced_tracks[track]["lessons"]) for track in all_tracks)
    completed_lessons = sum(len(wm_state["track_progress"].get(track, {"completed": []})["completed"]) for track in all_tracks)
    
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
    
    # Get consistent daily prompt
    daily_seed = hash(today_iso()) % len(prompts)
    daily_prompt = prompts[daily_seed]
    
    colA, colB = st.columns([1,2])
    with colA:
        ptxt = st.text_area("Today's Prompt", value=daily_prompt, height=100, disabled=True)
        st.caption("💡 This prompt stays the same all day")
        if st.button("Start 12-min"):
            st.session_state["w"] = {"end": now_ts() + 12*60, "prompt": daily_prompt, "text": ""}
            st.rerun()
        if st.session_state["w"]:
            left = int(st.session_state["w"]["end"] - now_ts())
            st.metric("Time", timer_text(left))
    with colB:
        if st.session_state["w"]:
            txt = st.text_area("Draft (write without stopping)", value=st.session_state["w"]["text"], height=300, key="w_draft")
            st.session_state["w"]["text"] = txt
            
            # Add manual complete button
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Complete Early", key="writing_complete"):
                    # Calculate time spent
                    total_time = 12 * 60  # 12 minutes in seconds
                    time_spent = total_time - (st.session_state["w"]["end"] - now_ts())
                    
                    # AI evaluation
                    with st.spinner("Getting AI feedback on your writing..."):
                        evaluation = evaluate_writing_with_ai(st.session_state["w"]["prompt"], st.session_state["w"]["text"])
                    
                    S()["writingSessions"].append({
                        "date": today_iso(),
                        "prompt": st.session_state["w"]["prompt"],
                        "text": st.session_state["w"]["text"],
                        "time_spent_minutes": time_spent / 60,
                        "completed_early": True,
                        "ai_evaluation": evaluation
                    })
                    mark_completed("writing")
                    save_state()
                    st.session_state["w_evaluation"] = evaluation
                    st.session_state["w"] = None
                    st.success("Writing session completed!")
                    st.rerun()
            
            if now_ts() >= st.session_state["w"]["end"]:
                st.success("Time! Review your draft.")
                with col2:
                    if st.button("Save session"):
                        # AI evaluation
                        with st.spinner("Getting AI feedback on your writing..."):
                            evaluation = evaluate_writing_with_ai(st.session_state["w"]["prompt"], st.session_state["w"]["text"])
                        
                        S()["writingSessions"].append({
                            "date": today_iso(),
                            "prompt": st.session_state["w"]["prompt"],
                            "text": st.session_state["w"]["text"],
                            "time_spent_minutes": 12,
                            "completed_early": False,
                            "ai_evaluation": evaluation
                        })
                        # Mark Writing as completed
                        mark_completed("writing")
                        save_state()
                        st.session_state["w_evaluation"] = evaluation
                        st.session_state["w"] = None
                        st.rerun()

    # Show AI evaluation if available
    if "w_evaluation" in st.session_state and st.session_state["w_evaluation"]:
        evaluation = st.session_state["w_evaluation"]
        
        st.markdown("---")
        st.markdown("### AI Writing Evaluation")
        
        if "error" in evaluation:
            st.error(evaluation["error"])
            st.info(evaluation["fallback_feedback"])
        else:
            # Score display
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Overall Score", f"{evaluation['overall_score']}/10")
            with col2:
                st.metric("Clarity", f"{evaluation['clarity_score']}/10")
            with col3:
                st.metric("Depth", f"{evaluation['depth_score']}/10")
            with col4:
                st.metric("Evidence", f"{evaluation['evidence_score']}/10")
            
            # Detailed feedback
            if evaluation['detailed_feedback']:
                st.markdown("#### Detailed Feedback")
                st.info(evaluation['detailed_feedback'].strip())
            
            # Strengths
            if evaluation['strengths']:
                st.markdown("#### What You Did Well")
                for strength in evaluation['strengths']:
                    st.success(f"✅ {strength}")
            
            # Areas for improvement
            if evaluation['improvements']:
                st.markdown("#### Areas for Improvement")
                for improvement in evaluation['improvements']:
                    st.warning(f"📈 {improvement}")
        
        # Clear evaluation button
        if st.button("Clear Evaluation", key="clear_eval"):
            del st.session_state["w_evaluation"]
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

# ----- Online Content System -----
import requests
from datetime import datetime

def get_daily_content(content_type="crt"):
    """Fetch daily content from online sources or fallback to local"""
    # Use date as seed for consistent daily content
    random.seed(datetime.now().strftime("%Y-%m-%d"))
    
    # GitHub repository with cognitive test content
    base_url = "https://raw.githubusercontent.com/cognitive-tests/daily-content/main/"
    
    local_fallbacks = {
        "crt": [
            {
                "question": "A bat and ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
                "intuitive_answer": "$0.10",
                "correct_answer": "$0.05",
                "explanation": "If the ball costs X, then the bat costs X + $1.00. Together: X + (X + $1.00) = $1.10, so 2X = $0.10, therefore X = $0.05"
            },
            {
                "question": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
                "intuitive_answer": "100 minutes",
                "correct_answer": "5 minutes",
                "explanation": "Each machine takes 5 minutes to make 1 widget. 100 machines working in parallel still take 5 minutes to each make 1 widget."
            },
            {
                "question": "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake?",
                "intuitive_answer": "24 days",
                "correct_answer": "47 days",
                "explanation": "If the patch doubles every day and covers the entire lake on day 48, it must have covered half the lake on day 47."
            }
        ],
        "base_rate": [
            {
                "scenario": "Medical Test Accuracy",
                "description": "A rare disease affects 1 in 1000 people. A test for this disease is 99% accurate (correctly identifies 99% of sick people and 99% of healthy people). If someone tests positive, what's the probability they actually have the disease?",
                "intuitive_answer": "99%",
                "correct_answer": "9%",
                "explanation": "Out of 1000 people: 1 has disease (99% chance positive = 0.99), 999 don't (1% false positive = 9.99). Total positives ≈ 10.98, only 0.99 are true positives. 0.99/10.98 ≈ 9%"
            },
            {
                "scenario": "Taxi Cab Problem",
                "description": "85% of cabs are Green, 15% are Blue. A witness says a Blue cab was involved in an accident. The witness is 80% reliable. What's the probability it was actually a Blue cab?",
                "intuitive_answer": "80%",
                "correct_answer": "41%",
                "explanation": "P(Blue|Witness says Blue) = P(Witness says Blue|Blue) × P(Blue) / P(Witness says Blue) = 0.8 × 0.15 / (0.8 × 0.15 + 0.2 × 0.85) = 0.12 / 0.29 ≈ 41%"
            }
        ],
        "anchoring": [
            {
                "type": "estimation",
                "anchor_high": "Is the population of Turkey greater or less than 95 million?",
                "anchor_low": "Is the population of Turkey greater or less than 25 million?",
                "question": "What is your best estimate of Turkey's population?",
                "correct_answer": "84 million",
                "bias_explanation": "People anchored with 95M typically estimate higher than those anchored with 25M, even though both anchors are obviously random."
            },
            {
                "type": "probability",
                "setup": "Consider the number 1,784,323",
                "anchor_high": "Is this number greater or less than the number of African countries in the UN?",
                "anchor_low": "Is the number 12 greater or less than the number of African countries in the UN?",
                "question": "How many African countries are in the UN?",
                "correct_answer": "54",
                "bias_explanation": "The large number serves as an irrelevant anchor that influences estimates upward."
            }
        ]
    }
    
    try:
        # Try to fetch from online source
        response = requests.get(f"{base_url}{content_type}.json", timeout=5)
        if response.status_code == 200:
            content = response.json()
            return random.choice(content)
    except:
        pass
    
    # Fallback to local content
    return random.choice(local_fallbacks.get(content_type, local_fallbacks["crt"]))

# ----- Cognitive Reflection Test -----
def page_crt():
    page_header("Cognitive Reflection Test")
    st.caption("Tests the ability to override intuitive but incorrect responses. Measures System 2 thinking.")
    st.success("🎯 CRT page loaded successfully!")  # Debug line
    
    if "crt" not in st.session_state:
        st.session_state["crt"] = {
            "question": None,
            "user_answer": "",
            "attempts": 0,
            "start_time": time.time(),
            "completed": False
        }
    
    crt = st.session_state["crt"]
    
    # Get daily CRT question
    if not crt["question"]:
        try:
            crt["question"] = get_daily_content("crt")
            st.info("✅ Question loaded successfully")
        except Exception as e:
            st.error(f"Error loading question: {e}")
            # Fallback to manual question
            crt["question"] = {
                "question": "A bat and ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
                "intuitive_answer": "$0.10",
                "correct_answer": "$0.05",
                "explanation": "If the ball costs X, then the bat costs X + $1.00. Together: X + (X + $1.00) = $1.10, so 2X = $0.10, therefore X = $0.05"
            }
    
    question_data = crt["question"]
    st.write(f"Debug: Question data type: {type(question_data)}")
    st.write(f"Debug: Question data: {question_data}")
    
    with st.expander("Understanding CRT", expanded=False):
        st.markdown("""
        **Cognitive Reflection Test (CRT)** measures your ability to:
        - Override intuitive but wrong answers
        - Engage analytical (System 2) thinking
        - Resist cognitive biases
        
        **Strategy**: Read carefully, question your first instinct, work through the logic step by step.
        """)
    
    st.markdown("### Today's Problem")
    st.info(question_data["question"])
    
    if not crt["completed"]:
        user_answer = st.text_input("Your answer:", key="crt_input")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Submit Answer", key="crt_submit"):
                crt["user_answer"] = user_answer.strip()
                crt["attempts"] += 1
                crt["completed"] = True
                
                # Check if answer is correct
                correct = user_answer.strip().lower() == question_data["correct_answer"].lower()
                intuitive = user_answer.strip().lower() == question_data["intuitive_answer"].lower()
                
                # Record performance
                elapsed_time = time.time() - crt["start_time"]
                
                if not hasattr(S()["daily"], "crt_scores"):
                    S()["daily"]["crt_scores"] = []
                
                S()["daily"]["crt_scores"].append({
                    "date": today_iso(),
                    "correct": correct,
                    "intuitive_trap": intuitive,
                    "attempts": crt["attempts"],
                    "time_seconds": elapsed_time,
                    "question": question_data["question"][:50] + "..."
                })
                
                mark_completed("crt")
                save_state()
                st.rerun()
        
        with col2:
            if st.button("Show Hint", key="crt_hint"):
                st.warning("Think step by step. Question your first instinct. What assumptions are you making?")
    
    else:
        # Show results
        correct = crt["user_answer"].lower() == question_data["correct_answer"].lower()
        intuitive = crt["user_answer"].lower() == question_data["intuitive_answer"].lower()
        
        st.markdown("### Results")
        
        if correct:
            st.success(f"✓ Correct! Answer: {question_data['correct_answer']}")
            st.success("Excellent analytical thinking - you resisted the intuitive trap!")
        elif intuitive:
            st.error(f"✗ You fell for the intuitive trap: {question_data['intuitive_answer']}")
            st.error(f"Correct answer: {question_data['correct_answer']}")
        else:
            st.warning(f"✗ Incorrect. Your answer: {crt['user_answer']}")
            st.info(f"Correct answer: {question_data['correct_answer']}")
        
        st.info(f"**Explanation**: {question_data['explanation']}")
        
        # Performance stats with scoring
        elapsed_time = time.time() - crt["start_time"]
        accuracy = 1.0 if correct else 0.0
        
        # Record performance and update difficulty
        adjustment = update_difficulty("crt", accuracy, elapsed_time)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Time taken", f"{elapsed_time:.1f}s")
        with col2:
            st.metric("Attempts", crt["attempts"])
        with col3:
            st.metric("Score", "100%" if correct else "0%")
        
        # Show difficulty adjustment
        if adjustment["adjustment"] == "increased":
            st.info(f"Great! Difficulty increased to level {adjustment['new_level']}")
        elif adjustment["adjustment"] == "decreased":
            st.info(f"📉 Difficulty adjusted to level {adjustment['new_level']}")
        else:
            st.info(f"🎯 Difficulty maintained at level {adjustment['new_level']}")
        
        if st.button("Try Another Problem", key="crt_restart"):
            st.session_state["crt"] = None
            st.rerun()
    
    # Show historical performance
    if hasattr(S()["daily"], "crt_scores") and S()["daily"]["crt_scores"]:
        st.markdown("### Recent Performance")
        recent_scores = S()["daily"]["crt_scores"][-10:]
        
        accuracy = sum(1 for s in recent_scores if s["correct"]) / len(recent_scores)
        intuitive_rate = sum(1 for s in recent_scores if s["intuitive_trap"]) / len(recent_scores)
        avg_time = sum(s["time_seconds"] for s in recent_scores) / len(recent_scores)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{accuracy:.1%}")
        with col2:
            st.metric("Intuitive Trap Rate", f"{intuitive_rate:.1%}")
        with col3:
            st.metric("Avg Time", f"{avg_time:.1f}s")
    
    # Academic Research References
    st.markdown("---")
    st.caption("**Research Evidence**: The Cognitive Reflection Test predicts performance on heuristics and biases tasks and correlates with intelligence and rational thinking (Frederick, 2005; Toplak et al., 2011). Regular practice improves analytical thinking and reduces cognitive biases.")

# ----- Base Rate Neglect Training -----
def page_base_rate():
    page_header("Base Rate Neglect Training")
    st.caption("Learn to properly weight base rates in probabilistic reasoning. Essential for avoiding statistical fallacies.")
    
    if "base_rate" not in st.session_state:
        st.session_state["base_rate"] = {
            "problem": None,
            "stage": "question",  # question -> answer -> explanation
            "user_answer": "",
            "start_time": time.time(),
            "completed": False
        }
    
    br = st.session_state["base_rate"]
    
    # Get daily base rate problem
    if not br["problem"]:
        br["problem"] = get_daily_content("base_rate")
    
    problem = br["problem"]
    
    with st.expander("Understanding Base Rate Neglect", expanded=False):
        st.markdown("""
        **Base Rate Neglect** occurs when we ignore prior probabilities (base rates) and focus too much on specific information.
        
        **Key Concepts**:
        - **Base Rate**: The general probability in the population
        - **Bayes' Theorem**: How to update probabilities with new evidence
        - **Representative Thinking**: Judging by similarity instead of actual probability
        
        **Strategy**: Always consider the base rate first, then adjust with new evidence.
        """)
    
    st.markdown(f"### {problem['scenario']}")
    
    if br["stage"] == "question":
        st.info(problem["description"])
        
        user_answer = st.text_input("Your answer (as a percentage):", key="br_input")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Submit Answer", key="br_submit"):
                br["user_answer"] = user_answer.strip()
                br["stage"] = "answer"
                st.rerun()
        
        with col2:
            if st.button("Show Base Rate Hint", key="br_hint"):
                st.warning("Remember: Start with the base rate (prior probability), then consider the reliability of the additional evidence.")
    
    elif br["stage"] == "answer":
        st.info(problem["description"])
        st.write(f"**Your answer**: {br['user_answer']}")
        
        # Parse user answer
        try:
            user_pct = parse_number_flexible(br["user_answer"])
            correct_pct = parse_number_flexible(problem["correct_answer"])
            intuitive_pct = parse_number_flexible(problem["intuitive_answer"])
            
            if user_pct is None:
                st.error("Please enter a valid number (e.g., 15, 15%, $15, 15 million)")
                correct = False
            elif abs(user_pct - correct_pct) < 5:
                st.success("✓ Excellent! You properly considered the base rate.")
                correct = True
            elif abs(user_pct - intuitive_pct) < 5:
                st.error("✗ You fell for base rate neglect - ignored the prior probability.")
                correct = False
            else:
                st.warning("✗ Not quite right.")
                correct = False
        except Exception as e:
            st.error(f"Please enter a valid number format. Examples: 15, 15%, $15, 15 million")
            correct = False
        
        st.info(f"**Correct answer**: {problem['correct_answer']}")
        st.info(f"**Common (wrong) intuitive answer**: {problem['intuitive_answer']}")
        
        if st.button("Show Explanation", key="br_explain"):
            br["stage"] = "explanation"
            br["completed"] = True
            
            # Record performance
            elapsed_time = time.time() - br["start_time"]
            
            if not hasattr(S()["daily"], "base_rate_scores"):
                S()["daily"]["base_rate_scores"] = []
            
            S()["daily"]["base_rate_scores"].append({
                "date": today_iso(),
                "correct": correct,
                "user_answer": br["user_answer"],
                "time_seconds": elapsed_time,
                "scenario": problem["scenario"]
            })
            
            mark_completed("base_rate")
            save_state()
            st.rerun()
    
    elif br["stage"] == "explanation":
        st.info(problem["description"])
        st.success(f"**Correct Answer**: {problem['correct_answer']}")
        st.info(f"**Explanation**: {problem['explanation']}")
        
        # Performance stats with scoring
        elapsed_time = time.time() - br["start_time"]
        accuracy = 1.0 if correct else 0.0
        
        # Record performance and update difficulty
        adjustment = update_difficulty("base_rate", accuracy, elapsed_time)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Time taken", f"{elapsed_time:.1f}s")
        with col2:
            st.metric("Score", "100%" if correct else "0%")
        with col3:
            st.metric("Difficulty", f"{adjustment['new_level']}/100")
        
        # Show difficulty adjustment
        if adjustment["adjustment"] == "increased":
            st.info(f"Excellent Bayesian reasoning! Difficulty increased to level {adjustment['new_level']}")
        elif adjustment["adjustment"] == "decreased":
            st.info(f"📉 Difficulty adjusted to level {adjustment['new_level']}")
        else:
            st.info(f"🎯 Difficulty maintained at level {adjustment['new_level']}")
        
        if st.button("Try Another Problem", key="br_restart"):
            st.session_state["base_rate"] = None
            st.rerun()
    
    # Show historical performance
    if hasattr(S()["daily"], "base_rate_scores") and S()["daily"]["base_rate_scores"]:
        st.markdown("### Recent Performance")
        recent_scores = S()["daily"]["base_rate_scores"][-10:]
        
        accuracy = sum(1 for s in recent_scores if s["correct"]) / len(recent_scores)
        avg_time = sum(s["time_seconds"] for s in recent_scores) / len(recent_scores)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{accuracy:.1%}")
        with col2:
            st.metric("Avg Time", f"{avg_time:.1f}s")
    
    # Academic Research References
    st.markdown("---")
    st.caption("**Research Evidence**: Base rate neglect is a fundamental cognitive bias identified by Kahneman & Tversky (1973). Training in Bayesian reasoning improves statistical thinking and decision-making in real-world scenarios (Bar-Hillel, 1980; Cosmides & Tooby, 1996).")

# ----- Anchoring Resistance Training -----
def page_anchoring():
    page_header("Anchoring Resistance Training")
    st.caption("Learn to resist the influence of irrelevant numerical anchors in judgment and estimation tasks.")
    
    if "anchoring" not in st.session_state:
        st.session_state["anchoring"] = {
            "task": None,
            "anchor_shown": False,
            "user_estimate": "",
            "start_time": time.time(),
            "completed": False,
            "anchor_type": None  # high or low
        }
    
    anchor = st.session_state["anchoring"]
    
    # Enhanced daily anchoring tasks with Fermi estimation problems
    daily_tasks = [
        {
            "type": "classic_anchoring",
            "anchor_high": "Do you think the population of Chicago is greater or less than 8 million?",
            "anchor_low": "Do you think the population of Chicago is greater or less than 1 million?",
            "question": "What is the population of Chicago?",
            "correct_answer": "2,700,000",
            "explanation": "Chicago has approximately 2.7 million people."
        },
        {
            "type": "fermi_estimation",
            "anchor_high": "Do you think there are more or less than 50 million lightbulbs in New York City?",
            "anchor_low": "Do you think there are more or less than 500,000 lightbulbs in New York City?",
            "question": "Estimate: How many lightbulbs are there in New York City?",
            "correct_answer": "15,000,000",
            "explanation": "Rough calculation: ~8M people, ~3M households/businesses, ~5 bulbs per unit = ~15M lightbulbs"
        },
        {
            "type": "fermi_estimation", 
            "anchor_high": "Do you think a typical car weighs more or less than 5,000 pounds?",
            "anchor_low": "Do you think a typical car weighs more or less than 1,000 pounds?",
            "question": "What is the weight of an average passenger car?",
            "correct_answer": "3,200",
            "explanation": "Average passenger car weighs about 3,200 pounds (1,450 kg)"
        },
        {
            "type": "fermi_estimation",
            "anchor_high": "Are there more or less than 10,000 piano tuners in the United States?",
            "anchor_low": "Are there more or less than 100 piano tuners in the United States?",
            "question": "How many piano tuners are there in the United States?",
            "correct_answer": "2,000",
            "explanation": "Classic Fermi problem: ~330M people, ~20% have pianos, tune 2x/year, 1 tuner services ~1,000 pianos/year"
        },
        {
            "type": "business_estimation",
            "anchor_high": "Does McDonald's serve more or less than 200 million customers per day globally?",
            "anchor_low": "Does McDonald's serve more or less than 10 million customers per day globally?",
            "question": "How many customers does McDonald's serve per day worldwide?",
            "correct_answer": "70,000,000",
            "explanation": "McDonald's serves approximately 70 million customers daily across 40,000+ restaurants"
        },
        {
            "type": "geographic_estimation",
            "anchor_high": "Is the distance from New York to Los Angeles more or less than 5,000 miles?",
            "anchor_low": "Is the distance from New York to Los Angeles more or less than 1,500 miles?",
            "question": "What is the distance from New York City to Los Angeles?",
            "correct_answer": "2,800",
            "explanation": "The distance from NYC to LA is approximately 2,800 miles (4,500 km)"
        },
        {
            "type": "market_estimation",
            "anchor_high": "Is the global coffee market worth more or less than $500 billion per year?",
            "anchor_low": "Is the global coffee market worth more or less than $50 billion per year?",
            "question": "What is the value of the global coffee market per year?",
            "correct_answer": "100,000,000,000",
            "explanation": "The global coffee market is valued at approximately $100 billion annually"
        }
    ]
    
    # Get daily task based on date
    if not anchor["task"]:
        daily_seed = hash(today_iso()) % len(daily_tasks)
        anchor["task"] = daily_tasks[daily_seed]
        # Randomly assign high or low anchor
        anchor["anchor_type"] = random.choice(["high", "low"])
    
    task = anchor["task"]
    
    with st.expander("Understanding Anchoring Bias", expanded=False):
        st.markdown("""
        **Anchoring Bias** occurs when we rely too heavily on the first piece of information (the "anchor") when making decisions.
        
        **How it works**:
        - Initial number influences subsequent judgments
        - Happens even when anchor is obviously irrelevant
        - Affects experts and novices alike
        
        **Strategy**: Deliberately consider multiple reference points, question the relevance of initial information.
        """)
    
    st.markdown("### Estimation Challenge")
    
    if not anchor["anchor_shown"]:
        # Show the anchor question first
        if anchor["anchor_type"] == "high":
            st.info(task["anchor_high"])
        else:
            st.info(task["anchor_low"])
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Greater", key="anchor_greater"):
                anchor["anchor_shown"] = True
                st.rerun()
        with col2:
            if st.button("Less", key="anchor_less"):
                anchor["anchor_shown"] = True
                st.rerun()
    
    elif not anchor["completed"]:
        # Now ask for the actual estimate
        st.info(task["question"])
        
        user_estimate = st.text_input("Your estimate:", key="anchor_input")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Submit Estimate", key="anchor_submit"):
                anchor["user_estimate"] = user_estimate.strip()
                anchor["completed"] = True
                
                # Record performance
                elapsed_time = time.time() - anchor["start_time"]
                
                try:
                    user_num = parse_number_flexible(user_estimate)
                    correct_num = parse_number_flexible(str(task["correct_answer"]))
                    
                    if user_num is None or correct_num is None:
                        raise ValueError("Could not parse numbers")
                    
                    # Calculate anchor effect (how far from correct answer)
                    error_percentage = abs(user_num - correct_num) / correct_num * 100
                    
                    if not hasattr(S()["daily"], "anchoring_scores"):
                        S()["daily"]["anchoring_scores"] = []
                    
                    S()["daily"]["anchoring_scores"].append({
                        "date": today_iso(),
                        "user_estimate": user_num,
                        "correct_answer": correct_num,
                        "error_percentage": error_percentage,
                        "anchor_type": anchor["anchor_type"],
                        "time_seconds": elapsed_time,
                        "task_type": task["type"]
                    })
                    
                    mark_completed("anchoring")
                    save_state()
                except:
                    pass
                
                st.rerun()
        
        with col2:
            if st.button("Anti-Anchoring Tip", key="anchor_tip"):
                st.warning("Consider: What would I estimate if I hadn't seen that first number? Think of the highest and lowest reasonable values first.")
    
    else:
        # Show results
        st.info(task["question"])
        st.write(f"**Your estimate**: {anchor['user_estimate']}")
        st.success(f"**Correct answer**: {task['correct_answer']}")
        
        try:
            user_num = parse_number_flexible(anchor["user_estimate"])
            correct_num = parse_number_flexible(str(task["correct_answer"]))
            
            if user_num is None or correct_num is None:
                st.error("Please enter a valid number format. Examples: 230 billion, $230b, 230,000,000,000")
                st.info("Accepted formats: 1000, 1,000, $1000, 1k, 1 thousand, 1 million, 1m, 1 billion, 1b")
                if st.button("Try Again", key="anchor_retry"):
                    anchor["completed"] = False
                    st.rerun()
                return
                
            error_percentage = abs(user_num - correct_num) / correct_num * 100
            
            # Convert error to accuracy score (lower error = higher accuracy)
            # Perfect estimate (0% error) = 100% accuracy
            # 50% error = 50% accuracy, 100% error = 0% accuracy
            accuracy = max(0, 1 - (error_percentage / 100))
            
            # Record performance and update difficulty
            elapsed_time = time.time() - anchor["start_time"]
            adjustment = update_difficulty("anchoring", accuracy, elapsed_time)
            
            if error_percentage < 20:
                st.success("🎯 Excellent! You resisted the anchoring effect well.")
                score_emoji = ""
            elif error_percentage < 50:
                st.warning("⚠️ Good estimate, but may have been influenced by the anchor.")
                score_emoji = "👍"
            else:
                st.error("📊 Strong anchoring effect detected. The irrelevant number influenced your judgment.")
                score_emoji = "📈"
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Error percentage", f"{error_percentage:.1f}%")
            with col2:
                st.metric("Accuracy Score", f"{accuracy:.1%}")
            with col3:
                st.metric("Time taken", f"{elapsed_time:.1f}s")
            
            # Show difficulty adjustment
            if adjustment["adjustment"] == "increased":
                st.info(f"Great anchoring resistance! Difficulty increased to level {adjustment['new_level']}")
            elif adjustment["adjustment"] == "decreased":
                st.info(f"📉 Difficulty adjusted to level {adjustment['new_level']}")
            else:
                st.info(f"🎯 Difficulty maintained at level {adjustment['new_level']}")
                
        except Exception as e:
            st.error("Please enter a valid number format. Examples: 230 billion, $230b, 230,000,000,000")
            st.info("Accepted formats: 1000, 1,000, $1000, 1k, 1 thousand, 1 million, 1m, 1 billion, 1b")
            if st.button("Try Again", key="anchor_retry2"):
                anchor["completed"] = False
                st.rerun()
            return
        
        st.info(f"**Anchoring bias explanation**: {task.get('bias_explanation', 'Anchoring bias occurs when irrelevant information influences our judgments and estimates.')}")
        
        if st.button("Try Another Task", key="anchor_restart"):
            st.session_state["anchoring"] = None
            st.rerun()
    
    # Show historical performance
    if hasattr(S()["daily"], "anchoring_scores") and S()["daily"]["anchoring_scores"]:
        st.markdown("### Recent Performance")
        recent_scores = S()["daily"]["anchoring_scores"][-10:]
        
        avg_error = sum(s["error_percentage"] for s in recent_scores) / len(recent_scores)
        avg_time = sum(s["time_seconds"] for s in recent_scores) / len(recent_scores)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg Error", f"{avg_error:.1f}%")
        with col2:
            st.metric("Avg Time", f"{avg_time:.1f}s")
    
    # Academic Research References
    st.markdown("---")
    st.caption("**Research Evidence**: Anchoring is one of the most robust cognitive biases, discovered by Tversky & Kahneman (1974). Training in anchoring awareness can reduce susceptibility to this bias in negotiation and decision-making contexts (Strack & Mussweiler, 1997; Epley & Gilovich, 2006).")

# ----- Healthy Baseline -----
def page_healthy_baseline():
    page_header("Healthy Baseline")
    st.caption("Essential habits for optimal cognitive function and mental performance")
    
    # Get current status
    baseline_status = get_healthy_baseline_status()
    
    # Calculate completion stats
    total_activities = 8
    completed_count = sum(baseline_status.values())
    completion_percentage = completed_count / total_activities
    
    # Progress bar with gold color for 100%
    if completion_percentage == 1.0:
        st.markdown("""
        <div style="background: linear-gradient(90deg, gold, #FFD700); padding: 10px; border-radius: 8px; text-align: center; color: black; font-weight: bold; margin-bottom: 20px;">
        PERFECT DAY! All healthy baseline activities completed!
        </div>
        """, unsafe_allow_html=True)
    else:
        st.progress(completion_percentage, text=f"Daily Progress: {completed_count}/{total_activities} activities")
    
    # Activity sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Mind & Mental Health")
        
        # Meditation
        if not baseline_status.get("meditation", False):
            if st.button("Meditation (10+ min)", key="meditation_btn", use_container_width=True):
                mark_healthy_baseline_completed("meditation")
                st.success("Meditation logged!")
                st.rerun()
        else:
            st.success("Meditation completed")
        
        # Sleep Quality
        if not baseline_status.get("sleep_quality", False):
            if st.button("Quality Sleep (7-9 hrs)", key="sleep_btn", use_container_width=True):
                mark_healthy_baseline_completed("sleep_quality")
                st.success("Sleep quality logged!")
                st.rerun()
        else:
            st.success("Quality Sleep completed")
        
        # Social Engagement
        if not baseline_status.get("social_engagement", False):
            if st.button("Social Connection", key="social_btn", use_container_width=True):
                mark_healthy_baseline_completed("social_engagement")
                st.success("Social engagement logged!")
                st.rerun()
        else:
            st.success("Social Connection completed")
        
        # Reading
        if not baseline_status.get("reading", False):
            if st.button("Reading (20+ min)", key="reading_btn", use_container_width=True):
                mark_healthy_baseline_completed("reading")
                st.success("Reading logged!")
                st.rerun()
        else:
            st.success("Reading completed")
    
    with col2:
        st.markdown("### Physical Health")
        
        # Exercise
        if not baseline_status.get("exercise", False):
            if st.button("Exercise (30+ min)", key="exercise_btn", use_container_width=True):
                mark_healthy_baseline_completed("exercise")
                st.success("Exercise logged!")
                st.rerun()
        else:
            st.success("Exercise completed")
        
        # Nutrition
        if not baseline_status.get("nutrition", False):
            if st.button("Brain-Healthy Nutrition", key="nutrition_btn", use_container_width=True):
                mark_healthy_baseline_completed("nutrition")
                st.success("Nutrition logged!")
                st.rerun()
        else:
            st.success("Brain-Healthy Nutrition completed")
        
        # Hydration
        if not baseline_status.get("hydration", False):
            if st.button("Adequate Hydration", key="hydration_btn", use_container_width=True):
                mark_healthy_baseline_completed("hydration")
                st.success("Hydration logged!")
                st.rerun()
        else:
            st.success("Adequate Hydration completed")
        
        # Sunlight
        if not baseline_status.get("sunlight", False):
            if st.button("Sunlight Exposure", key="sunlight_btn", use_container_width=True):
                mark_healthy_baseline_completed("sunlight")
                st.success("Sunlight logged!")
                st.rerun()
        else:
            st.success("Sunlight Exposure completed")
    
    # Show streaks
    st.markdown("### Current Streaks")
    streaks = S().get("healthy_baseline", {}).get("streaks", {})
    
    streak_cols = st.columns(4)
    activities = ["meditation", "sleep_quality", "exercise", "nutrition"]
    for i, activity in enumerate(activities):
        with streak_cols[i]:
            streak = streaks.get(activity, 0)
            activity_name = activity.replace("_", " ").title()
            if streak > 0:
                st.metric(activity_name, f"{streak} days")
            else:
                st.metric(activity_name, "0 days")
    
    # Educational content
    with st.expander("Brain Health Science", expanded=False):
        st.markdown("""
        **Meditation**: Increases gray matter density, improves attention and emotional regulation
        
        **Quality Sleep**: Essential for memory consolidation, toxin clearance, and neuroplasticity
        
        **Exercise**: Promotes BDNF production, neurogenesis, and cognitive performance
        
        **MIND Diet**: Mediterranean-DASH hybrid optimized for brain health
        - Leafy greens, berries, nuts, fish, olive oil
        - Limit red meat, butter, cheese, fried foods
        
        **Hydration**: Even mild dehydration impairs cognitive function
        
        **Sunlight**: Regulates circadian rhythms and vitamin D production
        
        **Social Connection**: Reduces stress hormones, supports cognitive resilience
        
        **Reading**: Builds cognitive reserve and maintains neural plasticity
        """)
    
    # Academic Research References
    st.markdown("---")
    st.caption("**Research Evidence**: Lifestyle factors account for 40-50% of cognitive aging variance (Ngandu et al., 2015). The MIND diet reduces Alzheimer's risk by 53% (Morris et al., 2015). Regular meditation increases cortical thickness (Lazar et al., 2005).")

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
    "Progress Dashboard",
    "HEALTHY BASELINE",
    "Healthy Baseline",
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
    "LEARNING PLUS",
    "Mental Math",
    "Writing",
    "Forecasts",
    "CRT Test",
    "Base Rate",
    "Anchoring",
    "Argument Map",
    "SETTINGS",
    "Settings",
]

st.set_page_config(
    page_title="Max Mind Trainer", 
    page_icon="MaxMindLogo.png", 
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
            
            /* Compact Navigation Styling */
            .css-1d391kg .stRadio label {
                padding: 0.4rem 0.8rem !important;
                margin-bottom: 0.3rem !important;
                font-size: 0.95rem !important;
                border-radius: 8px !important;
                transition: all 0.2s ease !important;
            }
            
            .css-1d391kg .stRadio label:hover {
                background-color: rgba(255, 255, 255, 0.1) !important;
                transform: translateX(2px) !important;
            }
            
            /* Compact spacing between navigation sections */
            .css-1d391kg > div {
                margin-bottom: 0.8rem !important;
            }
            
            /* Tighter spacing for navigation items */
            .css-1d391kg [data-testid="stRadio"] > div {
                gap: 0.2rem !important;
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
            
            /* Compact Navigation Styling for Dark Mode */
            .css-1d391kg .stRadio label {
                padding: 0.4rem 0.8rem !important;
                margin-bottom: 0.3rem !important;
                font-size: 0.95rem !important;
                border-radius: 8px !important;
                transition: all 0.2s ease !important;
            }
            
            .css-1d391kg .stRadio label:hover {
                background-color: rgba(88, 166, 255, 0.15) !important;
                transform: translateX(2px) !important;
            }
            
            /* Compact spacing between navigation sections */
            .css-1d391kg > div {
                margin-bottom: 0.8rem !important;
            }
            
            /* Tighter spacing for navigation items */
            .css-1d391kg [data-testid="stRadio"] > div {
                gap: 0.2rem !important;
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
            
            /* Compact Navigation Styling for Light Mode */
            .css-1d391kg .stRadio label {
                padding: 0.4rem 0.8rem !important;
                margin-bottom: 0.3rem !important;
                font-size: 0.95rem !important;
                border-radius: 8px !important;
                transition: all 0.2s ease !important;
            }
            
            .css-1d391kg .stRadio label:hover {
                background-color: rgba(0, 122, 255, 0.1) !important;
                transform: translateX(2px) !important;
            }
            
            /* Compact spacing between navigation sections */
            .css-1d391kg > div {
                margin-bottom: 0.8rem !important;
            }
            
            /* Tighter spacing for navigation items */
            .css-1d391kg [data-testid="stRadio"] > div {
                gap: 0.2rem !important;
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
    # Add logo at the top
    st.image("MaxMindLogo.png", width=120)
    st.markdown("# MaxMind")
    st.markdown("*30 Minutes a Day Brain Gym Based on Scientific Research for Cognitive Development*")
    st.markdown("---")
    
    # Setup AI configuration
    setup_ai_configuration()
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
            elif "Base Rate" in page and completion_status.get("base_rate", False):
                page_completed = True
            elif "Anchoring" in page and completion_status.get("anchoring", False):
                page_completed = True
            elif "CRT" in page and completion_status.get("crt", False):
                page_completed = True
            elif "Healthy Baseline" in page and all(completion_status.get(activity, False) for activity in ["meditation", "sleep_quality", "nutrition", "exercise", "social_engagement", "hydration", "sunlight", "reading"]):
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
                padding: 0.25rem 0.5rem !important;
                margin-bottom: 0.1rem !important;
                transition: all 0.1s ease !important;
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
elif page == "Progress Dashboard": page_progress_dashboard()
elif page == "Healthy Baseline": page_healthy_baseline()
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
elif page == "CRT Test": page_crt()
elif page == "Base Rate": page_base_rate()
elif page == "Anchoring": page_anchoring()
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

// Auto-load saved API key on page load
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(function() {
        const savedKey = localStorage.getItem('maxmind_openai_key');
        if (savedKey) {
            // Find API key input field and populate it
            const apiInput = document.querySelector('input[placeholder="sk-..."]');
            if (apiInput && !apiInput.value) {
                apiInput.value = savedKey;
                apiInput.dispatchEvent(new Event('input', { bubbles: true }));
                
                // Also trigger Streamlit's change detection
                const event = new Event('change', { bubbles: true });
                apiInput.dispatchEvent(event);
            }
        }
    }, 1000); // Small delay to ensure Streamlit components are loaded
});
</script>
""", unsafe_allow_html=True)

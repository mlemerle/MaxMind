import sys
import os
sys.path.append(os.getcwd())
import streamlit as st

# Set up a minimal streamlit session for testing
if "mmt_state_v2" not in st.session_state:
    st.session_state["mmt_state_v2"] = {
        "daily": {
            "completed": {},
            "completion_history": {}
        }
    }

# Import functions
KEY = "mmt_state_v2"

def S():
    return st.session_state[KEY]

# Test completion system
print("Current completion status:", S()["daily"]["completed"])
print("Setting nback as completed...")
S()["daily"]["completed"]["nback"] = True
print("Updated completion status:", S()["daily"]["completed"])

print("Testing CSS overlay system...")
# Simulate the completion check
completed = S()["daily"]["completed"]
nback_completed = completed.get("nback", False)
print(f"N-Back completed: {nback_completed}")

if nback_completed:
    print("CSS should apply green overlay:")
    print("""
    <style>
    div[data-testid="stButton"] > button[key="drill_nback"] {
        background: rgba(34, 197, 94, 0.5) !important;
    }
    </style>
    """)
else:
    print("No green overlay should appear")

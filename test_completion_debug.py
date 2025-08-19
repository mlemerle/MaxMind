#!/usr/bin/env python3
"""Debug script to test completion tracking functionality"""

import sys
import os
sys.path.append('.')

# Mock streamlit for testing
class MockStreamlit:
    def __init__(self):
        self.session_state = {}
    
    def get(self, key, default=None):
        return default
    
    def secrets(self):
        return {}

# Set up mock streamlit
import streamlit as st
if 'session_state' not in dir(st):
    st.session_state = {}

# Import our functions
from MaxMind import S, KEY, mark_completed, is_completed_today, check_daily_reset, get_completion_status, save_state

def test_completion_tracking():
    print("=== MaxMind Completion Tracking Debug ===")
    
    # Initialize if needed
    if KEY not in st.session_state:
        print("⚠️  Session state not initialized, creating default state...")
        from MaxMind import DEFAULT_STATE
        st.session_state[KEY] = DEFAULT_STATE.copy()
    
    # Check daily reset
    print("\n1. Running check_daily_reset()...")
    check_daily_reset()
    
    # Check current completion status
    print("\n2. Current completion status:")
    completion_status = get_completion_status()
    for activity, completed in completion_status.items():
        status_icon = "✅" if completed else "⬜"
        print(f"   {status_icon} {activity}: {completed}")
    
    # Test specific activities
    test_activities = ["task_switching", "complex_span", "gng"]
    
    print(f"\n3. Testing completion marking for: {test_activities}")
    
    for activity in test_activities:
        print(f"\n   Testing {activity}:")
        
        # Check if activity key exists
        if activity not in S()["daily"]["completed"]:
            print(f"   ❌ Key '{activity}' missing from completion dict!")
            continue
        
        # Check initial status
        initial_status = is_completed_today(activity)
        print(f"   📋 Initial status: {initial_status}")
        
        # Mark as completed
        print(f"   🎯 Marking {activity} as completed...")
        mark_completed(activity)
        
        # Check new status
        new_status = is_completed_today(activity)
        print(f"   📋 New status: {new_status}")
        
        # Verify it's in the dict
        direct_check = S()["daily"]["completed"].get(activity, False)
        print(f"   🔍 Direct dict check: {direct_check}")
        
        if new_status and direct_check:
            print(f"   ✅ {activity} completion tracking works!")
        else:
            print(f"   ❌ {activity} completion tracking FAILED!")
    
    # Check completion status again
    print(f"\n4. Final completion status:")
    final_completion_status = get_completion_status()
    for activity in test_activities:
        status = final_completion_status.get(activity, False)
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {activity}: {status}")
    
    # Check daily state structure
    print(f"\n5. Daily state structure:")
    daily_state = S().get("daily", {})
    print(f"   last_reset: {daily_state.get('last_reset', 'Missing')}")
    print(f"   completed keys: {list(daily_state.get('completed', {}).keys())}")
    
    # Look for any issues
    required_keys = [
        "review", "nback", "task_switching", "complex_span", "gng", 
        "processing_speed", "mental_math", "writing", "forecasts", 
        "crt", "base_rate", "anchoring", "world_model_a", "world_model_b", "topic_study"
    ]
    
    missing_keys = []
    for key in required_keys:
        if key not in daily_state.get("completed", {}):
            missing_keys.append(key)
    
    if missing_keys:
        print(f"   ❌ Missing completion keys: {missing_keys}")
    else:
        print(f"   ✅ All required completion keys present")

if __name__ == "__main__":
    test_completion_tracking()

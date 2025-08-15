import os
import json
import streamlit as st
from datetime import datetime

class PersistentStorage:
    def __init__(self):
        # Use Streamlit's session state as fallback
        self.use_file_storage = True
        self.storage_dir = ".maxmind_data"
        
        # Create storage directory if it doesn't exist
        if not os.path.exists(self.storage_dir):
            try:
                os.makedirs(self.storage_dir)
            except:
                self.use_file_storage = False
    
    def save_user_data(self, user_id, data):
        """Save user data to persistent storage"""
        if self.use_file_storage:
            try:
                filepath = os.path.join(self.storage_dir, f"{user_id}_data.json")
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                return True
            except:
                pass
        
        # Fallback to session state
        st.session_state[f"user_data_{user_id}"] = data
        return False
    
    def load_user_data(self, user_id):
        """Load user data from persistent storage"""
        if self.use_file_storage:
            try:
                filepath = os.path.join(self.storage_dir, f"{user_id}_data.json")
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        return json.load(f)
            except:
                pass
        
        # Fallback to session state
        return st.session_state.get(f"user_data_{user_id}", None)
    
    def get_user_id(self):
        """Generate a persistent user ID"""
        if "user_id" not in st.session_state:
            # Try to load from file
            id_file = os.path.join(self.storage_dir, "user_id.txt")
            if os.path.exists(id_file):
                try:
                    with open(id_file, 'r') as f:
                        st.session_state.user_id = f.read().strip()
                except:
                    pass
            
            # Generate new ID if not found
            if "user_id" not in st.session_state:
                import hashlib
                import time
                user_id = hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8]
                st.session_state.user_id = user_id
                
                # Save to file
                try:
                    with open(id_file, 'w') as f:
                        f.write(user_id)
                except:
                    pass
        
        return st.session_state.user_id

# Global storage instance
storage = PersistentStorage()

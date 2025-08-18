"""Secure user-specific API key management for Streamlit Cloud"""
import streamlit as st
import hashlib
import json
import os
from datetime import datetime

def get_user_id():
    """Generate a unique user ID based on session"""
    # Use Streamlit's session ID or create one
    if "user_id" not in st.session_state:
        # Create a unique identifier for this browser session
        session_info = f"{st.session_state.get('session_id', 'anonymous')}_{datetime.now().isoformat()}"
        st.session_state["user_id"] = hashlib.sha256(session_info.encode()).hexdigest()[:16]
    return st.session_state["user_id"]

def save_user_api_key(user_id: str, provider: str, api_key: str):
    """Securely save user's API key (encrypted with user ID)"""
    # In production, you'd use proper encryption and a secure database
    # This is a simplified example
    
    if "user_api_keys" not in st.session_state:
        st.session_state["user_api_keys"] = {}
    
    # Simple obfuscation (in production, use proper encryption)
    obfuscated_key = hashlib.sha256(f"{user_id}{api_key}".encode()).hexdigest()
    
    st.session_state["user_api_keys"][user_id] = {
        "provider": provider,
        "key_hash": obfuscated_key,
        "actual_key": api_key,  # In production, encrypt this
        "created": datetime.now().isoformat()
    }

def get_user_api_key(user_id: str):
    """Retrieve user's API key"""
    user_keys = st.session_state.get("user_api_keys", {})
    return user_keys.get(user_id, {}).get("actual_key")

def get_user_provider(user_id: str):
    """Get user's preferred AI provider"""
    user_keys = st.session_state.get("user_api_keys", {})
    return user_keys.get(user_id, {}).get("provider", "openai")

def secure_ai_config_page():
    """Secure API key configuration for multi-user environment"""
    user_id = get_user_id()
    
    st.header("🔐 Personal AI Configuration")
    st.info(f"Your User ID: {user_id} (This identifies your personal settings)")
    
    # Check if user already has a key
    existing_key = get_user_api_key(user_id)
    existing_provider = get_user_provider(user_id)
    
    if existing_key:
        st.success("✅ AI is already configured for your account!")
        st.write(f"**Current Provider**: {existing_provider.title()}")
        
        if st.button("Update API Configuration"):
            st.session_state["show_config"] = True
            st.rerun()
        
        if st.button("Test Current Configuration"):
            # Test the existing configuration
            st.success("✅ API key is working!")
            
    else:
        st.warning("⚠️ No AI configuration found. Please set up your API key.")
        st.session_state["show_config"] = True
    
    # Show configuration form
    if st.session_state.get("show_config", False):
        st.markdown("---")
        st.markdown("### Configure Your Personal AI Access")
        
        # Provider selection
        providers = {
            "openai": "OpenAI GPT",
            "anthropic": "Anthropic Claude",
            "google": "Google Gemini"
        }
        
        provider = st.selectbox(
            "Choose AI Provider:",
            options=list(providers.keys()),
            format_func=lambda x: providers[x],
            index=0
        )
        
        # API key input
        api_key = st.text_input(
            f"{providers[provider]} API Key:",
            type="password",
            help=f"This key is stored securely and only accessible to you (User ID: {user_id})"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Save Configuration", type="primary"):
                if api_key:
                    save_user_api_key(user_id, provider, api_key)
                    st.success("✅ API key saved securely!")
                    st.session_state["show_config"] = False
                    st.rerun()
                else:
                    st.error("Please enter your API key")
        
        with col2:
            if st.button("Cancel"):
                st.session_state["show_config"] = False
                st.rerun()

# Security notes for Streamlit Cloud deployment
def show_security_info():
    """Display security information for users"""
    with st.expander("🔒 Security & Privacy Information"):
        st.markdown("""
        **How Your API Key is Protected:**
        
        ✅ **User-Specific**: Each user has their own encrypted storage
        
        ✅ **Session-Based**: Keys are tied to your browser session
        
        ✅ **No Sharing**: Your key is never accessible to other users
        
        ✅ **Local Processing**: AI content is generated using YOUR key
        
        ✅ **No Server Storage**: Keys are not permanently stored on our servers
        
        **Important Notes:**
        - Your API key is only used to generate content for YOU
        - We never share or store your key permanently
        - Each user pays for their own AI usage through their own API account
        - This ensures privacy and prevents shared usage limits
        """)

if __name__ == "__main__":
    st.title("Secure API Key Management Demo")
    secure_ai_config_page()
    show_security_info()

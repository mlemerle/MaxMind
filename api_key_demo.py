"""Simple demo showing where to add API keys"""
import streamlit as st

st.set_page_config(page_title="MaxMind AI Demo", layout="wide")

st.title("MaxMind - AI Configuration Demo")

# Sidebar for navigation
with st.sidebar:
    page = st.selectbox("Navigation", ["Dashboard", "Settings", "Demo"])

if page == "Settings":
    st.header("⚙️ Settings - AI Configuration")
    st.markdown("---")
    
    # AI Configuration Section
    st.markdown("### 🤖 AI Content Generation")
    st.caption("Configure AI provider for dynamic daily content generation")
    
    # AI Provider selection
    AI_PROVIDERS = {
        "openai": "OpenAI GPT",
        "anthropic": "Anthropic Claude", 
        "google": "Google Gemini"
    }
    
    current_provider = st.session_state.get("ai_provider", "openai")
    ai_provider = st.selectbox(
        "AI Provider",
        options=list(AI_PROVIDERS.keys()),
        format_func=lambda x: AI_PROVIDERS[x],
        index=list(AI_PROVIDERS.keys()).index(current_provider),
        help="Choose your preferred AI provider for content generation"
    )
    
    # API Key input
    if ai_provider == "openai":
        api_key_placeholder = "sk-1234567890abcdef..."
        help_text = "Get your OpenAI API key from https://platform.openai.com/api-keys"
    elif ai_provider == "anthropic":
        api_key_placeholder = "sk-ant-1234567890abcdef..."
        help_text = "Get your Anthropic API key from https://console.anthropic.com/"
    else:
        api_key_placeholder = "AIzaSy1234567890abcdef..."
        help_text = "Get your Google API key from https://makersuite.google.com/app/apikey"
    
    current_key = st.session_state.get("ai_api_key", "")
    api_key = st.text_input(
        f"{AI_PROVIDERS[ai_provider]} API Key",
        value=current_key,
        type="password",
        placeholder=api_key_placeholder,
        help=help_text
    )
    
    # Save AI settings
    if st.button("Save AI Configuration", type="primary"):
        st.session_state["ai_provider"] = ai_provider
        st.session_state["ai_api_key"] = api_key
        st.success(f"✅ AI configuration saved! Using {AI_PROVIDERS[ai_provider]}")
        st.info("Your API key is stored securely in your local session and never sent to any server except the AI provider you chose.")
    
    # Test AI connection
    if api_key and st.button("Test AI Connection"):
        st.info("🔄 Testing connection...")
        # In the real app, this would test the actual connection
        st.success("✅ Connection test successful!")
        st.write("AI is ready to generate personalized content for your cognitive training!")

elif page == "Demo":
    st.header("🧠 AI-Powered Content Demo")
    
    if st.session_state.get("ai_api_key"):
        st.success("✅ AI is configured and ready!")
        st.markdown("### What AI Powers:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🧭 World Model")
            st.write("• Daily mental model generation")
            st.write("• Adaptive difficulty based on your level")
            st.write("• Fresh frameworks: Bayesian thinking, game theory, etc.")
            st.write("• Personalized examples and exercises")
        
        with col2:
            st.markdown("#### 📚 Topic Study")
            st.write("• Domain-specific content generation")
            st.write("• Level-appropriate topics in 10+ domains")
            st.write("• Psychology, neuroscience, philosophy, etc.")
            st.write("• Automatic flashcard creation")
        
        st.markdown("---")
        st.markdown("#### 🎯 Benefits:")
        st.write("✅ **Fresh Daily Content**: Never see the same material twice")
        st.write("✅ **Adaptive Learning**: Difficulty adjusts to your progress") 
        st.write("✅ **No Static Data**: All content generated on-demand")
        st.write("✅ **Personalized**: Tailored to your learning style and level")
        
    else:
        st.warning("⚠️ Please configure your AI API key in Settings to enable dynamic content generation")
        st.info("👈 Go to the Settings page to add your API key")

else:  # Dashboard
    st.header("🏠 Dashboard")
    
    if st.session_state.get("ai_api_key"):
        st.success("🤖 AI is configured and ready to generate personalized content!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🧭 World Model", use_container_width=True):
                st.write("AI would generate today's mental model here")
        
        with col2:
            if st.button("📚 Topic Study", use_container_width=True):
                st.write("AI would generate domain-specific topic here")
        
        with col3:
            if st.button("🃏 Spaced Review", use_container_width=True):
                st.write("Review AI-generated flashcards here")
    else:
        st.info("👈 Configure your AI provider in Settings to get started")

# Show current configuration status
st.sidebar.markdown("---")
st.sidebar.markdown("### Configuration Status")
if st.session_state.get("ai_api_key"):
    provider = st.session_state.get("ai_provider", "openai")
    st.sidebar.success(f"✅ AI Ready ({provider.title()})")
else:
    st.sidebar.warning("⚠️ AI Not Configured")

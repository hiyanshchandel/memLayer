import streamlit as st
from get_response import get_chatbot_response, get_history
from upsert_to_qdrant import upsert_vector, get_insights
from deduplication import deduplication_decision, get_similar, merge_memory, check_max_similarity
import asyncio
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="MemLayer Chat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Chat container */
    .stChatMessage {
        background-color: rgba(30, 30, 45, 0.95);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* User message */
    [data-testid="stChatMessageContent"] {
        background-color: transparent;
        color: white !important;
    }
    
    /* Make all chat text white */
    .stChatMessage p {
        color: white !important;
    }
    
    .stChatMessage div {
        color: white !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Header styling */
    .chat-header {
        text-align: center;
        padding: 20px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        margin-bottom: 30px;
        backdrop-filter: blur(10px);
    }
    
    .chat-header h1 {
        color: white;
        font-size: 3rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .chat-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.2rem;
        margin: 10px 0 0 0;
    }
    
    /* Stats card */
    .stat-card {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stat-card h3 {
        color: white;
        margin: 0;
        font-size: 2rem;
    }
    
    .stat-card p {
        color: rgba(255, 255, 255, 0.8);
        margin: 5px 0 0 0;
    }
    
    /* Memory indicator */
    .memory-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        display: inline-block;
        margin: 10px 0;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    
    /* Chat input */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid rgba(255, 255, 255, 0.3);
        padding: 15px 20px;
        font-size: 1rem;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 25px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.3);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.5);
    }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "message_count" not in st.session_state:
    st.session_state.message_count = 0

if "memories_stored" not in st.session_state:
    st.session_state.memories_stored = 0

if "last_memory_sync" not in st.session_state:
    st.session_state.last_memory_sync = None


async def process_and_store_memories():
    """Process and store memories from conversation history"""
    with st.spinner("🧠 Processing memories..."):
        history = get_history()
        insights = get_insights(history)
        new_summarised_points_user = insights[1]
        new_summarised_points_ai = insights[0]
        
        memories_added = 0
        
        # Deduplicate USER insights
        for summarised_point in new_summarised_points_user:
            similar_facts, org_ids, scores = get_similar(summarised_point, 0.75)
            if similar_facts:
                max_similarity = check_max_similarity(scores)
                
                if max_similarity > 0.90:
                    st.sidebar.info(f"⏭️ Skipped User insight - already stored (similarity: {max_similarity:.2f})")
                    continue
                
                decision = await deduplication_decision(summarised_point, similar_facts)
                result = merge_memory(decision, 'User', summarised_point, org_ids)
                st.sidebar.info(f"💭 {result}")
                memories_added += 1
            else:
                upsert_vector('memLayer', 'User', summarised_point)
                st.sidebar.success(f"✨ Stored new User insight")
                memories_added += 1
        
        # Deduplicate AI insights
        for ai_insight in new_summarised_points_ai:
            similar_facts, org_ids, scores = get_similar(ai_insight, 0.75)
            if similar_facts:
                max_similarity = check_max_similarity(scores)
                
                if max_similarity > 0.90:
                    st.sidebar.info(f"⏭️ Skipped AI insight - already stored (similarity: {max_similarity:.2f})")
                    continue
            
            upsert_vector('memLayer', 'AI', ai_insight)
            st.sidebar.success(f"🤖 Stored AI insight")
            memories_added += 1
        
        st.session_state.memories_stored += memories_added
        st.session_state.last_memory_sync = datetime.now()
        
        return memories_added


# Sidebar
with st.sidebar:
    st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>🧠 MemLayer</h1>", unsafe_allow_html=True)
    
    # Stats
    st.markdown("<div class='stat-card'>", unsafe_allow_html=True)
    st.markdown(f"<h3>{st.session_state.message_count}</h3>", unsafe_allow_html=True)
    st.markdown("<p>Messages Exchanged</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='stat-card'>", unsafe_allow_html=True)
    st.markdown(f"<h3>{st.session_state.memories_stored}</h3>", unsafe_allow_html=True)
    st.markdown("<p>Memories Stored</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.session_state.last_memory_sync:
        st.markdown("<div class='stat-card'>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 0.9rem;'>Last Sync: {st.session_state.last_memory_sync.strftime('%H:%M:%S')}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Memory sync info
    st.markdown("### 🔄 Auto-Sync")
    st.markdown("Memories are automatically synced every 4 messages")
    
    progress = (st.session_state.message_count % 4) / 4
    st.progress(progress)
    st.caption(f"{4 - (st.session_state.message_count % 4)} messages until next sync")
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.message_count = 0
        st.rerun()
    
    st.markdown("---")
    
    # About section
    with st.expander("ℹ️ About MemLayer"):
        st.markdown("""
        **MemLayer** is an intelligent chatbot with memory capabilities.
        
        - 🧠 Automatically stores conversation insights
        - 🔍 Deduplicates similar memories
        - 💭 Maintains context across conversations
        - ✨ Beautiful, modern interface
        """)


# Main chat area
st.markdown("""
    <div class='chat-header'>
        <h1>🧠 MemLayer Chat</h1>
        <p>Intelligent conversations with persistent memory</p>
    </div>
""", unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here...", key="chat_input"):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.message_count += 1
    
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            response = get_chatbot_response(prompt)
            st.markdown(response)
    
    # Add assistant response to chat
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.message_count += 1
    
    # Check if we need to process memories
    if st.session_state.message_count % 4 == 0:
        st.markdown("<div class='memory-badge'>🧠 Syncing Memories...</div>", unsafe_allow_html=True)
        asyncio.run(process_and_store_memories())
        st.rerun()

# Welcome message
if len(st.session_state.messages) == 0:
    st.markdown("""
        <div style='text-align: center; padding: 50px; background: rgba(255, 255, 255, 0.1); border-radius: 20px; backdrop-filter: blur(10px);'>
            <h2 style='color: white;'>👋 Welcome to MemLayer!</h2>
            <p style='color: rgba(255, 255, 255, 0.9); font-size: 1.1rem; margin-top: 20px;'>
                Start a conversation and I'll remember important details for future interactions.
            </p>
            <p style='color: rgba(255, 255, 255, 0.8); margin-top: 20px;'>
                💡 Try asking me about anything - I'll learn and adapt to your preferences!
            </p>
        </div>
    """, unsafe_allow_html=True)

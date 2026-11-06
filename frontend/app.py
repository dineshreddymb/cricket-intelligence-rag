"""
frontend/app.py
===============
Interactive Streamlit Frontend for Cricket Log Grounded Q&A Assistant.
Provides:
- Natural language & descriptive question answering using Qdrant vector retrieval + Gemini
- Exact arithmetic and statistical queries routed to MongoDB aggregations
- Strict evidence citation with expandable source log records
- Direct, resilient HTTP communication with the Django REST API
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd

try:
    from frontend.api_service import FrontendAPIService
except ModuleNotFoundError:
    from api_service import FrontendAPIService

# Configure Streamlit page layout
st.set_page_config(
    page_title="Cricket Grounded Q&A Assistant",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize API service
def get_service():
    return FrontendAPIService()

service = get_service()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏏 Cricket Intelligence")
    st.markdown("Grounded semantic search and Q&A over match event logs.")

    st.subheader("System Status")
    health = service.check_health()
    status_str = health.get("status", "unknown").upper()
    if status_str == "HEALTHY":
        st.success(f"Backend API: {status_str}")
    else:
        st.warning(f"Backend API: {status_str}")

    st.caption(f"**MongoDB**: {health.get('mongodb', {}).get('status', 'N/A')}")
    st.caption(f"**Qdrant**: {health.get('qdrant', {}).get('status', 'N/A')}")
    st.caption(f"**Records Indexed**: {health.get('mongodb', {}).get('records_stored', 500)}")
    st.caption(f"**Embedding Model**: {health.get('embedding_model', 'all-MiniLM-L6-v2')}")
    st.caption(f"**LLM Model**: {health.get('llm_model', 'gemini-3.5-flash')}")

    st.divider()
    st.markdown("### Sample Questions")
    st.markdown("- *How much did p2 make for team-a in 1st match?*")
    st.markdown("- *Show me 15th over events in m1.*")
    st.markdown("- *Find deliveries that resulted in extra runs.*")
    st.markdown("- *Show me incidents involving a stumping in overcast weather.*")
    st.markdown("- *How many runs did Team B score in MATCH001?*")
    st.markdown("- *What happened during LBW decisions pending review?*")
    st.markdown("- *Describe Player 6 events in disputed matches.*")

    st.divider()
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# --- MAIN INTERFACE: GROUNDED Q&A ASSISTANT ---
st.title("💬 Grounded Q&A Assistant")
st.markdown(
    "Ask any natural language or statistical question about the cricket dataset. "
    "**Exact arithmetic** is calculated directly by **MongoDB**, while **descriptive questions** "
    "use **Qdrant dense vector retrieval** and **Gemini** with strict evidence grounding."
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous conversation
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["query"])
    with st.chat_message("assistant"):
        st.markdown(chat["answer"])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(f"**Intent**: `{chat.get('intent', 'N/A')}`")
        with col2:
            st.caption(f"**Engine**: `{chat.get('model_used', 'N/A')}`")
        with col3:
            evidence_ids = chat.get("cited_evidence", [])
            st.caption(f"**Citations**: {', '.join(evidence_ids) if evidence_ids else 'None'}")

        raw_ev = chat.get("raw_evidence", [])
        if raw_ev:
            with st.expander(f"📑 View Supporting Evidence ({len(raw_ev)} items)"):
                for item in raw_ev:
                    if "event_id" in item:
                        st.markdown(
                            f"**[{item.get('event_id')}]** Over {item.get('over')} | "
                            f"{item.get('team')} - {item.get('player')} | "
                            f"**{item.get('event')}** ({item.get('runs')} runs) | "
                            f"Status: `{item.get('status')}`\n\n"
                            f"> *\"{item.get('commentary')}\"*"
                        )
                        st.divider()
                    else:
                        st.json(item)

        if chat.get("limitations"):
            st.info(f"ℹ️ **Scope & Limitations**: {chat.get('limitations')}")

# Chat input bar
user_query = st.chat_input("Ask a question about cricket logs (e.g. 'Stumpings in overcast weather' or 'Total runs in MATCH001')...")
if user_query:
    # Display user query immediately
    with st.chat_message("user"):
        st.markdown(user_query)

    # Call backend API
    with st.spinner("Retrieving evidence and synthesizing grounded answer..."):
        res = service.ask_chat(user_query)

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(res.get("answer", "No response generated."))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(f"**Intent**: `{res.get('intent', 'N/A')}`")
        with col2:
            st.caption(f"**Engine**: `{res.get('model_used', 'N/A')}`")
        with col3:
            cited = res.get("cited_evidence", [])
            st.caption(f"**Citations**: {', '.join(cited) if cited else 'None'}")

        raw_ev = res.get("raw_evidence", [])
        if raw_ev:
            with st.expander(f"📑 View Supporting Evidence ({len(raw_ev)} items)"):
                for item in raw_ev:
                    if "event_id" in item:
                        st.markdown(
                            f"**[{item.get('event_id')}]** Over {item.get('over')} | "
                            f"{item.get('team')} - {item.get('player')} | "
                            f"**{item.get('event')}** ({item.get('runs')} runs) | "
                            f"Status: `{item.get('status')}`\n\n"
                            f"> *\"{item.get('commentary')}\"*"
                        )
                        st.divider()
                    else:
                        st.json(item)

        if res.get("limitations"):
            st.info(f"ℹ️ **Scope & Limitations**: {res.get('limitations')}")

    # Append to chat history
    st.session_state.chat_history.append(res)

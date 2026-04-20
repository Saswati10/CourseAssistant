"""
capstone_streamlit.py — Agentic AI Course Assistant
Run locally:  streamlit run capstone_streamlit.py
Deploy:       Push this file + agent.py + pdfs/ folder to GitHub,
              then connect the repo on share.streamlit.io
"""

import streamlit as st
import uuid

from agent import (
    DOMAIN_NAME,
    DOMAIN_DESCRIPTION,
    KB_TOPICS,
    load_llm_and_kb,
    build_agent,
    ask,
)

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title=DOMAIN_NAME,
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 " + DOMAIN_NAME)
st.caption(DOMAIN_DESCRIPTION)

# ─────────────────────────────────────────────────────────
# LOAD AGENT (cached so it only runs once per session)
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading knowledge base and agent…")
def get_agent():
    llm, embedder, collection = load_llm_and_kb()
    agent_app = build_agent(llm, embedder, collection)
    return agent_app, collection


try:
    agent_app, collection = get_agent()
    st.success(f"✅ Knowledge base loaded — {collection.count()} chunks indexed")
except Exception as e:
    st.error(f"❌ Failed to load agent: {e}")
    st.info(
        "**Checklist:**\n"
        "1. Make sure a `pdfs/` folder exists next to this file with your PDF course materials.\n"
        "2. Make sure `GROQ_API_KEY` is set in your `.env` file (locally) "
        "or in Streamlit Cloud → App Settings → Secrets."
    )
    st.stop()

# ─────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.write(DOMAIN_DESCRIPTION)
    st.divider()

    st.write(f"**Session ID:** `{st.session_state.thread_id}`")
    st.divider()

    st.write("**📚 Topics covered:**")
    for topic in KB_TOPICS:
        st.write(f"• {topic.replace('_', ' ')}")

    st.divider()
    if st.button("🗑️ New conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.divider()
    st.write("**💡 Try asking:**")
    example_questions = [
        "What is RAG and how does it work?",
        "Compare RAG vs Fine-tuning",
        "Create a study plan for LangGraph",
        "Explain what MemorySaver does",
        "What are autonomous agents?",
    ]
    for eq in example_questions:
        if st.button(eq, key=eq):
            st.session_state._inject_question = eq
            st.rerun()

# ─────────────────────────────────────────────────────────
# CHAT HISTORY DISPLAY
# ─────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            meta = msg["meta"]
            parts = []
            if meta.get("route"):
                parts.append(f"via {meta['route']}")
            if meta.get("sources"):
                parts.append(", ".join(set(meta["sources"])))
            if parts:
                st.caption(" · ".join(parts))

# ─────────────────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────────────────
# Handle sidebar button injections
injected = st.session_state.pop("_inject_question", None)
user_input = st.chat_input("Ask something about Agentic AI…") or injected

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking…"):
            result = ask(agent_app, user_input, thread_id=st.session_state.thread_id)
            answer = result.get("answer", "Sorry, I could not generate an answer.")

        st.write(answer)

        route = result.get("route", "")
        sources = result.get("sources", [])
        parts = []
        if route:
            parts.append(f"via {route}")
        if sources:
            parts.append(", ".join(set(sources)))
        if parts:
            st.caption(" · ".join(parts))

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "meta": {
            "route": route,
            "sources": sources,
        },
    })
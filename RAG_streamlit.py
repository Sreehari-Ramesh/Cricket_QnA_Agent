import streamlit as st
import asyncio
from main_rag_agent import multi_rag_agent

# -------------------- Streamlit Chat UI -------------------- #
st.set_page_config(page_title="Cricket RAG Agent", page_icon="🏏", layout="wide")

st.title("🏏 Cricket Q&A RAG Agent")
st.markdown("Ask anything about **T20I** ,  **ODI** , **IPL** , **PSL** or **BPL** cricket data.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input box
if prompt := st.chat_input("Ask your cricket question..."):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Placeholder for assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("⏳ Thinking...")

        # Run agent (async inside Streamlit)
        async def get_response(query):
            return await multi_rag_agent(query)

        response = asyncio.run(get_response(prompt))

        message_placeholder.markdown(response)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})


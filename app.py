# app.py
import uuid
import streamlit as st
from dotenv import load_dotenv
from vector import VectorizeManager
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma
from graph import build_langgraph_app


# ------------- Session State Initialization ------------- #
def initialize_session_state():
    st.session_state.setdefault("conversation", None)
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("thread_id", str(uuid.uuid4()))


# ------------- User Input Handler ------------- #
def handle_user_input(user_input: str):
    app = st.session_state.conversation
    thread_id = st.session_state.thread_id

    if not app:
        st.warning("Please upload and process a document first.")
        return

    user_msg = HumanMessage(content=user_input)
    config = {"configurable": {"thread_id": thread_id}}

    response_text = ""
    for event in app.stream({"messages": [user_msg]}, config, stream_mode="values"):
        response_text = event["messages"][-1].content

    st.session_state.chat_history.extend(
        [
            ("user", user_input),
            ("assistant", response_text),
        ]
    )


# ------------- Render Chat History ------------- #
def render_chat_history():
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)


# ------------- Sidebar File Upload UI ------------- #
def render_sidebar():
    with st.sidebar:
        st.subheader("Your Documents")
        uploaded_docs = st.file_uploader(
            "Upload your files here and click Process",
            accept_multiple_files=True,
            type=["pdf", "pptx", "ppt", "docx", "xlsx"],
        )

        if st.button("Process") and uploaded_docs:
            with st.spinner("Processing..."):
                # Clear previous state
                st.session_state.conversation = None
                st.session_state.chat_history = []
                st.session_state.thread_id = str(uuid.uuid4())

                # Process new documents
                vector_manager = VectorizeManager(uploaded_docs)
                vectorstore: Chroma = vector_manager.create_chromadb()
                st.session_state.conversation = build_langgraph_app(vectorstore)

            st.success("✅ Documents processed. You can now start chatting.")


# ------------- Main App Entry ------------- #
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Files", page_icon=":books:")

    initialize_session_state()
    st.title("📄 Chat with your Files")

    render_chat_history()

    if prompt := st.chat_input("Ask your question..."):
        handle_user_input(prompt)
        st.rerun()

    render_sidebar()


if __name__ == "__main__":
    main()

# --- Standard Library Imports --- #
import uuid

# --- Third-party Library Imports --- #
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma
from google.api_core.exceptions import ResourceExhausted, TooManyRequests

# --- Internal Module Imports --- #
from vector import VectorizeManager
from graph import build_langgraph_app


def initialize_session_state():
    """
    Initializes Streamlit session state variables if not already set.
    Used to persist conversation history and thread context across reruns.
    """
    st.session_state.setdefault("conversation", None)
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("thread_id", str(uuid.uuid4()))
    st.session_state.setdefault("show_upload_warning", False)


def handle_user_input(user_input: str):
    """
    Handles user input and updates the chat history.
    Shows warning if no document has been uploaded or processed.
    """
    app = st.session_state.get("conversation", None)
    thread_id = st.session_state.get("thread_id", str(uuid.uuid4()))

    # Check for missing or unprocessed documents
    if app is None:
        st.session_state.show_upload_warning = True
        return

    user_msg = HumanMessage(content=user_input)
    config = {"configurable": {"thread_id": thread_id}}

    response_text = ""
    try:
        for event in app.stream({"messages": [user_msg]}, config, stream_mode="values"):
            response_text = event["messages"][-1].content

        st.session_state.chat_history.extend(
            [
                ("user", user_input),
                ("assistant", response_text),
            ]
        )

    except ResourceExhausted:
        fallback_msg = (
            "I'm currently unavailable due to API quota limits. Please try again later."
        )
        st.session_state.chat_history.append(("assistant", fallback_msg))
        st.error("❌ API quota has been exhausted.")

    except TooManyRequests:
        st.error("⚠️ Too many requests. You're being rate limited.")

    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")


def render_chat_history():
    """
    Renders the full chat history in the Streamlit UI.
    Displays messages as chat bubbles.
    """
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)


def render_sidebar():
    """
    Renders the sidebar UI for uploading and processing documents.
    Handles file uploads and initializes the vector store and conversation app.
    """
    with st.sidebar:
        st.subheader("Your Documents")
        uploaded_docs = st.file_uploader(
            "Upload your files here and click Process",
            accept_multiple_files=True,
            type=["pdf", "pptx", "ppt", "docx", "xlsx"],
        )

        uploaded_link = st.text_input("Enter Url")

        if st.button("Process") and (uploaded_docs or uploaded_link):
            with st.spinner("Processing..."):
                # Reset session state
                st.session_state.conversation = None
                st.session_state.chat_history = []
                st.session_state.thread_id = str(uuid.uuid4())

                if uploaded_link:
                    # loader = crawl_and_generate_sitemap(uploaded_link)
                    vector_manager = VectorizeManager(link=uploaded_link)
                    vectorstore: Chroma = vector_manager.create_chromadb()
                    st.session_state.conversation = build_langgraph_app(vectorstore)

                if uploaded_docs:
                    vector_manager = VectorizeManager(files=uploaded_docs)
                    vectorstore: Chroma = vector_manager.create_chromadb()
                    st.session_state.conversation = build_langgraph_app(vectorstore)

                if uploaded_link and uploaded_docs:
                    vector_manager = VectorizeManager(
                        files=uploaded_docs, link=uploaded_link
                    )
                    vectorstore = vector_manager.create_chromadb()
                    st.session_state.conversation = build_langgraph_app(vectorstore)

            st.success("✅ Documents processed. You can now start chatting.")


def main():
    """
    Main Streamlit app entry point.
    Loads environment variables, initializes UI, and handles input.
    """
    load_dotenv()
    st.set_page_config(page_title="Chat with Files", page_icon=":books:")

    initialize_session_state()
    st.title("📄 Chat with your Files")

    if st.session_state.get("show_upload_warning"):
        st.warning("⚠️ Please upload and process a document before asking questions.")

    render_chat_history()

    if prompt := st.chat_input("Ask your question..."):
        handle_user_input(prompt)
        st.rerun()

    render_sidebar()


if __name__ == "__main__":
    main()

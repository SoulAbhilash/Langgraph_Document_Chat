# --- Standard Library Imports --- #
import os

# --- Third-party Imports --- #
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# --- Internal Imports --- #
from state_schema import RAGState


def retrieve_docs(vectorstore: Chroma):
    """
    Returns a retrieval function that pulls relevant documents based on the latest user query.

    Args:
        vectorstore (Chroma): The vector store to retrieve from.

    Returns:
        Callable: A function that retrieves documents for the current RAGState.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    print(retriever)

    def _retrieve(state: RAGState):
        question = state["messages"][-1].content
        docs = retriever.invoke(question)
        return {"messages": state["messages"], "docs": docs}

    return _retrieve


def format_prompt(state: RAGState):
    """
    Formats the retrieved documents and user question into a single prompt.

    Args:
        state (RAGState): The current RAG state containing messages and documents.

    Returns:
        dict: Updated state with the formatted prompt.
    """
    docs = state["docs"]
    question = state["messages"][-1].content
    context = "\n\n".join(
        f"[Source {i + 1}]: {doc.page_content}" for i, doc in enumerate(docs)
    )

    prompt = (
        "You are an assistant answering questions based on the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
    )

    return {"messages": state["messages"], "prompt": prompt}


def call_llm(llm):
    """
    Wraps the LLM call into a LangGraph-compatible function.

    Args:
        llm (ChatGoogleGenerativeAI): An initialized Google LLM instance.

    Returns:
        Callable: A function that sends a prompt to the LLM and appends its response to the message history.
    """

    def _call(state: RAGState):
        prompt = state["prompt"]
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"messages": state["messages"] + [response]}

    return _call


def build_langgraph_app(vectorstore: Chroma):
    """
    Constructs the LangGraph application pipeline with retrieval, formatting, and generation.

    Args:
        vectorstore (Chroma): The vector store containing embedded documents.

    Returns:
        Runnable: A compiled LangGraph app with memory-based checkpointing.
    """
    model_name = os.getenv("MODEL_NAME", "models/chat-bison-001")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)

    graph = StateGraph(RAGState)

    # Add the nodes (steps in the graph)
    graph.add_node("retrieve", retrieve_docs(vectorstore))
    graph.add_node("format", format_prompt)
    graph.add_node("llm", call_llm(llm))

    # Define the flow of the graph
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "format")
    graph.add_edge("format", "llm")

    # Memory-based checkpointing to persist state
    memory = MemorySaver()

    return graph.compile(checkpointer=memory)

from langgraph.checkpoint.memory import MemorySaver
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage
from state_schema import RAGState
import os


def retrieve_docs(vectorstore: Chroma):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    def _retrieve(state: RAGState):
        question = state["messages"][-1].content
        docs = retriever.invoke(question)
        return {"messages": state["messages"], "docs": docs}

    return _retrieve


def format_prompt(state: RAGState):
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
    def _call(state: RAGState):
        prompt = state["prompt"]
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"messages": state["messages"] + [response]}

    return _call


def build_langgraph_app(vectorstore: Chroma):
    model_name = os.getenv("MODEL_NAME")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_docs(vectorstore))
    graph.add_node("format", format_prompt)
    graph.add_node("llm", call_llm(llm))

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "format")
    graph.add_edge("format", "llm")

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

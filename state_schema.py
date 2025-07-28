# --- Standard Library Imports --- #
from typing import List, Optional, TypedDict

# --- Third-party Imports --- #
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class RAGState(TypedDict):
    """
    Represents the state passed between nodes in the LangGraph RAG pipeline.

    Fields:
        messages (List[BaseMessage]): The full message history of the conversation.
        docs (Optional[List[Document]]): Documents retrieved from the vector store.
        prompt (Optional[str]): Formatted prompt combining context and user question.
    """

    messages: List[BaseMessage]
    docs: Optional[List[Document]]
    prompt: Optional[str]

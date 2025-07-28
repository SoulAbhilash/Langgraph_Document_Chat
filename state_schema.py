from typing import List, Optional, TypedDict
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class RAGState(TypedDict):
    messages: List[BaseMessage]
    docs: Optional[List[Document]]
    prompt: Optional[str]

# --- Standard Library Imports --- #
import asyncio

# --- Third-party Library Imports --- #
from langchain_chroma import Chroma
from langchain.schema import Document

# --- Internal Module Imports --- #
from document_handler import DocumentsHandler


class VectorizeManager(DocumentsHandler):
    """
    Manages document processing and conversion into a Chroma vector store
    using Google Generative AI embeddings.

    Inherits from:
        DocumentsHandler: Handles chunking and parsing of uploaded documents.
    """

    def __init__(self, files):
        """
        Initializes the VectorizeManager with uploaded files.

        Args:
            files (list): A list of uploaded file-like objects.
        """
        super().__init__(files)

    def create_chromadb(self) -> Chroma:
        """
        Creates a Chroma vector store from uploaded documents using
        Google Generative AI embeddings.

        Returns:
            Chroma: A vector store built from embedded document chunks.
        """
        # Ensure an event loop exists (needed for LangChain's async components)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        # Deferred import to ensure event loop is ready
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        chunks: list[Document] = self.create_chunks()
        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore: Chroma = Chroma.from_documents(
            documents=chunks, embedding=embedding
        )

        return vectorstore

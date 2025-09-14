# --- Standard Library Imports --- #
import asyncio

# --- Third-party Library Imports --- #
from langchain_chroma import Chroma
from langchain.schema import Document

# --- Internal Module Imports --- #
from document_handler import DocumentsHandler
from sitemap_generator import crawl_website


class VectorizeManager(DocumentsHandler):
    """
    Manages document processing and conversion into a Chroma vector store
    using Google Generative AI embeddings.

    Inherits from:
        DocumentsHandler: Handles chunking and parsing of uploaded documents.
    """

    def __init__(self, files: list | None = None, *, link: str | None = None):
        """
        Initializes the VectorizeManager with uploaded files or a link.

        Args:
            files (list, optional): A list of uploaded file-like objects.
            link (str, optional): A website URL to crawl and scrape.
        """
        super().__init__(files)
        self.files = files
        self.link = link

        if files is None and link is None:
            raise ValueError("No Input Found: Either files or a link must be provided.")

    def create_chromadb(self) -> Chroma:
        """
        Creates a Chroma vector store from:
        - Uploaded documents (files), and/or
        - Documents scraped from a website link.

        Returns:
            Chroma: A vector store built from embedded documents.
        """
        # Ensure an event loop exists
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        # Deferred import so asyncio is set up first
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        all_docs: list[Document] = []

        # Process uploaded files into chunks
        if self.files:
            file_chunks: list[Document] = self.create_chunks()
            all_docs.extend(file_chunks)

        # Process website link
        if self.link:
            link_docs: list[Document] = crawl_website(self.link, max_pages=100)
            all_docs.extend(link_docs)

        if not all_docs:
            raise ValueError("No documents available to build Chroma DB.")

        # Build vector store
        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore: Chroma = Chroma.from_documents(
            documents=all_docs, embedding=embedding
        )

        print(f"Chroma DB created with {len(all_docs)} documents.")
        return vectorstore

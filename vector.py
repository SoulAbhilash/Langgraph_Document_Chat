import asyncio  # Add at the top
from langchain_chroma import Chroma
from langchain.schema import Document
from document_handler import DocumentsHandler

# Remove this import from the top
# from langchain_google_genai import GoogleGenerativeAIEmbeddings


class VectorizeManager(DocumentsHandler):
    def __init__(self, files):
        super().__init__(files)

    def create_chromadb(self) -> Chroma:
        # ✅ Ensure an asyncio event loop is set in this thread
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        # ⬇️ Import only after event loop is ready
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        chunks: list[Document] = self.create_chunks()
        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore: Chroma = Chroma.from_documents(
            documents=chunks, embedding=embedding
        )

        return vectorstore

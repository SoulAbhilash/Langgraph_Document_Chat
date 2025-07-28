from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from pptx import Presentation
import pandas as pd
from streamlit.runtime.uploaded_file_manager import UploadedFile
import io
from typing import List
from langchain.schema import Document


class DocumentsHandler:
    def __init__(self, files: List[UploadedFile]) -> None:
        self._files: List[UploadedFile] = files

    def create_chunks(self) -> List[Document]:
        splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )

        documents: List[Document] = self._create_documents_list()
        chunks: List[Document] = splitter.split_documents(documents)
        return chunks

    def _create_documents_list(self) -> List[Document]:
        all_documents: List[Document] = []

        for file in self._files:
            filetype: str = file.type

            if filetype == "application/pdf":
                docs = self._extract_text_from_pdf(file)
            elif (
                filetype
                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ):
                docs = self._extract_text_from_word(file)
            elif (
                filetype
                == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            ):
                docs = self._extract_text_from_ppt(file)
            elif (
                filetype
                == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ):
                docs = self._extract_text_from_excel(file)
            else:
                continue  # Unsupported file type

            all_documents.extend(docs)

        return all_documents

    def _extract_text_from_pdf(self, file: UploadedFile) -> List[Document]:
        pdf_reader: PdfReader = PdfReader(file)
        to_documents: List[Document] = []

        for idx, page in enumerate(pdf_reader.pages):
            text: str = page.extract_text() or ""
            if text.strip():
                to_documents.append(
                    Document(
                        page_content=text,
                        metadata={"page": idx, "filename": file.name, "type": "pdf"},
                    )
                )
        return to_documents

    def _extract_text_from_word(self, file: UploadedFile) -> List[Document]:
        doc: DocxDocument = DocxDocument(file)
        full_text: str = "\n".join(
            [para.text for para in doc.paragraphs if para.text.strip()]
        )

        return [
            Document(
                page_content=full_text, metadata={"filename": file.name, "type": "word"}
            )
        ]

    def _extract_text_from_ppt(self, file: UploadedFile) -> List[Document]:
        presentation: Presentation = Presentation(file)
        to_documents: List[Document] = []

        for idx, slide in enumerate(presentation.slides):
            texts: List[str] = [
                shape.text.strip()
                for shape in slide.shapes
                if hasattr(shape, "text") and shape.text.strip()
            ]
            slide_text: str = "\n".join(texts)
            if slide_text:
                to_documents.append(
                    Document(
                        page_content=slide_text,
                        metadata={"slide": idx, "filename": file.name, "type": "ppt"},
                    )
                )

        return to_documents

    def _extract_text_from_excel(self, file: UploadedFile) -> List[Document]:
        to_documents: List[Document] = []
        in_memory_file = io.BytesIO(file.read())
        xls = pd.ExcelFile(in_memory_file)

        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name)
            text: str = df.to_string(index=False)
            if text.strip():
                to_documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "sheet": sheet_name,
                            "filename": file.name,
                            "type": "excel",
                        },
                    )
                )

        return to_documents


if __name__ == "__main__":
    DocumentsHandler(None)

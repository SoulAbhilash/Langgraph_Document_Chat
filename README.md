
# 📄 Chat with Your Files

A **Streamlit-based** interactive app that allows users to upload documents (PDFs, Word files, PowerPoint slides, Excel sheets) and interact with their content using natural language queries. Powered by LangChain, ChromaDB, and vector embeddings.

---

## 🚀 Features

- Upload and process multiple document types (`.pdf`, `.pptx`, `.ppt`, `.docx`, `.xlsx`)
- Ask questions about the content of your documents
- Conversational history saved per session
- Utilizes vector-based semantic search for context-aware answers
- Modular design using LangGraph and LangChain
- **NOTE: Tested only with gemini models, OpenAI is not tested yet so may not work as expected**

---

## 🧰 Tech Stack

- [Streamlit](https://streamlit.io/) for UI
- [LangChain](https://www.langchain.com/) for managing the language model and memory
- [ChromaDB](https://www.trychroma.com/) for vector storage
- `uuid`, `dotenv`, and `os` for session and environment management
- Custom modules: `graph.py` and `vector.py`

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/chat-with-files.git
cd chat-with-files
```
###  2. Create and Activate a Virtual Environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\\Scripts\\activate
```
###  3. Install Dependencies

```bash
pip install -r requirements.txt
```
### 4. Add Environment Variables
Create a .env file in the root directory and add the following variables:
```env
GOOGLE_API_KEY=""       # Required for Gemini-based models
MODEL_TYPE=""           # e.g., "openai" or "gemini"
MODEL_NAME=""           # e.g., "gpt-4", "gemini-pro"
OPENAI_API_KEY=""       # Required for OpenAI-based models
```
Note: Depending on the model provider, some variables may not be needed.

###  5. Run the App
```bash
streamlit run app.py
```
## 📌 How It Works
- Upload Files via the sidebar
- Click "Process" – files are vectorized and stored in ChromaDB
- Ask Questions in the chat input
- Get Responses based on your documents’ content
= The app creates a separate session for each user interaction using a thread_id, which is used to persist chat history.

## 🧠 Behind the Scenes
- VectorizeManager (from vector.py) handles parsing, chunking, and embedding your documents.
- build_langgraph_app (from graph.py) builds a LangGraph pipeline for intelligent query handling.
- Chat history is dynamically rendered and preserved between interactions.

R Abhilash — [GitHub](https://github.com/SoulAbhilash/Langgraph_Document_Chat.git) — [LinkedIn](https://www.linkedin.com/in/abhilash-r-199a48236)

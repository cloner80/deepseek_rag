# RAG Pipeline with Gradio and FAISS

This repository contains a **Retrieval-Augmented Generation (RAG) pipeline** with:
- **Multi-document ingestion**: Extracts text from PDFs, DOCX, XLSX, CSV, and TXT files.
- **Multi-folder support**: Indexes documents from multiple directories.
- **Vector storage with FAISS**: Uses embeddings for semantic search.
- **Conversational memory**: Supports multi-turn chat history.
- **Gradio-based UI**: Provides an easy-to-use chatbot interface.
- **LLM integration**: Uses an Ollama model for response generation.

## **Features**
- ✅ **Multiple folder support** (specify more than one directory in `config.yaml`).
- ✅ **Expanded file support**: Now supports **CSV** in addition to **PDF, DOCX, XLSX, and TXT**.
- ✅ **Conversation memory**: The chatbot remembers previous questions in the session.
- ✅ **Automatic FAISS index reuse**: Saves and reloads the index instead of rebuilding it every time.
- ✅ **Gradio-based UI**: Interactive web interface to ask questions.

---

## **1. Installation**

### **1.1. Clone the Repository**
```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo

1.2. Set Up Virtual Environment

python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# OR (Windows)
venv\Scripts\activate.bat

1.3. Install Dependencies

pip install --upgrade pip
pip install -r requirements.txt

2. Configuration

Before running, edit config.yaml to specify:

    The folders where your documents are stored.
    The LLM model to use.
    Whether to strip <think> blocks from responses.

Example config.yaml

model_name: "deepseek-r1:1.5b"
embedding_model_name: "sentence-transformers/all-mpnet-base-v2"
max_memory_gb: 3.0
index_path: "faiss_index"
remove_think: true

folder_paths:
  - "/XXX/YYY/ZZZ"
  - "/XXX2/YYY2/ZZZZ2"

🔹 Key Configurations
Parameter	Description
model_name	LLM model used for responses.
embedding_model_name	HuggingFace model for embeddings.
max_memory_gb	Memory threshold warning.
index_path	FAISS index storage path.
remove_think	Whether to remove <think> tags from responses.
folder_paths	List of folders to index documents from.
3. Running the App
3.1. First Run (Creates FAISS Index)

python rag_pipeline.py

    If faiss_index does not exist, it will build a new index from documents.
    If faiss_index already exists, it will reuse the saved index.

3.2. Updating Documents? Rebuild the Index

If you add new files or change the embedding model, delete the FAISS index to force a rebuild:

rm -rf faiss_index  # Linux/macOS
rmdir /s /q faiss_index  # Windows

Then, rerun the script:

python rag_pipeline.py

4. Using the Chat Interface

The chatbot runs on Gradio and will open a browser window at:

http://127.0.0.1:7860

4.1. Asking Questions

    The chatbot remembers previous questions in the same session.
    It searches for relevant document snippets and uses them to generate an answer.
    Example:

    User: What is mentioned in the documents about Company XYZ?
    Assistant: Based on the retrieved documents, Company XYZ was involved in...

4.2. Clearing the Chat

Click the "Clear Chat" button to reset conversation memory.
5. Supported File Formats
File Type	Supported?
✅ PDF	✔ Yes
✅ DOCX	✔ Yes
✅ XLSX	✔ Yes
✅ TXT	✔ Yes
✅ CSV	✔ Yes (new!)
6. Troubleshooting
6.1. SSH Host Key Error When Cloning Repo

If you see:

Host key verification failed.

Fix it by running:

ssh-keygen -R github.com
ssh -T git@github.com

6.2. FAISS Embedding Dimension Mismatch

If you get:

AssertionError: d == self.d

You changed the embedding model but are still using an old FAISS index. Delete and rebuild:

rm -rf faiss_index
python rag_pipeline.py

6.3. Gradio Error: 'dict' object has no attribute 'replace'

This happens if the retriever is passed a dictionary instead of a string. Make sure your pipeline extracts only the "question" key when calling retriever.
7. License

This project is licensed under the Apache License 2.0.
Patent Grant

This license provides an explicit grant of patent rights from contributors to users. See the LICENSE file for details.
Disclaimer

This software is provided on an "AS IS" basis, without warranties or conditions of any kind.
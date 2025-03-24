# RAG Pipeline with Gradio and FAISS

This repository contains a **Retrieval-Augmented Generation (RAG) pipeline** with:
- **Multi-document ingestion**: Extracts text from PDFs, DOCX, XLSX, CSV, MD, TXT, and images (JPG, PNG).
- **Multi-folder support**: Indexes documents from multiple directories.
- **Vector storage with FAISS**: Uses embeddings for semantic search.
- **Conversational memory**: Supports multi-turn chat history.
- **Gradio-based UI**: Provides an easy-to-use chatbot interface.
- **LLM integration**: Uses an Ollama model for response generation.

---

## Features
- ✅ **Multiple folder support** (`config.yaml`).
- ✅ **Expanded file support**: Images (**OCR**), **MD**, **CSV**, **PDF**, **DOCX**, **XLSX**, and **TXT**.
- ✅ **Conversation memory**: Remembers previous questions.
- ✅ **Automatic FAISS index reuse**.
- ✅ **Gradio-based UI**.

---

## Installation

### 1. Install Ollama

- Download from [Ollama](https://ollama.com/)
- Run Ollama app, then pull a model:

```bash
ollama pull deepseek-r1:1.5b
```

### 2. Clone Repository
```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

### 3. Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate.bat  # Windows
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Configuration

Edit `config.yaml`:

- Specify document directories
- Select LLM and embedding models
- Optionally remove `<think>` tags

**Example:**
```yaml
model_name: "deepseek-r1:1.5b"
embedding_model_name: "sentence-transformers/all-mpnet-base-v2"
max_memory_gb: 3.0
index_path: "faiss_index"
remove_think: true
folder_paths:
  - "/path/to/documents"
```

**Windows example:**
```yaml
folder_paths:
  - "C:\\path\\to\\documents"
```

| Parameter             | Description                            |
|-----------------------|----------------------------------------|
| model_name            | LLM model                              |
| embedding_model_name  | Embedding model                        |
| max_memory_gb         | Memory threshold                       |
| index_path            | Path for FAISS index                   |
| remove_think          | Remove `<think>` tags                  |
| folder_paths          | List of folders containing documents   |

---

## Running the App

### First Run (Creates FAISS Index)

```bash
python rag_pipeline.py
```

- Creates new index or loads existing one.

> **Screenshot:**
>
> ![Gradio UI](screenshots/gradio_ui.png)

### Updating Documents

Delete FAISS index to rebuild:

```bash
rm -rf faiss_index       # Linux/macOS
rmdir /s /q faiss_index  # Windows
python rag_pipeline.py
```

---

## Using the Chat Interface

Open in your browser:

```
http://127.0.0.1:7860
```

- Ask questions about indexed documents.
- The chatbot maintains session memory.

> **Example Conversation:**
>
> ![Example Chat](screenshots/example_chat.png)

Click "Clear Chat" to reset.

---

## Supported File Formats

| File Type | Supported? |
|-----------|------------|
| PDF       | ✅         |
| DOCX      | ✅         |
| XLSX      | ✅         |
| TXT       | ✅         |
| CSV       | ✅         |
| MD        | ✅         |
| JPG/PNG   | ✅ (OCR)   |

---

## Troubleshooting

### SSH Host Key Error

- Ensure correct repository URL or use HTTPS instead of SSH.

### FAISS Embedding Dimension Mismatch

```bash
rm -rf faiss_index
python rag_pipeline.py
```

### Gradio `'dict' object has no attribute 'replace'`

- Verify retriever inputs; ensure proper extraction of "question".

---

## License

Licensed under Apache License 2.0. See [LICENSE](LICENSE) for details.

---

## Screenshots

> **Document Indexing:**
>
> ![Indexing Process](screenshots/indexing_process.png)

> **Conversational Memory:**
>
> ![Conversation Memory](screenshots/conversation_memory.png)

---


# RAG Pipeline with Gradio and FAISS

This repository contains a **Retrieval-Augmented Generation (RAG) pipeline** with:
- **Multi-document ingestion**: Extracts text from PDFs, DOCX, XLSX, CSV, MD, TXT, and images (JPG, PNG).
- **Multi-folder support**: Indexes documents from multiple directories.
- **Vector storage with FAISS**: Uses embeddings for semantic search.
- **Conversational memory**: Supports multi-turn chat history.
- **Web search integration**: Automatically searches the internet for current information.
- **Gradio-based UI**: Provides an easy-to-use chatbot interface.
- **LLM integration**: Uses an Ollama model for response generation.

---

## Features
- ✅ **Multiple folder support** (`config.yaml`).
- ✅ **Expanded file support**: Images (**OCR**), **MD**, **CSV**, **PDF**, **DOCX**, **XLSX**, and **TXT**.
- ✅ **Conversation memory**: Remembers previous questions.
- ✅ **Automatic FAISS index reuse**.
- ✅ **Web search integration**: Searches DuckDuckGo for current information.
- ✅ **Smart web search triggers**: Automatically detects when web search is needed.
- ✅ **Gradio-based UI** with web search status indicators.

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
- Configure web search settings
- Optionally remove `<think>` tags

**Example:**
```yaml
model_name: "deepseek-r1:1.5b"
embedding_model_name: "sentence-transformers/all-mpnet-base-v2"
max_memory_gb: 3.0
index_path: "faiss_index"
remove_think: true

# Web search configuration
enable_web_search: true
max_web_results: 3
web_search_timeout: 10

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
| enable_web_search     | Enable web search functionality         |
| max_web_results       | Maximum web search results to process  |
| web_search_timeout    | Timeout for web requests (seconds)     |
| folder_paths          | List of folders containing documents   |

---

## Web Search Features

The system automatically detects when web search is needed based on keywords and patterns:

### Automatic Web Search Triggers
- **Time-related**: "latest", "recent", "current", "today", "yesterday"
- **News and updates**: "news", "update", "breaking", "trending"
- **Comparisons**: "best", "top", "compare", "vs", "versus"
- **Pricing**: "price", "cost", "how much", "where to buy"
- **Current events**: "weather", "forecast", "stock", "crypto"
- **Entertainment**: "movie", "film", "show", "release date"
- **Travel**: "restaurant", "hotel", "travel", "vacation"

### Example Web Search Queries
- "What are the latest news about AI?"
- "What's the current price of Bitcoin?"
- "What's the weather like in New York today?"
- "What are the best restaurants in Paris?"
- "Compare iPhone 15 vs Samsung Galaxy S24"

### Manual Web Search
The **🔍 Web Search** button allows you to force web search for any query, even if the automatic detection doesn't trigger it. This is useful for:
- Questions that don't match the automatic triggers
- When you want to ensure you get the most current information
- Research queries that might benefit from web sources
- Any question where you prefer web results over local documents

**Example manual web search queries:**
- "What is the history of the Eiffel Tower?" (might not trigger automatic search)
- "Tell me about quantum computing" (general knowledge query)
- "What are the benefits of meditation?" (wellness/health query)

---

## Running the App

### First Run (Creates FAISS Index)

```bash
python rag_pipeline.py
```

- Creates new index or loads existing one.
- Web search is enabled by default but can be toggled in the UI.

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

### Interface Features
- **Chat Interface**: Ask questions about indexed documents and current web information
- **Web Search Toggle**: Enable/disable automatic web search functionality
- **Manual Web Search Button**: Force web search for any query (bypasses automatic detection)
- **Status Indicator**: Shows whether the system is searching local documents or the web
- **Conversation Memory**: Maintains chat history across the session

### Web Search Status Messages
- 🔍 **"Searching the web for current information..."** - When automatic web search is active
- 🔍 **"Manually searching the web..."** - When manual web search is triggered
- ✅ **"Found X web sources"** - When automatic web search returns results
- ✅ **"Manual web search completed - Found X web sources"** - When manual web search returns results
- ⚠️ **"Web search failed, using local documents"** - When automatic web search fails
- ⚠️ **"Manual web search failed, using local documents"** - When manual web search fails
- 📚 **"Searching local documents..."** - When using local documents only
- ✅ **"Answered using local documents"** - When using local documents successfully

> **Example Conversation:**
>
> ![Example Chat](screenshots/example_chat.png)

Click "Clear Chat" to reset the conversation.

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

### Web Search Issues

- **Network connectivity**: Ensure internet connection is available
- **Rate limiting**: The system includes delays between requests to be respectful to websites
- **Timeout errors**: Increase `web_search_timeout` in config.yaml if needed
- **No results**: System will fall back to local documents if web search fails

### Memory Issues

- Reduce `max_web_results` in config.yaml to process fewer web pages
- Increase `max_memory_gb` if you have sufficient RAM

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

> **Web Search Interface:**
>
> ![Web Search UI](screenshots/web_search_ui.png)

---


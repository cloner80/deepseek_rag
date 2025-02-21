# RAG Pipeline with Gradio and FAISS

This repository contains a Retrieval-Augmented Generation (RAG) application that:
- **Extracts** text from multiple document types (PDF, DOCX, XLSX, TXT).
- **Embeds** them with HuggingFace embeddings.
- **Indexes** them in a FAISS vector store.
- **Runs** a local Large Language Model (LLM) via [Ollama](https://github.com/jmorganca/ollama) (Mac) or any other model you configure.
- **Serves** a Gradio interface to answer questions against these indexed documents.

## Features

1. **Config File**: All critical parameters (e.g., model name, folder path, etc.) are stored in `config.yaml`.
2. **Automatic Retrieval**: Your queries are turned into vector embeddings, and the most relevant chunks are retrieved and passed to the LLM.
3. **Gradio UI**: A simple interface for asking questions about your documents.
4. **Optional**: Remove `<think>` annotations from LLM output, controlled by a config parameter (`remove_think`).

## Getting Started

1. Clone this Repository

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo

2. Create a Virtual Environment

It is recommended to use a Python virtual environment:

python3 -m venv venv
source venv/bin/activate   # On macOS/Linux
# or
venv\Scripts\activate.bat  # On Windows

3. Install Dependencies

Install all required libraries from requirements.txt:

pip install --upgrade pip
pip install -r requirements.txt

4. Verify or Update config.yaml

Make sure your config.yaml has the correct paths and model names for your environment. Example:

model_name: "deepseek-r1:1.5b"
embedding_model_name: "sdadas/stella-pl-retrieval"
folder_path: "/data"
index_path: "faiss_index"
max_memory_gb: 3.0
remove_think: true

    model_name: The name or path of the model you want to load via Ollama.
    embedding_model_name: A HuggingFace model for generating embeddings.
    folder_path: Where the script will recursively look for PDF, DOCX, XLSX, and TXT files.
    index_path: Where the FAISS index will be created or loaded.
    max_memory_gb: A threshold for a memory check (you’ll see a warning if available memory is below this).
    remove_think: true means the script will remove <think>...</think> blocks from final answers.

5. Install & Run Ollama

Install and run Ollama on macOS:

brew install ollama/tap/ollama
ollama serve

If you are not on macOS use this link to get info how to run Ollama on different operating system 
https://github.com/ollama/ollama?tab=readme-ov-file

Run Ollama server in separate terminal window 



6. Run the Application

python rag_pipeline.py

Gradio will launch a local server. You will see an output such as:

Running on local URL:  http://127.0.0.1:7860

Open that link in your browser, and you can start asking questions about your documents.

Usage Notes

    First Run: The script will parse all documents in folder_path, build a FAISS index, and store it in index_path. On subsequent runs, it will reuse that index (which saves time). When there was a change in your docs you need to remove the index_path dir so that app will index the files again.
    Adding/Updating Documents: If you add new files or want to re-index, simply remove (or rename) the index_path folder and rerun. This forces a full rebuild.
    Memory Usage: The script logs your memory usage before answering each question. If you’re running large models on low-resources hardware, consider using smaller or quantized models.
    Removing <think>: If remove_think is true in config.yaml, the script will strip <think>...</think> from the final answers. If you want to see the full raw response from the LLM, set remove_think to false.

Troubleshooting

    Missing Packages: If you encounter an ImportError or ModuleNotFoundError, add the missing library to requirements.txt and reinstall.
    FAISS GPU: If you have a GPU and want faster similarity search, replace faiss-cpu with faiss-gpu (and ensure you have CUDA installed).
    Version Conflicts: If you see version mismatches (e.g., with torch or transformers), pin the versions in requirements.txt or upgrade/downgrade accordingly.


 ## License

This project is licensed under the [Apache License 2.0](LICENSE).


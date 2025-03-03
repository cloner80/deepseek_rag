import os
import logging
import psutil
import pdfplumber
import docx
import openpyxl
import re
import yaml
import csv
import gradio as gr

from typing import List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import FAISS


def load_config(config_file: str = "config.yaml") -> dict:
    """Load the YAML config."""
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class RAGPipeline:
    def __init__(
        self,
        model_name: str,
        embedding_model_name: str,
        max_memory_gb: float
    ):
        self.setup_logging()
        self.check_system_memory(max_memory_gb)
        
        # LLM
        self.llm = OllamaLLM(model=model_name)
        
        # Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'}
        )
        
        # Chat prompt template that includes chat_history
        self.chat_prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant that answers questions based on the provided documents.

Conversation so far:
{chat_history}

Relevant context from the documents:
{context}

Now answer the user's latest question: {question}
""")

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def check_system_memory(self, max_memory_gb: float):
        available_memory = psutil.virtual_memory().available / (1024 ** 3)
        self.logger.info(f"Available system memory: {available_memory:.1f} GB")
        if available_memory < max_memory_gb:
            self.logger.warning("Memory is below recommended threshold.")

    def extract_text_from_csv(self, filepath: str) -> str:
        """
        Reads the CSV, outputs:
        1) First line with comma-separated headers: "column1,column2"
        2) Each row as "column1: Text,column2: Text"
        """
        lines = []
        try:
            with open(filepath, mode="r", encoding="utf-8", errors="ignore") as f:
                csv_reader = csv.reader(f)
                headers = next(csv_reader, None)
                if not headers:
                    self.logger.warning(f"No headers found in CSV '{filepath}'")
                    return ""
                # First line: headers, comma-separated
                lines.append(",".join(headers))

                # Subsequent rows: "header1: val1,header2: val2"
                for row in csv_reader:
                    row_text = []
                    for i in range(min(len(row), len(headers))):
                        row_text.append(f"{headers[i]}: {row[i]}")
                    lines.append(",".join(row_text))
        except Exception as e:
            self.logger.warning(f"Failed to parse CSV '{filepath}' with error: {e}")
            return ""
        return "\n".join(lines)

    def extract_text_from_pdf(self, filepath: str) -> str:
        try:
            text_list = []
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text_list.append(page_text)
            return "\n".join(text_list)
        except Exception as e:
            self.logger.warning(f"pdfplumber failed on '{filepath}' with error: {e}")
            return ""

    def extract_text_from_docx(self, filepath: str) -> str:
        doc = docx.Document(filepath)
        return "\n".join(para.text for para in doc.paragraphs)

    def extract_text_from_xlsx(self, filepath: str) -> str:
        """
        Similar logic to CSV:
        1) First row => comma-separated headers
        2) Subsequent rows => "header1: val1,header2: val2"
        Combines data from all sheets one after another.
        """
        wb = openpyxl.load_workbook(filepath, read_only=True)
        lines = []
        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            header = None
            for row_index, row in enumerate(sheet.iter_rows(values_only=True)):
                row_values = list(row) if row else []
                # Skip empty rows
                if all(cell is None for cell in row_values):
                    continue
                # First row => set headers
                if row_index == 0:
                    header = [str(col) if col else "UNKNOWN_COLUMN" for col in row_values]
                    lines.append(",".join(header))
                else:
                    # If there's no header, we can't format properly
                    if not header:
                        self.logger.warning(f"No header found in sheet {sheet_name} for file {filepath}")
                        break
                    row_text = []
                    for i in range(min(len(header), len(row_values))):
                        row_text.append(f"{header[i]}: {row_values[i]}")
                    lines.append(",".join(row_text))
        return "\n".join(lines)

    def extract_text_from_txt(self, filepath: str) -> str:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def extract_text_from_md(self, filepath: str) -> str:
        """Reads Markdown (.md) files as plain text."""
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def extract_text(self, filepath: str) -> str:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".pdf":
            return self.extract_text_from_pdf(filepath)
        elif ext == ".docx":
            return self.extract_text_from_docx(filepath)
        elif ext == ".xlsx":
            return self.extract_text_from_xlsx(filepath)
        elif ext == ".txt":
            return self.extract_text_from_txt(filepath)
        elif ext == ".csv":
            return self.extract_text_from_csv(filepath)
        elif ext == ".md":
            return self.extract_text_from_md(filepath)
        return ""

    def gather_documents_from_folder(self, folder_path: str) -> List[Document]:
        if not os.path.exists(folder_path):
            self.logger.warning(f"Folder does not exist: {folder_path}")
            return []

        docs = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith((".pdf", ".docx", ".xlsx", ".txt", ".csv", ".md")):
                    path = os.path.join(root, file)
                    text = self.extract_text(path)
                    if text.strip():
                        docs.append(Document(page_content=text, metadata={"source": path}))
        
        if not docs:
            self.logger.warning(f"No recognized files found in folder: {folder_path}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            add_start_index=True,
        )
        splits = text_splitter.split_documents(docs)
        self.logger.info(f"Created {len(splits)} document chunks for folder: {folder_path}")
        return splits

    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        batch_size = 32
        if not documents:
            self.logger.warning("No documents provided to create_vectorstore. Creating an empty vector store.")
            return FAISS.from_texts([""], self.embeddings)

        vectorstore = FAISS.from_documents(documents[:batch_size], self.embeddings)
        for i in range(batch_size, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            vectorstore.add_documents(batch)
            self.logger.info(f"Processed batch {i // batch_size + 1}")
        return vectorstore
    
    def get_question(self, inputs: dict) -> str:
        return inputs["question"]

    def setup_rag_chain(self, vectorstore: FAISS):
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2, "fetch_k": 3}
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {
                "chat_history": RunnablePassthrough(),
                "context": RunnableLambda(self.get_question) | retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | self.chat_prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain
    
    def query(self, chain, chain_inputs: dict) -> str:
        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        self.logger.info(f"Memory usage: {memory_usage:.1f} MB")
        return chain.invoke(chain_inputs)


def build_pipeline(config: dict):
    rag = RAGPipeline(
        model_name=config["model_name"],
        embedding_model_name=config["embedding_model_name"],
        max_memory_gb=config["max_memory_gb"]
    )

    all_docs = []
    folder_paths = config.get("folder_paths", [])
    for folder_path in folder_paths:
        doc_splits = rag.gather_documents_from_folder(folder_path)
        all_docs.extend(doc_splits)

    if not all_docs:
        rag.logger.warning("No documents found in any specified folders. The vectorstore will be empty.")

    index_path = config["index_path"]
    if os.path.exists(index_path):
        rag.logger.info("Loading existing FAISS index from disk...")
        vectorstore = FAISS.load_local(index_path, rag.embeddings, allow_dangerous_deserialization=True)
    else:
        rag.logger.info("Creating a new FAISS index and saving it locally...")
        vectorstore = rag.create_vectorstore(all_docs)
        vectorstore.save_local(index_path)

    chain = rag.setup_rag_chain(vectorstore)
    return rag, chain


def main():
    config = load_config("config.yaml")
    rag, chain = build_pipeline(config)
    remove_think_flag = config.get("remove_think", True)

    def chat(user_message, history):
        conversation_text = ""
        for msg_dict in history:
            role = msg_dict["role"]
            content = msg_dict["content"]
            if role == "user":
                conversation_text += f"User: {content}\n"
            else:
                conversation_text += f"Assistant: {content}\n"

        chain_inputs = {
            "chat_history": conversation_text,
            "question": user_message
        }

        response = rag.query(chain, chain_inputs)
        if remove_think_flag:
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": response})
        return history, history, ""

    with gr.Blocks() as demo:
        gr.Markdown("## RAG Chat with Conversation History")

        chatbot = gr.Chatbot(label="Conversation", type="messages")
        msg = gr.Textbox(label="Your question")

        clear_btn = gr.Button("Clear Chat")
        state = gr.State([])

        msg.submit(
            fn=chat,
            inputs=[msg, state],
            outputs=[chatbot, state, msg]
        )

        def clear_chat():
            return [], [], ""

        clear_btn.click(
            fn=clear_chat,
            inputs=[],
            outputs=[chatbot, state, msg]
        )

        demo.launch()


if __name__ == "__main__":
    main()

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

    # Example CSV method
    def extract_text_from_csv(self, filepath: str) -> str:
        rows = []
        try:
            with open(filepath, mode="r", encoding="utf-8", errors="ignore") as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    rows.append(" ".join(row))
        except Exception as e:
            self.logger.warning(f"Failed to parse CSV '{filepath}' with error: {e}")
            return ""
        return "\n".join(rows)

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
        wb = openpyxl.load_workbook(filepath, read_only=True)
        extracted_data = []
        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            for row in sheet.iter_rows(values_only=True):
                row_text = [str(cell) for cell in row if cell]
                if row_text:
                    extracted_data.append(" ".join(row_text))
        return "\n".join(extracted_data)

    def extract_text_from_txt(self, filepath: str) -> str:
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
        return ""

    def gather_documents_from_folder(self, folder_path: str) -> List[Document]:
        docs = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith((".pdf", ".docx", ".xlsx", ".txt", ".csv")):
                    path = os.path.join(root, file)
                    text = self.extract_text(path)
                    if text.strip():
                        docs.append(Document(page_content=text, metadata={"source": path}))
        
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
        vectorstore = FAISS.from_documents(documents[:batch_size], self.embeddings)
        
        for i in range(batch_size, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            vectorstore.add_documents(batch)
            self.logger.info(f"Processed batch {i//batch_size + 1}")
        return vectorstore
    
    def get_question(self, inputs: dict) -> str:
    # Our chain inputs look like:
    #   {"chat_history": str, "question": str}
    # We only return the "question" string
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
                # The 'context' is derived from the question
                "context": RunnableLambda(self.get_question) | retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | self.chat_prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain
    
    def query(self, chain, chain_inputs: dict) -> str:
        """
        chain_inputs: a dict that must have:
          - 'chat_history': str
          - 'question': str
        The 'context' is auto-handled by the chain pipeline via the retriever step.
        """
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
    # 1. Load config
    config = load_config("config.yaml")

    # 2. Build pipeline
    rag, chain = build_pipeline(config)

    # 3. Whether to remove <think> blocks
    remove_think_flag = config.get("remove_think", True)

    # 4. Define chat function that references pipeline from closure
    def chat(user_message, history):
        """
        user_message: the new user input
        history: a list of message dicts in "role": "user"/"assistant", "content": ...
        (this is how Gradio stores it when type="messages")
        """
        # Convert the entire chat history to a text block
        conversation_text = ""
        for msg_dict in history:
            role = msg_dict["role"]
            content = msg_dict["content"]
            if role == "user":
                conversation_text += f"User: {content}\n"
            else:
                conversation_text += f"Assistant: {content}\n"

        # Prepare chain inputs
        chain_inputs = {
            "chat_history": conversation_text,
            "question": user_message
        }

        # Query the RAG pipeline
        response = rag.query(chain, chain_inputs)

        # Optionally remove <think> tags
        if remove_think_flag:
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

        # Append user message
        history.append({"role": "user", "content": user_message})
        # Append assistant reply
        history.append({"role": "assistant", "content": response})

        return history, history

    # 5. Build Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("## RAG Chat with Conversation History")

        chatbot = gr.Chatbot(
            label="Conversation",
            type="messages"  # use new 'messages' format
        )
        msg = gr.Textbox(label="Your question")

        clear_btn = gr.Button("Clear Chat")
        state = gr.State([])  # empty list => no conversation yet

        # The user hits enter on the textbox => call chat
        msg.submit(
            fn=chat,
            inputs=[msg, state],
            outputs=[chatbot, state]
        )

        def clear_chat():
            return [], []

        clear_btn.click(
            fn=clear_chat,
            inputs=[],
            outputs=[chatbot, state]
        )

        # Launch Gradio
        demo.launch()

if __name__ == "__main__":
    main()


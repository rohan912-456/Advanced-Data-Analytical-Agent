import os
import faiss
import numpy as np
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# Directory for the knowledge library (e.g., industry PDFs, whitepapers)
KNOWLEDGE_BASE_DIR = "knowledge_library"
INDEX_PATH = "knowledge_library/faiss_index"

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def get_rag_index():
    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Initialize empty index if not exists
    index = faiss.IndexFlatL2(768)
    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={}
    )

@tool
def search_knowledge_library(query: str) -> str:
    """
    Search the 'Expert Knowledge Library' for industry benchmarks, strategy whitepapers, 
    and architectural patterns. 
    
    Use this when you need context beyond the immediate dataset (e.g., "What are the standard churn rates for SaaS?").
    """
    try:
        if not os.path.exists(KNOWLEDGE_BASE_DIR):
            return "Knowledge library is empty. Please upload reference PDFs to the 'knowledge_library' folder."
            
        vectorstore = get_rag_index()
        docs = vectorstore.similarity_search(query, k=3)
        
        if not docs:
            return "No matching expert knowledge found for that query."
            
        results = []
        for d in docs:
            results.append(f"--- From: {d.metadata.get('source', 'Unknown')} ---\n{d.page_content}\n")
            
        return "\n".join(results)
    except Exception as e:
        return f"RAG search error: {str(e)}"

@tool
def ingest_knowledge_document(file_path: str) -> str:
    """
    Adds a document (PDF/Text) to the Expert Knowledge Library for future RAG searches.
    """
    try:
        from tools.pdf_tool import read_pdf
        
        if file_path.endswith(".pdf"):
            content = read_pdf(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
        vectorstore = get_rag_index()
        vectorstore.add_texts([content], metadatas=[{"source": os.path.basename(file_path)}])
        
        os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
        vectorstore.save_local(INDEX_PATH)
        
        return f"Successfully ingested '{os.path.basename(file_path)}' into the Knowledge Library."
    except Exception as e:
        return f"Ingestion error: {str(e)}"

import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.tools import tool
from langchain_core.documents import Document

# ── DEEP BRAIN (Vector Memory) ────────────────────────────────────────────────
DB_PATH = "memory/deep_brain_store"

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

class DeepBrain:
    """
    Persistent long-term memory system using FAISS and Google Embeddings.
    Allows the agent to store and retrieve insights across sessions.
    """
    def __init__(self):
        self.embeddings = get_embeddings()
        if os.path.exists(DB_PATH):
            self.vector_store = FAISS.load_local(DB_PATH, self.embeddings, allow_dangerous_deserialization=True)
        else:
            # Initialize with a dummy document to create the store
            self.vector_store = FAISS.from_documents(
                [Document(page_content="System Initialized", metadata={"source": "system"})],
                self.embeddings
            )
            self.vector_store.save_local(DB_PATH)

    def add_insight(self, text: str, metadata: dict = None):
        doc = Document(page_content=text, metadata=metadata or {})
        self.vector_store.add_documents([doc])
        self.vector_store.save_local(DB_PATH)

    def search(self, query: str, k: int = 3):
        return self.vector_store.similarity_search(query, k=k)

# Global instances
_deep_brain = None

def get_deep_brain():
    global _deep_brain
    if _deep_brain is None:
        _deep_brain = DeepBrain()
    return _deep_brain

@tool
def store_insight(insight: str):
    """
    Stores an important discovery or insight into the agent's long-term 'Deep Brain'.
    Use this for facts, data patterns, or strategic conclusions that should be 
    remembered in future sessions.
    """
    try:
        db = get_deep_brain()
        db.add_insight(insight, {"timestamp": os.getlogin()})
        return f"Insight successfully committed to Deep Brain: '{insight[:50]}...'"
    except Exception as e:
        return f"Error storing insight: {str(e)}"

@tool
def recall_past_insights(query: str):
    """
    Searches the agent's long-term 'Deep Brain' for past insights related to the query.
    Use this to recall findings from previous datasets or past conversations.
    """
    try:
        db = get_deep_brain()
        docs = db.search(query)
        if not docs:
            return "No matching insights found in Deep Brain."
        
        results = [f"- {d.page_content}" for d in docs if d.page_content != "System Initialized"]
        return "## Recalled Insights from Deep Brain:\n" + "\n".join(results)
    except Exception as e:
        return f"Error recalling insights: {str(e)}"

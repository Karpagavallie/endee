import os
import google.generativeai as genai

# gemini-embedding-001 outputs 3072-dimensional vectors
EMBEDDING_DIM = 3072

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def get_embedding(text: str) -> list[float]:
    """Generate a document embedding for storing."""
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
        task_type="retrieval_document",
    )
    return result["embedding"]


def get_query_embedding(text: str) -> list[float]:
    """Generate a query embedding for searching."""
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
        task_type="retrieval_query",
    )
    return result["embedding"]
"""
rag.py — Retrieval-Augmented Generation pipeline

Flow
────
1. User asks a question or types a search query.
2. Gemini embeds the query → 768-dim vector.
3. Endee performs ANN (cosine) search → top-K matching vocabulary records.
4. Retrieved records are formatted into a context block.
5. Gemini generates a natural-language answer grounded in that context.
"""

import os
import google.generativeai as genai
from database import get_index
from embedder import get_query_embedding

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ── Semantic Search (pure vector retrieval) ───────────────────────────────────

def semantic_search(
    query: str,
    top_k: int = 5,
    category: str | None = None,
) -> list[dict]:
    """
    Search the vocabulary index by meaning.

    Args:
        query    : natural-language query  e.g. "something that lasts a short time"
        top_k    : number of results to return
        category : optional Endee filter   e.g. "science" or "literature"

    Returns:
        List of Endee result dicts, each containing 'id', 'similarity', 'meta'.
    """
    index = get_index()
    query_vector = get_query_embedding(query)

    params = dict(
        vector=query_vector,
        top_k=top_k,
        ef=128,               # higher ef = better recall, slower search
        include_vectors=False,
    )

    if category:
        # Endee filter syntax: list of {field: {$operator: value}}
        params["filter"] = [{"category": {"$eq": category}}]

    return index.query(**params)


# ── RAG Pipeline ──────────────────────────────────────────────────────────────

def ask_vocabulary(question: str) -> str:
    """
    Full RAG pipeline:
      retrieve → build context → generate answer with Gemini.

    Args:
        question : anything the user wants to know about their vocabulary

    Returns:
        Gemini's answer as a plain string.
    """
    # Step 1 — Retrieve relevant vocabulary from Endee
    results = semantic_search(question, top_k=5)

    if not results:
        return "📭 No vocabulary found yet. Add some words first (option 1)!"

    # Step 2 — Build context block from retrieved Endee records
    lines = []
    for r in results:
        meta = r.get("meta", {})
        word       = meta.get("word", r.get("id", "?"))
        definition = meta.get("definition", "")
        example    = meta.get("example", "")
        sim        = r.get("similarity", 0.0)

        line = f"• {word}: {definition}"
        if example:
            line += f"  (e.g. \"{example}\")"
        lines.append(line)

    context = "\n".join(lines)

    # Step 3 — Construct RAG prompt
    prompt = f"""You are a friendly and knowledgeable vocabulary tutor.
The user has a personal vocabulary collection. Use ONLY the words listed below as your knowledge source.

--- Vocabulary Context ---
{context}
--------------------------

User question: {question}

Guidelines:
- If asked to use words in sentences, write vivid, memorable examples.
- If asked about meaning, give a clear, concise explanation.
- If asked to quiz the user, create a short quiz from the words above.
- If the question cannot be answered from the vocabulary, say so politely.
- Never invent words or definitions not in the context.
"""

    # Step 4 — Generate answer with Gemini
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text


# ── Quiz Generator ────────────────────────────────────────────────────────────

def generate_quiz(num_questions: int = 3) -> str:
    """
    Generate a multiple-choice quiz from the user's vocabulary using RAG.
    """
    question = (
        f"Create a {num_questions}-question multiple-choice quiz "
        "using the vocabulary words. For each question give 4 options "
        "labelled A-D and mark the correct answer. Keep it fun!"
    )
    return ask_vocabulary(question)

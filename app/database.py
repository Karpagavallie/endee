import os
import time
from endee import Endee, Precision
from embedder import get_embedding, EMBEDDING_DIM

INDEX_NAME = "vocabulary"

def get_client():
    client = Endee()
    endee_url = os.getenv("ENDEE_URL", "http://localhost:8080/api/v1")
    client.set_base_url(endee_url)
    return client

def wait_for_endee(retries=10, delay=2.0):
    import requests
    base = os.getenv("ENDEE_URL", "http://localhost:8080/api/v1")
    health_url = base.replace("/api/v1", "")
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(health_url, timeout=3)
            if r.status_code < 500:
                print("✅ Endee is ready.")
                return
        except Exception:
            pass
        print(f"⏳ Waiting for Endee... ({attempt}/{retries})")
        time.sleep(delay)
    raise RuntimeError("❌ Could not connect to Endee.")

def get_index():
    client = get_client()
    try:
        client.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            space_type="cosine",
            precision=Precision.FLOAT32
        )
        print("✅ Index created.")
    except Exception as e:
        if "conflict" not in str(e).lower() and "already exists" not in str(e).lower():
            raise e
    return client.get_index(name=INDEX_NAME)

def add_word(word, definition, example="", category="general"):
    index = get_index()
    word_id = word.lower().replace(" ", "_")
    embed_text = "Word: " + word + ". Definition: " + definition + "."
    if example:
        embed_text += " Example: " + example + "."
    vector = get_embedding(embed_text)
    index.upsert([{
        "id": word_id,
        "vector": vector,
        "meta": {
            "word": word.lower(),
            "definition": definition,
            "example": example
        },
        "filter": {
            "category": category
        }
    }])
    print("✅ " + word + " added!")
    return True

def delete_word(word):
    index = get_index()
    index.delete([word.lower().replace(" ", "_")])
    print("🗑️ " + word + " deleted.")

def list_words(top_k=100):
    from embedder import get_query_embedding
    index = get_index()
    vector = get_query_embedding("word vocabulary definition meaning language")
    return index.query(vector=vector, top_k=top_k, ef=200, include_vectors=False)
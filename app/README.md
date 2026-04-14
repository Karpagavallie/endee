# 🧠 AI Vocabulary

A personal AI-powered vocabulary builder using:

| Component | Role |
|---|---|
| **Endee** | Vector database (runs in Docker, stores embeddings) |
| **Gemini `text-embedding-004`** | Converts words & queries into 768-dim vectors |
| **Semantic Search** | Finds words by meaning, not just spelling |
| **RAG** | Retrieved vocab → Gemini prompt → natural AI answer |

---

## Project Structure

```
ai_vocabulary/
├── docker-compose.yml      # Endee server + Python app containers
├── .env                    # Your Gemini API key (never commit this)
├── .gitignore
├── data/                   # Endee persists its data here (auto-created)
└── app/
    ├── Dockerfile
    ├── requirements.txt
    ├── embedder.py         # Gemini embedding calls
    ├── database.py         # Endee index + CRUD operations
    ├── rag.py              # Semantic search + RAG pipeline
    └── main.py             # CLI entry point
```

---

## Quick Start

### 1. Get a Gemini API key (free)
Go to https://aistudio.google.com/app/apikey and copy your key.

### 2. Set your API key
Edit `.env`:
```
GEMINI_API_KEY=your_actual_key_here
```

### 3. Start the app
```bash
docker compose up --build
```

This starts two containers:
- **endee-server** — Endee vector DB on port 8080
- **ai-vocabulary-app** — your Python CLI

### 4. Use the CLI (in a second terminal)
```bash
docker compose exec app python main.py
```

Or attach interactively:
```bash
docker attach ai-vocabulary-app
```

---

## Features

| Menu Option | What it does |
|---|---|
| **1. Add word** | Embed word+definition with Gemini, store in Endee |
| **2. Semantic search** | Find words by meaning (e.g. "lasting a short time" → ephemeral) |
| **3. Ask AI (RAG)** | Retrieve relevant words → Gemini answers your question |
| **4. Filter by category** | Search within science / literature / business etc. |
| **5. Quiz** | Gemini generates a multiple-choice quiz from your words |
| **6. List all** | Browse your full vocabulary |
| **7. Delete** | Remove a word from Endee |

---

## Example

```
Add:    word="ephemeral"  definition="lasting a very short time"
        example="The ephemeral beauty of cherry blossoms"
        category="literature"

Search: "things that disappear quickly"
→ finds ephemeral even though you didn't type the word!

Ask:    "Write a short paragraph using my literature words"
→ Gemini writes using your actual stored words

Quiz:   3-question multiple-choice quiz generated from your vocabulary
```

---

## Stop & restart

```bash
docker compose down        # stop containers (data is kept in ./data/)
docker compose up          # restart
docker compose down -v     # stop AND delete all data
```

---

## Changing Gemini model

In `rag.py`, line with `GenerativeModel`, you can swap `gemini-1.5-flash`
for `gemini-1.5-pro` for higher quality answers (slower, uses more quota).

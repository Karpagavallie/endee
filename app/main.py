"""
main.py — AI Vocabulary CLI
Powered by: Endee (vector DB) + Gemini (embeddings + generation) + RAG
"""

import sys
from database import wait_for_endee, add_word, delete_word, list_words
from rag import semantic_search, ask_vocabulary, generate_quiz


# ── Display helpers ───────────────────────────────────────────────────────────

def print_word(r: dict, show_score: bool = False):
    meta       = r.get("meta", {})
    word       = meta.get("word", r.get("id", "?"))
    definition = meta.get("definition", "—")
    example    = meta.get("example", "")
    similarity = r.get("similarity", None)

    print(f"\n  📖 {word.upper()}")
    print(f"     Definition : {definition}")
    if example:
        print(f"     Example    : {example}")
    if show_score and similarity is not None:
        bar_len = int(similarity * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"     Match      : [{bar}] {similarity:.0%}")


def divider(char: str = "─", width: int = 54):
    print(char * width)


# ── Menu ──────────────────────────────────────────────────────────────────────

def print_menu():
    print()
    divider("═")
    print("   🧠  AI Vocabulary")
    print("   Endee  ·  Gemini Embeddings  ·  RAG")
    divider("═")
    print("   1  Add a new word")
    print("   2  Semantic search  (find by meaning)")
    print("   3  Ask AI a question  (RAG)")
    print("   4  Filter by category")
    print("   5  Take a quiz")
    print("   6  List all words")
    print("   7  Delete a word")
    print("   0  Exit")
    divider()


# ── Handlers ──────────────────────────────────────────────────────────────────

def handle_add():
    word = input("  Word            : ").strip()
    if not word:
        print("  ⚠️  Word cannot be empty.")
        return
    definition = input("  Definition      : ").strip()
    if not definition:
        print("  ⚠️  Definition cannot be empty.")
        return
    example  = input("  Example sentence (Enter to skip): ").strip()
    category = input("  Category [general]: ").strip() or "general"
    add_word(word, definition, example, category)


def handle_search():
    query = input("  Search query: ").strip()
    if not query:
        return
    results = semantic_search(query, top_k=5)
    if results:
        print(f"\n  🔍 Top results for: \"{query}\"")
        for r in results:
            print_word(r, show_score=True)
    else:
        print("  No results. Try adding some words first.")


def handle_ask():
    question = input("  Your question: ").strip()
    if not question:
        return
    print("\n  🤖 Thinking...\n")
    answer = ask_vocabulary(question)
    divider()
    print(answer)
    divider()


def handle_category():
    category = input("  Category to filter (e.g. science, literature): ").strip()
    query    = input("  Search query within this category: ").strip()
    if not query or not category:
        return
    from rag import semantic_search as ss
    results = ss(query, top_k=10, category=category)
    if results:
        print(f"\n  📂 Results in '{category}':")
        for r in results:
            print_word(r, show_score=True)
    else:
        print(f"  No words found in category '{category}'.")


def handle_quiz():
    try:
        n = int(input("  Number of questions [3]: ").strip() or "3")
    except ValueError:
        n = 3
    print("\n  🎯 Generating quiz...\n")
    quiz = generate_quiz(num_questions=n)
    divider()
    print(quiz)
    divider()


def handle_list():
    print("\n  📚 Fetching vocabulary...")
    words = list_words(top_k=100)
    if words:
        print(f"  Found {len(words)} word(s):\n")
        for w in words:
            print_word(w)
    else:
        print("  Your vocabulary is empty. Add some words!")


def handle_delete():
    word = input("  Word to delete: ").strip()
    if word:
        confirm = input(f"  Delete '{word}'? (y/N): ").strip().lower()
        if confirm == "y":
            delete_word(word)


# ── Main loop ─────────────────────────────────────────────────────────────────

HANDLERS = {
    "1": handle_add,
    "2": handle_search,
    "3": handle_ask,
    "4": handle_category,
    "5": handle_quiz,
    "6": handle_list,
    "7": handle_delete,
}


def main():
    print("\n  Connecting to Endee vector database...")
    wait_for_endee()

    while True:
        print_menu()
        choice = input("  Choose (0-7): ").strip()

        if choice == "0":
            print("\n  Goodbye! 👋\n")
            sys.exit(0)

        handler = HANDLERS.get(choice)
        if handler:
            try:
                handler()
            except KeyboardInterrupt:
                print("\n  (cancelled)")
            except Exception as e:
                print(f"\n  ❌ Error: {e}")
        else:
            print("  Invalid option — enter 0 to 7.")


if __name__ == "__main__":
    main()

import logging
import os

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from app.embedding import (
    process_directory, build_embeddings, save_embeddings_to_npy, load_embeddings_from_npy,
    build_faiss_index, save_faiss_index, load_faiss_index, semantic_search_faiss,
    save_chunks_meta, load_chunks_meta, compute_directory_fingerprint
)
from app.llm import stream_llm
from app.history import ConversationHistory
from app.config import DIRECTORY_PATH, EMBEDDINGS_FILE, INDEX_FILE, CHUNKS_META_FILE, MAX_TOKENS


if __name__ == "__main__":
    embeddings_file = EMBEDDINGS_FILE
    index_file = INDEX_FILE
    chunks_meta_file = CHUNKS_META_FILE

    current_fingerprint = compute_directory_fingerprint(DIRECTORY_PATH)
    cache_exists = os.path.exists(embeddings_file) and os.path.exists(index_file) and os.path.exists(chunks_meta_file)

    chunks, meta, cached_fingerprint = load_chunks_meta(chunks_meta_file) if cache_exists else (None, None, None)

    # On ne réutilise le cache que si data/files/ n'a pas changé depuis sa création.
    if cache_exists and cached_fingerprint == current_fingerprint:
        print("Chargement du cache (embeddings, index FAISS, chunks)...")
        embeddings = load_embeddings_from_npy(embeddings_file)
        index = load_faiss_index(index_file)
    else:
        print("Traitement des fichiers pour extraction et chunking...")
        chunks, meta = process_directory(DIRECTORY_PATH, max_tokens=MAX_TOKENS)
        save_chunks_meta(chunks, meta, current_fingerprint, chunks_meta_file)

        print(f"{len(chunks)} chunks extraits. Vectorisation en cours...")
        embeddings = build_embeddings(chunks)
        save_embeddings_to_npy(embeddings, embeddings_file)

        print("Construction de l'index FAISS...")
        index = build_faiss_index(embeddings)
        save_faiss_index(index, index_file)

    # Initialisation de l'historique
    history = ConversationHistory()

    # Boucle principale pour la recherche sémantique
    while True:
        print("\n--- Recherche sémantique ---\n")
        query = input("Pose ta question ('q' pour quitter) : ").strip()
        if query.lower() == "q":
            print("Au revoir !")
            break

        # Recherche sémantique dans l'index FAISS
        results = semantic_search_faiss(query, chunks, meta, index, top_k=5)
        full_context = "\n\n".join(res["chunk"] for res in results)

        sources = sorted({res["id_doc"] for res in results})
        if sources:
            print(f"\n Sources : {', '.join(sources)}")

        print("\n Réponse du LLM :\n")
        answer_parts = []
        for token in stream_llm(query, full_context, history):
            print(token, end="", flush=True)
            answer_parts.append(token)
        answer = "".join(answer_parts).strip()

        # Sauvegarde de l'interaction dans l'historique
        history.add_interaction(query, answer, full_context)

        print("\n\n--- Fin de réponse ---")
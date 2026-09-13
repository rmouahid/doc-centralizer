import logging
import os
import streamlit as st

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from app.embedding import (
    process_directory, build_embeddings, save_embeddings_to_npy,
    load_embeddings_from_npy, build_faiss_index, save_faiss_index,
    load_faiss_index, semantic_search_faiss, save_chunks_meta, load_chunks_meta,
    compute_directory_fingerprint
)
from app.llm import stream_llm
from app.history import ConversationHistory
from app.config import (
    DIRECTORY_PATH, EMBEDDINGS_FILE, INDEX_FILE, CHUNKS_META_FILE,
    MAX_TOKENS, MAX_TOKENS_RESPONSE
)

# Configuration de la page
st.set_page_config(
    page_title="Assistant Documentation",
    page_icon="📚",
    layout="wide"
)

# Initialisation des variables de session
if 'history' not in st.session_state:
    st.session_state.history = ConversationHistory()


if 'initialized' not in st.session_state:
    st.session_state.initialized = False

def initialize_system():
    """Initialize the document processing system"""
    current_fingerprint = compute_directory_fingerprint(DIRECTORY_PATH)
    cache_exists = os.path.exists(EMBEDDINGS_FILE) and os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_META_FILE)

    if cache_exists:
        chunks, meta, cached_fingerprint = load_chunks_meta(CHUNKS_META_FILE)
        # On ne réutilise le cache que si data/files/ n'a pas changé depuis sa création.
        if cached_fingerprint == current_fingerprint:
            embeddings = load_embeddings_from_npy(EMBEDDINGS_FILE)
            index = load_faiss_index(INDEX_FILE)
            return chunks, meta, index

    # Sinon, extraction complète puis construction et sauvegarde du cache
    chunks, meta = process_directory(DIRECTORY_PATH, max_tokens=MAX_TOKENS)
    save_chunks_meta(chunks, meta, current_fingerprint, CHUNKS_META_FILE)

    embeddings = build_embeddings(chunks)
    save_embeddings_to_npy(embeddings, EMBEDDINGS_FILE)

    index = build_faiss_index(embeddings)
    save_faiss_index(index, INDEX_FILE)

    return chunks, meta, index

# Interface principale
st.title("📚 Assistant Documentation")

# Initialisation du système
if not st.session_state.initialized:
    chunks, meta, index = initialize_system()
    st.session_state.chunks = chunks
    st.session_state.meta = meta
    st.session_state.index = index
    st.session_state.initialized = True

# Zone de chat
st.markdown("### 💬 Conversation")

# Affichage de l'historique
if st.session_state.history.active_conversation_id:
    active_conv = st.session_state.history.conversations[st.session_state.history.active_conversation_id]
    for message in active_conv.messages:
        with st.chat_message("user"):
            st.markdown(message.query)
        with st.chat_message("assistant"):
            st.markdown(message.response)

# Zone de saisie
query = st.chat_input("Posez votre question...")

if query:
    # Affichage de la question
    with st.chat_message("user"):
        st.markdown(query)
    
    # Recherche et réponse
    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours..."):
            results = semantic_search_faiss(
                query,
                st.session_state.chunks,
                st.session_state.meta,
                st.session_state.index,
                top_k=5
            )
            full_context = "\n\n".join(res["chunk"] for res in results)

        answer = st.write_stream(
            stream_llm(query, full_context, st.session_state.history, max_tokens=MAX_TOKENS_RESPONSE)
        )

        sources = sorted({res["id_doc"] for res in results})
        if sources:
            st.caption(f"📎 Sources : {', '.join(sources)}")

        # Sauvegarde dans l'historique
        st.session_state.history.add_interaction(query, answer, full_context)

# Sidebar avec informations et options
with st.sidebar:
    st.markdown("### 💬 Conversations")
    
    # Bouton pour créer une nouvelle conversation
    if st.button("➕ Nouvelle conversation"):
        st.session_state.history.create_conversation()
        st.rerun()
    
    # Liste des conversations
    conversations = st.session_state.history.get_conversations_list()
    for conv in conversations:
        col1, col2 = st.columns([4, 1])
        with col1:
            # Utiliser un bouton qui ressemble à un lien pour le titre
            if st.button(
                f"📝 {conv['title']} ({conv['message_count']} messages)",
                key=f"conv_{conv['id']}"
            ):
                st.session_state.history.active_conversation_id = conv['id']
                st.rerun()
        with col2:
            # Bouton de suppression
            if st.button("🗑️", key=f"del_{conv['id']}"):
                st.session_state.history.delete_conversation(conv['id'])
                st.rerun()
    
    st.divider()
    
    st.markdown("### ℹ️ Informations")
    st.markdown("""
    Cet assistant vous aide à explorer la documentation.
    - Posez vos questions en langage naturel
    - L'assistant recherche dans les documents
    - Les réponses sont générées à partir du contexte trouvé
    """)
    
    if st.button("🔄 Réinitialiser la conversation"):
        st.session_state.history.reset()
        st.rerun()

# Affichage du titre de la conversation active
if st.session_state.history.active_conversation_id:
    active_conv = st.session_state.history.conversations[st.session_state.history.active_conversation_id]
    st.markdown(f"### 💬 {active_conv.title}")
else:
    st.markdown("### 💬 Nouvelle conversation")
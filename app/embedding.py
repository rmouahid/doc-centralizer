import numpy as np
import faiss
import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from app.chunker import chunker, extract_excalidraw, extract_markdown, extract_text, extract_pdf, extract_image
from app.config import EMBEDDINGS_FILE, INDEX_FILE, CHUNKS_META_FILE, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

# === CONFIGURATION ===

EMBEDDINGS_PATH = EMBEDDINGS_FILE
INDEX_PATH = INDEX_FILE

# === Chargement du modèle d'embeddings ===

model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# === Extraction et chunking des fichiers ===

def process_file(file_path: str, file_type: str, max_tokens: int = 300) -> List[str]:
    """
    Extrait le contenu d'un fichier, le découpe en chunks, et retourne les chunks.
    """
    if file_type == "excalidraw":
        text = extract_excalidraw(file_path)
    elif file_type == "markdown":
        text = extract_markdown(file_path)
    elif file_type == "text":
        text = extract_text(file_path)
    elif file_type == "pdf":
        text = extract_pdf(file_path)
    elif file_type == "image":
        text = extract_image(file_path)
    else:
        raise ValueError(f"Type de fichier non supporté : {file_type}")
    
    return chunker(text, max_tokens=max_tokens)

def compute_directory_fingerprint(directory_path: str) -> str:
    """
    Calcule une empreinte du contenu d'un dossier (nom, taille, date de
    modification de chaque fichier), utilisée pour savoir si le cache
    (embeddings/index/chunks) est encore à jour ou doit être régénéré.
    """
    entries = []
    for file_name in sorted(os.listdir(directory_path)):
        file_path = os.path.join(directory_path, file_name)
        if os.path.isfile(file_path):
            stat = os.stat(file_path)
            entries.append(f"{file_name}:{stat.st_size}:{stat.st_mtime_ns}")
    return hashlib.sha256("|".join(entries).encode("utf-8")).hexdigest()

def process_directory(directory_path: str, max_tokens: int = 300) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Parcourt un répertoire, extrait et chunk les fichiers supportés.
    Retourne une liste de chunks et leurs métadonnées.
    """
    chunks = []
    meta = []
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if file_name.endswith(".excalidraw"):
            file_type = "excalidraw"
        elif file_name.endswith(".md"):
            file_type = "markdown"
        elif file_name.endswith(".txt"):
            file_type = "text"
        elif file_name.endswith(".pdf"):
            file_type = "pdf"
        elif file_name.endswith((".png", ".jpg", ".jpeg")):
            file_type = "image"
        else:
            continue  # Ignorer les fichiers non supportés
        
        try:
            file_chunks = process_file(file_path, file_type, max_tokens)
        except Exception as e:
            logger.warning("Erreur lors du traitement de '%s', fichier ignoré : %s", file_name, e)
            continue
        chunks.extend(file_chunks)
        meta.extend([{"id_doc": file_name, "chunk_id": i} for i in range(len(file_chunks))])

    return chunks, meta

# === Sauvegarde / chargement des chunks et métadonnées ===

def save_chunks_meta(chunks: List[str], meta: List[Dict[str, Any]], fingerprint: str, path: str = CHUNKS_META_FILE) -> None:
    """
    Sauvegarde les chunks, leurs métadonnées et l'empreinte du dossier source,
    pour éviter de refaire l'extraction/OCR quand rien n'a changé.
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({"chunks": chunks, "meta": meta, "fingerprint": fingerprint}, f, ensure_ascii=False)

def load_chunks_meta(path: str = CHUNKS_META_FILE) -> Tuple[List[str], List[Dict[str, Any]], Optional[str]]:
    """
    Charge les chunks, leurs métadonnées et l'empreinte précédemment sauvegardés.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["chunks"], data["meta"], data.get("fingerprint")

# === Construction des embeddings ===

def build_embeddings(chunks: List[str]) -> np.ndarray:
    """
    Génère les embeddings pour une liste de chunks.
    """
    embeddings = model.encode(chunks, convert_to_tensor=False)  # Convert to numpy array
    return embeddings

# === Sauvegarde / chargement des embeddings ===

def save_embeddings_to_npy(embeddings: np.ndarray, path: str = EMBEDDINGS_PATH) -> None:
    """
    Sauvegarde les embeddings dans un fichier .npy.
    """
    np.save(path, embeddings)

def load_embeddings_from_npy(path: str = EMBEDDINGS_PATH) -> np.ndarray:
    """
    Charge les embeddings depuis un fichier .npy.
    """
    return np.load(path)

# === Construction de l'index FAISS ===

def build_faiss_index(embeddings) -> faiss.Index:
    """
    Construit un index FAISS à partir des embeddings.
    """
    if len(embeddings) == 0:
        raise ValueError("Aucun embedding à indexer")
    
    if isinstance(embeddings, np.ndarray):
        dimension = embeddings.shape[1]
    else:
        embeddings = np.array(embeddings)
        if embeddings.ndim != 2:
            raise ValueError(f"Les embeddings doivent être un tableau 2D, reçu : {embeddings.ndim}D")
        dimension = embeddings.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# === Sauvegarde / chargement de l'index ===

def save_faiss_index(index: faiss.Index, path: str = INDEX_PATH) -> None:
    """
    Sauvegarde l'index FAISS dans un fichier.
    """
    faiss.write_index(index, path)

def load_faiss_index(path: str = INDEX_PATH) -> faiss.Index:
    """
    Charge un index FAISS depuis un fichier.
    """
    return faiss.read_index(path)

# === Recherche sémantique avec FAISS ===

def semantic_search_faiss(query: str, chunks: List[str], meta: List[Dict[str, Any]], index: faiss.Index, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Effectue une recherche sémantique dans l'index FAISS.
    """
    query_embedding = model.encode([query])  # Encode la requête
    D, I = index.search(np.array(query_embedding), top_k)  # Recherche dans l'index FAISS

    results = []
    for i, score in zip(I[0], D[0]):
        if i < len(meta):  # Vérifiez que l'indice est valide
            results.append({
                "score": float(score),
                "id_doc": meta[i]["id_doc"],
                "chunk_id": meta[i]["chunk_id"],
                "chunk": chunks[i]
            })
        else:
            logger.warning("Indice hors limites : %s, ignoré.", i)
    return results
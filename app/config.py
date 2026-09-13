"""Configuration centralisée du projet."""

# Chemins des données
DIRECTORY_PATH = "data/files"
EMBEDDINGS_FILE = "data/embeddings.npy"
INDEX_FILE = "data/faiss.index"
CHUNKS_META_FILE = "data/chunks_meta.json"
HISTORY_FILE = "data/history.json"

# Chunking / LLM
MAX_TOKENS = 300  # Taille max d'un chunk de document
MAX_TOKENS_RESPONSE = 1000  # Taille max d'une réponse du LLM
LLM_CONTEXT_SIZE = 2048

# Modèles
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOKENIZER_MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
LLM_MODEL_PATH = "models/phi-3-mini-128k-instruct.Q4_K_M.gguf"

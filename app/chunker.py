from transformers import AutoTokenizer
import json
import pymupdf as fitz
from PIL import Image
import pytesseract
from typing import List
from app.config import TOKENIZER_MODEL_NAME

# Initialisation du tokenizer (Phi-3 mini)
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_NAME)
except OSError as e:
    raise OSError(
        f"Impossible de charger le tokenizer '{TOKENIZER_MODEL_NAME}'. "
        "Vérifie ta connexion internet (téléchargement depuis Hugging Face Hub "
        "au premier lancement) ou le cache local Hugging Face."
    ) from e

def splitter(text: str) -> List[str]:
    """
    Divise le texte en paragraphes basés sur les sauts de ligne.
    Renvoie une liste de paragraphes non vides.
    """
    return [p.strip() for p in text.split('\n') if p.strip()]

def token_calculator(text: str) -> int:
    """
    Calcule le nombre de tokens dans un texte donné,
    sans compter les tokens spéciaux ajoutés par le tokenizer.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

def max_verifier(text: str, max_tokens: int) -> bool:
    """
    Vérifie que le texte ne dépasse pas la limite max_tokens.
    """
    return token_calculator(text) <= max_tokens

def chunker(text: str, max_tokens: int = 300) -> List[str]:
    """
    Découpe un texte en chunks (blocs) ne dépassant pas max_tokens tokens.
    Utilise un chevauchement d'un paragraphe entre les chunks.
    """
    paragraphs = splitter(text)
    chunks = []
    current_chunk = ""
    overlap = ""

    for paragraph in paragraphs:
        proposed_chunk = current_chunk + paragraph + "\n"
        if max_verifier(proposed_chunk, max_tokens):
            current_chunk = proposed_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # On recommence avec l'overlap (le paragraphe précédent), sauf si
            # cela dépasse déjà la limite à lui seul.
            overlapped_chunk = overlap + paragraph + "\n"
            if overlap and max_verifier(overlapped_chunk, max_tokens):
                current_chunk = overlapped_chunk
            else:
                current_chunk = paragraph + "\n"
        overlap = paragraph + "\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def extract_excalidraw(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [element['text'] for element in data['elements'] if 'text' in element]
    return "\n".join(texts)

def extract_markdown(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content  # Vous pouvez utiliser une bibliothèque comme `markdown-it` pour un traitement avancé.

def extract_text(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_image(file_path: str) -> str:
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)
    return text
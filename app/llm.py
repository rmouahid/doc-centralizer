from typing import Iterator, Optional
from llama_cpp import Llama
from .history import ConversationHistory
from .config import LLM_MODEL_PATH, LLM_CONTEXT_SIZE, MAX_TOKENS_RESPONSE

# Initialisation du modèle
llm = Llama(
    model_path=LLM_MODEL_PATH,
    n_ctx=LLM_CONTEXT_SIZE,
    n_gpu_layers=0
)

def build_prompt(query: str, context: str, history: Optional[ConversationHistory] = None) -> str:
    """Construit le prompt pour le LLM"""
    messages = []
    
    if history:
        messages = history.get_active_messages()
    
    # Format conversation history
    conversation_history = ""
    if messages:
        for msg in messages[-3:]:  # Get last 3 messages for context
            conversation_history += f"Human: {msg.query}\nAssistant: {msg.response}\n"
    
    # Build the prompt
    prompt = f"""Tu es un assistant AI qui aide à la compréhension de la documentation.
Utilise le contexte fourni pour répondre aux questions.

Contexte:
{context}

Historique de la conversation:
{conversation_history}

Question:
{query}

Réponse:"""

    return prompt

def ask_llm(query: str, context: str, history: Optional[ConversationHistory] = None, max_tokens: int = MAX_TOKENS_RESPONSE) -> str:
    """
    Interroge le LLM avec le contexte de la recherche et renvoie la réponse complète.
    """
    prompt = build_prompt(query, context, history)

    response = llm(
        prompt,
        max_tokens=max_tokens,
        stop=["### Question:", "### Contexte:", "### Réponse:"],
        echo=False
    )

    return response["choices"][0]["text"].strip()

def stream_llm(query: str, context: str, history: Optional[ConversationHistory] = None, max_tokens: int = MAX_TOKENS_RESPONSE) -> Iterator[str]:
    """
    Interroge le LLM et produit la réponse morceau par morceau (streaming).
    """
    prompt = build_prompt(query, context, history)

    for chunk in llm(
        prompt,
        max_tokens=max_tokens,
        stop=["### Question:", "### Contexte:", "### Réponse:"],
        echo=False,
        stream=True
    ):
        yield chunk["choices"][0]["text"]

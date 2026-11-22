from contextlib import contextmanager
from typing import Iterator, Optional, Sequence

from llama_cpp import Llama

from .history import ConversationHistory
from .config import LLM_MODEL_PATH, LLM_CONTEXT_SIZE, MAX_TOKENS_RESPONSE

# Initialisation du modèle
llm = Llama(
    model_path=LLM_MODEL_PATH,
    n_ctx=LLM_CONTEXT_SIZE,
    n_gpu_layers=0
)

_CONTEXT_ERROR = "La fenêtre de contexte du modèle est dépassée. Raccourcissez votre question."


class PromptContextError(ValueError):
    """Le prompt et la réponse réservée dépassent la fenêtre du modèle."""


def _tokenize_prompt(prompt: str) -> list[int]:
    """Utilise la même tokenisation pour mesurer le prompt et générer la réponse."""
    return llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)


def build_prompt(
    query: str,
    context: str | Sequence[str],
    history: Optional[ConversationHistory] = None,
    max_tokens: int = MAX_TOKENS_RESPONSE,
) -> str:
    """Construit le prompt en réservant les tokens demandés pour la réponse."""
    context_size = llm.n_ctx()
    if not 0 < max_tokens < context_size:
        raise PromptContextError(
            f"La réserve de réponse doit être comprise entre 1 et {context_size - 1} tokens."
        )
    budget = context_size - max_tokens
    chunks = context.split("\n\n") if isinstance(context, str) else list(context)
    messages = history.get_active_messages()[-3:] if history else []

    while True:
        context_text = "\n\n".join(chunks)
        history_text = "".join(
            f"Human: {msg.query}\nAssistant: {msg.response}\n" for msg in messages
        )
        prompt = f"""Tu es un assistant AI qui aide à la compréhension de la documentation.
Utilise le contexte fourni pour répondre aux questions.

Contexte:
{context_text}

Historique de la conversation:
{history_text}

Question:
{query}

Réponse:"""

        if len(_tokenize_prompt(prompt)) <= budget:
            return prompt
        if messages:
            messages = messages[1:]
        elif chunks:
            chunks.pop()  # Les résultats sont classés du plus au moins pertinent.
        else:
            raise PromptContextError(_CONTEXT_ERROR)


@contextmanager
def _handle_context_overflow() -> Iterator[None]:
    try:
        yield
    except ValueError as exc:
        if "exceed context window" in str(exc):
            raise PromptContextError(_CONTEXT_ERROR) from exc
        raise


def ask_llm(
    query: str,
    context: str | Sequence[str],
    history: Optional[ConversationHistory] = None,
    max_tokens: int = MAX_TOKENS_RESPONSE,
) -> str:
    """Interroge le modèle et renvoie la réponse complète."""
    prompt = build_prompt(query, context, history, max_tokens)

    with _handle_context_overflow():
        response = llm(
            _tokenize_prompt(prompt),
            max_tokens=max_tokens,
            stop=["### Question:", "### Contexte:", "### Réponse:"],
            echo=False
        )

    return response["choices"][0]["text"].strip()


def stream_llm(
    query: str,
    context: str | Sequence[str],
    history: Optional[ConversationHistory] = None,
    max_tokens: int = MAX_TOKENS_RESPONSE,
) -> Iterator[str]:
    """Interroge le modèle et produit la réponse morceau par morceau."""
    prompt = build_prompt(query, context, history, max_tokens)

    with _handle_context_overflow():
        for chunk in llm(
            _tokenize_prompt(prompt),
            max_tokens=max_tokens,
            stop=["### Question:", "### Contexte:", "### Réponse:"],
            echo=False,
            stream=True
        ):
            yield chunk["choices"][0]["text"]

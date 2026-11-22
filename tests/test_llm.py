import importlib.util
import sys
from types import SimpleNamespace

import pytest

from app.config import LLM_CONTEXT_SIZE, MAX_TOKENS_RESPONSE
from app.history import ConversationHistory


class FakeLlama:
    def __init__(self, **options):
        self.context_size = options["n_ctx"]

    def n_ctx(self):
        return self.context_size

    def tokenize(self, text, *, add_bos, special):
        assert special
        return [0] * (len(text.split()) + add_bos)


@pytest.fixture
def llm_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    spec = importlib.util.find_spec("app.llm")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_prompt_fits_with_three_long_responses(llm_module, tmp_path):
    history = ConversationHistory(history_file=str(tmp_path / "history.json"))
    for index in range(3):
        history.add_interaction(f"Ancienne question {index}", "mot " * 1000, "")
    chunks = [f"Extrait {index} " + "mot " * 298 for index in range(5)]
    query = "Quelle est la procédure ?"

    prompt = llm_module.build_prompt(query, chunks, history)

    assert len(llm_module.llm.tokenize(prompt.encode(), add_bos=True, special=True)) <= (
        LLM_CONTEXT_SIZE - MAX_TOKENS_RESPONSE
    )
    assert query in prompt
    assert "Ancienne question" not in prompt
    assert all(chunk in prompt for chunk in chunks[:3])
    assert all(chunk not in prompt for chunk in chunks[3:])
    assert len(history.get_active_messages()) == 3
    assert len(chunks) == 5


def test_build_prompt_drops_oldest_exchange_before_context(llm_module, tmp_path):
    history = ConversationHistory(history_file=str(tmp_path / "history.json"))
    history.add_interaction("Ancienne question", "mot " * 1000, "")
    history.add_interaction("Question récente", "Réponse récente", "")
    context = "Extrait pertinent " + "mot " * 298

    prompt = llm_module.build_prompt("Question actuelle", context, history)

    assert "Ancienne question" not in prompt
    assert "Question récente" in prompt
    assert context in prompt


def test_build_prompt_rejects_question_that_cannot_fit(llm_module):
    with pytest.raises(llm_module.PromptContextError, match="Raccourcissez votre question"):
        llm_module.build_prompt("mot " * LLM_CONTEXT_SIZE, "")


@pytest.mark.parametrize("stream", [False, True])
def test_generation_stop_sequences_match_prompt(llm_module, monkeypatch, tmp_path, stream):
    history = ConversationHistory(history_file=str(tmp_path / "history.json"))
    history.add_interaction("Première question", "Première réponse", "")
    history.add_interaction("Deuxième question", "Deuxième réponse", "")
    query = "Question actuelle"
    context = "Extrait pertinent"
    prompt = llm_module.build_prompt(query, context, history)

    class RecordingLlama(FakeLlama):
        def __call__(self, prompt_tokens, *, stop, **options):
            self.stop = stop
            response = {"choices": [{"text": "Réponse actuelle"}]}
            return iter([response]) if options.get("stream") else response

    model = RecordingLlama(n_ctx=LLM_CONTEXT_SIZE)
    monkeypatch.setattr(llm_module, "llm", model)

    if stream:
        assert "".join(llm_module.stream_llm(query, context, history)) == "Réponse actuelle"
    else:
        assert llm_module.ask_llm(query, context, history) == "Réponse actuelle"

    assert {"\nHuman:", "\nQuestion:"} <= set(model.stop)
    assert all(sequence in prompt for sequence in model.stop)


@pytest.mark.parametrize("stream", [False, True])
def test_generation_reserves_response_tokens_and_handles_residual_overflow(
    llm_module, monkeypatch, stream
):
    class OverflowingLlama(FakeLlama):
        def __call__(self, prompt, *, max_tokens, **options):
            assert max_tokens == 1500
            assert len(prompt) + max_tokens <= self.n_ctx()
            if options.get("stream"):
                return self.generate()
            raise ValueError("Requested tokens (2048) exceed context window of 2048")

        def generate(self):
            yield {"choices": [{"text": "Début"}]}
            raise ValueError("Requested tokens (2048) exceed context window of 2048")

    monkeypatch.setattr(llm_module, "llm", OverflowingLlama(n_ctx=LLM_CONTEXT_SIZE))
    chunks = ["un " * 300, "deux " * 300]

    if stream:
        response = llm_module.stream_llm("Question", chunks, max_tokens=1500)
        assert next(response) == "Début"
        with pytest.raises(llm_module.PromptContextError, match="Raccourcissez votre question"):
            next(response)
    else:
        with pytest.raises(llm_module.PromptContextError, match="Raccourcissez votre question"):
            llm_module.ask_llm("Question", chunks, max_tokens=1500)

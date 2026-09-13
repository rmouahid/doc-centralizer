from app.chunker import splitter, chunker, token_calculator


def test_splitter_strips_and_drops_empty_lines():
    text = "  premier paragraphe  \n\n\n   \ndeuxième paragraphe\n"
    assert splitter(text) == ["premier paragraphe", "deuxième paragraphe"]


def test_chunker_respects_max_tokens():
    text = "\n".join(f"Ligne de texte numéro {i}." for i in range(30))
    max_tokens = 20

    chunks = chunker(text, max_tokens=max_tokens)

    assert len(chunks) > 1
    for chunk in chunks:
        assert token_calculator(chunk) <= max_tokens


def test_chunker_keeps_all_paragraphs():
    paragraphs = [f"Paragraphe {i}." for i in range(5)]
    text = "\n".join(paragraphs)

    chunks = chunker(text, max_tokens=300)
    joined = "\n".join(chunks)

    for paragraph in paragraphs:
        assert paragraph in joined

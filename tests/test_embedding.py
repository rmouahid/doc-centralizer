import numpy as np
import pytest

from app.embedding import (
    build_faiss_index, save_faiss_index, load_faiss_index,
    save_chunks_meta, load_chunks_meta, semantic_search_faiss,
    compute_directory_fingerprint
)


def test_build_faiss_index_rejects_empty_embeddings():
    with pytest.raises(ValueError):
        build_faiss_index([])


def test_build_faiss_index_and_search_roundtrip():
    embeddings = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ], dtype="float32")
    chunks = ["chunk A", "chunk B", "chunk C"]
    meta = [{"id_doc": "doc.txt", "chunk_id": i} for i in range(len(chunks))]

    index = build_faiss_index(embeddings)

    D, I = index.search(np.array([[1.0, 0.0]], dtype="float32"), 1)
    assert I[0][0] == 0


def test_faiss_index_save_and_load_roundtrip(tmp_path):
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    index = build_faiss_index(embeddings)
    index_path = tmp_path / "faiss.index"

    save_faiss_index(index, str(index_path))
    reloaded = load_faiss_index(str(index_path))

    assert reloaded.ntotal == index.ntotal


def test_chunks_meta_save_and_load_roundtrip(tmp_path):
    chunks = ["chunk A", "chunk B"]
    meta = [{"id_doc": "doc.txt", "chunk_id": 0}, {"id_doc": "doc.txt", "chunk_id": 1}]
    path = tmp_path / "chunks_meta.json"

    save_chunks_meta(chunks, meta, "fake-fingerprint", str(path))
    loaded_chunks, loaded_meta, loaded_fingerprint = load_chunks_meta(str(path))

    assert loaded_chunks == chunks
    assert loaded_meta == meta
    assert loaded_fingerprint == "fake-fingerprint"


def test_directory_fingerprint_changes_when_a_file_is_modified(tmp_path):
    file_path = tmp_path / "doc.txt"
    file_path.write_text("contenu initial", encoding="utf-8")

    fingerprint_before = compute_directory_fingerprint(str(tmp_path))

    file_path.write_text("contenu modifié", encoding="utf-8")
    fingerprint_after = compute_directory_fingerprint(str(tmp_path))

    assert fingerprint_before != fingerprint_after


def test_directory_fingerprint_is_stable_when_nothing_changes(tmp_path):
    (tmp_path / "doc.txt").write_text("contenu", encoding="utf-8")

    assert compute_directory_fingerprint(str(tmp_path)) == compute_directory_fingerprint(str(tmp_path))

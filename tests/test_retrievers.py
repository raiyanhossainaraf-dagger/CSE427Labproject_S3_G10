import numpy as np
import pandas as pd
import pytest

from src.bm25_retrieval import PaperScopedBM25Retriever, tokenize
from src.hybrid_retrieval import weighted_rrf
from src.retrieval import DenseRetriever, PREDICTION_COLUMNS, format_predictions
from src.retrieval_metrics import validate_predictions
from src.vector_store import create_ip_index


def corpus():
    rows = [
        ("d1", "p1", "validation", "chunk", "c1", "c1", "r1", "s1", "", "apple apple fruit"),
        ("d2", "p1", "validation", "chunk", "c2", "c2", "r1", "s1", "", "apple fruit"),
        ("d3", "p1", "validation", "section", "s2", "", "", "s2", "", "banana methods"),
        ("d4", "p2", "validation", "float", "f1", "", "", "", "f1", "apple figure"),
        ("d5", "p1", "test", "section", "sx", "", "", "sx", "", "apple test"),
    ]
    columns = ["document_id", "paper_id", "split", "source_type", "source_id", "chunk_id",
               "paragraph_id", "section_id", "figure_table_id", "text"]
    return pd.DataFrame(rows, columns=columns)


def normalized(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_known_bm25_ordering_paper_split_isolation_and_empty_query():
    retriever = PaperScopedBM25Retriever(corpus())
    result = retriever.search("apple", "p1", "validation", 10, "q1")
    assert result.document_id.tolist()[:2] == ["d1", "d2"]
    assert set(result.paper_id) == {"p1"} and set(result.split) == {"validation"}
    assert len(result) == 3
    empty1 = retriever.search("", "p1", "validation", 10, "q1")
    empty2 = retriever.search("", "p1", "validation", 10, "q1")
    pd.testing.assert_frame_equal(empty1, empty2)
    assert tokenize("Hello, WORLD!") == ["hello", "world"]


def test_dense_fake_ordering_isolation_and_global_faiss():
    frame = corpus()
    embeddings = normalized([[1, 0], [.8, .2], [0, 1], [1, 0], [1, 0]])
    mapping = frame[["document_id", "paper_id", "split", "source_type", "source_id", "chunk_id",
                     "paragraph_id", "section_id", "figure_table_id"]].copy()
    retriever = DenseRetriever(frame, embeddings, index=create_ip_index(embeddings), document_map=mapping)
    result = retriever.search_embedding(np.array([1, 0], dtype=np.float32), "p1", "validation", 10, "q1")
    assert result.document_id.tolist() == ["d1", "d2", "d3"]
    assert set(result.paper_id) == {"p1"} and set(result.split) == {"validation"}
    global_result = retriever.global_search_embedding(np.array([1, 0], dtype=np.float32), 5)
    assert "d4" in set(global_result.document_id)


def test_source_deduplication_and_hand_calculated_rrf():
    frame = corpus().iloc[:3].copy()
    bm = frame.iloc[[0, 2, 1]].copy(); bm["score"] = [3., 2., 1.]
    dense = frame.iloc[[1, 2, 0]].copy(); dense["score"] = [.9, .8, .7]
    bm = format_predictions(bm, "q", "p1", "validation", "bm25", bm25=True)
    dense = format_predictions(dense, "q", "p1", "validation", "dense", dense=True)
    result = weighted_rrf(bm, dense, top_k=10, rrf_constant=60)
    assert len(result) == 2  # c1/c2 share canonical paragraph r1
    paragraph = result[result.paragraph_id.eq("r1")].iloc[0]
    assert paragraph.fused_score == pytest.approx(1 / 61 + 1 / 61)
    assert paragraph.bm25_rank == 1 and paragraph.dense_rank == 1
    validate_predictions(result)
    assert list(result.columns) == PREDICTION_COLUMNS


def test_deterministic_ties_missing_artifacts_and_empty_paper(tmp_path):
    frame = corpus().iloc[:3].copy()
    embeddings = normalized([[1, 0], [1, 0], [0, 1]])
    retriever = DenseRetriever(frame, embeddings)
    first = retriever.search_embedding(np.array([1, 0]), "p1", "validation", 2, "q")
    second = retriever.search_embedding(np.array([1, 0]), "p1", "validation", 2, "q")
    pd.testing.assert_frame_equal(first, second)
    assert first.document_id.tolist() == ["d1", "d2"]
    assert retriever.search_embedding(np.array([1, 0]), "missing", "validation").empty
    with pytest.raises(FileNotFoundError, match="Missing dense"):
        DenseRetriever.from_artifacts(tmp_path)

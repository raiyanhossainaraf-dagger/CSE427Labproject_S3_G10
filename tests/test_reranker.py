import numpy as np
import pandas as pd
import pytest

from src.reranker import CrossEncoderReranker, document_text, select_device


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores; self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs)); return np.asarray(self.scores[:len(pairs)])


def candidates():
    return pd.DataFrame([
        {"question_id": "q", "paper_id": "p", "split": "validation", "method": "hybrid", "rank": 1,
         "score": .03, "fused_score": .03, "document_id": "d1", "source_type": "chunk", "source_id": "c1",
         "chunk_id": "c1", "paragraph_id": "p1", "section_id": "s", "figure_table_id": "", "float_id": "",
         "bm25_score": 2., "bm25_rank": 1, "dense_score": .8, "dense_rank": 2,
         "title": "Paper", "section_name": "Methods", "text": "first source"},
        {"question_id": "q", "paper_id": "p", "split": "validation", "method": "hybrid", "rank": 2,
         "score": .02, "fused_score": .02, "document_id": "d2", "source_type": "float", "source_id": "f1",
         "chunk_id": "", "paragraph_id": "", "section_id": "", "figure_table_id": "f1", "float_id": "f1",
         "bm25_score": np.nan, "bm25_rank": pd.NA, "dense_score": .7, "dense_rank": 1,
         "title": "Paper", "section_name": "", "text": "figure caption"},
    ])


def test_known_order_batching_and_metadata_preservation():
    fake = FakeCrossEncoder([-.5, 2.0])
    result = CrossEncoderReranker(model=fake, device="cpu", batch_size=17).rerank("raw question", candidates())
    assert result.document_id.tolist() == ["d2", "d1"]
    assert result.reranked_rank.tolist() == [1, 2]
    assert result.set_index("document_id").loc["d1", "hybrid_rank"] == 1
    assert fake.calls[0][1] == {"batch_size": 17, "show_progress_bar": False}


def test_device_selection_is_injectable_without_gpu():
    assert select_device(lambda: True) == "cuda"
    assert select_device(lambda: False) == "cpu"


def test_model_input_has_only_allowed_source_fields_and_no_gold_leakage():
    row = candidates().iloc[0].to_dict()
    row.update({"answer": "SECRET ANSWER", "evidence_label": "GOLD", "annotation_id": "annotation"})
    text = document_text(row)
    assert text == "Title: Paper\nSection: Methods\nPassage: first source"
    assert not any(value in text for value in ("SECRET", "GOLD", "annotation"))


def test_empty_and_bad_model_output():
    reranker = CrossEncoderReranker(model=FakeCrossEncoder([]), device="cpu")
    assert reranker.rerank("q", candidates().iloc[:0]).empty
    with pytest.raises(ValueError, match="returned 0 scores"):
        reranker.rerank("q", candidates())

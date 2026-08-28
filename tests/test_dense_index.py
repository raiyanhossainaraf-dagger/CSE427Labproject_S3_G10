import numpy as np
import pandas as pd
import pytest

from src.embeddings import BGE_QUERY_INSTRUCTION, encode_passages, encode_queries
from src.vector_store import create_ip_index, search_ip_index, validate_embeddings


class FakeEmbedder:
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        values = []
        for text in texts:
            code = sum(ord(char) for char in text)
            values.append([code % 7 + 1, len(text) + 1, text.count(" ") + 1])
        return np.asarray(values, dtype=np.float64)


def test_passage_and_query_encoding_are_separate_normalized_float32():
    model = FakeEmbedder()
    passage = encode_passages(["hello"], model, show_progress=False)
    query = encode_queries(["hello"], model)
    assert passage.dtype == query.dtype == np.float32
    assert np.linalg.norm(passage[0]) == pytest.approx(1.0)
    assert not np.allclose(passage, query)
    captured = []
    class Capture(FakeEmbedder):
        def encode(self, texts, **kwargs):
            captured.extend(texts); return super().encode(texts, **kwargs)
    encode_queries(["q"], Capture())
    assert captured == [BGE_QUERY_INSTRUCTION + "q"]


def test_index_integrity_save_mapping_and_deterministic_search(tmp_path):
    embeddings = encode_passages(["alpha", "beta", "gamma"], FakeEmbedder(), show_progress=False)
    index = create_ip_index(embeddings)
    mapping = pd.DataFrame({"document_id": ["d0", "d1", "d2"]})
    first = search_ip_index(index, embeddings[:1], mapping, top_k=3)
    second = search_ip_index(index, embeddings[:1], mapping, top_k=3)
    pd.testing.assert_frame_equal(first, second)
    assert first.iloc[0].document_id == "d0"
    assert index.ntotal == len(mapping)


def test_embedding_validation_rejects_bad_shapes_values_and_norms():
    with pytest.raises(ValueError, match="row count"):
        validate_embeddings(np.ones((2, 2), dtype=np.float32) / np.sqrt(2), expected_rows=3)
    bad = np.array([[np.nan, 0]], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        validate_embeddings(bad)
    with pytest.raises(ValueError, match="L2-normalized"):
        validate_embeddings(np.ones((1, 2), dtype=np.float32))

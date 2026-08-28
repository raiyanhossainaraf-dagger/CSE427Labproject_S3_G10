import pandas as pd

from src.retrieval_corpus import build_retrieval_corpus, corpus_fingerprint


def frames():
    papers = pd.DataFrame([{"paper_id": "p1", "split": "train", "title": "Paper"},
                           {"paper_id": "p2", "split": "test", "title": "Other"}])
    chunks = pd.DataFrame([{"chunk_id": "c1", "paper_id": "p1", "split": "train", "section_id": "s1",
                            "section_name": "Intro", "paragraph_ids": ["r1"], "text": "Passage"}])
    sections = pd.DataFrame([{"section_id": "s1", "paper_id": "p1", "split": "train", "section_name": "Intro"},
                             {"section_id": "s2", "paper_id": "p2", "split": "test", "section_name": "Intro"}])
    floats = pd.DataFrame([{"float_id": "f1", "paper_id": "p1", "split": "train", "caption": "A caption"},
                           {"float_id": "f2", "paper_id": "p1", "split": "train", "caption": "A caption"},
                           {"float_id": "empty", "paper_id": "p1", "split": "train", "caption": ""}])
    return papers, chunks, sections, floats


def test_unified_corpus_sources_and_embedding_templates():
    corpus, invalid = build_retrieval_corpus(*frames())
    assert corpus.source_type.value_counts().to_dict() == {"section": 2, "float": 2, "chunk": 1}
    assert len(invalid) == 1 and invalid.iloc[0].source_id == "empty"
    chunk = corpus[corpus.source_type.eq("chunk")].iloc[0]
    assert chunk.source_id == chunk.chunk_id == "c1" and chunk.paragraph_id == "r1"
    assert chunk.embedding_text == "Title: Paper\nSection: Intro\nPassage: Passage"
    assert "question" not in " ".join(corpus.embedding_text).lower()
    assert set(corpus[corpus.source_type.eq("float")].source_id) == {"f1", "f2"}


def test_deterministic_ids_fingerprint_and_split_isolation():
    first, _ = build_retrieval_corpus(*frames())
    second, _ = build_retrieval_corpus(*frames())
    pd.testing.assert_frame_equal(first, second)
    assert corpus_fingerprint(first) == corpus_fingerprint(second)
    assert first.document_id.is_unique
    assert set(first[first.paper_id.eq("p2")].split) == {"test"}

import numpy as np
import pandas as pd

from src.evidence_scoring import EvidenceSelector, minmax_normalize, score_evidence
from src.retrieval_metrics import validate_predictions


def frame():
    rows = []
    specs = [
        ("d1", "chunk", "c1", "p1", 1., .1, 1, 1),
        ("d2", "chunk", "c2", "p1", 1., .9, 2, np.nan),  # repeated paragraph
        ("d3", "section", "s1", "", 0., .5, np.nan, 2),
        ("d4", "float", "f1", "", 0., .4, 3, 3),
        ("d5", "chunk", "c5", "p5", 0., .3, 4, np.nan),
        ("d6", "chunk", "c6", "p6", 0., .2, np.nan, 4),
    ]
    for rank, (doc, typ, source, paragraph, ce, hybrid, bm, dense) in enumerate(specs, 1):
        rows.append({"question_id": "q", "paper_id": "paper", "document_id": doc, "source_type": typ,
                     "source_id": source, "chunk_id": source if typ == "chunk" else "", "paragraph_id": paragraph,
                     "section_id": source if typ == "section" else "", "figure_table_id": source if typ == "float" else "",
                     "float_id": source if typ == "float" else "", "cross_encoder_score": ce, "hybrid_score": hybrid,
                     "hybrid_rank": rank, "bm25_rank": bm, "dense_rank": dense, "rank": rank, "score": ce,
                     "title": "Title", "section_name": "Section", "text": f"text {doc}"})
    return pd.DataFrame(rows)


def test_normalization_and_constant_scores():
    assert minmax_normalize(pd.Series([2., 4., 3.])).tolist() == [0., 1., .5]
    assert minmax_normalize(pd.Series([7., 7.])).tolist() == [0., 0.]


def test_agreement_bonus_and_fixed_fusion():
    scored = score_evidence(frame(), "fused").set_index("document_id")
    assert scored.loc["d1", "retrieved_by_both"]
    assert scored.loc["d1", "agreement_score"] == 1.
    assert scored.loc["d1", "final_evidence_score"] == .70 + .05
    assert scored.loc["d2", "final_evidence_score"] == .70 + .25


def test_deterministic_tie_breaking_canonical_dedup_and_top_five():
    scored = score_evidence(frame(), "cross_encoder")
    forward = EvidenceSelector().select(scored)
    reverse = EvidenceSelector().select(score_evidence(frame().iloc[::-1], "cross_encoder"))
    assert forward.document_id.tolist() == reverse.document_id.tolist()
    assert len(forward) == 5
    assert len(set(forward.paragraph_id) - {""}) == len(forward[forward.paragraph_id.ne("")])
    assert forward.evidence_text.tolist() == forward.text.tolist()


def test_empty_selection_and_t06_prediction_compatibility():
    assert EvidenceSelector().select(frame().iloc[:0]).empty
    selected = EvidenceSelector().select(score_evidence(frame(), "fused"))
    validate_predictions(selected)

import pandas as pd
import pytest

from src.data_validation import validate_foreign_keys


def _tables():
    return {
        "papers": pd.DataFrame({"paper_id": ["p1"]}),
        "sections": pd.DataFrame({"section_id": ["s1"], "paper_id": ["p1"]}),
        "paragraphs": pd.DataFrame({"paragraph_id": ["r1"], "section_id": ["s1"], "paper_id": ["p1"]}),
        "questions": pd.DataFrame({"question_id": ["q1"], "paper_id": ["p1"]}),
        "answers": pd.DataFrame({"answer_id": ["a1"], "question_id": ["q1"], "paper_id": ["p1"]}),
        "evidence": pd.DataFrame({"evidence_id": ["e1"], "answer_id": ["a1"], "paper_id": ["p1"]}),
        "figures_tables": pd.DataFrame({"float_id": ["f1"], "paper_id": ["p1"]}),
        "chunks": pd.DataFrame({"chunk_id": ["c1"], "paper_id": ["p1"]}),
        "evidence_mappings": pd.DataFrame({"mapping_id": ["m1"], "evidence_id": ["e1"], "paper_id": ["p1"], "section_id": ["s1"], "paragraph_id": ["r1"], "chunk_id": ["c1"], "float_id": [""]}),
    }


def test_foreign_key_integrity():
    validate_foreign_keys(_tables())
    tables = _tables()
    tables["evidence_mappings"].loc[0, "chunk_id"] = "bad"
    with pytest.raises(ValueError, match="Foreign key"):
        validate_foreign_keys(tables)


def test_no_cross_paper_mappings():
    tables = _tables()
    tables["papers"] = pd.DataFrame({"paper_id": ["p1", "p2"]})
    tables["chunks"].loc[0, "paper_id"] = "p2"
    with pytest.raises(ValueError, match="Cross-paper"):
        validate_foreign_keys(tables)


def test_valid_section_foreign_keys_and_cross_paper_rejection():
    tables = _tables()
    validate_foreign_keys(tables)
    tables["papers"] = pd.DataFrame({"paper_id": ["p1", "p2"]})
    tables["sections"].loc[0, "paper_id"] = "p2"
    with pytest.raises(ValueError, match="Cross-paper"):
        validate_foreign_keys(tables)


def test_invalid_section_foreign_key():
    tables = _tables()
    tables["evidence_mappings"].loc[0, "section_id"] = "missing-section"
    with pytest.raises(ValueError, match="Foreign key"):
        validate_foreign_keys(tables)

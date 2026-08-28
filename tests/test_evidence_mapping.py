import pandas as pd

from src.evidence_mapping import build_evidence_mappings, normalize_evidence_text


def _frames():
    evidence = pd.DataFrame([
        {"evidence_id": "e1", "paper_id": "p1", "split": "train", "evidence_type": "text", "evidence_text": "Café\n text"},
        {"evidence_id": "e2", "paper_id": "p2", "split": "train", "evidence_type": "text", "evidence_text": "Café text"},
        {"evidence_id": "e3", "paper_id": "p1", "split": "train", "evidence_type": "text", "evidence_text": "duplicate"},
        {"evidence_id": "e4", "paper_id": "p1", "split": "train", "evidence_type": "figure_table", "evidence_text": "FLOAT SELECTED: fig1.png"},
        {"evidence_id": "e5", "paper_id": "p1", "split": "train", "evidence_type": "figure_table", "evidence_text": "FLOAT SELECTED: Table 1 results"},
        {"evidence_id": "e6", "paper_id": "p1", "split": "train", "evidence_type": "text", "evidence_text": "missing"},
        {"evidence_id": "e7", "paper_id": "p1", "split": "train", "evidence_type": "text", "evidence_text": " Results\n and Discussion "},
        {"evidence_id": "e8", "paper_id": "p1", "split": "train", "evidence_type": "text", "evidence_text": "Repeated heading"},
    ])
    paragraphs = pd.DataFrame([
        {"paragraph_id": "a", "paper_id": "p1", "text": "Cafe\u0301   text"},
        {"paragraph_id": "foreign", "paper_id": "p2", "text": "Café text"},
        {"paragraph_id": "d1", "paper_id": "p1", "text": "duplicate"},
        {"paragraph_id": "d2", "paper_id": "p1", "text": "duplicate"},
    ])
    sections = pd.DataFrame([
        {"section_id": "s1", "paper_id": "p1", "section_name": "Results and Discussion"},
        {"section_id": "foreign-section", "paper_id": "p2", "section_name": "Results and Discussion"},
        {"section_id": "ds1", "paper_id": "p1", "section_name": "Repeated heading"},
        {"section_id": "ds2", "paper_id": "p1", "section_name": "Repeated heading"},
    ])
    floats = pd.DataFrame([
        {"float_id": "f1", "paper_id": "p1", "file": "fig1.png", "caption": "Figure 1"},
        {"float_id": "t1", "paper_id": "p1", "file": "tab1.png", "caption": "Table 1 results"},
    ])
    chunks = pd.DataFrame([
        {"chunk_id": "c1", "paper_id": "p1", "paragraph_ids": ["a", "d1"]},
        {"chunk_id": "c2", "paper_id": "p1", "paragraph_ids": ["a", "d2"]},
        {"chunk_id": "cx", "paper_id": "p2", "paragraph_ids": ["foreign"]},
    ])
    return evidence, paragraphs, sections, floats, chunks


def test_unicode_and_whitespace_normalization():
    assert normalize_evidence_text(" Cafe\u0301\r\n  text ") == "Café text"


def test_exact_paper_scoped_mapping_and_outcomes():
    mappings = build_evidence_mappings(*_frames())
    e1 = mappings[mappings.evidence_id.eq("e1")]
    assert set(e1.paragraph_id) == {"a"}
    assert set(e1.chunk_id) == {"c1", "c2"}
    assert "foreign" not in set(e1.paragraph_id)
    assert set(mappings[mappings.evidence_id.eq("e2")].paragraph_id) == {"foreign"}
    assert set(mappings[mappings.evidence_id.eq("e3")].mapping_status) == {"ambiguous"}
    assert set(mappings[mappings.evidence_id.eq("e3")].paragraph_id) == {"d1", "d2"}
    assert set(mappings[mappings.evidence_id.eq("e4")].float_id) == {"f1"}
    assert set(mappings[mappings.evidence_id.eq("e5")].float_id) == {"t1"}
    assert mappings[mappings.evidence_id.eq("e6")].iloc[0].mapping_status == "unmatched"


def test_exact_section_heading_is_paper_scoped():
    mappings = build_evidence_mappings(*_frames())
    section_mapping = mappings[mappings.evidence_id.eq("e7")]
    assert set(section_mapping.source_type) == {"section"}
    assert set(section_mapping.section_id) == {"s1"}
    assert "foreign-section" not in set(section_mapping.section_id)
    assert set(section_mapping.mapping_status) == {"matched"}


def test_duplicate_section_heading_retains_every_ambiguous_candidate():
    mappings = build_evidence_mappings(*_frames())
    section_mapping = mappings[mappings.evidence_id.eq("e8")]
    assert set(section_mapping.mapping_status) == {"ambiguous"}
    assert set(section_mapping.section_id) == {"ds1", "ds2"}
    assert set(section_mapping.candidate_count) == {2}


def test_deterministic_reproducible_results():
    frames = _frames()
    first = build_evidence_mappings(*frames)
    second = build_evidence_mappings(*frames)
    pd.testing.assert_frame_equal(first, second)
    assert first.mapping_id.is_unique

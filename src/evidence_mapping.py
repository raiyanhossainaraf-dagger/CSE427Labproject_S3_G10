"""Deterministic, paper-scoped exact mapping of QASPER evidence to sources."""

import re
import unicodedata
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd

from src.schemas import generate_mapping_id


def normalize_evidence_text(value: object) -> str:
    """Normalize Unicode and every whitespace/line-break run for exact comparison."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def _paragraph_chunk_index(chunks: pd.DataFrame) -> Dict[Tuple[str, str], List[str]]:
    index: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for row in chunks.itertuples(index=False):
        paragraph_ids = row.paragraph_ids
        if isinstance(paragraph_ids, str):
            paragraph_ids = [paragraph_ids]
        for paragraph_id in paragraph_ids:
            index[(str(row.paper_id), str(paragraph_id))].append(str(row.chunk_id))
    return {key: sorted(set(values)) for key, values in index.items()}


def _float_evidence_value(text: str) -> str:
    prefix = "FLOAT SELECTED:"
    return text[len(prefix):].strip() if text.startswith(prefix) else text


def build_evidence_mappings(
    evidence: pd.DataFrame,
    paragraphs: pd.DataFrame,
    sections: pd.DataFrame,
    figures_tables: pd.DataFrame,
    chunks: pd.DataFrame,
) -> pd.DataFrame:
    """Return normalized relation rows, including explicit unmatched/ambiguous rows."""
    paragraph_index: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for row in paragraphs.itertuples(index=False):
        paragraph_index[(str(row.paper_id), normalize_evidence_text(row.text))].append(str(row.paragraph_id))

    section_index: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for row in sections.itertuples(index=False):
        normalized = normalize_evidence_text(row.section_name)
        if normalized:
            section_index[(str(row.paper_id), normalized)].append(str(row.section_id))

    float_index: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for row in figures_tables.itertuples(index=False):
        paper_id = str(row.paper_id)
        # QASPER FLOAT SELECTED values are filenames; caption support is exact too.
        for value in (row.file, row.caption):
            normalized = normalize_evidence_text(value)
            if normalized:
                float_index[(paper_id, normalized)].append(str(row.float_id))

    chunk_index = _paragraph_chunk_index(chunks)
    rows = []
    for ev in evidence.sort_values("evidence_id", kind="stable").itertuples(index=False):
        paper_id, evidence_id = str(ev.paper_id), str(ev.evidence_id)
        raw_match_text = _float_evidence_value(ev.evidence_text) if ev.evidence_type == "figure_table" else ev.evidence_text
        normalized = normalize_evidence_text(raw_match_text)
        if ev.evidence_type == "figure_table":
            candidates = [("float", source_id) for source_id in sorted(set(float_index.get((paper_id, normalized), [])))]
        else:
            candidates = [
                *(('paragraph', source_id) for source_id in sorted(set(paragraph_index.get((paper_id, normalized), [])))),
                *(('section', source_id) for source_id in sorted(set(section_index.get((paper_id, normalized), [])))),
            ]
        status = "unmatched" if not candidates else ("matched" if len(candidates) == 1 else "ambiguous")

        if not candidates:
            source_type = "float" if ev.evidence_type == "figure_table" else "text"
            rows.append(_mapping_row(ev, status, source_type, "", "", "", "", 0, normalized))
        else:
            for source_type, source_id in candidates:
                if source_type == "paragraph":
                    chunk_ids = chunk_index.get((paper_id, source_id), [])
                    if chunk_ids:
                        for chunk_id in chunk_ids:
                            rows.append(_mapping_row(ev, status, source_type, "", source_id, chunk_id, "", len(candidates), normalized))
                    else:
                        rows.append(_mapping_row(ev, status, source_type, "", source_id, "", "", len(candidates), normalized))
                elif source_type == "section":
                    rows.append(_mapping_row(ev, status, source_type, source_id, "", "", "", len(candidates), normalized))
                else:
                    rows.append(_mapping_row(ev, status, source_type, "", "", "", source_id, len(candidates), normalized))

    columns = ["mapping_id", "evidence_id", "paper_id", "split", "mapping_status", "source_type",
               "section_id", "paragraph_id", "chunk_id", "float_id", "candidate_count", "normalized_evidence_text"]
    return pd.DataFrame(rows, columns=columns).sort_values("mapping_id", kind="stable").reset_index(drop=True)


def _mapping_row(ev, status, source_type, section_id, paragraph_id, chunk_id, float_id, count, normalized):
    source_id = "|".join(part for part in (section_id, paragraph_id, chunk_id, float_id) if part)
    return {
        "mapping_id": generate_mapping_id(str(ev.evidence_id), source_type, source_id),
        "evidence_id": str(ev.evidence_id), "paper_id": str(ev.paper_id), "split": str(ev.split),
        "mapping_status": status, "source_type": source_type, "section_id": section_id, "paragraph_id": paragraph_id,
        "chunk_id": chunk_id, "float_id": float_id, "candidate_count": count,
        "normalized_evidence_text": normalized,
    }

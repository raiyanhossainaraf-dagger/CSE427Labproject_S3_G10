"""Build the deterministic, leakage-free unified retrieval corpus."""

from __future__ import annotations

import hashlib
from typing import Dict, Tuple

import pandas as pd

from src.schemas import TABLE_SCHEMAS, get_stable_id, validate_table

CORPUS_VERSION = "2.0.0"
SOURCE_ORDER = {"chunk": 0, "section": 1, "float": 2}


def generate_document_id(source_type: str, source_id: str) -> str:
    """Return a stable ID tied to the corpus version and exact source foreign key."""
    return get_stable_id(f"retrieval-corpus|{CORPUS_VERSION}|{source_type}|{source_id}")


def build_retrieval_corpus(
    papers: pd.DataFrame,
    chunks: pd.DataFrame,
    sections: pd.DataFrame,
    figures_tables: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build corpus rows and an explicit report of excluded invalid source records."""
    titles = dict(zip(papers.paper_id.astype(str), papers.title.fillna("").astype(str)))
    rows, invalid = [], []

    def add(source_type: str, source_id: object, paper_id: object, split: object,
            text: object, section_name: object = "", chunk_id: object = "",
            paragraph_id: object = "", section_id: object = "", figure_table_id: object = ""):
        values = {"source_id": source_id, "paper_id": paper_id, "split": split, "text": text,
                  "section_name": section_name, "chunk_id": chunk_id, "paragraph_id": paragraph_id,
                  "section_id": section_id, "figure_table_id": figure_table_id}
        values = {key: _text(value) for key, value in values.items()}
        title = titles.get(values["paper_id"], "").strip()
        reasons = []
        if not values["source_id"]: reasons.append("missing_source_id")
        if not values["paper_id"] or values["paper_id"] not in titles: reasons.append("invalid_paper_id")
        if values["split"] not in {"train", "validation", "test"}: reasons.append("invalid_split")
        if not title: reasons.append("empty_title")
        if not values["text"].strip(): reasons.append("empty_text")
        if reasons:
            invalid.append({"source_type": source_type, "source_id": values["source_id"],
                            "paper_id": values["paper_id"], "reasons": "|".join(reasons)})
            return
        if source_type == "chunk":
            embedding_text = f"Title: {title}\nSection: {values['section_name']}\nPassage: {values['text']}"
        elif source_type == "section":
            embedding_text = f"Title: {title}\nSection: {values['text']}"
        else:
            embedding_text = f"Title: {title}\nFigure/Table: {values['text']}"
        rows.append({"document_id": generate_document_id(source_type, values["source_id"]),
                     "paper_id": values["paper_id"], "split": values["split"], "source_type": source_type,
                     "source_id": values["source_id"], "chunk_id": values["chunk_id"],
                     "paragraph_id": values["paragraph_id"], "section_id": values["section_id"],
                     "figure_table_id": values["figure_table_id"], "title": title,
                     "section_name": values["section_name"], "text": values["text"].strip(),
                     "embedding_text": embedding_text})

    for row in chunks.itertuples(index=False):
        paragraph_ids = list(row.paragraph_ids) if not isinstance(row.paragraph_ids, str) else [row.paragraph_ids]
        paragraph_id = str(paragraph_ids[0]) if len(paragraph_ids) == 1 else ""
        add("chunk", row.chunk_id, row.paper_id, row.split, row.text, row.section_name,
            chunk_id=row.chunk_id, paragraph_id=paragraph_id, section_id=row.section_id)
    for row in sections.itertuples(index=False):
        add("section", row.section_id, row.paper_id, row.split, row.section_name,
            section_name=row.section_name, section_id=row.section_id)
    for row in figures_tables.itertuples(index=False):
        add("float", row.float_id, row.paper_id, row.split, row.caption,
            figure_table_id=row.float_id)

    columns = TABLE_SCHEMAS["retrieval_corpus"]
    corpus = pd.DataFrame(rows, columns=columns)
    corpus["_source_order"] = corpus.source_type.map(SOURCE_ORDER)
    corpus = corpus.sort_values(["split", "paper_id", "_source_order", "source_id"], kind="stable").drop(columns="_source_order").reset_index(drop=True)
    validate_retrieval_corpus(corpus, papers, chunks, sections, figures_tables)
    invalid_df = pd.DataFrame(invalid, columns=["source_type", "source_id", "paper_id", "reasons"])
    return corpus, invalid_df


def validate_retrieval_corpus(corpus, papers, chunks, sections, figures_tables) -> None:
    """Validate schema, stable IDs, source foreign keys, split consistency and safe text."""
    validate_table(corpus, "retrieval_corpus")
    if corpus.embedding_text.str.strip().eq("").any() or corpus.text.str.strip().eq("").any():
        raise ValueError("Retrieval corpus contains empty documents")
    sources: Dict[str, Dict[str, Tuple[str, str]]] = {
        "chunk": {str(r.chunk_id): (str(r.paper_id), str(r.split)) for r in chunks.itertuples()},
        "section": {str(r.section_id): (str(r.paper_id), str(r.split)) for r in sections.itertuples()},
        "float": {str(r.float_id): (str(r.paper_id), str(r.split)) for r in figures_tables.itertuples()},
    }
    for row in corpus.itertuples(index=False):
        expected = sources[row.source_type].get(str(row.source_id))
        if expected != (str(row.paper_id), str(row.split)):
            raise ValueError(f"Invalid or cross-split source foreign key: {row.source_type}/{row.source_id}")
        if row.document_id != generate_document_id(row.source_type, row.source_id):
            raise ValueError(f"Unstable document_id for {row.source_id}")
    forbidden = ("question_id", "answer_id", "evidence_id", "annotation_id")
    if any(column in corpus.columns for column in forbidden):
        raise ValueError("Retrieval corpus contains label or annotation columns")


def corpus_fingerprint(corpus: pd.DataFrame) -> str:
    """Hash ordered document IDs and embedding text for manifest reproducibility."""
    digest = hashlib.sha256()
    for row in corpus[["document_id", "embedding_text"]].itertuples(index=False):
        digest.update(str(row.document_id).encode("utf-8")); digest.update(b"\0")
        digest.update(str(row.embedding_text).encode("utf-8")); digest.update(b"\n")
    return digest.hexdigest()


def _text(value: object) -> str:
    return "" if value is None or (not isinstance(value, str) and pd.isna(value)) else str(value)

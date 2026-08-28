"""Cross-table integrity checks for processed QASPER data."""

import pandas as pd


def _check_fk(df, column, reference, table_name, reference_name):
    values = set(df.loc[df[column].notna() & df[column].ne(""), column].astype(str))
    invalid = values - set(reference.astype(str))
    if invalid:
        raise ValueError(f"Foreign key violation: {table_name}.{column} -> {reference_name}: {sorted(invalid)[:5]}")


def validate_foreign_keys(tables: dict) -> None:
    papers = tables["papers"]["paper_id"]
    for name in ("sections", "paragraphs", "questions", "answers", "evidence", "figures_tables", "chunks", "evidence_mappings"):
        if name in tables:
            _check_fk(tables[name], "paper_id", papers, name, "papers.paper_id")
    _check_fk(tables["paragraphs"], "section_id", tables["sections"]["section_id"], "paragraphs", "sections.section_id")
    _check_fk(tables["answers"], "question_id", tables["questions"]["question_id"], "answers", "questions.question_id")
    _check_fk(tables["evidence"], "answer_id", tables["answers"]["answer_id"], "evidence", "answers.answer_id")
    if "evidence_mappings" not in tables:
        return
    mappings = tables["evidence_mappings"]
    _check_fk(mappings, "evidence_id", tables["evidence"]["evidence_id"], "evidence_mappings", "evidence.evidence_id")
    _check_fk(mappings, "section_id", tables["sections"]["section_id"], "evidence_mappings", "sections.section_id")
    _check_fk(mappings, "paragraph_id", tables["paragraphs"]["paragraph_id"], "evidence_mappings", "paragraphs.paragraph_id")
    _check_fk(mappings, "chunk_id", tables["chunks"]["chunk_id"], "evidence_mappings", "chunks.chunk_id")
    _check_fk(mappings, "float_id", tables["figures_tables"]["float_id"], "evidence_mappings", "figures_tables.float_id")

    paper_maps = {name: dict(zip(df[id_col].astype(str), df.paper_id.astype(str))) for name, df, id_col in (
        ("section", tables["sections"], "section_id"),
        ("paragraph", tables["paragraphs"], "paragraph_id"), ("chunk", tables["chunks"], "chunk_id"),
        ("float", tables["figures_tables"], "float_id"))}
    for row in mappings.itertuples(index=False):
        for kind, source_id in (("section", row.section_id), ("paragraph", row.paragraph_id), ("chunk", row.chunk_id), ("float", row.float_id)):
            if source_id and paper_maps[kind][str(source_id)] != str(row.paper_id):
                raise ValueError(f"Cross-paper mapping: {row.mapping_id} -> {source_id}")

    mapped_evidence = set(mappings.evidence_id.astype(str))
    missing = set(tables["evidence"].evidence_id.astype(str)) - mapped_evidence
    if missing:
        raise ValueError(f"Evidence records without mapping outcomes: {sorted(missing)[:5]}")

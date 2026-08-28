from typing import Dict, Set, List, Optional
import pandas as pd
import hashlib

SCHEMA_VERSION = "1.0.0"

SPLITS: Set[str] = {"train", "validation", "test"}

ANSWER_TYPES: Set[str] = {
    "extractive",
    "abstractive",
    "yes",
    "no",
    "unanswerable",
}

EVIDENCE_TYPES: Set[str] = {
    "text",
    "figure_table",
}

SOURCE_TYPES: Set[str] = {
    "abstract",
    "full_text",
}

TABLE_SCHEMAS: Dict[str, List[str]] = {
    "papers": [
        "paper_id", "split", "title", "abstract", 
        "num_full_text_sections", "num_source_sections", 
        "num_paragraphs", "num_questions", "full_text_word_count"
    ],
    "sections": [
        "section_id", "paper_id", "split", 
        "section_index", "section_name", "source_type"
    ],
    "paragraphs": [
        "paragraph_id", "section_id", "paper_id", "split", 
        "section_index", "paragraph_index", "section_name", 
        "source_type", "text", "normalized_text", "word_count"
    ],
    "questions": [
        "question_id", "paper_id", "split", "question_index", 
        "question", "nlp_background", "topic_background", 
        "paper_read", "search_query", "question_writer", "num_answers"
    ],
    "answers": [
        "answer_id", "annotation_id", "question_id", "paper_id", "split", 
        "annotation_index", "worker_id", "answer_type", "unanswerable", 
        "yes_no", "extractive_spans", "free_form_answer", "highlighted_evidence"
    ],
    "evidence": [
        "evidence_id", "answer_id", "annotation_id", "question_id", 
        "paper_id", "split", "evidence_index", "evidence_type", 
        "evidence_text", "evidence_word_count"
    ],
    "figures_tables": [
        "float_id", "paper_id", "split", "float_index", 
        "file", "caption", "float_type"
    ]
}

def validate_table(df: pd.DataFrame, table_name: str) -> None:
    """Validates a DataFrame against its schema."""
    if table_name not in TABLE_SCHEMAS:
        raise ValueError(f"Unknown table name: {table_name}")
    
    expected_cols = TABLE_SCHEMAS[table_name]
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Table '{table_name}' missing columns: {missing_cols}")
    
    # Check for unique IDs
    id_col = f"{table_name.rstrip('s')}_id"
    if table_name == "figures_tables":
        id_col = "float_id"
        
    if id_col in df.columns:
        if not df[id_col].is_unique:
            duplicates = df[df[id_col].duplicated()][id_col].unique()
            raise ValueError(f"Table '{table_name}' has duplicate {id_col} values: {duplicates[:5]}")

    # Check for split validity
    if "split" in df.columns:
        invalid_splits = set(df["split"].unique()) - SPLITS
        if invalid_splits:
            raise ValueError(f"Table '{table_name}' has invalid splits: {invalid_splits}")

def get_stable_id(payload: str) -> str:
    """Generates a stable SHA-256 digest from a string."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def generate_section_id(paper_id: str, source_type: str, section_index: int) -> str:
    return f"{paper_id}::section::{source_type}::{section_index}"

def generate_paragraph_id(section_id: str, paragraph_index: int) -> str:
    return f"{section_id}::paragraph::{paragraph_index}"

def generate_evidence_id(annotation_id: str, evidence_index: int) -> str:
    return f"{annotation_id}::evidence::{evidence_index}"

def generate_float_id(paper_id: str, float_index: int) -> str:
    return f"{paper_id}::float::{float_index}"

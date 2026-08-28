import re
import pandas as pd
from typing import List, Dict
from tqdm.auto import tqdm

def normalize_text(text: str) -> str:
    """Performs basic text normalization."""
    if not isinstance(text, str):
        text = str(text)
    # Normalize repeated whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip surrounding whitespace
    text = text.strip()
    return text

def chunk_text(text: str, chunk_size: int = 350, overlap: int = 50) -> List[str]:
    """Chunks text into units based on word count."""
    words = text.split()
    chunks = []
    if not words:
        return []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

from src.schemas import (
    generate_section_id, generate_paragraph_id,
    generate_evidence_id, generate_float_id,
    ANSWER_TYPES, EVIDENCE_TYPES
)

def get_answer_type(ans_data: Dict) -> str:
    """Deterministic answer-type rule."""
    if ans_data.get('unanswerable', False):
        return 'unanswerable'
    yes_no = ans_data.get('yes_no')
    if yes_no is True:
        return 'yes'
    if yes_no is False:
        return 'no'
    if ans_data.get('extractive_spans'):
        return 'extractive'
    if ans_data.get('free_form_answer'):
        return 'abstractive'
    return 'unknown' # Should be caught by validation

def process_qasper_to_tables(ds: Dict) -> Dict[str, pd.DataFrame]:
    """
    Converts nested QASPER dataset into structured DataFrames.
    Phase 1 implementation.
    """
    paper_rows = []
    section_rows = []
    paragraph_rows = []
    question_rows = []
    answer_rows = []
    evidence_rows = []
    float_rows = []

    # Deterministic split order
    split_order = ["train", "validation", "test"]

    for split in split_order:
        if split not in ds:
            continue

        split_data = ds[split]
        # Deterministic paper order
        sorted_paper_ids = sorted(split_data.keys())

        for paper_id in tqdm(sorted_paper_ids, desc=f"Processing {split} split"):
            paper = split_data[paper_id]

            title = paper['title']
            abstract = paper['abstract']

            # --- Sections & Paragraphs ---
            paper_sections = []

            # 1. Abstract as a section
            if abstract and abstract.strip():
                sec_id = generate_section_id(paper_id, "abstract", 0)
                sec_row = {
                    "section_id": sec_id,
                    "paper_id": paper_id,
                    "split": split,
                    "section_index": 0,
                    "section_name": "Abstract",
                    "source_type": "abstract"
                }
                section_rows.append(sec_row)
                paper_sections.append(sec_row)

                para_id = generate_paragraph_id(sec_id, 0)
                paragraph_rows.append({
                    "paragraph_id": para_id,
                    "section_id": sec_id,
                    "paper_id": paper_id,
                    "split": split,
                    "section_index": 0,
                    "paragraph_index": 0,
                    "section_name": "Abstract",
                    "source_type": "abstract",
                    "text": abstract,
                    "normalized_text": normalize_text(abstract),
                    "word_count": len(abstract.split())
                })
            
            # 2. Full text sections
            full_text = paper.get('full_text', [])
            for sec_idx, sec in enumerate(full_text):
                sec_name = sec.get('section_name', f"Section {sec_idx}")
                sec_id = generate_section_id(paper_id, "full_text", sec_idx)
                sec_row = {
                    "section_id": sec_id,
                    "paper_id": paper_id,
                    "split": split,
                    "section_index": sec_idx,
                    "section_name": sec_name,
                    "source_type": "full_text"
                }
                section_rows.append(sec_row)
                paper_sections.append(sec_row)
                
                paragraphs = sec.get('paragraphs', [])
                for para_idx, para_text in enumerate(paragraphs):
                    para_id = generate_paragraph_id(sec_id, para_idx)
                    paragraph_rows.append({
                        "paragraph_id": para_id,
                        "section_id": sec_id,
                        "paper_id": paper_id,
                        "split": split,
                        "section_index": sec_idx,
                        "paragraph_index": para_idx,
                        "section_name": sec_name,
                        "source_type": "full_text",
                        "text": para_text,
                        "normalized_text": normalize_text(para_text),
                        "word_count": len(para_text.split())
                    })
            
            # --- Figures and Tables ---
            figs_tables = paper.get('figures_and_tables', [])
            for float_idx, ft in enumerate(figs_tables):
                float_id = generate_float_id(paper_id, float_idx)
                caption = ft.get('caption', '')
                filename = ft.get('file', '')
                
                # float_type logic
                float_type = 'unknown'
                combined = (caption + ' ' + filename).lower()
                if 'fig' in combined:
                    float_type = 'figure'
                elif 'tab' in combined:
                    float_type = 'table'
                
                float_rows.append({
                    "float_id": float_id,
                    "paper_id": paper_id,
                    "split": split,
                    "float_index": float_idx,
                    "file": filename,
                    "caption": caption,
                    "float_type": float_type
                })

            # --- Questions, Answers, Evidence ---
            qas = paper.get('qas', [])
            for q_idx, q_entry in enumerate(qas):
                q_id = q_entry['question_id']
                
                question_rows.append({
                    "question_id": q_id,
                    "paper_id": paper_id,
                    "split": split,
                    "question_index": q_idx,
                    "question": q_entry['question'],
                    "nlp_background": q_entry.get('nlp_background'),
                    "topic_background": q_entry.get('topic_background'),
                    "paper_read": q_entry.get('paper_read'),
                    "search_query": q_entry.get('search_query'),
                    "question_writer": q_entry.get('question_writer'),
                    "num_answers": len(q_entry.get('answers', []))
                })
                
                for ans_idx, ann in enumerate(q_entry.get('answers', [])):
                    ann_id = ann['annotation_id']
                    ans_data = ann.get('answer', {})
                    
                    answer_rows.append({
                        "answer_id": ann_id,
                        "annotation_id": ann_id,
                        "question_id": q_id,
                        "paper_id": paper_id,
                        "split": split,
                        "annotation_index": ans_idx,
                        "worker_id": ann.get('worker_id'),
                        "answer_type": get_answer_type(ans_data),
                        "unanswerable": ans_data.get('unanswerable', False),
                        "yes_no": ans_data.get('yes_no'),
                        "extractive_spans": ans_data.get('extractive_spans', []),
                        "free_form_answer": ans_data.get('free_form_answer', ""),
                        "highlighted_evidence": ans_data.get('highlighted_evidence', [])
                    })
                    
                    evidence_list = ans_data.get('evidence', [])
                    for ev_idx, ev_text in enumerate(evidence_list):
                        ev_id = generate_evidence_id(ann_id, ev_idx)
                        ev_type = "figure_table" if ev_text.startswith("FLOAT SELECTED:") else "text"
                        
                        evidence_rows.append({
                            "evidence_id": ev_id,
                            "answer_id": ann_id,
                            "annotation_id": ann_id,
                            "question_id": q_id,
                            "paper_id": paper_id,
                            "split": split,
                            "evidence_index": ev_idx,
                            "evidence_type": ev_type,
                            "evidence_text": ev_text,
                            "evidence_word_count": len(ev_text.split())
                        })

            # --- Paper Summary ---
            all_para_words = [p['word_count'] for p in paragraph_rows if p['paper_id'] == paper_id]
            paper_rows.append({
                "paper_id": paper_id,
                "split": split,
                "title": title,
                "abstract": abstract,
                "num_full_text_sections": len(full_text),
                "num_source_sections": len(paper_sections),
                "num_paragraphs": len(all_para_words),
                "num_questions": len(qas),
                "full_text_word_count": sum(all_para_words)
            })

    return {
        "papers": pd.DataFrame(paper_rows),
        "sections": pd.DataFrame(section_rows),
        "paragraphs": pd.DataFrame(paragraph_rows),
        "questions": pd.DataFrame(question_rows),
        "answers": pd.DataFrame(answer_rows),
        "evidence": pd.DataFrame(evidence_rows),
        "figures_tables": pd.DataFrame(float_rows)
    }

def process_papers_to_chunks(ds: Dict, chunk_size: int = 350, overlap: int = 50) -> pd.DataFrame:
    """
    Transforms all papers in the dataset into retrievable textual units (chunks).
    """
    chunks_list = []
    
    for split in ds.keys():
        for paper_id, paper in ds[split].items():
            paper_title = paper['title']
            full_text = paper.get('full_text', [])
            
            for sec_idx, section in enumerate(full_text):
                section_name = section.get('section_name', f'Section_{sec_idx}')
                paragraphs = section.get('paragraphs', [])
                
                section_text = " ".join(paragraphs)
                section_text = normalize_text(section_text)
                
                if not section_text:
                    continue
                    
                text_chunks = chunk_text(section_text, chunk_size, overlap)
                
                for chk_idx, text in enumerate(text_chunks):
                    # Deterministic chunk_id
                    chunk_id = f"{paper_id}_{sec_idx}_{chk_idx}"
                    chunks_list.append({
                        "chunk_id": chunk_id,
                        "paper_id": paper_id,
                        "paper_title": paper_title,
                        "split": split,
                        "section_name": section_name,
                        "chunk_index": chk_idx,
                        "text": text,
                        "word_count": len(text.split())
                    })
            
    return pd.DataFrame(chunks_list)

def process_paper_to_chunks(paper: Dict, chunk_size: int = 350, overlap: int = 50) -> List[Dict]:
    """
    Transforms a single paper into retrievable textual units (chunks) while preserving metadata.
    Keep for backward compatibility.
    """
    paper_id = paper['id']
    paper_title = paper['title']
    split = paper.get('split', 'unknown')
    
    chunks_metadata = []
    
    # Process full_text sections
    full_text = paper.get('full_text', [])
    for sec_idx, section in enumerate(full_text):
        section_name = section.get('section_name', f'Section_{sec_idx}')
        paragraphs = section.get('paragraphs', [])
        
        section_text = " ".join(paragraphs)
        section_text = normalize_text(section_text)
        
        if not section_text:
            continue
            
        text_chunks = chunk_text(section_text, chunk_size, overlap)
        
        for chk_idx, text in enumerate(text_chunks):
            chunk_id = f"{paper_id}_{sec_idx}_{chk_idx}"
            chunks_metadata.append({
                "chunk_id": chunk_id,
                "paper_id": paper_id,
                "paper_title": paper_title,
                "split": split,
                "section_name": section_name,
                "chunk_index": chk_idx,
                "text": text,
                "word_count": len(text.split())
            })
            
    return chunks_metadata

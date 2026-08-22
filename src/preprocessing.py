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

def create_structured_dataframes(ds: Dict) -> Dict[str, pd.DataFrame]:
    """
    Converts nested QASPER dataset into structured DataFrames for papers, questions, and evidence.
    """
    paper_data = []
    question_data = []
    evidence_data = []

    for split in ds.keys():
        split_data = ds[split]
        for paper_id, paper in split_data.items():
            paper['id'] = paper_id
            title = paper['title']
            abstract = paper['abstract']
            
            sections = paper.get('full_text', [])
            all_paragraphs = []
            for sec in sections:
                all_paragraphs.extend(sec.get('paragraphs', []))
            
            full_text_str = " ".join(all_paragraphs)
            
            paper_data.append({
                "paper_id": paper_id,
                "split": split,
                "title": title,
                "abstract": abstract,
                "num_sections": len(sections),
                "num_paragraphs": len(all_paragraphs),
                "full_text_word_count": len(full_text_str.split()),
                "num_questions": len(paper.get('qas', []))
            })
            
            for q_entry in paper.get('qas', []):
                q_id = q_entry['question_id']
                question = q_entry['question']
                answers = q_entry.get('answers', [])
                
                for ans_idx, annotation in enumerate(answers):
                    ans_data = annotation.get('answer', {})
                    evidence_texts = ans_data.get('evidence', [])
                    for ev_text in evidence_texts:
                        evidence_data.append({
                            "question_id": q_id,
                            "paper_id": paper_id,
                            "split": split,
                            "evidence_text": ev_text,
                            "evidence_word_count": len(ev_text.split()),
                            "annotation_id": annotation.get('annotation_id', f"{q_id}_{ans_idx}")
                        })
                
                question_data.append({
                    "question_id": q_id,
                    "paper_id": paper_id,
                    "split": split,
                    "question": question,
                    "question_word_count": len(question.split()),
                    "num_answers": len(answers)
                })

    return {
        "papers": pd.DataFrame(paper_data),
        "questions": pd.DataFrame(question_data),
        "evidence": pd.DataFrame(evidence_data)
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

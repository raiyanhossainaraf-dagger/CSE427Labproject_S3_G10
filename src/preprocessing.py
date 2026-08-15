import re
import pandas as pd
from typing import List, Dict

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
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

def process_paper_to_chunks(paper: Dict, chunk_size: int = 350, overlap: int = 50) -> List[Dict]:
    """
    Transforms a paper into retrievable textual units (chunks) while preserving metadata.
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

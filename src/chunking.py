import hashlib
from typing import List, Dict, Optional, Union
import pandas as pd
from src.schemas import get_stable_id

class TokenizerAdapter:
    """Mock or adapter for a real tokenizer to avoid heavy downloads during tests."""
    def __init__(self, tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2", use_hf: bool = True,
                 require_hf: bool = False):
        self.tokenizer_name = tokenizer_name
        self._tokenizer = None
        if use_hf:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            except Exception as exc:
                if require_hf:
                    raise RuntimeError(
                        f"Configured tokenizer '{tokenizer_name}' is unavailable; refusing a non-reproducible fallback"
                    ) from exc

    def tokenize(self, text: str) -> List[Union[int, str]]:
        if self._tokenizer:
            return self._tokenizer.encode(text, add_special_tokens=False)
        # Fallback to word-like split for testing if transformers unavailable or disabled
        return text.split()

    def decode(self, tokens: List[Union[int, str]]) -> str:
        if self._tokenizer:
            return self._tokenizer.decode(tokens)
        return " ".join(tokens)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

def generate_chunk_id(paper_id: str, section_id: str, paragraph_ids: List[str], offsets: Optional[tuple] = None) -> str:
    payload = f"{paper_id}|{section_id}|{','.join(paragraph_ids)}"
    if offsets:
        payload += f"|{offsets[0]}:{offsets[1]}"
    return get_stable_id(payload)

def chunk_paragraphs(
    paragraphs_df: pd.DataFrame,
    papers_df: pd.DataFrame,
    tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_tokens: int = 384,
    overlap_tokens: int = 32,
    chunking_version: str = "1.0.0",
    use_hf: bool = True,
    require_hf: bool = False,
) -> pd.DataFrame:
    tokenizer = TokenizerAdapter(tokenizer_name, use_hf=use_hf, require_hf=require_hf)
    chunks = []
    
    # Metadata reservation (Title and Section name)
    # Template: "Title: {title}\nSection: {section}\nPassage: "
    
    # Group by paper
    for paper_id, paper_group in paragraphs_df.groupby("paper_id", sort=False):
        paper_title = papers_df[papers_df["paper_id"] == paper_id]["title"].iloc[0]
        title_prefix = f"Title: {paper_title}\n"
        
        # Group by section
        for section_id, section_group in paper_group.groupby("section_id", sort=False):
            section_name = section_group["section_name"].iloc[0]
            section_prefix = f"Section: {section_name}\nPassage: "
            
            metadata_text = title_prefix + section_prefix
            metadata_token_count = tokenizer.count_tokens(metadata_text)
            
            available_tokens = max_tokens - metadata_token_count
            if available_tokens <= 0:
                # Should not happen with MiniLM and normal titles
                available_tokens = max_tokens // 2 

            current_chunk_tokens = []
            current_chunk_paras = []
            chunk_idx = 0

            # Iterate through paragraphs in the section
            for _, row in section_group.iterrows():
                para_id = row["paragraph_id"]
                para_text = row["text"]
                if not para_text or not para_text.strip():
                    continue
                para_tokens = tokenizer.tokenize(para_text)
                if not para_tokens:
                    continue
                
                # If a single paragraph exceeds available tokens, it must be split
                if len(para_tokens) > available_tokens:
                    # Flush current chunk if not empty
                    if current_chunk_paras:
                        chunks.append(_create_chunk_record(
                            paper_id, section_id, section_name, row["split"], 
                            chunk_idx, current_chunk_paras, 
                            tokenizer.decode(current_chunk_tokens),
                            metadata_text, len(current_chunk_tokens),
                            chunking_version, tokenizer_name
                        ))
                        chunk_idx += 1
                        current_chunk_tokens = []
                        current_chunk_paras = []
                    
                    # Split the long paragraph
                    start = 0
                    previous_start = -1
                    while start < len(para_tokens):
                        if start <= previous_start:
                             raise RuntimeError(f"Infinite loop detected in chunking para {para_id}. Start: {start}, Previous: {previous_start}")
                        previous_start = start
                        
                        end = start + available_tokens
                        chunk_para_tokens = para_tokens[start:end]
                        
                        chunks.append(_create_chunk_record(
                            paper_id, section_id, section_name, row["split"],
                            chunk_idx, [para_id],
                            tokenizer.decode(chunk_para_tokens),
                            metadata_text, len(chunk_para_tokens),
                            chunking_version, tokenizer_name,
                            is_partial=True,
                            parent_para_id=para_id,
                            token_start=start,
                            token_end=start + len(chunk_para_tokens)
                        ))
                        chunk_idx += 1
                        
                        if end >= len(para_tokens):
                            break
                            
                        # Advance with overlap
                        advance = available_tokens - overlap_tokens
                        if advance <= 0:
                            # Fallback if overlap >= budget
                            advance = 1
                        
                        start += advance
                
                # Otherwise, try to fit into current chunk
                elif len(current_chunk_tokens) + len(para_tokens) > available_tokens:
                    # Flush and start new
                    chunks.append(_create_chunk_record(
                        paper_id, section_id, section_name, row["split"], 
                        chunk_idx, current_chunk_paras, 
                        tokenizer.decode(current_chunk_tokens),
                        metadata_text, len(current_chunk_tokens),
                        chunking_version, tokenizer_name
                    ))
                    chunk_idx += 1
                    current_chunk_tokens = para_tokens
                    current_chunk_paras = [para_id]
                else:
                    # Add to current
                    if current_chunk_tokens:
                        current_chunk_tokens.extend(tokenizer.tokenize(" ")) # Para separator
                    current_chunk_tokens.extend(para_tokens)
                    current_chunk_paras.append(para_id)

            # Flush last chunk
            if current_chunk_paras:
                chunks.append(_create_chunk_record(
                    paper_id, section_id, section_name, row["split"], 
                    chunk_idx, current_chunk_paras, 
                    tokenizer.decode(current_chunk_tokens),
                    metadata_text, len(current_chunk_tokens),
                    chunking_version, tokenizer_name
                ))

    return pd.DataFrame(chunks)

def _create_chunk_record(
    paper_id, section_id, section_name, split, chunk_index, paragraph_ids, 
    text, metadata_prefix, token_count, chunking_version, tokenizer_name,
    is_partial=False, parent_para_id=None, token_start=None, token_end=None
):
    embedding_text = metadata_prefix + text
    offsets = (token_start, token_end) if is_partial else None
    
    return {
        "chunk_id": generate_chunk_id(paper_id, section_id, paragraph_ids, offsets),
        "paper_id": paper_id,
        "split": split,
        "section_id": section_id,
        "section_name": section_name,
        "source_type": "full_text" if "full_text" in section_id else "abstract",
        "chunk_index": chunk_index,
        "paragraph_ids": paragraph_ids,
        "text": text,
        "embedding_text": embedding_text,
        "token_count": token_count,
        "word_count": len(text.split()),
        "is_partial_paragraph": is_partial,
        "parent_paragraph_id": parent_para_id,
        "paragraph_token_start": token_start,
        "paragraph_token_end": token_end,
        "chunking_version": chunking_version,
        "tokenizer_name": tokenizer_name
    }

import pandas as pd
from typing import List

def format_rag_prompt(question: str, retrieved_chunks: pd.DataFrame) -> str:
    """
    Formats the prompt for the future LLM generation step.
    """
    context_blocks = []
    for i, (_, row) in enumerate(retrieved_chunks.iterrows()):
        block = f"Source {i+1} (Paper: {row['paper_title']}, Section: {row['section_name']}):\n{row['text']}"
        context_blocks.append(block)
    
    context = "\n\n".join(context_blocks)
    
    prompt = f"""
Use the following retrieved scientific context to answer the question. 
If the answer is not in the context, state that you do not have enough information.
Always cite the source numbers in your answer.

Context:
{context}

Question:
{question}

Answer:
"""
    return prompt

def prepare_baseline_rag(question: str, retrieved_chunks: pd.DataFrame):
    """
    Prepares and prints the formatted RAG prompt.
    """
    prompt = format_rag_prompt(question, retrieved_chunks)
    print("--- BASELINE RAG PROMPT PREPARED ---")
    print(prompt)
    return prompt

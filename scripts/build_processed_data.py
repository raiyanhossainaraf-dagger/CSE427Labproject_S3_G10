import argparse
import sys
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_project_paths, ensure_project_directories
from src.data_loader import load_qasper_dataset
from src.preprocessing import process_qasper_to_tables
from src.schemas import validate_table, SCHEMA_VERSION
from src.chunking import chunk_paragraphs

def build_processed_data(args):
    paths = get_project_paths(explicit_root=args.project_root)
    ensure_project_directories(paths)
    
    raw_dir = Path(args.raw_dir) if args.raw_dir else paths.raw_data_dir
    output_dir = Path(args.output_dir) if args.output_dir else paths.processed_data_dir
    
    if args.validate_only:
        print("Running validation only...")
        validate_existing_tables(output_dir)
        return

    print(f"Loading raw QASPER data from {raw_dir}...")
    ds = load_qasper_dataset(raw_dir)
    
    print("Building schema tables (Phase 1)...")
    tables = process_qasper_to_tables(ds)
    
    # Staging validation
    print("Validating tables...")
    for name, df in tables.items():
        validate_table(df, name)
    
    # Cross-table foreign key validation
    validate_foreign_keys(tables)
    
    # Save tables
    print(f"Saving tables to {output_dir}...")
    for name, df in tables.items():
        file_path = output_dir / f"{name}.parquet"
        df.to_parquet(file_path, index=False)
        print(f" - Saved {name}.parquet ({len(df)} rows)")

    # Summary generation
    generate_schema_summary(tables, paths.summaries_dir)
    
    # Phase 2: Chunking
    print("Building paragraph-aware chunks (Phase 2)...")
    chunks_df = chunk_paragraphs(
        tables["paragraphs"], 
        tables["papers"],
        tokenizer_name=args.tokenizer_name,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens
    )
    
    print("Validating chunks...")
    validate_chunks(chunks_df)
    
    # Save chunks
    chunks_file = output_dir / "chunks.parquet"
    chunks_df.to_parquet(chunks_file, index=False)
    print(f" - Saved chunks.parquet ({len(chunks_df)} rows)")
    
    # Manifest generation
    generate_chunk_manifest(chunks_df, tables["paragraphs"], output_dir, args)
    
    # Chunking summary
    generate_chunking_summary(chunks_df, paths.summaries_dir, args)
    
    print("Phase 2 complete.")

def validate_chunks(df: pd.DataFrame):
    if not df["chunk_id"].is_unique:
        raise ValueError("Duplicate chunk IDs found!")
    if df["text"].str.strip().eq("").any():
        raise ValueError("Empty chunks found!")
    if df["token_count"].max() > 512: # Soft limit check, should respect max_tokens
         print(f"Warning: Observed max token count {df['token_count'].max()}")

def generate_chunk_manifest(chunks_df: pd.DataFrame, paragraphs_df: pd.DataFrame, output_dir: Path, args):
    import hashlib
    
    def get_df_checksum(df):
        return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values).hexdigest()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "chunking_version": "1.0.0",
        "tokenizer_name": args.tokenizer_name,
        "max_tokens": args.max_tokens,
        "metadata_template_version": "1.0.0",
        "long_paragraph_overlap_tokens": args.overlap_tokens,
        "chunk_count": len(chunks_df),
        "paragraph_count": len(paragraphs_df),
        "chunk_checksum": get_df_checksum(chunks_df[["chunk_id", "text"]]),
        "source_paragraph_checksum": get_df_checksum(paragraphs_df[["paragraph_id", "text"]]),
        "created_with_python": sys.version
    }
    
    with open(output_dir / "chunk_manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)
    print(f"Manifest saved to {output_dir / 'chunk_manifest.json'}")

def generate_chunking_summary(chunks_df: pd.DataFrame, summary_dir: Path, args):
    summary = {
        "total_chunks": len(chunks_df),
        "chunks_per_split": chunks_df["split"].value_counts().to_dict(),
        "token_stats": chunks_df["token_count"].describe().to_dict(),
        "is_partial_stats": chunks_df["is_partial_paragraph"].value_counts().to_dict(),
        "chunking_config": {
            "tokenizer_name": args.tokenizer_name,
            "max_tokens": args.max_tokens,
            "overlap_tokens": args.overlap_tokens
        }
    }
    with open(summary_dir / "chunking_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Chunking summary saved to {summary_dir / 'chunking_summary.json'}")

def validate_existing_tables(output_dir: Path):
    table_names = ["papers", "sections", "paragraphs", "questions", "answers", "evidence", "figures_tables"]
    for name in table_names:
        file_path = output_dir / f"{name}.parquet"
        if not file_path.exists():
            print(f"Warning: {file_path} does not exist.")
            continue
        df = pd.read_parquet(file_path)
        validate_table(df, name)
        print(f" - {name}.parquet is valid.")

def validate_foreign_keys(tables: dict):
    # sections.paper_id -> papers.paper_id
    paper_ids = set(tables["papers"]["paper_id"])
    
    def check_fk(df, col, ref_set, df_name, ref_name):
        invalid = set(df[col]) - ref_set
        if invalid:
            raise ValueError(f"Foreign key violation: {df_name}.{col} contains values not in {ref_name}.paper_id: {list(invalid)[:5]}")

    check_fk(tables["sections"], "paper_id", paper_ids, "sections", "papers")
    check_fk(tables["paragraphs"], "paper_id", paper_ids, "paragraphs", "papers")
    check_fk(tables["questions"], "paper_id", paper_ids, "questions", "papers")
    check_fk(tables["answers"], "paper_id", paper_ids, "answers", "papers")
    check_fk(tables["figures_tables"], "paper_id", paper_ids, "figures_tables", "papers")
    
    # paragraphs.section_id -> sections.section_id
    section_ids = set(tables["sections"]["section_id"])
    check_fk(tables["paragraphs"], "section_id", section_ids, "paragraphs", "sections")
    
    # answers.question_id -> questions.question_id
    question_ids = set(tables["questions"]["question_id"])
    check_fk(tables["answers"], "question_id", question_ids, "answers", "questions")
    
    # evidence.answer_id -> answers.answer_id
    answer_ids = set(tables["answers"]["answer_id"])
    check_fk(tables["evidence"], "answer_id", answer_ids, "evidence", "answers")

def generate_schema_summary(tables: dict, summary_dir: Path):
    summary = {
        "schema_version": SCHEMA_VERSION,
        "table_counts": {name: len(df) for name, df in tables.items()},
        "split_counts": tables["papers"]["split"].value_counts().to_dict(),
        "answer_types": tables["answers"]["answer_type"].value_counts().to_dict(),
        "evidence_types": tables["evidence"]["evidence_type"].value_counts().to_dict(),
        "status": "validated"
    }
    
    summary_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_dir / "qasper_schema_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Summary saved to {summary_dir / 'qasper_schema_summary.json'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build processed QASPER data.")
    parser.add_argument("--project-root", type=str, help="Explicit project root path.")
    parser.add_argument("--raw-dir", type=str, help="Raw data directory.")
    parser.add_argument("--output-dir", type=str, help="Output directory for processed data.")
    parser.add_argument("--tokenizer-name", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Tokenizer name.")
    parser.add_argument("--max-tokens", type=int, default=384, help="Maximum tokens per chunk.")
    parser.add_argument("--overlap-tokens", type=int, default=32, help="Overlap tokens for split paragraphs.")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing processed tables.")
    
    args = parser.parse_args()
    build_processed_data(args)

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
from src.evidence_mapping import build_evidence_mappings
from src.data_validation import validate_foreign_keys

def build_processed_data(args):
    paths = get_project_paths(explicit_root=args.project_root)
    ensure_project_directories(paths)
    
    raw_dir = Path(args.raw_dir) if args.raw_dir else paths.raw_data_dir
    output_dir = Path(args.output_dir) if args.output_dir else paths.processed_data_dir
    
    if args.validate_only:
        print("Running validation only...")
        validate_existing_tables(output_dir)
        return

    if args.mapping_only:
        print("Building exact evidence mappings from existing processed sources...")
        names = ["papers", "sections", "paragraphs", "questions", "answers", "evidence", "figures_tables", "chunks"]
        tables = {name: pd.read_parquet(output_dir / f"{name}.parquet") for name in names}
        mappings_df = build_evidence_mappings(
            tables["evidence"], tables["paragraphs"], tables["sections"], tables["figures_tables"], tables["chunks"]
        )
        tables["evidence_mappings"] = mappings_df
        validate_table(mappings_df, "evidence_mappings")
        validate_foreign_keys(tables)
        mappings_df.to_parquet(output_dir / "evidence_mappings.parquet", index=False)
        generate_evidence_mapping_summary(mappings_df, tables["evidence"], paths.summaries_dir)
        print(f" - Saved evidence_mappings.parquet ({len(mappings_df)} rows)")
        return

    print(f"Loading raw QASPER data from {raw_dir}...")
    ds = load_qasper_dataset(raw_dir)
    
    print("Building schema tables (Phase 1)...")
    tables = process_qasper_to_tables(ds)
    
    # Staging validation
    print("Validating tables...")
    for name, df in tables.items():
        validate_table(df, name)
    
    # Cross-table foreign key validation before derived tables are built.
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
        overlap_tokens=args.overlap_tokens,
        require_hf=True,
    )
    
    print("Validating chunks...")
    validate_chunks(chunks_df)
    tables["chunks"] = chunks_df

    print("Building exact evidence mappings (T05)...")
    mappings_df = build_evidence_mappings(
        tables["evidence"], tables["paragraphs"], tables["sections"], tables["figures_tables"], chunks_df
    )
    tables["evidence_mappings"] = mappings_df
    validate_table(mappings_df, "evidence_mappings")
    validate_foreign_keys(tables)
    
    # Save chunks
    chunks_file = output_dir / "chunks.parquet"
    chunks_df.to_parquet(chunks_file, index=False)
    print(f" - Saved chunks.parquet ({len(chunks_df)} rows)")
    mappings_file = output_dir / "evidence_mappings.parquet"
    mappings_df.to_parquet(mappings_file, index=False)
    print(f" - Saved evidence_mappings.parquet ({len(mappings_df)} rows)")
    
    # Manifest generation
    generate_chunk_manifest(chunks_df, tables["paragraphs"], output_dir, args)
    
    # Chunking summary
    generate_chunking_summary(chunks_df, paths.summaries_dir, args)
    generate_evidence_mapping_summary(mappings_df, tables["evidence"], paths.summaries_dir)
    
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
    table_names = ["papers", "sections", "paragraphs", "questions", "answers", "evidence", "figures_tables", "chunks", "evidence_mappings"]
    tables = {}
    for name in table_names:
        file_path = output_dir / f"{name}.parquet"
        if not file_path.exists():
            print(f"Warning: {file_path} does not exist.")
            continue
        df = pd.read_parquet(file_path)
        if name != "chunks":
            validate_table(df, name)
        else:
            validate_chunks(df)
        tables[name] = df
        print(f" - {name}.parquet is valid.")
    validate_foreign_keys(tables)

def generate_evidence_mapping_summary(mappings_df, evidence_df, summary_dir):
    outcomes = mappings_df[["evidence_id", "mapping_status"]].drop_duplicates()
    observed_counts = outcomes["mapping_status"].value_counts().to_dict()
    counts = {status: int(observed_counts.get(status, 0)) for status in ("matched", "ambiguous", "unmatched")}
    total = len(evidence_df)
    summary = {
        "total_evidence": total,
        "status_counts": counts,
        "matched_coverage": counts.get("matched", 0) / total if total else 1.0,
        "candidate_mapping_coverage": (total - counts.get("unmatched", 0)) / total if total else 1.0,
        "mapping_rows": len(mappings_df),
        "source_type_rows": mappings_df["source_type"].value_counts().to_dict(),
    }
    with open(summary_dir / "evidence_mapping_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

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
    parser.add_argument("--mapping-only", action="store_true", help="Build T05 mappings from existing processed tables.")
    
    args = parser.parse_args()
    build_processed_data(args)

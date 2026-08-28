# Evidence-Aware Multi-Agent RAG for Scientific Literature Review

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raiyanhossainaraf-dagger/CSE427Labproject_S3_G10/blob/main/CSE427_Final_Project.ipynb)

## Project overview and research problem

This CSE427 project studies evidence-aware retrieval-augmented generation (RAG) over QASPER scientific papers. Scientific QA needs useful answers and inspectable support; we test structure-aware retrieval, explicit evidence selection, and bounded critique without overstating answer-quality gains.

## Five-agent architecture

`Question → Query → Retrieval → Evidence → Answer → Critic → final answer`

- **Query Agent** normalizes the question and response style.
- **Retrieval Agent** combines paper-scoped BM25 and BGE dense rankings using weighted reciprocal-rank fusion.
- **Evidence Agent** cross-encoder reranks and labels five sources E1–E5 using the frozen evidence-fusion rule.
- **Answer Agent** generates from selected evidence and returns citation labels.
- **Critic Agent** checks citation structure and grounding, with at most one revision.

## Installation and Colab

Python 3.10+ and CUDA are recommended. The exact requested models are used; loading failures are reported rather than silently replaced.

```bash
git clone https://github.com/raiyanhossainaraf-dagger/CSE427Labproject_S3_G10.git
cd CSE427Labproject_S3_G10
python -m venv .venv
python -m pip install -r requirements-dev.txt -r requirements-llm.txt
```

For Colab, open the badge and run all cells. The notebook installs `requirements-colab.txt` and `requirements-llm.txt`.

## Frozen models and settings

- Embeddings: `BAAI/bge-small-en-v1.5`
- Reranking: `cross-encoder/ms-marco-MiniLM-L6-v2`
- Generation: `Qwen/Qwen2.5-1.5B-Instruct`
- Seed 427; greedy generation; `max_new_tokens=256`; batch size 1; CUDA FP16 for Qwen when CUDA is available.

## Dataset and artifact policy

QASPER source and processed tabular data are retained for reproducibility. Live inference reads validation question metadata only and does not load gold answers or evidence. The test split is not evaluated. Generated embeddings, FAISS indexes, model weights, caches, and executed-notebook artifacts are ignored; committed aggregate CSV/JSON results and predictions support auditing.

## Local execution and main commands

```bash
python scripts/build_processed_data.py --validate-only
python scripts/build_retrieval_corpus.py
python scripts/build_dense_index.py --device cuda
python scripts/run_multi_agent_demo.py --mode real --limit 5 --device cuda
python scripts/evaluate_retrieval.py --smoke
python scripts/validate_submission.py
python -m pytest -q
```

The final notebook defaults to `QUICK_DEMO=True`, `RUN_FULL_GENERATION=False`, `REBUILD_INDEX_IF_MISSING=True`, and `DEMO_QUESTION_COUNT=3`. Enabling the 100-question generation run is optional and expensive; committed results are displayed by default.

## Verified results

Full-validation retrieval covers 927 evidence-eligible validation questions:

| Method | Recall@5 | nDCG@5 |
|---|---:|---:|
| Dense | 0.5754 | 0.4312 |
| Hybrid RRF | 0.5965 | 0.4522 |
| Evidence-fused reranking | 0.6580 | 0.5010 |

Generation uses a separate fixed deterministic 100-question validation sample:

| Configuration | Answer F1 | Citation-label validity | Insufficient |
|---|---:|---:|---:|
| Dense single-agent | 0.2954 | 100% | 5 |
| Evidence-aware | 0.2695 | 100% | 4 |
| Full multi-agent | 0.2707 | 100% | 1 |

Citation-label validity is structural—it confirms labels refer to selected evidence, not that citations are semantically correct.

## Statistical interpretation

Full minus dense Answer F1 is −2.47 percentage points, with 20/50/30 wins/ties/losses. The bootstrap 95% CI is −8.41 to +3.49 points and paired permutation p = 0.419.

> The full multi-agent system produced statistically comparable Answer F1 to the dense single-agent baseline on the fixed 100-question sample, while achieving higher evidence retrieval Recall@5, fewer insufficient responses, traceable evidence selection, and validated citation labels.

No retrieval significance is claimed because no paired retrieval significance test was reported.

## Limitations

- The 100-question result is sampled validation generation, not full-validation generation.
- Full-system Answer F1 is not higher than the dense baseline on that sample.
- Structural citation validity is not semantic entailment or factuality.
- QASPER is domain-specific; real generation and index building require substantial compute, and CPU execution is slow.

## Repository structure

```text
CSE427_Final_Project.ipynb  final Colab workflow and results
src/                        reusable processing, retrieval, agent, and evaluation modules
scripts/                    builders, experiments, diagnostics, submission validation
tests/                      unit, integration, and submission tests
data/                       QASPER source and processed metadata
outputs/                    committed predictions, tables, summaries
figures/                    final academic result charts
report/final_project_report.md
```

## Reproducibility

Use seed 427 and the frozen settings above. In a fresh clone, install dependencies, build the corpus and dense index, and run the notebook or small demo. Load committed files under `outputs/tables/` for verified full results; do not rerun generation merely to view them. Before submission run the validator and complete test suite.

## License

Academic project for CSE427.

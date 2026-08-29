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

Python 3.10+ is required. The final Colab demonstration requires CUDA and `transformers>=4.51.0`; loading failures are reported rather than silently replaced.

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
- Backend compatibility default and original baseline: `Qwen/Qwen2.5-1.5B-Instruct`
- Recommended final Colab generator: `Qwen/Qwen3-4B-Instruct-2507`
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

The final notebook defaults to `QUICK_DEMO=True`, `RUN_FULL_GENERATION=False`, `REBUILD_INDEX_IF_MISSING=True`, and `DEMO_QUESTION_COUNT=3`. It explicitly uses Qwen3 for the small live CUDA demonstration and only displays the committed 100-question results. Full ablation reproduction is isolated in `CSE427_LLM_Ablation.ipynb`.

## Verified results

Full-validation retrieval covers 927 evidence-eligible validation questions:

| Method | Recall@5 | nDCG@5 |
|---|---:|---:|
| Dense | 0.5754 | 0.4312 |
| Hybrid RRF | 0.5965 | 0.4522 |
| Evidence-fused reranking | 0.6580 | 0.5010 |

Generation uses a separate fixed deterministic 100-question validation sample:

| Model / configuration | Answer F1 | Citation-label validity | Runtime/question | Outcome counts |
|---|---:|---:|---:|---|
| Qwen2.5 dense single-agent (original baseline) | 0.2954 | 100% | 1.8384 s | 5 insufficient |
| Qwen2.5 evidence-aware (original diagnostic) | 0.2695 | 100% | — | 4 insufficient |
| Qwen2.5 full multi-agent (original baseline) | 0.2707 | 100% | 3.2485 s | 1 insufficient |
| Qwen3 dense single-agent | 0.3199 | 100% | 5.8316 s | 100 accepted |
| Qwen3 full multi-agent | **0.4082** | 100% | 9.8402 s | 89 first-attempt, 4 revised, 7 rejected |

Citation-label validity is structural—it confirms labels refer to selected evidence, not that citations are semantically correct.

## Statistical interpretation

On the fixed 100-question validation sample, the Qwen3 full multi-agent system achieved 40.82% Answer F1 compared with 31.99% for the Qwen3 dense single-agent baseline. The paired difference was +8.84 percentage points, with a 95% bootstrap confidence interval of +3.29 to +14.95 points and p=0.0019.

This does not mean every multi-agent/model combination improves F1: the preserved Qwen2.5 full result (0.2707) is below its dense baseline (0.2954). The exploratory ablation indicates that the stronger Qwen3 generator was better able to use the evidence-aware pipeline. Qwen3 full also exceeded Qwen2.5 full by 13.75 points (95% CI +4.96 to +22.48, p=0.0020); Qwen3 dense exceeded Qwen2.5 dense by 2.44 points, but its CI includes zero (p=0.5098).

No retrieval significance is claimed because no paired retrieval significance test was reported.

## Limitations

- This is an exploratory validation ablation, not test-set confirmation, and it uses only 100 validation questions; the test split remains untouched.
- Structural citation validity is not semantic entailment or factuality.
- Seven Qwen3 full-system records were rejected.
- Qwen3 full is slower (9.8402 s/question) and both Qwen3 configurations used approximately 9 GB peak GPU memory. Local 8 GB FP16 execution is not recommended; use a suitable Colab CUDA GPU.
- QASPER is domain-specific, and Answer F1 does not capture every aspect of answer quality.

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

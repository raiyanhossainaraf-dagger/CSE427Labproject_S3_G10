# Evidence-Aware Multi-Agent Retrieval-Augmented Generation for Scientific Literature Review

## Abstract

This project evaluates evidence-aware multi-agent RAG over QASPER scientific papers. It unifies paragraph, section, and figure/table sources; combines BM25 and dense retrieval; applies cross-encoder evidence reranking; and coordinates five traceable agents. On 927 evidence-eligible validation questions, evidence-fused reranking achieved Recall@5 0.6580 and nDCG@5 0.5010. In an exploratory frozen-evidence ablation on 100 validation questions, Qwen3 full multi-agent achieved Answer F1 0.4082 versus 0.3199 for Qwen3 dense single-agent. The original Qwen2.5 results remain the baseline.

## Introduction

Scientific QA must locate relevant material across long papers and show what supports an answer. Dense RAG can hide retrieval failures and provide little visibility into evidence choice. This work adds structure-aware retrieval and explicit agent contracts so answers can be inspected through E1–E5 evidence, source metadata, citations, and public statuses.

## Research questions

1. Does hybrid and evidence-aware ranking improve evidence retrieval over dense retrieval?
2. How does the five-agent workflow compare with dense single-agent and evidence-aware configurations on Answer F1?
3. What transparency and insufficiency behavior follows from explicit evidence selection and critique?

## Dataset

QASPER supplies questions, answers, and supporting evidence over NLP papers. The validated schema contains 1,585 papers, 23,494 sections, 83,087 paragraphs, 5,049 questions, 7,993 answers, 12,761 evidence items, and 11,364 figures/tables. Retrieval uses 927 evidence-eligible validation questions; 78 without eligible mapped evidence are excluded by the evaluation protocol. Generation uses a separate deterministic sample of exactly 100 validation questions drawn with seed 427. No test-split evaluation is reported.

## Methodology

The pipeline separates preprocessing, inference, and gold evaluation. Stable identifiers connect papers, structure, chunks, questions, annotations, and figures/tables. Inference reads question metadata and corpus text. Official gold answers are accessed only after prediction files exist at the evaluation boundary.

## Evidence mapping

Evidence annotations map to paragraph/chunk, section, or figure/table records. Of 12,761 items, 12,732 are uniquely matched and 29 ambiguous; none is unmatched. Candidate mapping coverage is 100% and matched coverage is 99.77%. Ambiguity stays explicit rather than being silently discarded.

## Paragraph-aware chunking

The processed data contains 39,398 chunks with a 384-token maximum and 32-token overlap. Paragraph and section identifiers preserve provenance instead of treating papers as undifferentiated fixed-width text.

## Retrieval and reranking

The corpus unifies paragraph chunks, sections, and figures/tables. BM25 supplies lexical retrieval. Dense retrieval uses `BAAI/bge-small-en-v1.5`, normalized embeddings, and exact inner-product FAISS search. Weighted reciprocal-rank fusion combines BM25 and dense candidates. The Evidence Agent uses `cross-encoder/ms-marco-MiniLM-L6-v2` and the frozen evidence fusion rule to select five labeled sources. Generated embeddings and indexes are ignored and rebuilt with repository scripts.

## Multi-agent architecture

The Query Agent normalizes questions and identifies response style. Retrieval obtains paper-scoped candidates. Evidence reranks and labels E1–E5. The original baseline prompts `Qwen/Qwen2.5-1.5B-Instruct`; the final recommended Colab demonstration explicitly uses `Qwen/Qwen3-4B-Instruct-2507`. Critic applies deterministic checks and bounded model review with at most one revision. Public traces record statuses and decisions without exposing hidden reasoning.

## Experimental setup

Experiments use seed 427. Qwen generation is greedy (`do_sample=False`), `max_new_tokens=256`, batch size 1, and CUDA FP16. Qwen3 requires `transformers>=4.51.0`; the final Colab workflow requires CUDA. Full-validation retrieval and 100-question generation are distinct scopes. The final notebook loads committed results instead of rerunning the expensive generation experiment; full reproduction is directed to `CSE427_LLM_Ablation.ipynb`.

## Results

### Full-validation retrieval (927 evidence-eligible questions)

| Method | Recall@5 | nDCG@5 |
|---|---:|---:|
| Dense | 0.5754 | 0.4312 |
| Hybrid RRF | 0.5965 | 0.4522 |
| Evidence-fused | 0.6580 | 0.5010 |

### Fixed 100-question validation generation sample

| Model / configuration | Answer F1 | Citation-label validity | Runtime/question |
|---|---:|---:|---:|
| Qwen2.5 dense single-agent | 0.2954 | 100% | 1.8384 s |
| Qwen2.5 evidence-aware diagnostic | 0.2695 | 100% | — |
| Qwen2.5 full multi-agent | 0.2707 | 100% | 3.2485 s |
| Qwen3 dense single-agent | 0.3199 | 100% | 5.8316 s |
| Qwen3 full multi-agent | **0.4082** | 100% | 9.8402 s |

Citation-label validity only verifies that emitted labels reference selected evidence; it does not establish semantic support.

## Statistical diagnostic

On the fixed 100-question validation sample, the Qwen3 full multi-agent system achieved 40.82% Answer F1 compared with 31.99% for the Qwen3 dense single-agent baseline. The paired difference was +8.84 percentage points, with a 95% bootstrap confidence interval of +3.29 to +14.95 points and p=0.0019. Wins/ties/losses were 40/36/24.

Qwen3 full minus Qwen2.5 full was +13.75 points (95% CI +4.96 to +22.48; p=0.0020). Qwen3 dense minus Qwen2.5 dense was +2.44 points, but its interval includes zero (p=0.5098).

## Discussion

The results do not show that every multi-agent/model combination improves F1: preserved Qwen2.5 full (0.2707) is below Qwen2.5 dense (0.2954). The stronger Qwen3 generator was better able to use the evidence-aware pipeline on this fixed exploratory sample. Separately, evidence-fused ranking improves Recall@5 on full-validation retrieval.

## Limitations

This is an exploratory validation ablation, not test-set confirmation, and covers only 100 validation questions. Citation validity is structural rather than semantic. Seven Qwen3 full-system records were rejected. Qwen3 full takes 9.8402 seconds/question, and both Qwen3 configurations peak at approximately 9 GB GPU memory; local 8 GB FP16 execution is not recommended. QASPER is domain-specific, Answer F1 is incomplete, and retrieval improvements lack a paired significance test. The test split remains untouched.

## Ethical and reproducibility considerations

Generated answers are not authoritative scientific conclusions and should be checked against source text. Public traces avoid hidden reasoning while retaining audit fields. Reproducibility uses frozen models and parameters, seed 427, committed aggregates, validation-only evaluation, and rebuildable ignored retrieval artifacts. Inference predictions contain no gold fields, secrets, model weights, or caches.

## Conclusion

Evidence-aware reranking improves evidence retrieval and the five-agent design provides traceability. The Qwen3 ablation supports higher full-system Answer F1 than its Qwen3 dense baseline on this fixed validation sample, while the Qwen2.5 baseline cautions against generalizing that result to every generator or multi-agent combination.

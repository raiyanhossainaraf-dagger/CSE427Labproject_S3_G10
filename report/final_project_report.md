# Evidence-Aware Multi-Agent Retrieval-Augmented Generation for Scientific Literature Review

## Abstract

This project evaluates evidence-aware multi-agent RAG over QASPER scientific papers. It unifies paragraph, section, and figure/table sources; combines BM25 and dense retrieval; applies cross-encoder evidence reranking; and coordinates five traceable agents. On 927 evidence-eligible validation questions, evidence-fused reranking achieved Recall@5 0.6580 and nDCG@5 0.5010. On a separate fixed 100-question validation generation sample, the full system achieved Answer F1 0.2707 versus 0.2954 for the dense baseline, with a paired interval crossing zero. The demonstrated benefits are stronger evidence retrieval, fewer insufficient responses, and inspectable traces—not higher Answer F1.

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

The Query Agent normalizes questions and identifies response style. Retrieval obtains paper-scoped candidates. Evidence reranks and labels E1–E5. Answer prompts `Qwen/Qwen2.5-1.5B-Instruct` using selected evidence. Critic applies deterministic checks and bounded model review with at most one revision. Public traces record statuses and decisions without exposing hidden reasoning.

## Experimental setup

Experiments use seed 427. Qwen generation is greedy (`do_sample=False`), `max_new_tokens=256`, batch size 1, and CUDA FP16 where CUDA is available. Full-validation retrieval and 100-question generation are distinct scopes. The final notebook loads committed results instead of rerunning the expensive generation experiment.

## Results

### Full-validation retrieval (927 evidence-eligible questions)

| Method | Recall@5 | nDCG@5 |
|---|---:|---:|
| Dense | 0.5754 | 0.4312 |
| Hybrid RRF | 0.5965 | 0.4522 |
| Evidence-fused | 0.6580 | 0.5010 |

### Fixed 100-question validation generation sample

| Configuration | Answer F1 | Citation-label validity | Insufficient |
|---|---:|---:|---:|
| Dense single-agent | 0.2954 | 100% | 5 |
| Evidence-aware | 0.2695 | 100% | 4 |
| Full multi-agent | 0.2707 | 100% | 1 |

Citation-label validity only verifies that emitted labels reference selected evidence; it does not establish semantic support.

## Statistical diagnostic

Full minus dense Answer F1 is −2.47 percentage points with 20 wins, 50 ties, and 30 losses. A 10,000-sample bootstrap gives a 95% CI of −8.41 to +3.49 points; a 10,000-sample paired permutation diagnostic gives p = 0.419. The interval includes zero and the diagnostic is not statistically significant.

## Discussion

The full system does not improve mean Answer F1 over the dense baseline on the fixed sample. It reduces insufficient responses from five to one and provides inspectable evidence and citation traces. Separately, evidence-fused ranking improves Recall@5 on full-validation retrieval.

The full multi-agent system produced statistically comparable Answer F1 to the dense single-agent baseline on the fixed 100-question sample, while achieving higher evidence retrieval Recall@5, fewer insufficient responses, traceable evidence selection, and validated citation labels.

## Limitations

The generation sample is not full-validation generation. QASPER is domain-specific. Answer F1 is incomplete, and citation-label validity is structural. Retrieval improvements lack a paired significance test, so no retrieval significance claim is made. Model downloads, dense indexing, and generation require substantial compute.

## Ethical and reproducibility considerations

Generated answers are not authoritative scientific conclusions and should be checked against source text. Public traces avoid hidden reasoning while retaining audit fields. Reproducibility uses frozen models and parameters, seed 427, committed aggregates, validation-only evaluation, and rebuildable ignored retrieval artifacts. Inference predictions contain no gold fields, secrets, model weights, or caches.

## Conclusion

Evidence-aware reranking improves evidence retrieval and the five-agent design yields fewer insufficient outputs and greater traceability. The generation evidence does not justify claiming higher Answer F1; it supports statistical comparability on the fixed sample alongside evidence-access benefits.

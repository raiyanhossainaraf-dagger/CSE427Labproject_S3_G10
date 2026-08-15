# Milestone 1 Report Notes

## Introduction

The rapid growth of scientific literature presents a significant challenge for researchers who must keep pace with the latest developments. Manually reviewing large volumes of research is time-consuming and prone to oversight. While Large Language Models (LLMs) offer a potential solution for automating literature reviews, they suffer from the "hallucination" problem, where they may generate claims that are unsupported by the source material.

Retrieval-Augmented Generation (RAG) addresses this by grounding LLM generation in retrieved documents. However, the quality of a RAG-based review depends heavily on the accuracy of retrieval and the proper attribution of evidence. This project proposes an **Evidence-Aware Multi-Agent Retrieval-Augmented Generation** system. By employing specialized agents for retrieval, evidence verification, and generation, we aim to produce reliable, grounded scientific literature reviews.

## Problem Statement

Current LLM-based scientific review generation systems can produce fluent and professional-sounding text that nonetheless contains factual inaccuracies. These systems often make claims that are unsupported by the cited research, weakly supported, or incorrectly attributed to the wrong source. While standard RAG systems attempt to ground generation in retrieved documents, their performance is limited by the quality of the initial retrieval and the lack of explicit evidence verification. There is a critical need for a more robust framework that not only retrieves relevant documents but also rigorously verifies that every claim in the generated review is backed by explicit evidence from the research papers.

## Project Objectives

1.  **Scientific Document Retrieval:** Develop a high-precision retrieval system capable of identifying relevant paragraphs within long scientific papers.
2.  **Evidence Extraction:** Implement methods to extract explicit supporting evidence for scientific claims.
3.  **Multi-Agent Architecture:** Design a multi-agent framework where specialized agents collaborate on retrieval, verification, and synthesis.
4.  **Evidence Verification:** Create a verification mechanism to cross-reference generated claims against extracted evidence labels.
5.  **Literature-Review Generation:** Produce coherent and grounded literature reviews that cite verifiable sources.
6.  **Comparison against Baselines:** Evaluate the system against standard RAG baselines to quantify improvements in grounding and factual accuracy.
7.  **Quantitative Evaluation:** Use metrics like ROUGE, BERTScore, and evidence-match ratios to assess system performance.

## Dataset Identified

### Primary Dataset: QASPER
- **Source:** Allen Institute for AI (Hugging Face / Official Website)
- **Format:** Nested JSON with paper sections, questions, answers, and evidence annotations.
- **Description:** QASPER is a dataset for Question Answering on Scientific Papers. It contains over 5,000 questions grounded in 1,585 NLP papers.
- **Selection Reason:** Unlike standard QA datasets, QASPER provides evidence annotations (highlighted paragraphs) that show exactly where the answer comes from. This is crucial for our evidence-aware RAG system.
- **Important Fields:** `id`, `title`, `abstract`, `full_text` (sections and paragraphs), `qas` (questions and answers with evidence).

### EDA Statistics (to be filled from notebook)
- **Total Papers:** [Placeholder]
- **Total Questions:** [Placeholder]
- **Total Evidence Items:** [Placeholder]
- **Average Paper Length:** [Placeholder] words.

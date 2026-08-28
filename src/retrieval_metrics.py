"""Evidence-aware retrieval evaluation over exact T05 source mappings.

Metrics are macro-averaged over eligible questions. A question is eligible when at
least one answer annotation has one or more mapped evidence units. The default
``best`` reference policy scores every annotation independently at each cutoff and
selects the annotation with the highest (Evidence F1, Recall, AP), breaking ties by
annotation ID. ``union`` merges evidence units from all eligible annotations and is
available as a diagnostic.

At cutoff k: Hit Rate is one when any evidence unit is matched; Precision is matched
units/k; Recall is matched units/all gold units; Evidence F1 is their harmonic mean;
MRR is the reciprocal rank of the first match; AP is the mean precision at relevant
ranks divided by min(k, gold count); and nDCG is binary-gain DCG divided by ideal DCG.
Each ranked prediction and each evidence unit can participate in at most one match.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd

DEFAULT_K_VALUES = (1, 3, 5, 10, 20)
METRIC_NAMES = ("hit_rate", "precision", "recall", "evidence_f1", "mrr", "map", "ndcg")
SourceKey = Tuple[str, str]


def validate_predictions(predictions: pd.DataFrame) -> None:
    """Validate input columns and rank values; duplicate/gapped ranks are normalized later."""
    if predictions.empty:
        return
    required = {"question_id", "paper_id", "rank"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions missing required columns: {sorted(missing)}")
    source_columns = {"chunk_id", "paragraph_id", "section_id", "float_id"}
    if not source_columns & set(predictions.columns):
        raise ValueError(f"Predictions require at least one source column: {sorted(source_columns)}")
    ranks = pd.to_numeric(predictions["rank"], errors="coerce")
    if ranks.isna().any() or (ranks < 1).any() or (ranks % 1 != 0).any():
        raise ValueError("Prediction ranks must be positive integers starting at 1")


def normalize_predictions(predictions: pd.DataFrame, chunks: pd.DataFrame) -> pd.DataFrame:
    """Resolve chunk sources, remove duplicate results, and assign deterministic ranks from 1."""
    validate_predictions(predictions)
    columns = ["question_id", "paper_id", "rank", "retrieval_score", "source_keys"]
    if predictions.empty:
        return pd.DataFrame(columns=columns)

    chunk_lookup = {str(row.chunk_id): row for row in chunks.itertuples(index=False)}
    rows = []
    score_column = "retrieval_score" if "retrieval_score" in predictions else ("score" if "score" in predictions else None)
    frame = predictions.copy()
    frame["_input_order"] = np.arange(len(frame))
    frame["_rank"] = pd.to_numeric(frame["rank"]).astype(int)
    frame["_score"] = pd.to_numeric(frame[score_column], errors="coerce").fillna(float("-inf")) if score_column else 0.0
    tie_columns = [column for column in ("chunk_id", "paragraph_id", "section_id", "float_id") if column in frame]
    frame["_source_tie_key"] = frame[tie_columns].fillna("").astype(str).agg("|".join, axis=1) if tie_columns else ""
    frame = frame.sort_values(
        ["question_id", "_rank", "_score", "_source_tie_key", "paper_id", "_input_order"],
        ascending=[True, True, False, True, True, True], kind="stable"
    )

    for question_id, group in frame.groupby("question_id", sort=True):
        seen = set()
        normalized_rank = 0
        for row in group.to_dict("records"):
            paper_id = str(row["paper_id"])
            keys: Set[SourceKey] = set()
            chunk_id = _record_value(row, "chunk_id")
            if chunk_id:
                if chunk_id not in chunk_lookup:
                    raise ValueError(f"Unknown prediction chunk_id: {chunk_id}")
                chunk = chunk_lookup[chunk_id]
                if str(chunk.paper_id) != paper_id:
                    raise ValueError(f"Prediction paper_id disagrees with chunk {chunk_id}")
                keys.add(("section", str(chunk.section_id)))
                paragraph_ids = chunk.paragraph_ids
                if isinstance(paragraph_ids, str):
                    paragraph_ids = [paragraph_ids]
                keys.update(("paragraph", str(pid)) for pid in paragraph_ids)
            for source_type, column in (("paragraph", "paragraph_id"), ("section", "section_id"), ("float", "float_id")):
                source_id = _record_value(row, column)
                if source_id:
                    keys.add((source_type, source_id))
            identity = (paper_id, chunk_id or "", tuple(sorted(keys)))
            if identity in seen:
                continue
            seen.add(identity)
            normalized_rank += 1
            rows.append({"question_id": str(question_id), "paper_id": paper_id, "rank": normalized_rank,
                         "retrieval_score": None if row["_score"] == float("-inf") else float(row["_score"]),
                         "source_keys": frozenset(keys)})
    return pd.DataFrame(rows, columns=columns)


def build_annotation_gold(
    questions: pd.DataFrame, answers: pd.DataFrame, evidence: pd.DataFrame, mappings: pd.DataFrame
) -> Tuple[Dict[str, Dict[str, Dict[str, Set[SourceKey]]]], Dict[str, str], Dict[str, str]]:
    """Build question -> annotation -> evidence-unit -> alternative exact source keys."""
    question_papers = dict(zip(questions.question_id.astype(str), questions.paper_id.astype(str)))
    question_splits = dict(zip(questions.question_id.astype(str), questions.split.astype(str)))
    answer_question = dict(zip(answers.annotation_id.astype(str), answers.question_id.astype(str)))
    evidence_annotation = dict(zip(evidence.evidence_id.astype(str), evidence.annotation_id.astype(str)))
    gold: Dict[str, Dict[str, Dict[str, Set[SourceKey]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    for row in mappings.itertuples(index=False):
        evidence_id = str(row.evidence_id)
        annotation_id = evidence_annotation.get(evidence_id)
        question_id = answer_question.get(annotation_id or "")
        if not question_id or str(row.paper_id) != question_papers.get(question_id):
            continue
        source_id = {"paragraph": row.paragraph_id, "section": row.section_id, "float": row.float_id}.get(row.source_type, "")
        if source_id:
            gold[question_id][annotation_id][evidence_id].add((str(row.source_type), str(source_id)))
    return {q: {a: dict(units) for a, units in refs.items()} for q, refs in gold.items()}, question_papers, question_splits


def evaluate_retrieval(
    predictions: pd.DataFrame,
    questions: pd.DataFrame,
    answers: pd.DataFrame,
    evidence: pd.DataFrame,
    evidence_mappings: pd.DataFrame,
    chunks: pd.DataFrame,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    reference_policy: str = "best",
    include_union: bool = True,
    split: str | None = None,
) -> Dict[str, object]:
    """Evaluate ranked retrieval with macro averages and explicit eligibility counts."""
    if reference_policy not in {"best", "union"}:
        raise ValueError("reference_policy must be 'best' or 'union'")
    ks = tuple(sorted(set(int(k) for k in k_values)))
    if not ks or ks[0] < 1:
        raise ValueError("k_values must contain positive integers")
    split_values = set(questions["split"].dropna().astype(str))
    if split is None:
        if len(split_values) != 1:
            raise ValueError("Evaluate exactly one split at a time or pass split=...")
        split = next(iter(split_values))
    questions = questions[questions.split.astype(str).eq(str(split))].copy()
    question_ids = set(questions.question_id.astype(str))
    answers = answers[answers.question_id.astype(str).isin(question_ids) & answers.split.astype(str).eq(str(split))]
    evidence = evidence[evidence.question_id.astype(str).isin(question_ids) & evidence.split.astype(str).eq(str(split))]
    evidence_mappings = evidence_mappings[
        evidence_mappings.evidence_id.astype(str).isin(set(evidence.evidence_id.astype(str)))
        & evidence_mappings.split.astype(str).eq(str(split))
    ]
    if not predictions.empty and "split" in predictions and set(predictions.split.dropna().astype(str)) - {str(split)}:
        raise ValueError("Predictions contain rows from another split")
    if not predictions.empty:
        predictions = predictions[predictions.question_id.astype(str).isin(question_ids)]

    normalized = normalize_predictions(predictions, chunks[chunks.split.astype(str).eq(str(split))])
    gold, paper_by_question, _ = build_annotation_gold(questions, answers, evidence, evidence_mappings)
    prediction_groups = {qid: group for qid, group in normalized.groupby("question_id", sort=False)}
    aggregate = {k: {name: [] for name in METRIC_NAMES} for k in ks}
    union_aggregate = {k: {name: [] for name in METRIC_NAMES} for k in ks}
    evaluated = 0
    excluded = 0

    for question_id in sorted(question_ids):
        references = {aid: units for aid, units in gold.get(question_id, {}).items() if units}
        if not references:
            excluded += 1
            continue
        evaluated += 1
        pred_group = prediction_groups.get(question_id, pd.DataFrame(columns=normalized.columns))
        pred_sources = [row.source_keys if str(row.paper_id) == paper_by_question[question_id] else frozenset()
                        for row in pred_group.itertuples(index=False)]
        for k in ks:
            reference_scores = [(aid, _score_reference(pred_sources, units, k)) for aid, units in sorted(references.items())]
            if reference_policy == "best":
                _, selected = max(reference_scores, key=lambda item: (
                    item[1]["evidence_f1"], item[1]["recall"], item[1]["map"]
                ))
            else:
                selected = _score_reference(pred_sources, _union_units(references), k)
            for name in METRIC_NAMES:
                aggregate[k][name].append(selected[name])
            if include_union:
                union_score = _score_reference(pred_sources, _union_units(references), k)
                for name in METRIC_NAMES:
                    union_aggregate[k][name].append(union_score[name])

    result = {"split": str(split), "reference_policy": reference_policy, "evaluated_questions": evaluated,
              "excluded_questions": excluded, "k_values": list(ks), "metrics": _macro(aggregate)}
    if include_union:
        result["union_diagnostic"] = _macro(union_aggregate)
    return result


def _score_reference(predictions: List[Set[SourceKey]], units: Mapping[str, Set[SourceKey]], k: int) -> Dict[str, float]:
    relevant = _ranked_relevance(predictions[:k], units)
    hits = sum(relevant)
    gold_count = len(units)
    precision = hits / k
    recall = hits / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mrr = next((1.0 / rank for rank, hit in enumerate(relevant, 1) if hit), 0.0)
    ap = sum((sum(relevant[:rank]) / rank) for rank, hit in enumerate(relevant, 1) if hit) / min(k, gold_count)
    dcg = sum(hit / np.log2(rank + 1) for rank, hit in enumerate(relevant, 1))
    ideal = sum(1.0 / np.log2(rank + 1) for rank in range(1, min(k, gold_count) + 1))
    return {"hit_rate": float(hits > 0), "precision": precision, "recall": recall, "evidence_f1": f1,
            "mrr": mrr, "map": ap, "ndcg": dcg / ideal if ideal else 0.0}


def _ranked_relevance(predictions: List[Set[SourceKey]], units: Mapping[str, Set[SourceKey]]) -> List[int]:
    unit_ids = sorted(units)
    edges = [[unit_id for unit_id in unit_ids if sources & units[unit_id]] for sources in predictions]
    unit_to_prediction: Dict[str, int] = {}
    relevance = []
    for prediction_index in range(len(predictions)):
        visited = set()
        def augment(index: int) -> bool:
            for unit_id in edges[index]:
                if unit_id in visited:
                    continue
                visited.add(unit_id)
                if unit_id not in unit_to_prediction or augment(unit_to_prediction[unit_id]):
                    unit_to_prediction[unit_id] = index
                    return True
            return False
        relevance.append(int(augment(prediction_index)))
    return relevance


def _union_units(references: Mapping[str, Mapping[str, Set[SourceKey]]]) -> Dict[str, Set[SourceKey]]:
    return {f"{annotation_id}|{evidence_id}": candidates for annotation_id, units in references.items()
            for evidence_id, candidates in units.items()}


def _macro(values):
    return {str(k): {name: float(np.mean(scores)) if scores else 0.0 for name, scores in metrics.items()}
            for k, metrics in values.items()}


def _record_value(row: Mapping[str, object], column: str) -> str:
    value = row.get(column, "")
    return "" if value is None or (not isinstance(value, str) and pd.isna(value)) else str(value)

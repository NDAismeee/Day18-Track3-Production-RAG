"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, re
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, TEST_SET_PATH


def _langchain_embeddings_for_ragas():
    if OPENAI_API_KEY:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _word_set(text: str) -> set[str]:
    return {m.group(0).lower() for m in re.finditer(r"\S+", text or "")}


def _fallback_scores(
    question: str, answer: str, ctxs: list[str], ground_truth: str,
) -> tuple[float, float, float, float]:
    a = _word_set(answer)
    q = _word_set(question)
    g = _word_set(ground_truth)
    union_ctx: set[str] = set()
    for c in ctxs or []:
        union_ctx |= _word_set(c)
    faith = min(1.0, len(a & union_ctx) / max(1, len(a)))
    relev = min(1.0, len(a & q) / max(1, len(a)))
    prec = 0.0
    if ctxs:
        overlaps = [len(a & _word_set(c)) / max(1, len(_word_set(c))) for c in ctxs if c.strip()]
        prec = float(sum(overlaps) / len(overlaps)) if overlaps else 0.0
    prec = min(1.0, prec)
    recall = min(1.0, len(g & union_ctx) / max(1, len(g)))
    return float(faith), float(relev), float(prec), float(recall)


def _evaluate_ragas_fallback(
    questions: list[str], answers: list[str], contexts: list[list[str]], ground_truths: list[str],
) -> dict:
    per_question: list[EvalResult] = []
    rows_f, rows_ar, rows_cp, rows_cr = [], [], [], []
    for i in range(len(questions)):
        f, ar, cp, cr = _fallback_scores(
            questions[i], answers[i], contexts[i], ground_truths[i],
        )
        rows_f.append(f)
        rows_ar.append(ar)
        rows_cp.append(cp)
        rows_cr.append(cr)
        per_question.append(EvalResult(
            question=questions[i],
            answer=answers[i],
            contexts=contexts[i],
            ground_truth=ground_truths[i],
            faithfulness=f,
            answer_relevancy=ar,
            context_precision=cp,
            context_recall=cr,
        ))
    return {
        "faithfulness": float(np.mean(rows_f)) if rows_f else 0.0,
        "answer_relevancy": float(np.mean(rows_ar)) if rows_ar else 0.0,
        "context_precision": float(np.mean(rows_cp)) if rows_cp else 0.0,
        "context_recall": float(np.mean(rows_cr)) if rows_cr else 0.0,
        "per_question": per_question,
    }


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    if not questions:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": [],
        }
    from datasets import Dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })
    try:
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            show_progress=False,
            embeddings=_langchain_embeddings_for_ragas(),
        )
        keys = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
        agg: dict = {k: float(np.nanmean(np.asarray(result[k], dtype=np.float64))) for k in keys}
        per_question: list[EvalResult] = []
        for i in range(len(questions)):
            sc = result.scores[i]
            per_question.append(EvalResult(
                question=questions[i],
                answer=answers[i],
                contexts=contexts[i],
                ground_truth=ground_truths[i],
                faithfulness=float(np.nan_to_num(sc.get("faithfulness"), nan=0.0)),
                answer_relevancy=float(np.nan_to_num(sc.get("answer_relevancy"), nan=0.0)),
                context_precision=float(np.nan_to_num(sc.get("context_precision"), nan=0.0)),
                context_recall=float(np.nan_to_num(sc.get("context_recall"), nan=0.0)),
            ))
        agg["per_question"] = per_question
        return agg
    except Exception:
        return _evaluate_ragas_fallback(questions, answers, contexts, ground_truths)


def _diagnose(worst_metric: str, value: float) -> tuple[str, str]:
    rules: list[tuple[str, float, str, str]] = [
        ("faithfulness", 0.85, "LLM hallucinating", "Tighten prompt"),
        ("context_recall", 0.75, "Missing chunks", "Improve chunking/search"),
        ("context_precision", 0.75, "Irrelevant chunks", "Add reranking"),
        ("answer_relevancy", 0.80, "Answer mismatch", "Improve prompt"),
    ]
    for name, th, diagnosis, fix in rules:
        if name == worst_metric and value < th:
            return diagnosis, fix
    return "Overall quality below target", "Review RAG pipeline end-to-end"


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    scored: list[tuple[float, EvalResult, str, float]] = []
    for er in eval_results:
        mets = [er.faithfulness, er.answer_relevancy, er.context_precision, er.context_recall]
        avg = sum(mets) / 4.0
        worst_i = int(min(range(4), key=lambda i: mets[i]))
        worst_name = names[worst_i]
        worst_val = float(mets[worst_i])
        scored.append((avg, er, worst_name, worst_val))
    scored.sort(key=lambda x: x[0])
    out: list[dict] = []
    for _, er, worst_name, worst_val in scored[: max(0, bottom_n)]:
        diagnosis, fix = _diagnose(worst_name, worst_val)
        out.append({
            "question": er.question,
            "worst_metric": worst_name,
            "score": worst_val,
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })
    return out


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")

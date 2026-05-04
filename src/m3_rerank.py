"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []
        model = self._load_model()
        pairs = [(query, d["text"]) for d in documents]
        scores = np.asarray(model.predict(pairs), dtype=np.float64).ravel()
        combined = sorted(
            zip(scores.tolist(), documents),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]
        out: list[RerankResult] = []
        for rank, (score, doc) in enumerate(combined):
            out.append(RerankResult(
                text=doc["text"],
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=dict(doc.get("metadata") or {}),
                rank=rank,
            ))
        return out


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents:
            return []
        try:
            from flashrank import Ranker, RerankRequest
        except ImportError:
            return []
        if self._model is None:
            self._model = Ranker()
        passages = [{"id": i, "text": d["text"]} for i, d in enumerate(documents)]
        ranked = self._model.rerank(RerankRequest(query=query, passages=passages))
        out: list[RerankResult] = []
        for rank, item in enumerate(ranked[:top_k]):
            i = item.get("id", rank)
            if isinstance(i, str) and str(i).isdigit():
                i = int(i)
            idx = int(i) if isinstance(i, (int, float)) else rank
            if not (0 <= idx < len(documents)):
                idx = rank
            doc = documents[idx]
            out.append(RerankResult(
                text=item.get("text", doc["text"]),
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(item.get("score", 0.0)),
                metadata=dict(doc.get("metadata") or {}),
                rank=rank,
            ))
        return out


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs."""
    times_ms: list[float] = []
    for _ in range(max(1, n_runs)):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times_ms.append((time.perf_counter() - start) * 1000.0)
    return {
        "avg_ms": sum(times_ms) / len(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")

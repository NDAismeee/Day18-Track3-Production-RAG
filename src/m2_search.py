"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, socket, sys, warnings
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


def _qdrant_tcp_open(timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((QDRANT_HOST, int(QDRANT_PORT)), timeout=timeout):
            return True
    except OSError:
        return False


def _connect_qdrant():
    from qdrant_client import QdrantClient
    if os.getenv("QDRANT_USE_MEMORY", "").strip().lower() in ("1", "true", "yes"):
        return QdrantClient(location=":memory:"), True
    if not _qdrant_tcp_open():
        warnings.warn(
            f"Qdrant at {QDRANT_HOST}:{QDRANT_PORT} is unreachable; using in-memory "
            f"Qdrant (:memory:) for dense vectors. Start Docker Qdrant for a persistent index.",
            UserWarning,
            stacklevel=2,
        )
        return QdrantClient(location=":memory:"), True
    timeout = int(os.getenv("QDRANT_TIMEOUT", "60"))
    remote = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        timeout=timeout,
        prefer_grpc=False,
        check_compatibility=False,
    )
    return remote, False


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    from underthesea import word_tokenize
    return word_tokenize(text, format="text")


class BM25Search:
    def __init__(self):
        self.corpus_tokens: list[list[str]] = []
        self.documents: list[dict] = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = list(chunks)
        self.corpus_tokens = []
        for chunk in self.documents:
            seg = segment_vietnamese(chunk.get("text") or "")
            toks = [t for t in seg.split() if t]
            self.corpus_tokens.append(toks if toks else ["__empty__"])
        if self.corpus_tokens:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if not self.bm25 or not self.documents:
            return []
        seg = segment_vietnamese(query)
        tokenized_query = [t for t in seg.split() if t]
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)
        n = len(scores)
        if n == 0:
            return []
        order = np.argsort(np.asarray(scores, dtype=np.float64))[::-1][:top_k]
        out: list[SearchResult] = []
        for i in order:
            idx = int(i)
            doc = self.documents[idx]
            out.append(SearchResult(
                text=doc["text"],
                score=float(scores[idx]),
                metadata=dict(doc.get("metadata") or {}),
                method="bm25",
            ))
        return out


class DenseSearch:
    def __init__(self):
        self.client, self._memory_backend = _connect_qdrant()
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, PointStruct, VectorParams
        if not chunks:
            self.client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            return
        texts = [c.get("text") or "" for c in chunks]
        enc = self._get_encoder()
        vectors = enc.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        if hasattr(vectors, "tolist"):
            rows = vectors.tolist()
        else:
            rows = list(vectors)
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        points = [
            PointStruct(
                id=i,
                vector=rows[i],
                payload={**(chunks[i].get("metadata") or {}), "text": chunks[i].get("text") or ""},
            )
            for i in range(len(chunks))
        ]
        self.client.upsert(collection_name=collection, points=points)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        qv = self._get_encoder().encode(query, normalize_embeddings=True)
        if hasattr(qv, "tolist"):
            vec = qv.tolist()
        else:
            vec = list(qv)
        if hasattr(self.client, "query_points"):
            resp = self.client.query_points(
                collection_name=collection,
                query=vec,
                limit=top_k,
            )
            hits = list(getattr(resp, "points", None) or [])
        else:
            hits = self.client.search(
                collection_name=collection,
                query_vector=vec,
                limit=top_k,
            )
        out: list[SearchResult] = []
        for hit in hits:
            payload = hit.payload or {}
            text = payload.get("text", "")
            meta = {k: v for k, v in payload.items() if k != "text"}
            out.append(SearchResult(
                text=text,
                score=float(hit.score),
                metadata=meta,
                method="dense",
            ))
        return out


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores: defaultdict[str, float] = defaultdict(float)
    meta_by_text: dict[str, dict] = {}
    for results in results_list:
        for rank, r in enumerate(results):
            rrf_scores[r.text] += 1.0 / (k + rank + 1)
            if r.text not in meta_by_text:
                meta_by_text[r.text] = dict(r.metadata or {})
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        SearchResult(text=t, score=s, metadata=meta_by_text.get(t, {}), method="hybrid")
        for t, s in ranked
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")

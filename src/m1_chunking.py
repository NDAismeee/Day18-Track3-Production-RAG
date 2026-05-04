"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all markdown/text files from data/. (Đã implement sẵn)"""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    return docs


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


def _split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\n", text) if s.strip()]
    if not parts:
        t = text.strip()
        return [t] if t else []
    return parts


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}
    sentences = _split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(
            text=sentences[0],
            metadata={**metadata, "chunk_index": 0, "strategy": "semantic"},
        )]
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences, convert_to_numpy=True)
    groups: list[list[str]] = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = _cosine_sim(embeddings[i - 1], embeddings[i])
        if sim < threshold:
            groups.append([])
        groups[-1].append(sentences[i])
    chunks: list[Chunk] = []
    for idx, g in enumerate(groups):
        body = " ".join(g).strip()
        if body:
            chunks.append(Chunk(
                text=body,
                metadata={**metadata, "chunk_index": idx, "strategy": "semantic"},
            ))
    return chunks


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Chars per parent chunk.
        child_size: Chars per child chunk.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        t = text.strip()
        paragraphs = [t] if t else [""]
    parents: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        add_len = len(para) + (2 if current else 0)
        if current and current_len + add_len > parent_size:
            parent_text = "\n\n".join(current)
            pid = f"parent_{len(parents)}"
            parents.append(Chunk(
                text=parent_text,
                metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
            ))
            current = []
            current_len = 0
        current.append(para)
        current_len += add_len
    if current:
        pid = f"parent_{len(parents)}"
        parents.append(Chunk(
            text="\n\n".join(current),
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
        ))
    children: list[Chunk] = []
    for p in parents:
        pid = p.metadata.get("parent_id")
        ptext = p.text
        if not ptext.strip():
            children.append(Chunk(
                text=ptext,
                metadata={**metadata, "chunk_type": "child"},
                parent_id=pid,
            ))
            continue
        start = 0
        n = len(ptext)
        while start < n:
            end = min(start + child_size, n)
            seg = ptext[start:end].strip()
            if seg:
                children.append(Chunk(
                    text=seg,
                    metadata={**metadata, "chunk_type": "child"},
                    parent_id=pid,
                ))
            if end >= n:
                break
            start = end
    return parents, children


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}
    lines = text.split("\n")
    header_re = re.compile(r"^#{1,3}\s+")
    sections: list[tuple[str, list[str]]] = []
    cur_header: str | None = None
    cur_body: list[str] = []
    for line in lines:
        if header_re.match(line):
            if cur_header is not None:
                sections.append((cur_header, cur_body))
            cur_header = line.strip()
            cur_body = []
        else:
            cur_body.append(line)
    if cur_header is not None:
        sections.append((cur_header, cur_body))
    chunks: list[Chunk] = []
    for h, body_lines in sections:
        body = "\n".join(body_lines).strip()
        block = f"{h}\n{body}".strip() if body else h
        chunks.append(Chunk(
            text=block,
            metadata={**metadata, "section": h, "strategy": "structure"},
        ))
    return chunks


def _length_stats(chunks: list[Chunk]) -> dict:
    if not chunks:
        return {"num_chunks": 0, "avg_length": 0.0, "min_length": 0, "max_length": 0}
    lens = [len(c.text) for c in chunks]
    return {
        "num_chunks": len(chunks),
        "avg_length": round(sum(lens) / len(lens), 1),
        "min_length": min(lens),
        "max_length": max(lens),
    }


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    all_basic: list[Chunk] = []
    all_semantic: list[Chunk] = []
    all_parents: list[Chunk] = []
    all_children: list[Chunk] = []
    all_structure: list[Chunk] = []
    for doc in documents:
        meta = doc.get("metadata") or {}
        t = doc.get("text") or ""
        all_basic.extend(chunk_basic(t, metadata=meta))
        all_semantic.extend(chunk_semantic(t, metadata=meta))
        ps, cs = chunk_hierarchical(t, metadata=meta)
        all_parents.extend(ps)
        all_children.extend(cs)
        all_structure.extend(chunk_structure_aware(t, metadata=meta))
    basic_stats = _length_stats(all_basic)
    semantic_stats = _length_stats(all_semantic)
    child_stats = _length_stats(all_children)
    parent_stats = _length_stats(all_parents)
    hierarchical_stats = {
        "num_parents": len(all_parents),
        "num_children": len(all_children),
        "avg_parent_length": parent_stats["avg_length"],
        "avg_child_length": child_stats["avg_length"],
        "min_child_length": child_stats["min_length"],
        "max_child_length": child_stats["max_length"],
    }
    structure_stats = _length_stats(all_structure)
    rows = [
        ("basic", basic_stats["num_chunks"], basic_stats["avg_length"],
         basic_stats["min_length"], basic_stats["max_length"]),
        ("semantic", semantic_stats["num_chunks"], semantic_stats["avg_length"],
         semantic_stats["min_length"], semantic_stats["max_length"]),
        ("hierarchical", f"{hierarchical_stats['num_parents']}p/{hierarchical_stats['num_children']}c",
         hierarchical_stats["avg_child_length"],
         hierarchical_stats["min_child_length"], hierarchical_stats["max_child_length"]),
        ("structure", structure_stats["num_chunks"], structure_stats["avg_length"],
         structure_stats["min_length"], structure_stats["max_length"]),
    ]
    print(f"{'Strategy':<14} | {'Chunks':^8} | {'Avg Len':^8} | {'Min':^5} | {'Max':^5}")
    print("-" * 52)
    for name, nch, avg, mn, mx in rows:
        print(f"{name:<14} | {str(nch):^8} | {str(avg):^8} | {str(mn):^5} | {str(mx):^5}")
    return {
        "basic": basic_stats,
        "semantic": semantic_stats,
        "hierarchical": hierarchical_stats,
        "structure": structure_stats,
    }


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")

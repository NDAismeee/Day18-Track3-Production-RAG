# Individual Reflection — Lab 18

**Tên:** Nguyễn Đức Anh  
**Mã học viên**: 2A202600146
**Module phụ trách:** M1 Chunking · M2 Hybrid Search · M3 Reranking · M4 RAGAS Evaluation · M5 Enrichment (toàn bộ pipeline)

---

## 1. Đóng góp kỹ thuật

- Module đã implement: **M1–M5** — chunking (semantic / hierarchical / structure-aware + so sánh chiến lược), hybrid search BM25 + dense Qdrant + RRF, reranker cross-encoder, pipeline RAGAS + failure analysis, enrichment (tóm tắt / HyQA / contextual prepend / metadata).
- Các hàm/class chính đã viết (trọng tâm): `chunk_semantic`, `chunk_hierarchical`, `chunk_structure_aware`, `compare_strategies`; `BM25Search`, `DenseSearch`, `HybridSearch`, `reciprocal_rank_fusion`; `CrossEncoderReranker`, `benchmark_reranker`; `evaluate_ragas`, `failure_analysis`; `summarize_chunk`, `generate_hypothesis_questions`, `contextual_prepend`, `extract_metadata`, `enrich_chunks`; chỉnh `pipeline.run_query` dùng OpenAI khi có API key; xử lý Qdrant (`query_points`) và fallback `:memory:` khi không có Docker.
- Số tests pass: **37/37** (pytest: M1 13, M2 5, M3 5, M4 4, M5 10 — theo cấu trúc hiện tại của repo).

## 2. Kiến thức học được

- Khái niệm mới nhất: **RRF** gộp xếp hạng BM25 và dense; **parent–child chunking** (retrieve child, hiểu parent); **RAGAS** và diagnostic tree (faithfulness vs recall vs precision vs relevancy); **HyQA / contextual prepend** để giảm vocabulary gap trước khi embed.
- Điều bất ngờ nhất: Điểm **context precision/recall** có thể ~1 trong khi **answer relevancy** vẫn thấp — lỗi không chỉ ở retrieval mà còn ở **cách diễn đạt câu trả lời** so với ground truth ngắn.
- Kết nối với bài giảng (slide nào): Production RAG — chunking nâng cao, hybrid retrieval, reranking, đánh giá RAGAS và phân tích lỗi (đúng với flow Lab 18 / buổi Production RAG).

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: **Qdrant** không chạy local → cần fallback hoặc Docker; **qdrant-client** đổi API (`search` → **`query_points`**); **RAGAS 0.4** gán embedding mặc định không tương thích **`embed_query`** với metric `answer_relevancy` → lỗi `OpenAIEmbeddings` / `embed_query`.
- Cách giải quyết: Kiểm tra cổng TCP + **`QdrantClient(location=":memory:")`** khi cần; dense search chuyển sang **`query_points`** và đọc **`resp.points`**; truyền **`langchain_openai.OpenAIEmbeddings`** vào **`evaluate(..., embeddings=...)`**; **`.env`** load từ root project và **`OPENAI_API_KEY`** cho LLM/embedding.
- Thời gian debug: Ước lượng **2–4 giờ** rải rác (môi trường, dependency, RAGAS/Qdrant).

## 4. Nếu làm lại

- Sẽ làm khác điều gì: Chuẩn hóa **prompt trả lời ngắn / extractive** sớm hơn để khớp ground truth; chạy **Docker Qdrant** ngay từ đầu để tránh lệch hành vi giữa in-memory và server; ghi **`requirements.txt`** cố định phiên bản package quan trọng.
- Module nào muốn thử tiếp: Tinh chỉnh **M4** (thử metrics/benchmark khác hoặc generator có kiểm soát định dạng); hoặc **M2** với filter metadata sau enrichment.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 2 |
| Problem solving | 4 |


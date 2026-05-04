# Group Report — Lab 18: Production RAG

**Nhóm:** nhóm 1 thành viên
**Ngày:** 04/05/2026

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-------------|------------|
| Nguyễn Đức Anh | M1: Chunking | ☑ | 13/13 |
| Nguyễn Đức Anh | M2: Hybrid Search | ☑ | 5/5 |
| Nguyễn Đức Anh | M3: Reranking | ☑ | 5/5 |
| Nguyễn Đức Anh | M4: Evaluation | ☑ | 4/4 |
| Nguyễn Đức Anh | M5: Enrichment | ☑ | 10/10 |

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|--------|-------|------------|---|
| Faithfulness | 1.0000 | 0.9250 | −0.0750 |
| Answer Relevancy | 0.2828 | 0.3382 | +0.0554 |
| Context Precision | 1.0000 | 1.0000 | 0.0000 |
| Context Recall | 1.0000 | 1.0000 | 0.0000 |

*(20 câu trong `test_set.json`; báo cáo từ `reports/naive_baseline_report.json` và `reports/ragas_report.json`.)*

## Key Findings

1. **Biggest improvement:** **Answer Relevancy** tăng nhẹ (+0.055) nhờ pipeline production (chunk hierarchical + enrichment HyQA/context/metadata + hybrid BM25–dense–RRF + rerank + LLM trả lời có điều kiện khi có API). Context **precision/recall** giữ ~1.0 — retrieval không làm mất hoặc nhét nhầm chunk trên bộ test hiện tại.
2. **Biggest challenge:** **Answer Relevancy** vẫn thấp tuyệt đối (~0.28–0.34): ground truth ngắn, một dòng; câu trả lời thường là đoạn policy hoặc paraphrase dài → RAGAS penalize dù context đúng. **Faithfulness** production thấp hơn naive một chút — trade-off khi thêm bước sinh câu trả lời / wording không khớp span đánh giá.
3. **Surprise finding:** Naive baseline đã đạt **precision/recall ~1** trên test set; bottleneck không phải “không tìm thấo chunk” mà là **định dạng và độ ngắn của câu trả lời** so với tiêu chí benchmark.

## Presentation Notes (5 phút)

1. **RAGAS scores (naive vs production):** So sánh bảng trên — nhấn mạnh Δ **Answer Relevancy** dương nhỏ; **Faithfulness** giảm nhẹ; **Precision/Recall** không đổi → pipeline “đủ context”, chưa tối ưu “answer giống ground truth một dòng”.
2. **Biggest win — module nào, tại sao:** **M2 Hybrid Search** + **M5 Enrichment** — BM25 tiếng Việt + dense `bge-m3` + RRF giữ được recall tốt trên câu hỏi biến thể; enrichment (HyQA/context) giúp bắc cầu từ vựng. **M3 Rerank** tinh chỉnh top-k trước khi đưa vào LLM.
3. **Case study — 1 failure, Error Tree walkthrough:** Ví dụ *“Ai phê duyệt đơn xin nghỉ phép không lương?”* — context có “Giám đốc bộ phận” nhưng output có thể dài hoặc không trích đúng một cụm → faithfulness/relevancy kém. Error Tree: retrieval OK → generation/format chưa khớp benchmark.
4. **Next optimization nếu có thêm 1 giờ:** Prompt LLM trả lời **một câu ngắn / trích đúng span**; hoặc extractive QA cho câu hỏi factoid; giảm noise trong text làm giàu khi chunk đã đủ ngắn.

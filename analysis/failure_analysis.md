# Failure Analysis — Lab 18: Production RAG

**Nhóm:** nhóm đơn 1 thành viên  
**Thành viên:** Nguyễn Đức Anh → M1 Chunking · M2 Hybrid Search · M3 Rerank · M4 RAGAS Eval · M5 Enrichment (toàn bộ pipeline)

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 1.0000 | 0.9250 | −0.0750 |
| Answer Relevancy | 0.2828 | 0.3382 | +0.0554 |
| Context Precision | 1.0000 | 1.0000 | 0.0000 |
| Context Recall | 1.0000 | 1.0000 | 0.0000 |

Ghi chú: Production **không giảm** precision/recall so với baseline; context luôn chứa đủ thông tin (recall = 1). Điểm kém chủ yếu nằm ở **answer relevancy** (cách diễn đạt câu trả lời so với cách đặt câu hỏi / ground truth ngắn) và một ca **faithfulness** do pipeline đôi khi trả về câu trả lời dài / không khớp định dạng ground truth.

---

## Bottom-5 Failures

### #1
- **Question:** Ai phê duyệt đơn xin nghỉ phép không lương?
- **Expected:** Giám đốc bộ phận.
- **Got:** Câu trả lời lấy từ context / đoạn chunk có thể là nguyên đoạn policy hoặc LLM diễn giải dài hơn so với một cụm danh từ ngắn trong ground truth.
- **Worst metric:** faithfulness (0.0 trong báo cáo failure slice).
- **Error Tree:** Output không khớp ràng buộc “chỉ fact trong context” ở dạng ngắn → Context vẫn chứa đúng ý → Retrieval và chunk đủ → Root cause: **answer generator / heuristic trả lời** (pipeline đang ưu tiên context đầu tiên hoặc wording không trích đúng một mệnh đề).
- **Root cause:** RAGAS faithfulness so khớp giữa câu trả lời và context; trả lời dài hoặc lệch diễn đạt làm điểm giảm thậm chí khi nội dung đúng.
- **Suggested fix:** Prompt LLM: “Trả lời một cụm ngắn: ai phê duyệt?” + ép format noun phrase; hoặc post-process trích entity sau rerank.

### #2
- **Question:** Chính sách nói gì về việc tăng ngày phép theo thâm niên?
- **Expected:** Tăng thêm 1 ngày cho mỗi 5 năm thâm niên.
- **Got:** Có thể là cả đoạn policy hoặc paraphrase không trùng cấu trúc với ground truth một dòng.
- **Worst metric:** answer_relevancy (~0.14).
- **Error Tree:** Output sai format so với “ground_truth ngắn” → Context có đủ số “5 năm”, “1 ngày” → Query khớp chủ đề → Root cause: **style mismatch** giữa answer và cách RAGAS đo độ “relevant” với câu hỏi.
- **Root cause:** Ground truth là tóm tắt một dòng; model/context trả về câu dài hoặc thứ tự từ khác.
- **Suggested fix:** Fine-tune prompt trả lời kiểu “một câu, đủ số và đơn vị”; thử rerank context ngắn hơn (child chunk + parent).

### #3
- **Question:** Nghỉ phép không lương tối đa bao nhiêu ngày một năm?
- **Expected:** 30 ngày mỗi năm.
- **Got:** Có thể nhắc đúng “30 ngày” nhưng kèm thêm câu khác trong chunk → độ dài / embedding similarity với câu hỏi thấp.
- **Worst metric:** answer_relevancy.
- **Error Tree:** Output không compact → Context đúng → Query OK → Root cause: **answer không được rút gọn về “30 ngày…”**.
- **Suggested fix:** Extractive QA: regex / span “30 ngày”; hoặc HyQA đã index thêm câu hỏi gần user query để dense match tốt hơn.

### #4
- **Question:** Có được gửi mật khẩu qua email hoặc chat không mã hóa không?
- **Expected:** Không được chia sẻ mật khẩu qua email hoặc chat không mã hóa.
- **Got:** Trả lời kiểu có/không hoặc trích nguyên đoạn IT policy — wording khác ground truth chuẩn.
- **Worst metric:** answer_relevancy.
- **Error Tree:** Output đúng ý “không” nhưng không literal match → Context có đủ → Root cause: **binary QA vs verbatim GT**.
- **Suggested fix:** Chuẩn hóa câu trả lời yes/no + một mệnh đề từ policy; hoặc nới ground truth trong eval (semantic match).

### #5
- **Question:** Khi nghỉ ốm, trong bao lâu phải nộp giấy xác nhận y tế?
- **Expected:** Trong vòng 3 ngày làm việc.
- **Got:** Có thể nhắc “3 ngày làm việc” trong chunk dài hơn → relevancy embedding so với câu hỏi trung bình.
- **Worst metric:** answer_relevancy (~0.18).
- **Error Tree:** Tương tự #2–#4 — retrieval tốt, **generation/format** chưa khớp benchmark ngắn.
- **Suggested fix:** Prompt bắt buộc format “Trong vòng … ngày làm việc”; giảm noise trong enriched_text nếu enrichment làm lệch embedding.

---

## Case Study (cho presentation)

**Question chọn phân tích:** Ai phê duyệt đơn xin nghỉ phép không lương?

**Error Tree walkthrough:**
1. Output đúng? → Nội dung đúng (Giám đốc bộ phận) có trong corpus nhưng câu trả lời có thể không được RAGAS faithfulness chấm cao nếu so khớp literal/context span kém.
2. Context đúng? → Có — precision và recall aggregate = 1 cho pipeline.
3. Query rewrite OK? → Không cần rewrite mạnh; BM25 + dense đã kéo đúng chunk tiếng Việt.
4. Fix ở bước: **generation / extractive answer** — không phải chunking hay search.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Thêm bước LLM chỉ trích **một cụm danh từ** làm final answer khi câu hỏi là “Ai…?” / “Bao nhiêu…?” / “Trong bao lâu…?”.
- Giảm độ dài enriched chunk trong M5 cho các chunk factual ngắn (tránh dilute embedding relevancy).

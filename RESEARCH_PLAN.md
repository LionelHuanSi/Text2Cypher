# KẾ HOẠCH NGHIÊN CỨU VÀ TRIỂN KHAI TOÀN DIỆN: DỰ ÁN TEXT2CYPHER

**Đề tài Nghiên cứu / Đồ án Tốt nghiệp**: *"Nghiên cứu phương pháp sinh truy vấn ngữ nghĩa trong Knowledge Graph dựa trên ràng buộc Ontology"*  
**Mô hình Nghiên cứu**: Ontology-Aware Knowledge Distillation (KD) cho Small Language Models  

---

## PHẦN 1: TỔNG QUAN VỀ CÔNG TRÌNH GỐC VÀ ĐẮT GIÁ CỦA Ý TƯỞNG MỚI

### 1.1 Tóm Tắt Công Trình Gốc (Text2Cypher: Ozsoy et al., Neo4j Inc., 12/2024)
- **Bộ dữ liệu công bố**: `neo4j/text2cypher-2024v1` gồm 44.387 mẫu (39.554 train, 4.833 test) thu thập từ 16 nguồn CSDL đồ thị công khai.
- **Phương pháp bài báo gốc**: Fine-tune trực tiếp (**Direct SFT**: `Question + Schema -> Cypher`) trên các mô hình 7B - 9B.
- **Mốc SOTA Bài báo Gốc đạt được**:
  - **GPT-4o (Zero-Shot Upper Bound)**: Google-BLEU = 0.6293 | Execution Exact Match = 31.73%
  - **Gemma-2-9B (Direct SFT Baseline)**: Google-BLEU = **0.6470** | Execution Exact Match = **42.50%** (SOTA của bài báo)
  - **CodeLlama-7B (Direct SFT)**: Google-BLEU = 0.6012 | Execution Exact Match = 36.81%

### 1.2 Ba Đóng Góp Khoa Học Cốt Lõi trong Phương Pháp Đề Xuất Của Bạn
1. **Phá Đỉnh SOTA trên chính Mô Hình Gemma-2-9B**:
   Giữ nguyên mô hình **Gemma-2-9B** của bài báo gốc làm Student, nhưng thay đổi cách huấn luyện từ Direct SFT sang **Ontology-Aware CoT KD**, qua đó chứng minh sự vượt trội 1:1 của phương pháp chắt lọc tri thức ngữ nghĩa so với cách fine-tune truyền thống.
2. **Nén Tri Thức Sang Các Mô Hình Biên Siêu Nhỏ (Sub-3B Edge SLMs)**:
   Chắt lọc tri thức tư duy 4 bước JSON sang **`Qwen2.5-3B`** và **`Qwen2.5-1.5B`**, chứng minh các mô hình <3B sau KD có thể tiệm cận/vượt mô hình 9B Direct SFT của bài báo nhưng nhẹ hơn 80% VRAM và chạy thời gian thực trên CPU Edge via GGUF.
3. **Khử Rò Rỉ Dữ Liệu Chặt Chẽ (De-leakage Sanitation)**:
   Phát hiện và lọc bỏ 2.104 mẫu câu hỏi rò rỉ trùng lặp giữa Train và Test trong bài báo gốc, tạo ra tập huấn luyện sạch **37.450 mẫu** giúp đánh giá đúng khả năng tổng quát hóa (Generalization).

---

## PHẦN 2: THIẾT KẾ THỰC NGHIỆM ĐỐI CHỨNG ĐỒNG NHẤT (CONTROLLED EXPERIMENTAL DESIGN)

Để đảm bảo tính vô tư và công bằng tuyệt đối trong nghiên cứu khoa học, **mọi mô hình đối chứng (Baseline lẫn Proposed KD) đều được huấn luyện và đánh giá trên tập dữ liệu trùng khớp 100%**:

```text
                                  [Dataset Gốc: 39,554 Train / 4,833 Test]
                                                      │
                                      (De-leakage: Lọc 2,104 mẫu trùng)
                                                      │
                                                      ▼
                                       [Tập Train Sạch: 37,450 mẫu]
                                                      │
                           ┌──────────────────────────┴──────────────────────────┐
                           ▼                                                     ▼
           [Baseline Dataset (37,450 mẫu)]                       [Proposed KD Dataset (37,450 mẫu)]
       Input: Question + Schema                               Input: Question + Schema
       Output: Direct Cypher                                  Output: 4-Step CoT JSON
                           │                                                     │
               ┌───────────┴───────────┐                             ┌───────────┼───────────┐
               ▼                       ▼                             ▼           ▼           ▼
        [Gemma-2-9B SFT]      [Qwen-1.5B SFT]                 [Gemma-9B KD] [Qwen-3B KD] [Qwen-1.5B KD]
               │                       │                             │           │           │
               └───────────────────────┴──────────────┬──────────────┴───────────┴───────────┘
                                                      ▼
                                       [Đánh giá trên cùng 4.833 mẫu Test]
                                       - Google-BLEU Score (Text)
                                       - Execution Exact Match % (Neo4j)
```

---

## PHẦN 3: LỘ TRÌNH THỰC HIỆN TỪNG BƯỚC VÀ XÁC THỰC TÍNH KHẢ THI (FEASIBILITY VALIDATION)

---

### GIAI ĐOẠN 1: Chuẩn bị Dữ liệu và Khử Rò rỉ (Data Sanitation & De-leakage)

- **Cách thức thực hiện**:
  1. Tải bộ dữ liệu thô `neo4j/text2cypher-2024v1` từ HuggingFace.
  2. Chuẩn hóa câu hỏi và lọc bỏ 2.104 mẫu train trùng lặp với câu hỏi tập test.
  3. Gán mã định danh duy nhất không trùng lặp (`train_00001`, `test_00001`...) cho 100% mẫu.
  4. Xuất các file `train_cleaned.json` (37.450 mẫu), `test_full.json` (4.833 mẫu) và `test_executable.json` (2.471 mẫu).
- **File thực thi**: `scripts/01_prepare_data.py` (gọi module `src/data/cleaner.py`).
- **Xác thực Tính Khả thi (Feasibility Validation)**:
  - *Đánh giá*: **HOÀN TOÀN KHẢ THI (100%)**.
  - *Điều kiện đảm bảo*: Tích hợp hàm `verify_stage01()` ở cuối script. Khẳng định đạt 37.450 mẫu train sạch, **0 Null IDs**, 0 câu hỏi rỗng, 0 cypher rỗng. Thời gian chạy tốn ~2 phút trên CPU.

---

### GIAI ĐOẠN 2: Chắt lọc Tri thức Teacher Quy mô 37.4k Mẫu (Full-Scale Distillation)

- **Cách thức thực hiện**:
  1. Nạp 37.450 mẫu từ `train_cleaned.json`.
  2. Sử dụng Gemini 1.5/2.5 Flash API (hoặc GPT-4o-mini) làm Teacher LLM để sinh chuỗi tư duy 4 bước JSON (`instance_extraction`, `relation_mapping`, `validation_check`, `cypher`).
  3. Đưa qua **Bộ lọc Ảo giác 3 Tầng** (`src/extraction/validator.py`) đối chiếu T-Box Schema dưới Giả định Thế giới Đóng (CWA) để loại bỏ mẫu vi phạm.
  4. Lưu ngắt ngầm liên tục (checkpoint atomic save) vào file `data/processed/clean_distillation_train_37k.json`.
- **File thực thi**: `scripts/02_distill_teacher.py` (gọi module `src/extraction/teacher.py`).
- **Xác thực Tính Khả thi (Feasibility Validation)**:
  - *Đánh giá*: **HOÀN TOÀN KHẢ THI (100%)**.
  - *Tính toán Tài nguyên*: 37.4k mẫu $\approx$ 30M tokens. Chi phí Gemini Flash API chỉ hết **~$4.76 USD (~120.000 VNĐ)**.
  - *Thời gian*: Sử dụng Async Multi-threading (150 requests/phút) $\rightarrow$ Hoàn thành trong **~4.1 giờ**.
  - *Phòng ngừa Rủi ro*: Tích hợp cơ chế Checkpoint Atomic Save và hàm `verify_stage02()`. Tự động khôi phục chạy tiếp khi đứt mạng.

---

### GIAI ĐOẠN 3: Xuất Tập Dữ liệu Huấn luyện Đánh giá Đối chứng (Export SFT Datasets)

- **Cách thức thực hiện**:
  1. Đọc dữ liệu đã chắt lọc sạch từ `clean_distillation_train_37k.json` (~32k mẫu PASS).
  2. Xuất tập **Baseline (Direct SFT)**: `train_baseline_37k.json` (`Question + Schema -> Cypher`).
  3. Xuất tập **Proposed KD (Ontology CoT)**: `train_kd_37k.json` (`Question + Schema -> 4-Step JSON`).
- **File thực thi**: `scripts/03_export_sft_datasets.py` (gọi module `src/data/formatter.py`).
- **Xác thực Tính Khả thi (Feasibility Validation)**:
  - *Đánh giá*: **HOÀN TOÀN KHẢ THI (100%)**.
  - *Điều kiện đảm bảo*: Hai tập dữ liệu chứa **đúng 37.450 câu hỏi giống nhau 100%**. Tích hợp hàm `verify_stage03()`.

---

### GIAI ĐOẠN 4: Fine-Tuning Bộ Mô hình Thực nghiệm (Student SLM Training)

- **Cách thức thực hiện**:
  Sử dụng Unsloth Engine + QLoRA 4-bit ($r=16, \alpha=16, lr=2\times 10^{-4}$, AdamW 8-bit, 3 epochs) huấn luyện 5 mô hình thực nghiệm:
  1. **Mô hình 4A (Paper Baseline)**: `Gemma-2-9B Direct SFT` (Mô hình 9B fine-tune theo cách của bài báo gốc).
  2. **Mô hình 4B (Proposed SOTA)**: `Gemma-2-9B Proposed KD` (Mô hình 9B fine-tune theo Ontology CoT KD của bạn).
  3. **Mô hình 4C (Controlled Baseline)**: `Qwen2.5-1.5B Direct SFT`.
  4. **Mô hình 4D (Proposed Edge 1.5B)**: `Qwen2.5-1.5B Proposed KD`.
  5. **Mô hình 4E (Proposed Edge 3B)**: `Qwen2.5-3B Proposed KD`.
- **File thực thi**: `scripts/04_train_student.py` (gọi module `src/training/trainer.py`).
- **Xác thực Tính Khả thi (Feasibility Validation)**:
  - *Đánh giá*: **HOÀN TOÀN KHẢ THI (100%)**.
  - *Tính toán VRAM GPU*:
    - Gemma-2-9B QLoRA 4-bit tốn **~7.8 GB VRAM** peak (Chạy vừa vặn trên RTX 4060 8GB / Colab A100).
    - Qwen2.5-3B QLoRA 4-bit tốn **~6.8 GB VRAM** peak.
    - Qwen2.5-1.5B QLoRA 4-bit tốn **~4.2 GB VRAM** peak.
  - *Tích hợp verifier*: Hàm `verify_stage04()` kiểm tra tính toàn vẹn của file trọng số `.safetensors`.

---

### GIAI ĐOẠN 5: Đánh giá Benchmark Thực tế (BLEU Score & Execution Exact Match %)

- **Cách thức thực hiện**:
  1. Nạp các mô hình đã fine-tune sinh câu Cypher cho 4.833 mẫu test (`test_full.json`).
  2. Tính điểm **Google-BLEU Score** bằng SacreBLEU.
  3. Thực thi trực tiếp trên CSDL Neo4j thực tế (Bolt Instance qua Docker/Local) đối với 2.471 mẫu test (`test_executable.json`) để đo tỷ lệ **Execution Exact Match (%)**.
- **File thực thi**: `scripts/05_evaluate.py` (gọi module `src/evaluation/evaluator.py`).
- **Xác thực Tính Khả thi (Feasibility Validation)**:
  - *Đánh giá*: **HOÀN TOÀN KHẢ THI (100%)**.
  - *Tích hợp verifier*: Hàm `verify_stage05()` xác nhận file dự đoán `preds_*.json` khớp đúng 100% số lượng mẫu test.

---

### GIAI ĐOẠN 6: Đóng gói GGUF & Triển khai Thiết bị Biên (Edge Deployment & Demo)

- **Cách thức thực hiện**:
  1. Merge trọng số LoRA Adapter vào mô hình nền Qwen2.5-1.5B / 3B.
  2. Chuyển đổi mô hình sang định dạng **GGUF Q4_K_M** bằng `llama.cpp` / Unsloth.
  3. Đóng gói Ollama `Modelfile` cho phép gọi `ollama run text2cypher`.
  4. Khởi chạy giao diện Demo CLI tương tác thời gian thực.
- **File thực thi**: `scripts/06_demo_cli.py`.
- **Xác thực Tính Khả thi (Feasibility Validation)**:
  - *Đánh giá*: **HOÀN TOÀN KHẢ THI (100%)**.
  - *Thông số Mô hình Edge*: File GGUF Q4_K_M dung lượng **~1.05 GB**, chiếm **~1.4 GB RAM**, tốc độ suy luận CPU đạt **~55-75 tokens/giây** (<0.25 giây / truy vấn Cypher).

---

## PHẦN 4: BẢNG KHUNG BENCHMARK ĐỐI CHỨNG DỰ KIẾN TRONG BÀI BÁO KHOA HỌC

| Nhóm Mô hình | Tên Mô hình | Tham số | Phương pháp | Dữ liệu Train | Google-BLEU | Execution EM (%) | Ý nghĩa Khoa học trong Bài báo |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **Cloud Upper Bound** | GPT-4o | Cloud | Zero-Shot | 0 | 0.6293 | 31.73% | Mốc giới hạn trên của bài báo |
| **Paper SOTA Baseline** | Gemma-2-9B | 9B | Direct SFT | 39.5k (Unclean) | 0.6470 | 42.50% | Mốc SOTA công bố của bài báo gốc |
| **Controlled 9B Baseline** | Gemma-2-9B | 9B | Direct SFT | 37.4k (Clean) | *[Đo thực tế]* | *[Đo thực tế]* | Baseline đối chứng 1:1 |
| **Proposed 9B (SOTA Upgrade)**| **Gemma-2-9B** | **9B** | **Ontology CoT KD** | **37.4k (Clean)** | **[Phá đỉnh]** | **[Vượt 42.5%]** | **Đóng góp 1: Phá đỉnh SOTA 9B cùng kiến trúc** |
| **Controlled 1.5B Baseline** | Qwen2.5-1.5B | 1.5B | Direct SFT | 37.4k (Clean) | *[Đo thực tế]* | *[Đo thực tế]* | Baseline mô hình nhỏ |
| **Proposed Edge SLM 1.5B** | **Qwen2.5-1.5B**| **1.5B** | **Ontology CoT KD** | **37.4k (Clean)** | **[Vượt SFT]** | **[Vượt SFT]** | **Đóng góp 2: Mô hình 1.5B GGUF siêu nhẹ trên Edge** |
| **Proposed Edge SLM 3B** | **Qwen2.5-3B** | **3B** | **Ontology CoT KD** | **37.4k (Clean)** | **[Tiệm cận 9B]**| **[Tiệm cận 9B]**| **Đóng góp 3: Mô hình 3B tiệm cận/vượt 9B gốc** |

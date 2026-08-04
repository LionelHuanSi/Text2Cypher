# Ontology-Aware Knowledge Distillation for Text2Cypher

**Đề tài Nghiên cứu / Đồ án Tốt nghiệp**: *"Nghiên cứu phương pháp sinh truy vấn ngữ nghĩa trong Knowledge Graph dựa trên ràng buộc Ontology"*  
**Tác giả**: Nguyễn Văn Huấn (Công nghệ thông tin Việt-Nhật - Đại học Bách khoa Hà Nội)  
**Giảng viên hướng dẫn**: PGS. TS. Cao Tuấn Dũng  
**Bộ dữ liệu Benchmark**: `neo4j/text2cypher-2024v1`  

---

## 1. Tổng Quan Bài Toán và Đặt Vấn Đề

Bài toán **Text-to-Cypher** đóng vai trò cốt lõi trong các hệ thống khai thác Đồ thị tri thức (Knowledge Graph - KG) và cơ sở dữ liệu Neo4j. Tuy nhiên, việc tinh chỉnh các mô hình ngôn ngữ nhỏ (Small Language Models - SLMs $\le$ 3B) theo phương pháp Fine-Tuning có giám sát truyền thống (Direct SFT: `Question + Schema -> Cypher`) thường thất bại do hiện tượng ảo giác cấu trúc đồ thị (tự bịa đặt nhãn nút, thuộc tính không tồn tại, ngược chiều quan hệ).

Hệ thống **Ontology-Aware Knowledge Distillation (KD)** đề xuất giải pháp chắt lọc tri thức tư duy Ontology 4 bước dạng JSON từ Mô hình Mẹ (Teacher LLM: Gemini Flash / GPT-4o) sang Mô hình Con siêu nhỏ (Student SLM: `Qwen2.5-1.5B` / `Qwen2.5-3B`), kết hợp **Bộ lọc ảo giác 3 tầng** kiểm duyệt nghiêm ngặt dưới **Giả định Thế giới Đóng (Closed World Assumption - CWA)**.

---

## 2. Điểm Mới và Đóng Góp Khoa Học Cốt Lõi

| Tiêu chí | Công trình gốc (Ozsoy et al., 2024) | Phương pháp Đề xuất (Proposed Method) |
| :--- | :--- | :--- |
| **Kích thước Mô hình** | Mô hình lớn (7B - 9B) | **Mô hình nhỏ (1.5B - 3B)** - Giảm 85% VRAM, triển khai Edge GGUF. |
| **Chiến lược Huấn luyện** | Direct SFT (`Question -> Cypher`) | **Ontology-Aware CoT Distillation** (Nén chuỗi tư duy 4 bước JSON). |
| **Kiểm soát Ảo giác** | Chỉ kiểm tra cú pháp bằng `EXPLAIN` | **Bộ lọc 3 tầng + SHACL Validation** (Triệt tiêu lỗi sai T-Box Schema). |
| **Chất lượng Dữ liệu** | Tồn tại rò rỉ 2.104 câu hỏi trùng | **Khử trùng rò rỉ triệt để (De-leakage)** -> Tập train sạch **37.450 mẫu**. |

---

## 3. Bảng Tính Toán Tài Nguyên và Ngân Sách Thực Hiện

| Giai đoạn | Tác vụ | Dịch vụ / Phần cứng | Đòi hỏi RAM/VRAM | Chi phí ước tính | Thời gian dự kiến |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Giai đoạn 1** | Tải và Khử rò rỉ dữ liệu (37.4k) | CPU Laptop / PC | RAM 4GB | **0 VNĐ** | 2 phút |
| **Giai đoạn 2** | Chắt lọc Teacher 37.4k mẫu | Gemini 1.5/2.5 Flash API | - | **~120.000 VNĐ** ($4.76) | ~4.1 giờ |
| **Giai đoạn 3** | Xuất tập dữ liệu SFT Baseline & KD | CPU Laptop / PC | RAM 4GB | **0 VNĐ** | 5 phút |
| **Giai đoạn 4A** | Train Qwen2.5-1.5B KD (3 epoch) | GPU RTX 4060 8GB / Colab T4 | VRAM ~4.2 GB | **0 VNĐ** | ~2.5 giờ |
| **Giai đoạn 4B** | Train Qwen2.5-3B KD (3 epoch) | GPU RTX 4060 8GB / Colab A100 | VRAM ~6.8 GB | **0 VNĐ** | ~1.8 giờ |
| **Giai đoạn 5** | Eval BLEU & Exec EM (4.8k test) | GPU RTX 4060 + Neo4j Docker | VRAM ~3.5 GB | **0 VNĐ** | ~35 phút |
| **Giai đoạn 6** | Convert GGUF & Demo Edge | CPU Laptop / Ollama CLI | RAM ~1.5 GB | **0 VNĐ** | 10 phút |
| **TỔNG CỘNG** | **Toàn bộ Quy trình từ A - Z** | **GPU RTX 4060 + Gemini API** | **VRAM max 6.8GB** | **~120.000 VNĐ** | **~9.5 giờ** |

---

## 4. Cấu Trúc Mã Nguồn Chuẩn Hóa

```text
Text2Cypher/
├── configs/
│   └── config.py                 # Cấu hình API Key, Paths, Hyperparameters, Model IDs, Neo4j
├── data/
│   ├── raw/                      # Dữ liệu thô tải từ HuggingFace
│   └── processed/                # train_cleaned.json (37.4k), test_full.json (4.8k), test_executable.json (2.4k)
├── src/                          # THƯ MỤC MÃ NGUỒN CỐT LÕI (CORE ENGINE)
│   ├── data/
│   │   ├── cleaner.py            # Khử trùng rò rỉ dữ liệu giữa tập Train và Test (39.5k -> 37.4k clean)
│   │   └── formatter.py          # Đóng gói dữ liệu SFT Baseline vs Proposed Ontology CoT KD
│   ├── extraction/
│   │   ├── teacher.py            # API Extractor đa nhà cung cấp (Gemini Flash / OpenAI / vLLM)
│   │   └── validator.py          # Bộ lọc 3 tầng kiểm duyệt tri thức under Closed World Assumption (CWA)
│   ├── prompts/
│   │   └── teacher_prompts.py    # System prompt chuẩn hóa ép tư duy 4 bước dạng JSON
│   ├── training/
│   │   └── trainer.py            # Fine-tuning Student SLMs (Qwen2.5-1.5B/3B) với Unsloth + QLoRA 4-bit
│   ├── evaluation/
│   │   ├── metrics.py            # Thang đo Google-BLEU (SacreBLEU) & Execution Exact Match (Neo4j)
│   │   └── evaluator.py          # Quy trình đánh giá benchmark tự động trên tập test
│   └── utils/
│       ├── logger.py             # Utility logging sạch
│       ├── json_parser.py        # Robust parser bóc tách 4-step CoT JSON
│       └── cleanup.py            # Utility tự động dọn dẹp file script cũ
├── scripts/                      # CHUỖI SCRIPT THỰC THI CHUẨN NỐI TIẾP (01 -> 06)
│   ├── 01_prepare_data.py        # Stage 1: Tải & làm sạch 37.4k train, 4.8k test
│   ├── 02_distill_teacher.py     # Stage 2: Chắt lọc tri thức Teacher 37.4k mẫu với 3-tier validation
│   ├── 03_export_sft_datasets.py # Stage 3: Xuất file train_baseline_37k.json & train_kd_37k.json
│   ├── 04_train_student.py       # Stage 4: Fine-tune Student SLM (Baseline vs Proposed KD)
│   ├── 05_evaluate.py            # Stage 5: Đánh giá Benchmark (Google-BLEU & Execution EM)
│   └── 06_demo_cli.py            # Stage 6: Giao diện Playground Demo tương tác CLI
├── outputs/                      # Thư mục chứa checkpoints, weights adapter và báo cáo benchmark
├── .gitignore                    # Cấu hình loại bỏ dữ liệu lớn, venv & cache khỏi Git
├── requirements.txt              # Danh sách thư viện phụ thuộc
└── README.md                     # Tài liệu hướng dẫn dự án
```

---

## 5. Hướng Dẫn Vận Hành Hệ Thống

### Bước 1: Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### Bước 2: Tải và Khử rò rỉ dữ liệu (Stage 1)
```bash
python scripts/01_prepare_data.py
```
*Kết quả*: Tạo file `data/processed/train_cleaned.json` (37.450 mẫu sạch) và `data/processed/test_full.json` (4.833 mẫu test).

### Bước 3: Chắt lọc tri thức Teacher 37.4k mẫu (Stage 2)
Điền `GEMINI_API_KEY` vào `configs/config.py` (hoặc đặt biến môi trường), sau đó chạy:
```bash
python scripts/02_distill_teacher.py
```
*Tính năng*: Tự động lưu checkpoint ngắt ngầm vào `clean_distillation_train_37k.json` và khôi phục thông minh khi bị đứt mạng/hết quota.

### Bước 4: Đóng gói dữ liệu huấn luyện SFT (Stage 3)
```bash
python scripts/03_export_sft_datasets.py
```
*Kết quả*: Xuất hai tập dữ liệu SFT: `train_baseline_37k.json` và `train_kd_37k.json`.

### Bước 5: Fine-tune mô hình học viên Student SLM (Stage 4)
- **Huấn luyện mô hình Proposed KD (Qwen2.5-1.5B)**:
  ```bash
  python scripts/04_train_student.py --mode kd --model_id unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit
  ```
- **Huấn luyện mô hình Baseline (Direct SFT)**:
  ```bash
  python scripts/04_train_student.py --mode baseline --model_id unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit
  ```

### Bước 6: Đánh giá Benchmark Thực tế (Stage 5)
```bash
python scripts/05_evaluate.py --adapter qwen2.5_1.5b_student_kd
```
*Kết quả*: Đo đạc thực tế chỉ số **Google-BLEU Score** và **Execution Exact Match (%)** trên CSDL Neo4j.

### Bước 7: Chạy Giao diện Demo Playground tương tác CLI (Stage 6)
```bash
python scripts/06_demo_cli.py
```

---

## 6. License và Tham Chiếu Nguồn

* **Dataset Reference**: `neo4j/text2cypher-2024v1`
* **Original Paper Reference**: M. Ozsoy et al., *"Text2Cypher: Bridging natural language and graph databases"*, arXiv:2412.10064, Dec 2024.
* **Trường đào tạo**: Trường Công nghệ thông tin và Truyền thông – Đại học Bách khoa Hà Nội (HUST).

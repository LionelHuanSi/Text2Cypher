# Báo cáo tiến độ: Ontology-Constrained Knowledge Distillation cho Text-to-Cypher

*Cập nhật lần cuối: 20/11/2026. File này được cập nhật sau mỗi bước; mọi số liệu đều có đường dẫn tới kết quả gốc.*

## 0. Mục tiêu, câu hỏi nghiên cứu, tiến độ

**Luận điểm cần chứng minh:** một ontology **xây tự động** cho một KG bất kỳ giúp **mọi model** sinh Cypher đúng hơn, và phần lợi này **vượt hơn những gì schema đã cho**. Sau đó, ontology được dùng để chưng cất xuống Student 1,5B chạy trên thiết bị biên.

| Câu hỏi nghiên cứu | Cách chứng minh | Trạng thái |
|---|---|---|
| **RQ0.** Baseline và điểm yếu là gì? | Tái lập trên CypherBench, phân loại lỗi | Xong (mục 1) |
| **RQ1.** Ontology tự xây có giúp **vượt hơn schema** không? | So 4 nhánh cùng model, cùng ngân sách: schema-only / schema + `graph_info` (chú thích tay của tác giả) / **ontology tự xây** / placebo | **Chưa chứng minh được với thẻ v0.1.** Đủ 4 nhánh trên 2 backend: C (ontology) = D (placebo) trên cả hai (chênh 0,04 và 0,00 điểm). Mức tăng +3,9 trên MLX đến từ khuôn thẻ (liệt kê lại các mẫu quan hệ đúng hướng), không phải từ tri thức ontology (mục 2.4) |
| **RQ2.** Có giúp **model nào cũng lợi** không? | Lặp RQ1 trên ≥ 3 họ model (Qwen, Llama, Gemma/Phi), nhiều cỡ (1,5B → 7B) | Chưa bắt đầu |
| **RQ3.** Có **tổng quát cho KG bất kỳ** không? | Builder chỉ được nhìn đồ thị train khi thiết kế, rồi chạy nguyên trạng trên 7 đồ thị test chưa thấy; báo kết quả theo từng đồ thị | Chưa bắt đầu |
| **RQ4.** Có **chưng cất** xuống 1,5B mà giữ được lợi ích không? | Teacher + ontology sinh dữ liệu → lọc/DPO theo ontology → Student 1,5B; so EX và tỷ lệ vi phạm với Teacher | Chưa bắt đầu (mục 3) |

**Kết quả chính đến nay:**
- Teacher (Qwen2.5-Coder-7B-Instruct, zero-shot) đạt **21,59% EX** trên test CypherBench.
- **28,5% số câu** sai vì vi phạm schema: đảo hướng, sai kiểu đầu cạnh, hoặc dùng phần tử không tồn tại. Nhóm này **schema đã đủ** để bắt, nên tính cho nhánh schema-only.
- Các nhóm cần **tri thức ngoài schema** có EX gần 0:
  - quan hệ phụ thuộc thời gian: 1,7%;
  - hai quan hệ cùng nối một cặp thực thể, mẫu `(n)=(m)`: 8,5%;
  - UNION: 2,9%;
  - OPTIONAL MATCH: 4,5%.

  Đây là chỗ ontology phải chứng minh giá trị riêng.
- **Thẻ ontology đưa vào prompt (v0.1) chưa cho lợi ích riêng.** Trên MLX, C đạt 25,51 so với A 21,59, nhưng placebo D (cùng khuôn, nội dung bị xáo) cũng đạt 25,47. Trên Kaggle (fp16), C = D = 18,31, cả hai thấp hơn A (20,87). Kết luận: phần lợi trên MLX là do khuôn thẻ (liệt kê lại `(:A)-[:R]->(:B)` đúng hướng), là thông tin schema; các tri thức ontology (nghĩa, bản số, bắt buộc, thời gian) đọc qua prompt **không** giúp model 7B. Hướng tiếp theo: dùng ontology **chủ động** (bộ kiểm/sửa, ràng buộc khi sinh, tín hiệu huấn luyện) thay vì chỉ đưa vào prompt (mục 2.4).

*Đối chiếu với đề cương của thầy:*
- Giai đoạn 1 (chọn baseline, tái hiện, phân tích điểm yếu) tương ứng mục 1.
- Giai đoạn 2:
  - distillation và ràng buộc ontology qua loss/DPO: mục 3;
  - so sánh Student với Teacher theo EX và tỷ lệ vi phạm ontology: mục 3.
- Mục 2 (chứng minh ontology) là phần nền, phải có trước để ràng buộc ở Giai đoạn 2 có cơ sở.

---

## 1. Baseline, tái lập và điểm yếu (RQ0)

### 1.1 Lựa chọn baseline

| | Lựa chọn | Lý do |
|---|---|---|
| **Benchmark** | **CypherBench**: Feng, Papicchio, Rahman, *ACL 2025*, [arXiv:2412.18702](https://arxiv.org/abs/2412.18702) | Dữ liệu đầy đủ, công khai (Apache-2.0); 11 đồ thị Neo4j thật từ Wikidata; **test trên đồ thị chưa từng thấy**, nên đúng bài toán "ontology cho KG bất kỳ"; có metric thực thi chính thức |
| **Phương pháp SOTA cần vượt** | **Cypher-RI**: Su et al., *NeurIPS 2025* | SOTA trên CypherBench: **69,59% EX**, vượt GPT-4o (60,18); **cùng base model 7B** với đề tài |
| **Teacher** | Qwen2.5-Coder-7B (Instruct và base) | Là base model của Cypher-RI, nên so sánh cùng điều kiện; chạy được trên máy cá nhân |
| **Đã loại** | MedT2C (dữ liệu y tế cũ) | Chỉ 2 KG; nhãn gold nhiễu; ontology gần như trùng schema, nên khó chứng minh đóng góp. Giữ y tế làm **ứng dụng demo** của đồ án (cần thầy đồng ý) |

**Mốc công bố cần vượt (EX trên test CypherBench):**

| Model | EX (%) | Nguồn |
|---|---:|---|
| Claude-3.5-Sonnet zero-shot | 61,58 | CypherBench |
| GPT-4o zero-shot | 60,18 | CypherBench |
| Llama3.1-8B zero-shot | 18,82 | CypherBench |
| Llama3.2-3B zero-shot | 11,20 | CypherBench |
| Qwen2.5-Coder-7B zero-shot | 13,12 | Cypher-RI |
| Qwen2.5-Coder-7B SFT (train 85,66, test 44,34) | 44,34 | Cypher-RI |
| **Qwen2.5-Coder-7B Cypher-RI** (RL, 4×A800) | **69,59** | Cypher-RI |

**Khoảng trống nghiên cứu:**
- Chưa có công trình nào trên CypherBench làm distillation xuống model ≤ 3B.
- Chưa có công trình nào dùng ontology tự xây để ràng buộc truy vấn.
- Cypher-RI chỉ học *chọn schema*, không có ràng buộc ontology, và không có model nhỏ.

### 1.2 Dữ liệu

| | Số lượng | Ghi chú |
|---|---|---|
| Train | 8.534 câu, 4 đồ thị (art, biology, soccer, terrorist_attack) | Bài báo ghi 8.817; bản HF hiện tại ít hơn |
| Test | 2.348 câu, 7 đồ thị **khác** (company 347, fictional_character 385, flight_accident 189, geography 366, movie 401, nba 270, politics 390) | Bài báo ghi 2.488 |
| Đồ thị test (bản đầy đủ) | Tổng khoảng 1,8 GB JSON; lớn nhất là politics: 885k thực thể, 1,55 triệu cạnh | Nạp vào 7 Neo4j; kiểm tra bằng cách chạy gold: **2.348/2.348 ra đúng đáp án** |

- **Nguồn:** HF `megagonlabs/cypherbench` revision `fd80c26`, code tác giả commit `4cc727d`. Checksum lưu tại `data/cypherbench/manifest.json`.
- **Mỗi câu gồm:** câu hỏi, Cypher gold, đáp án, loại mẫu MATCH (12 loại) và loại RETURN (6 nhóm).
- **Mỗi thực thể** có QID Wikidata. Nhờ vậy có thể lấy thêm tri thức ontology (phân cấp lớp, mô tả quan hệ, ràng buộc) từ Wikidata, ngoài schema.
- **`graph_info.json`** (tác giả chú thích tay: đối xứng, phụ thuộc thời gian, bản số) được coi là **tri thức có sẵn** của benchmark, **không tính là đóng góp** của đề tài.

### 1.3 Metric (dùng nguyên code chính thức, không sửa)

- **EX (Execution Accuracy):** kết quả thực thi khớp đáp án, cho phép hoán vị hàng/cột. Đây là metric chính.
- **PSJS:** độ trùng (Jaccard) giữa subgraph mà phần MATCH lấy ra và subgraph của gold.
- **Exec.:** tỷ lệ truy vấn chạy được.
- **Bổ sung, do đề tài đặt ra:**
  - **tỷ lệ vi phạm ontology/schema** (mục 1.6);
  - khoảng tin cậy 95% bằng bootstrap.

### 1.4 Thiết lập tái hiện

| | Giá trị |
|---|---|
| Model | `Qwen/Qwen2.5-Coder-7B-Instruct` và `Qwen/Qwen2.5-Coder-7B` (base), trọng số bf16 gốc |
| Phần cứng | Apple M5, 32 GB RAM; suy luận bằng MLX |
| Giải mã | Greedy (temperature 0); tối đa 512 token (CypherBench) hoặc 2.000 token (Cypher-RI) |
| Prompt zero-shot | Nguyên văn từ code CypherBench (`src/cb_common.py`): yêu cầu dịch câu hỏi sang Cypher theo schema + 4 quy tắc + schema JSON + câu hỏi |
| Hạ tầng | 7 Neo4j bằng image chính thức `megagonlabs/neo4j-with-loader:2.4`. Loader trong container bị tràn bộ nhớ với movie/politics, nên được chạy từ host bằng **chính loader.py của tác giả**; chấm lần lượt từng đồ thị |

### 1.5 Kết quả tái hiện

**(a) Teacher zero-shot, giao thức CypherBench, đủ 7 đồ thị (2.348 câu):**

| Đồ thị | n | EX | PSJS | Exec. |
|---|---:|---:|---:|---:|
| company | 347 | 16,71 | 23,58 | 80,12 |
| fictional_character | 385 | 22,60 | 35,96 | 76,88 |
| flight_accident | 189 | 43,92 | 52,47 | 91,01 |
| geography | 366 | 23,50 | 34,50 | 81,97 |
| movie | 401 | 19,95 | 28,74 | 73,32 |
| nba | 270 | 22,96 | 36,40 | 77,78 |
| politics | 390 | 13,08 | 21,50 | 72,31 |
| **Tổng** | **2.348** | **21,59** (CI95 19,9–23,3) | **31,65** | **78,02** |

Nguồn: `results/cypherbench/zero_shot/qwen2.5-coder-7b-instruct/metrics.json`

**(b) Đối chiếu mốc 13,12 của Cypher-RI.** Cypher-RI không công bố code và không nói mốc 13,12 được đo bằng prompt nào. Table 6 của bài cho biết họ dùng **bản base**. Ta thử 3 giao thức trên cùng 210 câu (30 câu mỗi đồ thị):

| Giao thức | EX | PSJS | Exec. |
|---|---:|---:|---:|
| Instruct + prompt CypherBench + chat template | 21,90 | 31,01 | 76,19 |
| Base + prompt CypherBench (văn bản thô) | 22,86 | 34,38 | 92,38 |
| **Base + template Cypher-RI** (tự chọn schema 2 lượt, Table 1) | **16,19** (CI95 ≈ 11–21) | 25,97 | 86,67 |
| *Bài báo Cypher-RI* | *13,12* | *22,65* | *62,95* |

Nguồn: `results/cypherbench/protocol_probe/`

**Kết luận:**
- Chỉ "base + template Cypher-RI" có khoảng tin cậy chứa 13,12, nên nhiều khả năng mốc của họ được đo theo cách này.
- Riêng template tự chọn schema đã làm model chưa huấn luyện **giảm khoảng 6 điểm EX** so với đưa toàn bộ schema.
- Exec. vẫn chênh (87 so với 63), có thể do cách giải mã hoặc tách đáp án mà bài không mô tả.

**Mốc dùng trong luận văn:**
1. **21,59**: zero-shot theo giao thức CypherBench. Đây là mốc chính, khắt khe hơn 13,12.
2. **Base + template Cypher-RI**: dùng khi so cùng điều kiện với 13,12 và 69,59.

### 1.6 Điểm yếu của Teacher (2.348 câu)

**Phân loại lỗi tự động** (`src/cb_error_analysis.py`):
- **Cách làm:** đối chiếu Cypher dự đoán với schema, không cần chạy truy vấn.
- **Độ tin cậy:** trên gold không báo nhầm câu nào (0/2.348). Trong 507 câu Teacher trả lời đúng, chỉ 4 câu bị báo vi phạm, và đều là vi phạm thật nằm trong một nhánh `UNION`.

| Loại | Số câu | % toàn bộ | Ví dụ |
|---|---:|---:|---|
| Đúng | 507 | 21,59 | |
| Không chạy được (cú pháp/runtime) | 516 | 21,98 | `OR` giữa hai pattern; `{name IN [...]}` trong node |
| **Đảo hướng quan hệ** | **448** | **19,08** | `(Person)-[:hasBoardMember]->(Company)`, đúng ra phải là `Company → Person` |
| **Sai kiểu hai đầu cạnh** | **171** | **7,28** | `(Country)-[:operatesIn]->(Industry)` |
| **Label/quan hệ/thuộc tính không tồn tại** | **50** | **2,13** | `c.foundedBy.country_of_citizenship`, `[:place_of_birth]` |
| Hợp lệ schema nhưng sai kết quả | 656 | 27,94 | Sai ngữ nghĩa, lọc, gộp nhóm, liên kết thực thể |

- **Vi phạm schema/ontology (3 dòng in đậm):** 669 câu, tức **28,5%**.
- Nếu tính cả các câu không chạy được mà vẫn có vi phạm, có **815 câu sai có ít nhất một vi phạm (34,7%)**.
- Chi tiết: `results/cypherbench/zero_shot/qwen2.5-coder-7b-instruct/error_analysis.json`.

**EX theo loại mẫu truy vấn (yếu nhất):**

| Mẫu | n | EX (%) |
|---|---:|---:|
| Phụ thuộc thời gian (time-sensitive) | 60 | **1,7** |
| UNION | 310 | **2,9** |
| OPTIONAL MATCH | 268 | **4,5** |
| Hai quan hệ cùng nối một cặp thực thể `(n)=(m)` | 94 | 8,5 |
| Lọc bằng WHERE trên kết quả (RETURN `n_where`) | 73 | 5,5 |
| Sắp xếp ORDER BY | 57 | 10,5 |
| *Truy vấn 1 node (để so sánh)* | 61 | *88,5* |

**Theo đồ thị:** khó nhất là politics (13,1) và company (16,7); dễ nhất là flight_accident (43,9).

**Hệ quả cho Giai đoạn 2:**
1. **Đảo hướng, sai kiểu đầu cạnh và phần tử không tồn tại chiếm 28,5% số câu, và đều phát hiện được tự động.** Bộ kiểm vi phạm này dùng được ngay làm (i) bộ lọc dữ liệu chưng cất, (ii) tín hiệu phạt trong loss/DPO, và (iii) metric "tỷ lệ vi phạm ontology".
2. Các nhóm *time-sensitive*, OPTIONAL MATCH (cần biết quan hệ nào không bắt buộc) và cặp hai quan hệ cần tri thức mà schema không nói rõ. Đây là chỗ **ontology chứng minh giá trị riêng**, vượt hơn schema thuần.
3. Lỗi "đảo hướng" bắt được chỉ bằng schema. Vì vậy thí nghiệm phải có nhánh đối chứng **schema-only**, để tách phần đóng góp riêng của ontology.

### 1.7 Chưa làm / cần thầy quyết định

- **Chưa tái hiện huấn luyện RL của Cypher-RI.** Họ cần 4×A800 huấn luyện toàn bộ tham số, và không công bố code hay checkpoint.

**Cách so sánh với Cypher-RI khi không chạy lại được phương pháp của họ:**

1. **So với số công bố.** Đây là cách làm chuẩn khi đối thủ không công bố code. Điều kiện là ta dùng cùng benchmark, cùng tập test (7 đồ thị), cùng metric EX/PSJS/Exec. và cùng code chấm chính thức. Trong bảng kết quả, số của Cypher-RI được đánh dấu "lấy từ bài báo".
2. **Chứng minh thiết lập của ta khớp thiết lập của họ**, bằng các điểm neo tự chạy lại:

| Điểm neo | Bài Cypher-RI | Ta tự chạy | Trạng thái |
|---|---:|---:|---|
| Base 7B zero-shot theo template của họ | 13,12 | 16,19 (CI95 ≈ 11–21, 210 câu) | Khớp trong sai số |
| GPT-4o, Llama3.1-8B… zero-shot | chép từ CypherBench | (dùng cùng code chấm) | Cùng nguồn |
| SFT 7B trên tập train | 44,34 | sẽ chạy (đây cũng là Teacher) | Kế hoạch |

   Nếu SFT tự chạy cũng ra khoảng 44, thiết lập hai bên được coi là tương đương. Khi đó so sánh với 69,59 là có cơ sở.
3. **Nêu rõ chỗ khác biệt** trong bài viết:
   - bản HF có 2.348 câu test, bài báo dùng 2.488;
   - Exec. zero-shot của ta cao hơn;
   - ta báo cả khoảng tin cậy.
4. **Không tuyên bố đánh bại Cypher-RI trên cùng cỡ model** nếu chưa đạt 69,59. Luận điểm chính của đề tài là **hiệu quả**: Student 1,5B, không cần RL trên nhiều GPU, chạy được trên thiết bị biên, có tỷ lệ vi phạm ontology thấp. Câu hỏi đặt ra là Student đạt được bao nhiêu phần trăm EX của Cypher-RI, với số tham số ít hơn bao nhiêu lần.
5. **(Tuỳ chọn, nếu có GPU)** dựng lại Cypher-RI ở quy mô nhỏ, tức RL kiểu GRPO với LoRA trên cùng dữ liệu, để có một baseline do chính ta chạy. Việc này tốn tài nguyên nên chỉ làm khi thầy/hội đồng yêu cầu.

**Đề xuất:** Teacher của đề tài là Qwen2.5-Coder-7B được **SFT/LoRA trên tập train CypherBench**. Thí nghiệm này đồng thời là điểm neo thứ hai ở bảng trên (đối chiếu với 44,34). Làm ở bước chưng cất (mục 3), **sau** khi đã chứng minh ontology ở mục 2.

- **Template Cypher-RI chỉ mới chạy trên 210 câu.** Chạy đủ 2.348 câu cần khoảng 20 giờ máy; sẽ chạy nếu cần một con số đầy đủ.

---

## 2. Chứng minh ontology (RQ1–RQ3, trọng tâm)

### 2.1 Ontology Builder: tự động, cho KG bất kỳ

Đầu vào chỉ gồm: một Neo4j bất kỳ, cộng với các QID Wikidata gắn trên thực thể nếu KG có. **Không dùng `graph_info.json`, không chú thích tay.**

| Thành phần | Nguồn tự động | Dùng để |
|---|---|---|
| Kiểu, quan hệ, **hướng**, cặp kiểu hai đầu | Introspection Neo4j | Kiểm hướng/kiểu (phần này schema đã có; tính cho nhánh schema-only) |
| **Bản số** (một/nhiều), bắt buộc/tuỳ chọn | Thống kê bậc trên dữ liệu Neo4j + ràng buộc Wikidata (P2302 single-value) | Biết khi nào cần `OPTIONAL MATCH`, `DISTINCT`, gộp nhóm |
| **Quan hệ đối xứng** | Wikidata (thuộc tính đối xứng) + kiểm cạnh hai chiều trên dữ liệu | Viết `(a)-[:R]-(b)` vô hướng |
| **Quan hệ phụ thuộc thời gian** | Thuộc tính `start/end_year` trên cạnh + qualifier P580/P582 của Wikidata | Điều kiện thời gian (EX hiện tại: 1,7%) |
| **Nghĩa của quan hệ/thuộc tính** | Mô tả và bí danh của property Wikidata (PID) | Chọn đúng quan hệ theo câu hỏi |
| **Phân cấp lớp** | Wikidata P31/P279 | Chọn đúng label |
| **Từ vựng giá trị** | Giá trị phổ biến của thuộc tính kiểu chuỗi (vd. `gender`) | Lọc đúng giá trị |

**Đầu ra:**
- (a) một **thẻ ontology** ngắn đưa vào prompt;
- (b) một **bộ luật kiểm** (mở rộng `src/cb_error_analysis.py`) dùng để lọc, xếp hạng và sửa truy vấn.

### 2.1b Kết quả Builder v0.1 (đã xong, 19/11/2026)

**Đã chạy** trên 10 đồ thị: 3 đồ thị train (art, soccer, terrorist_attack) và 7 đồ thị test. Biology (2,4 GB) chưa nạp.
- Mỗi đồ thị mất 1–80 giây, không cần người can thiệp.
- Kết quả: `ontology/<graph>.json`. Cache Wikidata (2.150 thực thể, 1,1 MB) lưu ở `ontology/cache/` để tái lập được.

**Cách lấy ngữ nghĩa từ Wikidata:**
- Lấy 12 cạnh mẫu của mỗi quan hệ.
- Tìm thuộc tính Wikidata (PID) nối hai QID ở hai đầu cạnh, theo chiều thuận hoặc ngược.
- Chọn PID có độ phủ cao nhất cộng với độ khớp tên giữa PID và tên quan hệ trong KG.
- Lấy nhãn, mô tả, bí danh và các ràng buộc của PID đó.
- Dùng Action API chứ không dùng SPARQL, vì SPARQL đang bị giới hạn còn 1 truy vấn/phút.

Ví dụ kết quả:

| Quan hệ trong KG | PID tự tìm | Nghĩa (Wikidata) |
|---|---|---|
| `nba.playsFor` | P54 | member of sports team |
| `nba.partOfDivision` | P361 | part of |
| `politics.hasHeadOfGovernment` | P6 | head of government |
| `fictional_character.hasMother` | P25 | mother |
| `flight_accident.involves` | P1876 | vessel |
| `geography.flowsInto` | P403 | mouth of the watercourse (Wikidata: bắc cầu) |

**Đối chiếu với chú thích tay của tác giả** (`graph_info.json`, `ontology/eval_vs_graph_info.json`). Ngưỡng chỉ được chỉnh trên đồ thị train.

| Trường | Train (13 quan hệ) | **Test (51 quan hệ)** |
|---|---:|---:|
| Đối xứng | 100 | **100** |
| Phụ thuộc thời gian | 100 | **100** |
| Bản số phía chủ thể | 92,3 | **96,1** |
| Bản số phía đối tượng | 84,6 | **84,3** |
| Bắt buộc (chủ thể) | 30,8 | 49,0 |
| Bắt buộc (đối tượng) | 53,8 | 78,4 |
| Tìm được PID Wikidata | 100 | **98,0** |

**Nhận xét:**
- Đối xứng, thời gian và bản số khớp 84–100% mà không cần chú thích tay. Như vậy phần tự động thay thế được công chú thích tay cho các trường này.
- Hai đồ thị có quan hệ đối xứng đều được phát hiện đúng bằng hai nguồn độc lập (dữ liệu hai chiều và Wikidata): `hasSpouse`, `hasDiplomaticRelationWith`.
- **Tính "bắt buộc" lệch nhiều là có lý do.** Tác giả chú thích theo khái niệm, ví dụ "mỗi vụ tấn công đều có mục tiêu". Trên dữ liệu thật thì chỉ 6% vụ có cạnh `targets`, vì Wikidata thiếu dữ liệu. Builder báo theo **dữ liệu thật**. Đây mới là điều quyết định truy vấn có cần `OPTIONAL MATCH` hay không, nên ta **không chỉnh để khớp chú thích tay**.
- Một số lệch về bản số cũng do khác nhau giữa khái niệm và dữ liệu. Ví dụ `movie.directedBy`: trên dữ liệu, ≥ 95% phim chỉ có một đạo diễn.

**Các lỗi đã sửa trong quá trình làm** (để minh bạch):
- Mã lớp "symmetric property" ghi nhầm: Q18647515 là *transitive*, mã đúng là Q18647518.
- Ràng buộc "một giá trị" của Wikidata không được áp cho quan hệ có khoảng hiệu lực.
- Chọn PID có xét tên. Trước đó `hasHeadOfGovernment` bị gán nhầm thành "country of citizenship".

**Thẻ ontology** (`src/ontology/card.py`):
- Ba nhánh `auto`, `graphinfo`, `placebo` dùng chung một khuôn thẻ.
- Độ dài thẻ: 430–840 token (auto, placebo) và 160–410 token (graphinfo).
- Placebo có cùng độ dài với auto, nhưng nội dung bị xáo giữa các quan hệ.

### 2.2 Thí nghiệm chứng minh (suy luận, chưa huấn luyện; rẻ, chạy trên máy cá nhân)

Mọi nhánh dùng cùng model, cùng prompt gốc, cùng số token ngữ cảnh và cùng số ứng viên K:

| Nhánh | Model nhận thêm gì | Trả lời câu hỏi |
|---|---|---|
| A. schema-only | Schema gốc + bộ kiểm hướng/kiểu từ schema | Riêng schema đã giúp được bao nhiêu |
| B. schema + `graph_info` | A + chú thích tay của tác giả | Mức trần "tri thức thủ công" |
| **C. ontology tự xây** | A + thẻ ontology + bộ luật ontology | **Đóng góp chính** |
| D. placebo | A + thẻ có cùng độ dài nhưng nội dung ngẫu nhiên hoặc sai | Loại trừ hiệu ứng "prompt dài hơn" |

**Tiêu chí thành công:**
- **C > A có ý nghĩa thống kê** (bootstrap ghép cặp), trên nhiều họ model.
- **C ≈ B hoặc C > B.**
- **D ≈ A.**
- Phần lợi tập trung ở các nhóm cần tri thức ngoài schema: thời gian, OPTIONAL MATCH, UNION, hai quan hệ cùng nối một cặp.

**Model:**
- Qwen2.5-Coder 1,5B / 7B;
- Llama-3.2-3B / Llama-3.1-8B;
- Gemma-2 hoặc Phi.

Đủ để trả lời RQ2.

**Thực hiện thực tế (lần chạy đầu):**
- Nhánh A = kết quả zero-shot ở mục 1.5: prompt chính thức của CypherBench với schema rút gọn, chưa có bộ kiểm.
- Nhánh C = cùng prompt đó, thêm thẻ ontology tự xây ngay sau schema (`--protocol cb_chat_card --card auto`).
- Cả hai: Qwen2.5-Coder-7B-Instruct (MLX), giải mã tham lam, 1 ứng viên/câu, không bộ kiểm, không chọn lọc. Như vậy khác biệt duy nhất là thẻ ontology.

### 2.3 Kết quả nhánh C so với nhánh A (19/11/2026)

*Lưu ý: kết luận của mục này đã bị thay thế bởi mục 2.4. Nhánh placebo cho mức tăng như nhánh C, nên mức tăng dưới đây không phải do nội dung ontology.*

Qwen2.5-Coder-7B-Instruct, 2.348 câu test, so ghép cặp theo từng câu. CI 95% là bootstrap ghép cặp (10.000 lần lấy mẫu lại câu hỏi); p là McNemar chính xác.

| | A. schema-only | C. + ontology tự xây | Chênh (C − A) | CI 95% | p |
|---|---:|---:|---:|---|---:|
| **EX** | 21,59 | **25,51** | **+3,92** | [2,60; 5,24] | 1,2e-8 |
| PSJS | 31,65 | 36,66 | +5,01 | | |
| Exec. | 78,02 | 81,26 | +3,24 | | |
| Câu có ít nhất một vi phạm schema | 34,88% | 30,41% | −4,47 | [−6,09; −2,81] | |
| Câu sai do vi phạm schema (loại 2–4) | 28,49% | 25,26% | −3,24 | [−4,86; −1,62] | |

Số câu chỉ A đúng: 84; chỉ C đúng: 176.

**Theo đồ thị (EX):**

| Đồ thị | A | C | Chênh | CI 95% |
|---|---:|---:|---:|---|
| company | 16,71 | 19,31 | +2,59 | [−0,29; 5,76] |
| fictional_character | 22,60 | 28,31 | +5,71 | [2,86; 8,58] |
| flight_accident | 43,92 | 45,50 | +1,59 | [−4,23; 7,41] |
| geography | 23,50 | 25,68 | +2,19 | [−1,09; 5,74] |
| movie | 19,95 | 25,19 | +5,24 | [1,75; 8,73] |
| nba | 22,96 | 30,74 | +7,78 | [3,70; 11,85] |
| politics | 13,08 | 15,13 | +2,05 | [−1,03; 4,87] |

Cả 7 đồ thị đều tăng; 3 đồ thị tăng có ý nghĩa. Đây là bằng chứng ban đầu cho RQ3: Builder không được chỉnh trên các đồ thị này.

**Phân loại lỗi (số câu / 2.348):**

| Loại | A | C |
|---|---:|---:|
| Đúng | 507 | 599 |
| Không chạy được | 516 | 440 |
| Đảo hướng quan hệ | 448 | 373 |
| Sai kiểu đầu cạnh | 171 | 156 |
| Phần tử không tồn tại | 50 | 64 |
| Hợp lệ schema nhưng sai | 656 | 716 |

**Theo loại câu hỏi (EX), các nhóm thay đổi rõ:**

| Nhóm | n | A | C | Chênh | CI 95% |
|---|---:|---:|---:|---:|---|
| `(n)-(m0*)` | 356 | 38,20 | 49,44 | +11,24 | [7,30; 15,73] |
| `(n)-(m0*),(n)-(m1*)` | 333 | 16,82 | 28,83 | +12,01 | [7,21; 16,82] |
| three-node-groupby | 260 | 20,38 | 25,38 | +5,00 | [1,15; 8,85] |
| time-sensitive | 60 | 1,67 | 8,33 | +6,67 | [1,67; 13,33] |
| `(n)=(m0)` (hai quan hệ cùng nối một cặp) | 94 | 8,51 | 11,70 | +3,19 | [0,00; 7,45] |
| OPTIONAL MATCH | 268 | 4,48 | 6,34 | +1,87 | [−0,75; 4,48] |
| UNION | 310 | 2,90 | 3,23 | +0,32 | [−1,61; 2,58] |
| `(n*)` (chỉ một thực thể) | 59 | 76,27 | 59,32 | **−16,95** | [−27,12; −8,47] |
| comparison | 147 | 40,82 | 34,01 | **−6,80** | [−11,56; −2,72] |

**Diễn giải:**
1. Thẻ ontology giúp rõ nhất ở câu hỏi đi qua quan hệ (các mẫu `(n)-(m)`): model chọn đúng hướng và đúng quan hệ hơn (đảo hướng giảm từ 448 xuống 373 câu).
2. Nhóm thời gian tăng từ 1,7% lên 8,3% nhưng vẫn thấp. Nhóm OPTIONAL MATCH và UNION gần như không đổi: chỉ đưa thông tin "không phải thực thể nào cũng có quan hệ này" vào prompt là chưa đủ để model tự dùng OPTIONAL MATCH.
3. **Tác dụng phụ: nối thừa.** Với câu hỏi chỉ về thuộc tính của một hoặc hai thực thể, model có thẻ hay chèn thêm quan hệ không cần thiết, khiến kết quả rỗng hoặc sai. Ví dụ: "Ukraine International Airlines đặt ở nước nào?" thì C sinh thêm `-[:operatedBy]->(:FlightAccident)`. Số câu có nhiều quan hệ hơn gold: A 148, C 234; riêng nhóm `(n*)` và comparison: 46 lên 76.
4. **Chưa kết luận được "vượt hơn schema" một cách chặt chẽ.** Mức tăng có thể một phần do prompt dài hơn hoặc do nhắc lại schema dưới dạng khác. Cần nhánh D (placebo, cùng độ dài) và nhánh B (`graph_info`).

**Việc tiếp theo cho RQ1:**
- Chạy nhánh D (placebo) và nhánh B (`graph_info`) với cùng model.
- Thẻ v0.2: chỉ đưa các dòng ontology liên quan đến câu hỏi (chọn theo thực thể/quan hệ được nhắc tới), để giảm nối thừa. Thêm gợi ý tường minh cho OPTIONAL MATCH và quan hệ thời gian.
- Thêm bộ kiểm hướng/kiểu sau khi sinh (sửa đảo hướng tự động), áp cho cả A và C.

Kết quả gốc: `results/cypherbench/rq1/qwen7b_instruct_C_auto/` (metrics.json, error_analysis.json), so sánh ghép cặp `results/cypherbench/rq1/compare_A_vs_C.json`.

### 2.4 Kết quả đủ 4 nhánh, 2 backend (20/11/2026): nội dung thẻ không có tác dụng

Hai môi trường chạy độc lập, cùng 9.392 prompt (dựng sẵn bởi `src/kaggle_build_prompts.py`; thẻ nhánh C giống hệt lần chạy MLX):
- **MLX**: Apple M5, bf16, sinh từng câu (A, C, D).
- **Kaggle**: 2x T4, transformers fp16, sinh theo lô có đệm trái (A, B, C, D). Dự kiến dùng vLLM nhưng vLLM 0.6.6 không cài được trên Python 3.13 của Kaggle nên chạy phương án dự phòng; mất 5,6 giờ GPU.

Hai backend cho dự đoán khác nhau nhiều (chỉ 17,8% dự đoán nhánh A giống hệt nhau; bản fp16 hay bọc ```` ```cypher ```` dù prompt cấm), nên **chỉ so trong cùng backend**.

**EX (2.348 câu):**

| Nhánh | MLX bf16 | Kaggle fp16 |
|---|---:|---:|
| A. schema-only | 21,59 | 20,87 |
| B. + `graph_info` (chú thích tay) | chưa chạy | 19,63 |
| C. + ontology tự xây | 25,51 | 18,31 |
| D. + placebo (cùng khuôn, nội dung xáo) | 25,47 | 18,31 |

**So ghép cặp (EX, CI 95% bootstrap, McNemar):**

| So sánh | MLX | Kaggle |
|---|---|---|
| C − A | +3,92 [2,60; 5,24], p = 1,2e-8 | −2,56 [−3,92; −1,24], p = 2e-4 |
| D − A | +3,88 [2,60; 5,20], p = 5e-9 | −2,56 [−3,88; −1,28], p = 1e-4 |
| **C − D** (tác dụng riêng của nội dung ontology) | **+0,04 [−0,77; 0,89]**, p = 1,0 | **0,00 [−0,81; 0,81]**, p = 1,0 |
| C − B | | −1,32 [−2,43; −0,26] |
| B − A | | −1,24 [−2,51; 0,00] |

Vi phạm schema (câu có ít nhất một vi phạm): C − D = −0,26 điểm (MLX) và −0,43 (Kaggle), đều không có ý nghĩa. Theo từng loại câu hỏi (time-sensitive, OPTIONAL MATCH, UNION, `(n)=(m)`), C − D đều nằm trong khoảng ±4 điểm và CI chứa 0.

**Diễn giải:**
1. **Nội dung ontology đọc qua prompt không có tác dụng đo được.** Đây là so sánh quyết định (C − D) và ra 0 trên cả hai backend. Model 7B không tận dụng được nghĩa Wikidata, bản số, tính bắt buộc hay thông tin thời gian khi chúng chỉ là văn bản trong prompt.
2. **Phần tăng +3,9 trên MLX là hiệu ứng khuôn thẻ.** Placebo chỉ xáo thuộc tính, vẫn giữ đúng các dòng `(:A)-[:R]->(:B)`. Việc liệt kê lại hướng quan hệ dưới dạng mẫu Cypher giúp giảm đảo hướng. Đó là thông tin **schema**, không phải ontology. Placebo này vì vậy kiểm soát được nội dung ontology, chưa kiểm soát độ dài thuần (văn bản ngẫu nhiên).
3. **Hiệu ứng khuôn không ổn định:** +3,9 trên bf16, −2,6 trên fp16. Thay đổi prompt có kết quả phụ thuộc môi trường số học, nên một kết quả prompting đơn lẻ không đủ tin cậy cho bài báo.
4. Chú thích tay của tác giả (B) cũng không giúp (−1,24 so với A trên Kaggle). Vậy vấn đề không riêng ở chất lượng ontology tự xây, mà ở **cách dùng**: đưa tri thức vào prompt.

**Hệ quả cho kế hoạch (cần thầy cho ý kiến):**
- Bỏ hướng "thẻ ontology trong prompt" làm đóng góp chính.
- Dùng ontology **chủ động**, ở chỗ model không cần tự đọc hiểu:
  1. *Bộ kiểm + sửa sau khi sinh*: phát hiện vi phạm (đảo hướng, sai đầu cạnh, sai bản số, thiếu OPTIONAL MATCH ở quan hệ không bắt buộc, thiếu điều kiện thời gian) rồi sửa tất định hoặc cho model sinh lại kèm thông báo lỗi cụ thể.
  2. *Chọn ứng viên*: sinh K câu, xếp hạng theo số vi phạm ontology.
  3. *Tín hiệu huấn luyện* (mục 3): lọc dữ liệu chưng cất và cặp DPO theo vi phạm ontology. Đây chính là hướng "Ontology Constraints qua Loss/DPO" trong đề cương; kết quả này là lập luận cho việc phải đưa ontology vào **huấn luyện** thay vì prompt.
- Phép thử vẫn giữ nguyên: mọi cách dùng phải thắng placebo (ontology xáo) trên ít nhất 2 họ model.

Kết quả gốc: `results/cypherbench/rq1/` (MLX) và `results/cypherbench/rq1_kaggle/` (Kaggle), các file `compare_*.json`.

### 2.5 Tham khảo tài liệu: các pipeline nhiều tầng (20/11/2026)

Tra cứu nhanh (arXiv, ACL Anthology, web), tập trung vào Text-to-Cypher và Text-to-SQL chia nhiều bước. Mức kiểm chứng: (đọc) = đã đọc trang bài báo; (tóm tắt) = chỉ qua tóm tắt/nguồn thứ cấp.

| Công trình | Các tầng | Kết quả báo cáo | Liên quan đến mình |
|---|---|---|---|
| **CypherGuard** (Chen, Wen, Zhang, Ozsoy; arXiv 2610.41755, 10/2026) (đọc) | Sinh nháp có thư viện luật + ví dụ truy xuất -> sửa tất định theo schema (sửa kiểu quan hệ, **lật hướng**, xử lý quan hệ hai chiều qua đo tỷ lệ cạnh đối ứng) -> chỉ khi lỗi/rỗng: lật toàn bộ hướng, **dò giá trị literal trong DB**, LLM sửa theo mã lỗi (tối đa 3 vòng) | **CypherBench test 2.348 câu**: gpt-5.4-mini 63,42 -> 88,50; claude-sonnet-4.6 71,76 -> 94,68. Bỏ vòng thực thi mất 10,8 điểm; bỏ few-shot 6,6; bỏ thư viện luật 4,4; bỏ dò literal 2,3 | Đối thủ trực tiếp, cùng benchmark. **Chỉ thử model API**, không thử model mở nhỏ; không báo theo loại câu hỏi; không dùng OPTIONAL MATCH/tính bắt buộc; thời gian chỉ là một luật (B3); không có đối chứng cùng lượng tính toán |
| **LAST-CQ** (Prokopiou và cs.; arXiv 2609.12746, EMNLP 2026 workshop) (đọc) | 5 agent, không huấn luyện: sinh -> chạy -> phát hiện lỗi -> sinh lại kèm phản hồi | Tầng **sửa lỗi** là nguồn lợi chính; phản hồi chi tiết theo schema **không hơn** chuỗi lỗi thô của DB; sinh song song cùng ngân sách làm **giảm** 10–11% | Phần quan trọng là **phát hiện đúng câu sai**, không phải lời nhắc dài. Ontology nên dùng để **phát hiện** lỗi mà schema không bắt được |
| **Multi-Agent GraphRAG** (Gusarov và cs.; arXiv 2511.08274) (đọc) | Sinh -> chạy -> trích thực thể/nhãn/giá trị -> **kiểm tồn tại trong DB**, đề xuất thay thế (Levenshtein + LLM xếp hạng) -> gộp phản hồi, tối đa 4 vòng | 5 đồ thị CypherBench, 150 câu/đồ thị: Qwen3-Coder 45,7 -> 53,4; GPT-4o 56,1 -> 62,9 (chấm bằng LLM, không ablation) | Schema viết dạng mẫu Cypher + giá trị mẫu giúp sinh tốt hơn, khớp với quan sát "khuôn thẻ" ở mục 2.4 |
| **DIN-SQL** (Pourreza, Rafiei; arXiv 2304.11015) (tóm tắt) | Liên kết schema -> **phân loại độ khó/loại câu** -> sinh theo loại -> tự sửa | Mốc gốc của hướng chia nhỏ | Tầng phân loại loại câu: ứng với các nhóm yếu của mình (thời gian, OPTIONAL, UNION) |
| **MAC-SQL, CHESS, CHASE-SQL** (tóm tắt, nguồn thứ cấp) | Chọn/tỉa schema -> tách câu hỏi -> sinh -> tinh chỉnh; CHASE-SQL sinh nhiều ứng viên + bộ xếp hạng | Độ phủ liên kết schema tương quan mạnh với câu đúng (DIN-SQL 69% khi đúng so với 39% khi sai) | Liên kết schema là tầng then chốt |
| **SERL-SQL, LitE-SQL, MSc-SQL, SQuaD-SQL** (tóm tắt) | Model nhỏ: chỉ mục giá trị + truy xuất giá trị theo câu hỏi + liên kết lai; chưng cất từ teacher | Không kiểm chứng số liệu | Hướng cho Student 1,5B: chưng cất cả pipeline |
| **cypher_validator** (PyPI), **CyVer** (Neo4j) (đọc tóm tắt) | Thư viện kiểm Cypher theo schema: nhãn, kiểu quan hệ, thuộc tính, **hướng**; vòng sửa bằng LLM | Công cụ, không có số liệu | Có thể dùng làm đối chứng cho bộ kiểm schema của mình |

**Kết luận cho thiết kế:**
1. Hướng "pipeline nhiều tầng + sửa lỗi theo schema" **đã có người làm mạnh trên đúng CypherBench** (CypherGuard 88,5 với model API). Đóng góp của mình không thể là "pipeline nhiều tầng" nói chung.
2. Khoảng trống còn lại, khớp với đề tài:
   - **Model mở nhỏ** (7B, rồi Student 1,5B chạy trên máy biên): chưa ai báo cáo pipeline kiểu này trên CypherBench.
   - **Tri thức ontology ngoài schema** dùng để *phát hiện và sửa* lỗi: tính bắt buộc -> OPTIONAL MATCH, khoảng thời gian -> lọc thời gian, đối xứng, bản số -> DISTINCT/gộp nhóm, nghĩa Wikidata -> liên kết quan hệ. CypherGuard chỉ có một phần nhỏ dưới dạng luật rời.
   - **Đối chứng chặt**: cùng pipeline chỉ dùng schema (tái hiện lớp sửa tất định kiểu CypherGuard) so với pipeline dùng ontology so với ontology placebo; báo theo loại câu hỏi. Các bài trên đều thiếu.
3. Theo LAST-CQ, đầu tư vào **bộ phát hiện lỗi** quan trọng hơn lời nhắc dài; theo mục 2.4, ontology trong prompt không tác dụng. Vậy ontology nên nằm ở tầng **kiểm tra/sửa** và **chọn ngữ cảnh**, không ở dạng văn bản mô tả.

**Pipeline đề xuất (mỗi tầng có bản schema-only và bản ontology để so):**

| Tầng | Việc | Phần ontology thêm vào (so với schema-only) |
|---|---|---|
| T0 (ngoại tuyến) | Builder ontology; chỉ mục giá trị; thư viện luật mined từ 4 đồ thị train | Bản số, bắt buộc, thời gian, đối xứng, nghĩa Wikidata |
| T1. Liên kết | Trích nhãn/quan hệ/thuộc tính/giá trị trong câu hỏi; dò giá trị trong Neo4j; cắt schema thành lát nhỏ dạng mẫu Cypher | Ghép từ trong câu với quan hệ qua nghĩa và tên Wikidata |
| T2. Phân loại | Loại câu hỏi (cơ bản / so sánh / thời gian / OPTIONAL / UNION / gộp nhóm) | Quan hệ có khoảng thời gian + mốc thời gian -> loại thời gian; quan hệ không bắt buộc -> cân nhắc OPTIONAL |
| T3. Sinh | 1 lần gọi, với lát schema + chỉ dẫn theo loại | |
| T4. Kiểm tĩnh | Hướng, kiểu hai đầu, phần tử lạ, thuộc tính sai nhãn | Thiếu lọc thời gian; đếm không DISTINCT qua quan hệ nhiều-nhiều; quan hệ đối xứng ghi hướng; MATCH bắt buộc trên quan hệ không bắt buộc khi câu hỏi cần giữ cả thực thể không có quan hệ |
| T5. Sửa | Sửa tất định (lật hướng, bỏ hướng) -> chạy -> lỗi/rỗng thì LLM sửa theo mã lỗi, tối đa 3 vòng | Mã lỗi ontology cụ thể |

**Thứ tự thực hiện:** T4 + T5 phần tất định trước, chạy trên dự đoán nhánh A đã có (không cần sinh lại), so 3 bản: schema-only, ontology, ontology placebo. Sau đó T1–T2, rồi vòng sửa có LLM (cần sửa vLLM trên Kaggle).

*Giới hạn của phần tra cứu:* tra nhanh, không phải tổng quan hệ thống; các bài 2026 phần lớn là preprint chưa bình duyệt; số liệu của MAC-SQL/CHESS/CHASE-SQL và nhóm model nhỏ lấy từ nguồn thứ cấp. Phần tra cứu có sự hỗ trợ của công cụ AI.

### 2.6 Thử nghiệm nhỏ: Teacher + ontology ở tầng kiểm/sửa (20/11/2026)

Mục đích: trước khi cài đặt đầy đủ, kiểm tra trên vài câu thật xem ontology có sửa được lỗi mà schema không sửa được không. Đồ thị company, dự đoán zero-shot của Teacher (Qwen2.5-Coder-7B-Instruct).

**Câu hỏi thời gian (3 câu, Teacher sai cả 3).** Lỗi chung: model viết `[:hasCEO {start_year: 1999, end_year: 1999}]` (đòi nhiệm kỳ bắt đầu và kết thúc đúng năm 1999) thay vì lọc khoảng `start <= 1999 AND (end >= 1999 OR end IS NULL)`.

| Cách sửa | Đúng |
|---|---:|
| Chỉ sửa theo ontology (viết lại điều kiện khoảng thời gian, tự động) | 1/3 |
| Chỉ sửa theo schema (lật hướng quan hệ) | 0/3 |
| **Schema (lật hướng) rồi ontology (khoảng thời gian)** | **3/3** |

Hai câu còn sai cả hướng quan hệ; mỗi tầng sửa một lỗi khác nhau, phải ghép mới đúng.

**Câu hỏi OPTIONAL MATCH (3 câu, Teacher sai cả 3).** Lỗi chung: đếm qua quan hệ không bắt buộc trong MATCH bắt buộc nên mất các thực thể có số đếm 0. Bộ phát hiện dùng ontology (quan hệ không bắt buộc, tỷ lệ tham gia thật trong dữ liệu) sinh thông báo lỗi; Teacher sửa lại một lần.

| Thông báo đưa cho Teacher | Đúng |
|---|---:|
| Chung chung ("câu có thể sai, kiểm tra lại") | 0/3 |
| **Từ ontology** ("chỉ 0,2% Company có hasBoardMember ... dùng OPTIONAL MATCH") | **2/3** |

Câu còn sai do đảo **nghĩa** của `subsidiaryOf` giữa hai Company: schema không bắt được (hai đầu cùng kiểu), cần tầng ngữ nghĩa (nghĩa Wikidata) của ontology.

**Nhận xét:**
- Tín hiệu ban đầu tốt nhưng mẫu rất nhỏ (6 câu), chưa kết luận.
- Khác với thẻ ở mục 2.4: ở đây ontology **chỉ ra lỗi cụ thể của câu đang xét** và sửa tự động khi được, thay vì mô tả chung cho model tự đọc.
- Bộ phát hiện bản thử còn báo thừa (đánh dấu cả quan hệ neo có tên riêng); sửa trong bản chính thức.
- Phát hiện thêm loại kiểm tra cần có: hướng **theo nghĩa** cho quan hệ nối cùng kiểu (`subsidiaryOf`, `parentOf`, ...).

**Bước tiếp theo:** cài đặt bộ kiểm/sửa đầy đủ (`src/ontology/verifier.py`), chạy trên toàn bộ 2.348 câu với các nhánh đối chứng: chỉ schema / schema + ontology / schema + ontology placebo / thông báo chung chung.

### 2.7 Pipeline trên tập dev, lần 1 (20/11/2026)

**Thiết lập.** Tập dev = 300 câu lấy ngẫu nhiên từ 3 đồ thị TRAIN (art, soccer, terrorist_attack; 100 câu/đồ thị). Mọi luật chỉ được chỉnh trên tập dev; 7 đồ thị test chỉ chạy một lần khi đã chốt. Teacher = Qwen2.5-Coder-7B-Instruct (MLX), cùng prompt CypherBench. Cùng một dự đoán của Teacher, 8 nhánh:

| Nhánh | Sửa tự động | Teacher sửa lại (1 lần) | EX dev |
|---|---|---|---:|
| A0 | không | không | 27,00 |
| S | schema | không | 31,67 |
| O | schema + ontology | không | 33,67 |
| Op | schema + ontology placebo | không | 32,67 |
| S+R | schema | thông báo schema + lỗi Neo4j | 32,67 |
| O+R | schema + ontology | thông báo schema/ontology + lỗi Neo4j | 34,33 |
| O+G | schema + ontology | thông báo chung chung (cùng tập câu bị đánh dấu) | 34,33 |
| Op+R | ontology placebo | thông báo từ ontology placebo | 33,33 |

**Theo từng luật (thắng / thua so với bước trước):**
- Lật hướng theo schema: kích hoạt 66 câu, +14 / −0.
- Sửa khoảng thời gian theo ontology: kích hoạt 28 câu, +6 / −0.
- Thông báo OPTIONAL từ ontology: kích hoạt 22 câu, +0 / −1.
- Teacher sửa lại nhờ lỗi Neo4j: 55 câu, +2.

**Vấn đề phát hiện được nhờ tập dev (chưa đụng tới test):**
1. Tập dev ngẫu nhiên có quá ít câu ontology nhắm tới (5 câu thời gian, 16 câu OPTIONAL / 300). Bổ sung **dev2** (154 câu chọn có chủ đích từ train: 40 câu thời gian, 60 câu OPTIONAL, 30 câu gom nhóm 3 nút và 24 câu một thực thể để đo báo nhầm).
2. Luật thời gian hiểu `{end_year: null}` là "hiện tại" kể cả khi câu hỏi nói "at any time / either currently or in the past". Theo quy ước rút từ đáp án chuẩn của đồ thị train (173 câu "at any time", 75 câu "either currently or in the past" đều **không** lọc thời gian; 40 câu "currently" lọc `end IS NULL`), luật được sửa để đọc tín hiệu thời gian trong câu hỏi; thêm luật O_TIME_CURRENT.
3. Luật OPTIONAL báo nhầm khi thực thể không phải khóa gom nhóm; sửa: chỉ báo khi thực thể nằm ngoài hàm gộp trong RETURN/WITH.
4. Placebo cũ tình cờ giữ lại dữ kiện thật (khi đồ thị có ít quan hệ, xáo ngẫu nhiên có thể trả `hasHomeVenue` về đúng chỗ). Placebo mới chọn hoán vị không giữ lại dữ kiện then chốt nào (0 quan hệ giữ đúng trên cả 10 đồ thị).

Đang chạy lại 8 nhánh trên dev1 + dev2 (454 câu).

## 3. Chưng cất xuống Student 1,5B (RQ4; chưa thực hiện)

```
Teacher 7B + ontology (nhánh C) sinh K Cypher/câu trên tập train (4 đồ thị train)
 → thực thi trên Neo4j + bộ luật ontology → giữ câu đúng và không vi phạm
 → SFT Student (Qwen2.5-Coder-1,5B)
 → DPO: cặp (đúng, hợp lệ) tốt hơn (vi phạm ontology hoặc sai); hoặc loss phạt vi phạm
 → đánh giá trên 7 đồ thị test
```

| Nhánh (cùng Student, cùng ngân sách) | Mục đích |
|---|---|
| SFT thường (lọc chỉ theo thực thi) | Mốc distillation |
| Lọc/DPO theo schema-only | Tách phần lợi từ schema |
| Lọc/DPO theo **ontology tự xây** | **Đóng góp chính** |
| Placebo | Loại trừ hiệu ứng "thêm dữ liệu" |

**Thước đo:**
- EX, PSJS, Exec.;
- **tỷ lệ vi phạm ontology**, so giữa Student và Teacher;
- kích thước model, độ trễ trên máy biên (Apple M5).

**So với Cypher-RI:** theo cách ở mục 1.7, tức điểm neo SFT 44,34 cộng với lập luận hiệu quả.

---

## 4. Nhật ký quyết định

| Ngày | Quyết định | Lý do |
|---|---|---|
| 18/11/2026 | Chuyển từ MedT2C sang **CypherBench** | Có test trên đồ thị chưa thấy, có QID Wikidata, có SOTA mạnh để so sánh |
| 18/11/2026 | Baseline SOTA = **Cypher-RI** (69,59) | Cùng base model 7B; mới nhất (NeurIPS 2025) |
| 18/11/2026 | Dùng **đồ thị đầy đủ** (không dùng bản sampled) khi chấm | Đúng thiết lập của bài báo |
| 18/11/2026 | Mốc zero-shot chính = **21,59** (tự đo), 13,12 chỉ để tham khảo | 13,12 không tái lập được theo giao thức CypherBench; mốc tự đo khắt khe hơn |
| 18/11/2026 | `graph_info.json` = tri thức có sẵn, **không** tính là đóng góp | Tránh bị phản biện "ontology chỉ là chép lại chú thích tay của tác giả" |
| 18/11/2026 | **Ưu tiên chứng minh ontology (mục 2) trước distillation** | Đóng góp chính của đề tài là ontology; thí nghiệm suy luận rẻ, cho bằng chứng sớm trên nhiều model |
| 19/11/2026 | Ontology báo **bản số/tính bắt buộc theo dữ liệu thật**, không chỉnh theo chú thích tay | Truy vấn đúng phụ thuộc dữ liệu thật (vd. có cần OPTIONAL MATCH không) |
| 19/11/2026 | So nhánh bằng **bootstrap ghép cặp + McNemar** trên cùng câu hỏi | Hai nhánh chạy trên cùng 2.348 câu; ghép cặp nhạy hơn so hai CI riêng |
| 19/11/2026 | Builder chỉ dùng Neo4j + Wikidata (qua QID), **không dùng câu hỏi benchmark hay graph_info**; ngưỡng chỉnh trên đồ thị train | Đảm bảo tổng quát cho KG bất kỳ (RQ3) và tránh rò rỉ từ test |
| 20/11/2026 | Chạy sinh trên **Kaggle 2x T4** (dataset prompt dựng sẵn) | Máy cá nhân 5 giờ/nhánh và ngủ giữa chừng; chấm vẫn làm trên máy vì cần Neo4j |
| 20/11/2026 | **Chỉ so trong cùng backend** (MLX bf16 riêng, Kaggle fp16 riêng) | Hai backend cho dự đoán khác nhau ở 82% câu |
| 20/11/2026 | Đối thủ trực tiếp mới: **CypherGuard** (88,5 trên CypherBench, model API). Định vị lại: model mở nhỏ + tri thức ontology ngoài schema + đối chứng chặt | Pipeline nhiều tầng nói chung đã có người làm trên cùng benchmark |
| 20/11/2026 | **Thẻ ontology trong prompt không phải đóng góp chính**; chuyển sang dùng ontology chủ động (kiểm/sửa, chọn ứng viên, huấn luyện) | C = placebo trên cả hai backend |

## 5. Tái lập kết quả

```bash
# Sinh + nạp đồ thị + chấm 7 đồ thị (zero-shot Teacher)
nohup scripts/cb_run_zero_shot_all7.sh > results/cypherbench/zero_shot_all7_pipeline.log 2>&1 &
# Đối chiếu giao thức (cb_chat | cb_raw | cypherri), 30 câu/đồ thị
.venv-cb/bin/python src/cb_run_protocol.py --model Qwen/Qwen2.5-Coder-7B --protocol cypherri --per_graph 30 \
    --out results/cypherbench/protocol_probe/base_cypherri
scripts/cb_eval_sequential.sh results/cypherbench/protocol_probe/base_cypherri
# Phân loại lỗi
.venv-cb/bin/python src/cb_error_analysis.py --pred_dir results/cypherbench/zero_shot/qwen2.5-coder-7b-instruct
# Nhánh C (thẻ ontology tự xây) + chấm + so ghép cặp với nhánh A
.venv-cb/bin/python src/cb_run_protocol.py --model Qwen/Qwen2.5-Coder-7B-Instruct --protocol cb_chat_card --card auto \
    --out results/cypherbench/rq1/qwen7b_instruct_C_auto
scripts/cb_eval_sequential.sh results/cypherbench/rq1/qwen7b_instruct_C_auto
cd src && ../.venv-cb/bin/python cb_compare.py --a ../results/cypherbench/zero_shot/qwen2.5-coder-7b-instruct \
    --b ../results/cypherbench/rq1/qwen7b_instruct_C_auto --out ../results/cypherbench/rq1/compare_A_vs_C.json; cd ..
# Sinh 4 nhánh trên Kaggle (2x T4) rồi chấm trên máy
cd src && ../.venv-cb/bin/python kaggle_build_prompts.py && cd ..
scripts/kaggle_rq1.sh push "Qwen/Qwen2.5-Coder-7B-Instruct" "A B C D"; scripts/kaggle_rq1.sh status; scripts/kaggle_rq1.sh pull
K=results/cypherbench/rq1_kaggle/qwen2.5-coder-7b-instruct; scripts/cb_eval_multi.sh $K/A $K/B $K/C $K/D
# Ontology Builder cho 10 đồ thị + đối chiếu với graph_info.json
scripts/build_ontologies.sh
# Xem thẻ ontology của một đồ thị (auto | graphinfo | placebo)
cd src && ../.venv-cb/bin/python -m ontology.card politics auto
```

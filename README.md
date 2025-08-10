# ENTRep-ACMMM25-TRACK-3: KGER - Cầu nối ngữ nghĩa trong truy xuất hình ảnh nội soi

Chào mừng bạn đến với dự án **KGER (Knowledge-Grounded Endoscopic Retrieval)**!

## ✨ Tóm tắt dự án

Trong lĩnh vực chẩn đoán Tai Mũi Họng (ENT), việc tìm kiếm hình ảnh nội soi chính xác và liên quan là vô cùng quan trọng. Tuy nhiên, các hệ thống truyền thống thường gặp phải "khoảng trống ngữ nghĩa" - chúng chỉ nhìn thấy những điểm tương đồng về màu sắc, hình dạng, mà bỏ qua các chi tiết lâm sàng tinh tế.

**KGER** ra đời để giải quyết vấn đề này. Đây là một framework đa giai đoạn đột phá, kết hợp sức mạnh của:
* **Học Metric Sâu (Deep Metric Learning):** Sử dụng kiến trúc NanoCLIP để nhanh chóng tìm kiếm các ứng viên tiềm năng từ cơ sở dữ liệu lớn.
* **Mô hình đa phương thức lớn (LMM):** Tận dụng Gemini để tái xếp hạng kết quả, phân tích hình ảnh dựa trên kiến thức y khoa chuyên sâu.
* **Chiến lược kết hợp điểm số thông minh:** Giới thiệu Pos-Fuse, một thuật toán độc đáo để tổng hợp điểm số từ cả hai giai đoạn, đảm bảo kết quả cuối cùng vừa trực quan, vừa chính xác về mặt lâm sàng.

KGER không chỉ hiệu quả mà còn đáng tin cậy, giúp các bác sĩ và nhà nghiên cứu nhanh chóng tìm thấy những trường hợp tương tự, hỗ trợ chẩn đoán chính xác hơn.

---

## 🚀 Hướng dẫn khởi động nhanh

Để bắt đầu với dự án, bạn chỉ cần một vài bước đơn giản để thiết lập môi trường.

1.  **Tạo môi trường ảo:**
    ```bash
    python -m venv venv
    ```

2.  **Kích hoạt môi trường:**
    * **macOS/Linux:** `source venv/bin/activate`
    * **Windows:** `.\venv\Scripts\activate`

3.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```

---

## ⚙️ Các bước thực thi

Sau khi cài đặt xong, hãy làm theo các bước dưới đây để chạy toàn bộ quy trình của KGER.

1.  **Chuẩn bị dữ liệu**
    * **Lệnh:** `python -m scripts.prepare_data`
    * **Mô tả:** Lệnh này sẽ chuẩn bị và sắp xếp bộ dữ liệu train, validation và test từ thư mục `data`.
    * **Output:** Dữ liệu đã xử lý sẽ được lưu tại `./data/dataset`.

2.  **Huấn luyện mô hình**
    * **Lệnh:** `python -m scripts.train`
    * **Mô tả:** Bắt đầu quá trình fine-tune mô hình trên K-Fold để tạo ra các checkpoint hiệu quả.
    * **Output:** Các checkpoint và logs sẽ được lưu tại `./checkpoint/nano_clip/logs`.

3.  **Tải và giải nén dữ liệu test**
    * **Lệnh:** `python -m scripts.download_and_extract_dataset`
    * **Mô tả:** Tải về và giải nén bộ dữ liệu test phục vụ cho việc inference.
    * **Output:** Dữ liệu test sẽ được lưu tại `./data/downloaded_files`.

4.  **Truy xuất hình ảnh (Initial Retrieval)**
    * **Lệnh:** `python -m scripts.infer_retrieval`
    * **Mô tả:** Hệ thống sẽ truy xuất 5 hình ảnh hàng đầu dựa trên truy vấn của bạn.
    * **Output:** Kết quả sẽ được lưu tại `./data/result`.

5.  **Tái xếp hạng kết quả (Reranking)**
    * **Lệnh:** `python -m scripts.rerank_posfuse`
    * **Mô tả:** Bước cuối cùng và quan trọng nhất! Sử dụng chiến lược Pos-Fuse và Gemini để tinh chỉnh lại kết quả, đảm bảo sự chính xác về mặt lâm sàng.
    * **Output:** Kết quả cuối cùng sẽ nằm trong `./data/rerank_posfuse_result`.

---

## 📂 Cấu trúc thư mục

Dưới đây là cấu trúc dự án để bạn dễ dàng theo dõi và quản lý.
ENTREP-ACMMM25-TRACK-3/
├── checkpoints/
├── data/
├── nano_clip/
│   ├── __pycache__/
│   ├── postfuse_combiner.py
│   └── utils.py
├── config.yaml
├── inference/
│   └── ...
├── LICENSE
├── postprocess/
│   ├── __pycache__/
│   ├── postfuse_combiner.py
│   └── utils.py
├── README.md
├── requirements.txt
├── scripts/
│   ├── __pycache__/
│   ├── download_and_extract_dataset.py
│   ├── infer_retrieval.py
│   ├── prepare_data.py
│   ├── rerank_postfuse.py
│   ├── train.py
│   └── utils/
│       ├── __pycache__/
│       ├── config.py
│       ├── drive_utils.py
│       ├── get_path.py
│       └── logger.py
└── utils/
    ├── __init__.py
    ├── dataset.py
    ├── encoders.py
    ├── loss.py
    └── model.py

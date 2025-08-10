# # ENTRep-ACMMM25-TRACK-3: KGER: A Knowledge-Grounded Endoscopic Retrieval Framework
with a Fused Bi-Encoder and Gemini Re-ranking Pipeline

## ✨ Mô tả ngắn

Dự án này giới thiệu **KGER (Knowledge-Grounded Endoscopic Retrieval)**, một framework đa giai đoạn được thiết kế để giải quyết bài toán truy xuất hình ảnh nội soi tai mũi họng (ENT). KGER kết hợp hiệu quả giữa **học metric sâu (deep metric learning)**, khả năng suy luận của **mô hình đa phương thức lớn (LMM)** như Gemini, và một chiến lược kết hợp điểm số thông minh.

Hệ thống bắt đầu bằng việc sử dụng kiến trúc **NanoCLIP** để truy xuất các ứng viên tiềm năng một cách nhanh chóng. Tiếp theo, một giai đoạn tái xếp hạng tinh vi sẽ sử dụng **Gemini** để phân tích và suy luận về các hình ảnh, vượt qua những hạn chế về sự tương đồng hình ảnh bề mặt. Cuối cùng, chiến lược **Pos-Fuse** tổng hợp điểm số từ cả hai giai đoạn, đảm bảo kết quả cuối cùng vừa trực quan vừa chính xác về mặt lâm sàng. KGER đã chứng minh hiệu suất vượt trội trên bộ dữ liệu ENTRep, giúp thu hẹp khoảng cách ngữ nghĩa và cung cấp một công cụ mạnh mẽ cho lĩnh vực Tai Mũi Họng.

---

## 📁 Cấu trúc thư mục

```
.
├── requirements.txt             # Thư viện Python
├── config.py                   # Thiết lập chung
├── .env                        # (tuỳ chọn) biến môi trường
├── blacklist_builder/          # Tạo blacklist từ dữ liệu báo chí
│   └── blacklist_builder_app.py
│   └── builder/
├── llm_model/                  # Gọi và sử dụng mô hình LLM
├── utils/                      # Tiện ích dùng chung
├── dynamodb/                   # Tương tác với DynamoDB
├── mongodb/                    # Tương tác với MongoDB
├── agent/                      # Backend API agent để truy vấn
│   └── app.py
├── Frontend/                   # Giao diện người dùng
    ├── package.json
    └── ...
```

---

## 🛠️ Yêu cầu hệ thống

- Python >= 3.10
- pip

---

## 🧪 Thiết lập môi trường Python

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
```

---

## 🧱 Các bước thực thi dự án

Sau khi đã thiết lập môi trường, bạn có thể thực hiện các bước sau để chạy dự án:

### Bước 1️⃣  **Chuẩn bị dữ liệu**
    -   **Lệnh:** `python -m scripts.prepare_data`
    -   **Mô tả:** Chuẩn bị bộ dữ liệu **train**, **validation** và **test** từ thư mục `data`.
    -   **Lưu output tại:** `./data/dataset`

2.  **Huấn luyện mô hình**
    -   **Lệnh:** `python -m scripts.train`
    -   **Mô tả:** Tiến hành fine-tune mô hình trên K-Fold để tạo các checkpoint.
    -   **Lưu output tại:** `./checkpoint/nano_clip/logs`

3.  **Tải và giải nén bộ dữ liệu test**
    -   **Lệnh:** `python -m scripts.download_and_extract_dataset`
    -   **Mô tả:** Tải và giải nén bộ dữ liệu test để chuẩn bị cho quá trình truy xuất.
    -   **Lưu output tại:** `./data/downloaded_files`

4.  **Truy xuất hình ảnh**
    -   **Lệnh:** `python -m scripts.infer_retrieval`
    -   **Mô tả:** Thực hiện truy xuất ban đầu để lấy ra 5 hình ảnh hàng đầu dựa trên truy vấn đầu vào.
    -   **Lưu output tại:** `./data/result`

5.  **Tái xếp hạng kết quả**
    -   **Lệnh:** `python -m scripts.rerank_posfuse`
    -   **Mô tả:** Sử dụng chiến lược **Pos-Fuse** và mô hình **Gemini** để tái xếp hạng kết quả từ Bước 4, cho ra kết quả cuối cùng.
    -   **Lưu output tại:** `./data/rerank_posfuse_result`

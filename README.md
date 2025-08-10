# ENTRep-ACMMM25-TRACK-3

## Mô tả ngắn

Dự án này giới thiệu **KGER (Knowledge-Grounded Endoscopic Retrieval)**, một framework đa giai đoạn để truy xuất hình ảnh nội soi tai mũi họng (ENT). KGER kết hợp hiệu quả giữa **học metric sâu (deep metric learning)**, khả năng suy luận của **mô hình đa phương thức lớn (LMM)** như Gemini, và một chiến lược kết hợp điểm số thông minh.

Hệ thống bắt đầu bằng việc sử dụng kiến trúc **NanoCLIP** để truy xuất các ứng viên tiềm năng một cách nhanh chóng. Tiếp theo, một giai đoạn tái xếp hạng tinh vi sẽ sử dụng **Gemini** để phân tích và suy luận về các hình ảnh, vượt qua những hạn chế về sự tương đồng hình ảnh bề mặt. Cuối cùng, chiến lược **Pos-Fuse** tổng hợp điểm số từ cả hai giai đoạn, đảm bảo kết quả cuối cùng vừa trực quan vừa chính xác về mặt lâm sàng. KGER đã chứng minh hiệu suất vượt trội trên bộ dữ liệu ENTRep, giúp thu hẹp khoảng cách ngữ nghĩa và cung cấp một công cụ mạnh mẽ cho lĩnh vực Tai Mũi Họng.

---

## Hướng dẫn cài đặt

Bạn cần tạo một môi trường ảo để cài đặt các thư viện cần thiết cho dự án.

1.  **Tạo môi trường ảo `venv`:**
    ```bash
    python -m venv venv
    ```

2.  **Kích hoạt môi trường ảo:**
    -   Trên macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
    -   Trên Windows:
        ```bash
        .\venv\Scripts\activate
        ```

3.  **Cài đặt các gói thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Các bước thực thi dự án

Sau khi đã thiết lập môi trường, bạn có thể thực hiện các bước sau để chạy dự án:

### Bước 1: Chuẩn bị dữ liệu
Chuẩn bị bộ dữ liệu **train**, **validation** và **test** từ thư mục `data`.
```bash
python -m scripts.prepare_data

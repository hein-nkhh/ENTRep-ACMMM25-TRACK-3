# ENTRep-ACMMM25-TRACK-3: KGER: A Knowledge-Grounded Endoscopic Retrieval Framework with a Fused Bi-Encoder and Gemini Re-ranking Pipeline

## ✨ Overview

This project presents **KGER (Knowledge-Grounded Endoscopic Retrieval)**, a multi-stage framework designed to tackle the retrieval of endoscopic images in ENT (Ear-Nose-Throat) domain. KGER effectively combines **deep metric learning**, reasoning capabilities of **Large Multimodal Models (LMM)** like Gemini, and a smart score fusion strategy.

The system begins with **10 model NanoCLIP** (Kfold) architecture to quickly retrieve candidate images. Then, an advanced re-ranking phase uses **Gemini** to analyze and reason about the images, overcoming surface-level visual similarity limitations. Finally, the **Pos-Fuse** strategy fuses scores from both stages to ensure results are both visually intuitive and clinically accurate. KGER achieves state-of-the-art performance on the ENTRep dataset, bridging semantic gaps and providing a powerful tool for ENT applications.

---

## 📁 Project Structure

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

## 🛠️ System Requirements

- Python >= 3.10
- pip

---

## 🧪 Python Environment Setup

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

## 🧱 Execution Steps

After setting up the environment, follow these steps to run the project:

### Step 1️⃣: Prepare Data
```bash
python -m scripts.prepare_data
```
- Prepare **training**, **validation**, and **test** datasets from the `data` folder.
- Output location: `./data/dataset`

### Step 2️⃣: Train NanoCLIP Model
```bash
python -m scripts.train
```
- Fine-tune the model using K-Fold cross-validation to generate checkpoints.
- Checkpoints saved at: `./checkpoint/nano_clip/logs`

### Step 3️⃣: Download and Extract Test Dataset
```bash
python -m scripts.download_and_extract_dataset
```
-  Download and extract test images for retrieval experiments.
-  Output saved at: `./data/downloaded_files`

### Step 4️⃣: Initial Image Retrieval
```bash
python -m scripts.infer_retrieval
```
- Perform initial retrieval to get the top 5 candidate images based on the input query.
- Results saved at: `./data/result`

### Step 5️⃣: Re-rank Results
```bash
python -m scripts.rerank_posfuse
```
-  Use the **Pos-Fuse** strategy combined with the **Gemini** model to re-rank the results from Step 4, producing the final ranked list.
-  Re-ranked results saved at: `./data/rerank_posfuse_result`

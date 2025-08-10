# H3N1 at ENTRep-ACMMM25-TRACK-3: KGER: A Knowledge-Grounded Endoscopic Retrieval Framework with a Fused Bi-Encoder and Gemini Re-ranking Pipeline

## ✨ Overview

This project presents **KGER (Knowledge-Grounded Endoscopic Retrieval)**, a multi-stage framework designed to tackle the retrieval of endoscopic images in ENT (Ear-Nose-Throat) domain. KGER effectively combines **deep metric learning**, reasoning capabilities of **Large Multimodal Models (LMM)** like Gemini, and a smart score fusion strategy.

The system begins with **10 model NanoCLIP** (Kfold) architecture to quickly retrieve candidate images. Then, an advanced re-ranking phase uses **Gemini** to analyze and reason about the images, overcoming surface-level visual similarity limitations. Finally, the **Pos-Fuse** strategy fuses scores from both stages to ensure results are both visually intuitive and clinically accurate. KGER achieves state-of-the-art performance on the ENTRep dataset, bridging semantic gaps and providing a powerful tool for ENT applications.

---

## 📁 Project Structure

```
.
|── checkpoints/                                  # Stores trained model parameters and checkpoints from fine-tuning processes.
|── data/                                         # Contains the ENTRep-ACMMM-TRACK-3 dataset used for training and evaluation.
|── nano_clip/                                    # Core module implementing the NanoCLIP model pipeline: dataset creation, training, inference, and post-processing.
|   └── inference/
|   |   └── inference_model.py                    # Defines the inference model architecture and loading pre-trained weights.
|   |   └── retrieval.py                          # Implements the image retrieval logic and query processing.
|   └── postprocess/
|   |   └── posfuse_combiner.py                   # Implements the Pos-Fuse score fusion strategy for re-ranking retrieved results.
|   |   └── utils.py                              # Utility functions supporting post-processing operations.
|   └── utils/
|   |   └── circle_cropper.py                     # Helper functions for circular cropping of endoscopic images.
|   |   └── transforms.py                         # Image transformation and augmentation utilities used during training and inference.
|   └── __init__.py
|   └── dataset.py                                # Dataset class definitions and data loading pipeline for NanoCLIP.
|   └── encoders.py                               # Encoder model architectures (e.g., image and text encoders) used in NanoCLIP.
|   └── loss.py                                   # Loss function implementation.
|   └── model.py                                  # Main NanoCLIP model definition and forward pass logic.
|── scripts/                                      
|   └── download_and_extract_dataset.py           # Script to download and extract the test dataset automatically.
|   └── infer_retrieval.py                        # Script to run the initial image retrieval process.
|   └── prepare_data.py                           # Script for preparing and organizing datasets before training.
|   └── rerank_posfuse.py                         # Script to perform re-ranking of retrieval results using Pos-Fuse and Gemini.
|   └── train.py                                  # Script for training and fine-tuning the NanoCLIP model.
|── utils/                                        
|   └── config.py                                 # Configuration settings and parameter definitions.
|   └── drive_utils.py                            # Utility functions for interacting with Google Drive or other storage services.
|   └── get_path.py                               # Functions to handle and resolve dataset and file paths consistently.
|   └── logger.py                                 # Logger setup for tracking experiments and debugging.
|── .env                                          # (Optional) Environment variables to configure runtime parameters securely.
|── .gitinore
|── config.yaml                                   # YAML configuration file for project-wide settings and hyperparameters.
|── requirements.txt
```

---

## 🛠️ System Requirements

- Python >= 3.10
- pip

---

## 🧪 Python Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
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

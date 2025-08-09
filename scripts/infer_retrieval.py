import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from nano_clip.model import NanoCLIP
from nano_clip.inference.retrieval import ensemble_image_retrieval_topk 
from utils.config import load_config

cfg = load_config()

def main():
    image_dir = '/kaggle/input/acmmm2025/ENTRep_Private_Dataset_Update/imgs/ENTRep Private Dataset'
    RESULT_DIR = os.makedirs(cfg['data']['RESULT_TOP_K'], exist_ok=True)
    model_paths = [
        "/kaggle/input/t2i-acmmm-bi-encoder/logs/nano_clip_fold0/version_0/checkpoints/fold0_epoch=05_recall@5=1.0000.ckpt",
        "/kaggle/input/t2i-acmmm-bi-encoder/logs/nano_clip_fold1/version_0/checkpoints/fold1_epoch=07_recall@5=1.0000.ckpt",
        "/kaggle/input/t2i-acmmm-bi-encoder/logs/nano_clip_fold2/version_0/checkpoints/fold2_epoch=03_recall@5=1.0000.ckpt"
    ]
    
    # Chuẩn bị dữ liệu test
    df = pd.read_csv(cfg['data']['TEST_CSV'])
    text_queries = df["text"].tolist()
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
    results_df = ensemble_image_retrieval_topk(text_queries, image_paths, model_paths, k=5)

    results_df.to_csv("image_retrieval_results.csv", index=False)
    print("✅ Saved results to image_retrieval_results.csv")

if __name__ == "__main__":
    main()
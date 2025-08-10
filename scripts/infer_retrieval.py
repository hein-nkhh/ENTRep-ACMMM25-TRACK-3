import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from nano_clip.model import NanoCLIP
from nano_clip.inference.retrieval import ensemble_image_retrieval_topk 
from utils.config import load_config
from utils.get_path import get_model_paths

cfg = load_config()

def main():
    image_dir = cfg['data']['LIST_IMAGES']
    
    os.makedirs(cfg['data']['RESULT_TOP_K'], exist_ok=True)
    
    RESULT_DF = os.path.join(cfg['data']['RESULT_TOP_K'], "image_retrieval_results.csv")
    
    model_paths = get_model_paths(root_dir=cfg['logging']['save_dir'])
    
    # Chuẩn bị dữ liệu test
    df = pd.read_csv(cfg['data']['TEST_CSV'])
    text_queries = df["text"].tolist()
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
    results_df = ensemble_image_retrieval_topk(text_queries, image_paths, model_paths, k=5)

    results_df.to_csv(RESULT_DF, index=False)
    print("✅ Saved results to image_retrieval_results.csv")

if __name__ == "__main__":
    main()
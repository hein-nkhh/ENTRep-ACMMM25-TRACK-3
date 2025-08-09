import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from .inference_model import InferenceModel  
from nano_clip.model import NanoCLIP

def ensemble_image_retrieval_topk(text_queries, image_paths, model_paths, k=5):
    
    # Khởi tạo tất cả model
    models = [InferenceModel(path) for path in model_paths]
    
    # Tính image embeddings trước để tránh load lại nhiều lần
    all_model_image_embeddings = []
    for model in tqdm(models, desc="Computing image embeddings"):
        img_embeds = [model.get_image_embedding(p) for p in image_paths]
        img_embeds = torch.cat(img_embeds, dim=0)  # (num_images, embed_dim)
        all_model_image_embeddings.append(img_embeds)
        
    
    results = []
    for text in tqdm(text_queries, desc="Processing text queries"):
        image_scores = {} 

        for model, model_img_embeds in zip(models, all_model_image_embeddings):
            txt_embed = model.get_text_embedding(text=text)

            sim = F.cosine_similarity(txt_embed, model_img_embeds, dim=1).cpu().numpy()

            # Top k cho từng model
            top_k_idx = np.argsort(sim)[-k:][::-1]
            for idx in top_k_idx:
                img_path = image_paths[idx]
                image_scores[img_path] = image_scores.get(img_path, 0) + sim[idx]

        # Rerank theo tổng điểm
        sorted_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        top_k_image_names = [os.path.basename(img_path) for img_path, _ in sorted_images]

        # Lưu kết quả
        results.append({
            "text": text,
            "image_name": top_k_image_names
        })

    return pd.DataFrame(results)
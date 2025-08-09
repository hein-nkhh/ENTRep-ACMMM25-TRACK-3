import os
os.environ['HF_HOME'] = './checkpoints'

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from transformers import AutoTokenizer
from utils.logger import default_logger as logger
from utils.config import load_config
from nano_clip.model import NanoCLIP
from nano_clip.utils.transforms import get_transforms
from nano_clip.utils.circle_cropper import Circle_Cropper

cfg = load_config()


class InferenceModel:
    def __init__(self, model_path):
        self.device = torch.device(cfg['DEVICE']['preferred'] if torch.cuda.is_available() else "cpu")

        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = NanoCLIP.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.to(self.device)

        # Prepare text tokenizer
        txt_model = cfg['MODEL']['Lang_encoder']
        self.tokenizer = AutoTokenizer.from_pretrained(
            txt_model,
            cache_dir="./checkpoints/all-MiniLM-L6-v2/tokenizer"
        )

        # Circle Cropper
        self.cropper = Circle_Cropper()

        # Prepare image transform
        self.image_transform = get_transforms(mode="valid")

    def get_image_embedding(self, image_path):
        # image = Image.open(image_path).convert("RGB")
        image = self.cropper(image_path)
        image = Image.fromarray(image)
        image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_embed = self.model.img_encoder(image_tensor)
            # img_embed, _ = self.model(image_tensor, None, None)

        return img_embed  # shape: (1, embed_dim)

    def get_text_embedding(self, text):
        tokenized = self.tokenizer(
            text,
            padding='max_length',
            max_length=cfg['MODEL']['max_length'],
            truncation=True,
            return_tensors='pt'
        )
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        with torch.no_grad():
            # _, txt_embed = self.model(None, input_ids, attention_mask)
            txt_embed = self.model.txt_encoder(input_ids, attention_mask)

        return txt_embed
    
    def __call__(self, image_path=None, text=None):
        img_emb = self.get_image_embedding(image_path) if image_path else None
        txt_emb = self.get_text_embedding(text) if text else None
        return img_emb, txt_emb
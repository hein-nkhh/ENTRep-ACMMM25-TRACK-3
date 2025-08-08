import os
os.environ["TORCH_HOME"] = "./checkpoints"

import torch
import torch.nn as nn
from transformers import AutoModel
from utils.logger import default_logger as logger

class ImageEncoder(nn.Module):
    """
    Image encoder using DINOv2 backbone from torch.hub.
    Allows partial fine-tuning of last N blocks.
    """
    SUPPORTED_MODELS = [
        'dinov2_vits14',
        'dinov2_vitb14',
        'dinov2_vitl14',
        # 'dinov2_vitg14'  # optionally excluded due to size
    ]

    def __init__(self, output_dim=64, img_model='dinov2_vits14', unfreeze_n_blocks=4):
        super().__init__()

        if img_model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Invalid model '{img_model}'. Choose one of: {self.SUPPORTED_MODELS}")
        
        logger.info(f"🧠 Loading DINOv2 image model: {img_model}")
        self.encoder = torch.hub.load('facebookresearch/dinov2', img_model)

        # Freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Unfreeze last N transformer blocks
        for block in self.encoder.blocks[-unfreeze_n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        
        # Unfreeze normalization layer
        for param in self.encoder.norm.parameters():
            param.requires_grad = True

        # Output projection
        self.fc = nn.Linear(self.encoder.embed_dim, output_dim)
        logger.info("🛠️ ImageEncoder initialized with output_dim=%d", output_dim)

    def forward(self, x):
        dino_output = self.encoder.forward_features(x)
        x = dino_output['x_norm_clstoken']  # Use CLS token
        x = self.fc(x)
        return x
    
class TextEncoder(nn.Module):
    """
    Text encoder using HuggingFace transformer backbone.
    Allows partial fine-tuning of last N encoder layers.
    """
    def __init__(self, output_dim=64, lang_model="sentence-transformers/all-MiniLM-L6-v2", unfreeze_n_blocks=4):
        super().__init__()
        logger.info(f"🧠 Loading language model: {lang_model}")
        self.encoder = AutoModel.from_pretrained(lang_model, cache_dir = "./checkpoints/all-MiniLM-L6-v2")

        # Freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Unfreeze last N transformer layers
        for layer in self.encoder.encoder.layer[-unfreeze_n_blocks:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Unfreeze the pooler layer if it exists
        if hasattr(self.encoder, "pooler"):
            for param in self.encoder.pooler.parameters():
                param.requires_grad = True

        self.fc = nn.Linear(self.encoder.config.hidden_size, output_dim)
        logger.info("🛠️ TextEncoder initialized with output_dim=%d", output_dim)

    def forward(self, input_ids, attention_mask=None):
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        x = self.fc(x)
        return x

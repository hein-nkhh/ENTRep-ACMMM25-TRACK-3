import os
os.environ['HF_HOME'] = './checkpoints'

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold
from transformers.utils.hub import cached_file
import numpy as np
import os

from nano_clip.dataset import ImageTextDataset, CollateImageText
from nano_clip.model import NanoCLIP
from utils.logger import default_logger as logger
from nano_clip.utils.transforms import get_transforms
from utils.config import load_config
from omegaconf import OmegaConf

cfg = load_config()

def main():
    
    # Get transform
    train_transform = get_transforms(mode="train")
    valid_transform = get_transforms(mode = "valid")
    
    # Get encoder model
    txt_model = cfg['MODEL']['Lang_encoder']
    img_model = cfg['MODEL']['ViT_encoder']
    
    # Load dataset and prepare K-Fold
    dataset = ImageTextDataset(csv_file=cfg['data']['TRAIN_CSV'], split="train")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        logger.info("➡️ Fold %d bắt đầu", fold)
        
        # Dataset
        train_dataset = ImageTextDataset(
            csv_file=cfg['data']['TRAIN_CSV'],
            split='train',
            img_transform=train_transform,
            indices=train_idx
        )

        val_dataset = ImageTextDataset(
            csv_file=cfg['data']['TRAIN_CSV'],
            split='val',
            img_transform=valid_transform,
            indices=val_idx
        )
        
        # Tokenizer cho Collate
        logger.info("📥 Tải tokenizer từ mô hình: %s", txt_model)
        tokenizer = AutoTokenizer.from_pretrained(
            txt_model,
            cache_dir="./checkpoints/all-MiniLM-L6-v2/tokenizer"
        )
        resolved_file = cached_file(txt_model, "tokenizer_config.json")
        logger.info("📦 Đường dẫn tokenizer: %s", resolved_file)
        
        # Dataloader
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=64, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True, 
            collate_fn=CollateImageText(tokenizer)
        )
        
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=64, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True,
            collate_fn=CollateImageText(tokenizer)
        )
        
        # Tensorboard training
        logger.info("📊 Khởi tạo TensorBoard logger cho fold %d", fold)
        tensorboard_logger = TensorBoardLogger(
            save_dir=cfg['logging']['save_dir'],
            name=f"nano_clip_fold{fold}",
            default_hp_metric=False
        )
        
        # Checkpoint
        checkpoint_cb = ModelCheckpoint(
            monitor="recall@5",
            filename=f"fold{fold}_epoch={{epoch:02d}}_recall@5={{recall@5:.4f}}",
            auto_insert_metric_name=False,
            save_weights_only=True,
            save_top_k=1,
            mode="max",
        )
        
        logger.info("📦 Khởi tạo mô hình NanoCLIP")
        model = NanoCLIP(
            txt_model=cfg['MODEL']['Lang_encoder'],
            img_model=cfg['MODEL']['ViT_encoder'],
            embed_size=cfg['MODEL']['embed_size'],
            unfreeze_n_blocks=cfg['MODEL']['unfreeze_n_blocks'],
            lr=cfg['OPTIMIZER']['lr'],
            warmup_epochs=cfg['OPTIMIZER']['weight_decay'],
            weight_decay=cfg['OPTIMIZER']['warmup_epochs'],
            milestones=tuple(cfg['OPTIMIZER']['milestones']),
            lr_mult=cfg['OPTIMIZER']['lr_mult'],
        )
        
        logger.info("🚀 Bắt đầu huấn luyện mô hình cho fold %d", fold)
        trainer = Trainer(
            accelerator="auto",
            devices="auto",
            logger=tensorboard_logger,      
            precision=cfg['TRAINING']['precision'],
            max_epochs=cfg['TRAINING']['max_epochs'],
            check_val_every_n_epoch=cfg['TRAINING']['check_val_every_n_epoch'],
            callbacks=[
                checkpoint_cb,              # this callback saves the best model based on the metric we monitor (recall@5)
                # early_stopping,
                # RichProgressBar()           # comment this line if you want classic progress bar
            ],
            log_every_n_steps=cfg['TRAINING']['log_every_n_steps'],
            fast_dev_run=cfg['TRAINING']['fast_dev_run'],
            enable_model_summary=True,
        )
        
        trainer.fit(model, train_dataloader, val_dataloader)
        
        best_score = checkpoint_cb.best_model_score
        logger.info("✅ [Fold %d] Best Recall@5: %.4f", fold, best_score.item())
        fold_scores.append(best_score.item())
    
    logger.info("📈 Recall@5 trung bình trên %d folds: %.4f", len(fold_scores), np.mean(fold_scores))

if __name__ == "__main__":
    main()
        


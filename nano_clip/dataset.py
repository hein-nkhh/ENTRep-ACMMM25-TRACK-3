import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from .utils.circle_cropper import Circle_Cropper
from utils.logger import default_logger as logger

class ImageTextDataset(Dataset):
    """
    Dataset hỗ trợ train/val/test dựa vào flag split.
    CSV chứa 2 cột: image_path và text (có thể thiếu ở test).
    """
    def __init__(self, csv_file, img_root_dir=None, split='train', img_transform=None, txt_transform=None, indices=None):
        
        logger.info("[{dataset}] 🔍 Đang load dataset từ %s", csv_file)
        self.data = pd.read_csv(csv_file)
        logger.info("[{dataset}] 📦 Tổng cộng %d samples", len(self.data))
        
        self.img_root_dir = Path(img_root_dir) if img_root_dir else None
        self.img_transform = img_transform
        self.txt_transform = txt_transform
        self.split = split.lower()
        self.cropper = Circle_Cropper()
        
        assert self.split in ['train', 'val', 'test'], "split phải là 'train', 'val', hoặc 'test'"
        
        if indices is not None:
            self.data = self.data.iloc[indices].reset_index(drop=True)
            logger.info("✂️ Lấy subset %d samples theo indices", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['image_path']
        # caption = self.data.iloc[idx]['text'] if 'text' in self.data.columns else None
        caption = row.get('text', None)

        if pd.isna(caption) or self.split == 'test':
            caption = None

        if self.img_root_dir:
            img_path = self.img_root_dir / img_path

        # image = Image.open(img_path).convert('RGB')
        image = self.cropper(img_path)
        if self.img_transform:
            image = self.img_transform(image)

        if caption is not None and self.txt_transform:
            caption = self.txt_transform(caption)

        return image, caption
    
class CollateImageText:
    """
    Collate function để dùng trong DataLoader.
    Tokenize caption nếu có tokenizer, nếu không sẽ chỉ trả về ảnh (cho test).
    """
    def __init__(self, tokenizer=None, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info("🧱 Tạo CollateImageText với tokenizer=%s, max_length=%d", 
                    tokenizer.__class__.__name__ if tokenizer else "None", max_length)

    def __call__(self, batch):
        images, captions = zip(*batch)
        images = torch.stack(images)

        if self.tokenizer is None or captions[0] is None:
            logger.debug("🧪 Test mode: chỉ trả về ảnh")
            return (images,)  # test mode

        encoding = self.tokenizer(
            list(captions),
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )

        return images, encoding['input_ids'], encoding['attention_mask']
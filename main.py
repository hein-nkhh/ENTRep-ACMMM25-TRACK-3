# import os
# os.environ['HF_HOME'] = './checkpoints'

# from nano_clip.utils import Circle_Cropper
# from nano_clip.dataset import ImageTextDataset, CollateImageText
# import torchvision.transforms as T
# from torchvision.transforms import InterpolationMode
# import torch
# from torch.utils.data import DataLoader
# from transformers import AutoTokenizer
# from transformers.utils.hub import cached_file
# import logging

# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# logger = logging.getLogger(__name__)

# def main():
    
#     logger.info("🚀 Bắt đầu")
    
#     txt_model = "sentence-transformers/all-MiniLM-L6-v2"
#     logger.info("📥 Tải tokenizer từ mô hình: %s", txt_model)
#     tokenizer = AutoTokenizer.from_pretrained(
#         txt_model,
#         cache_dir="./checkpoints/all-MiniLM-L6-v2/tokenizer"
#     )
#     resolved_file = cached_file(txt_model, "tokenizer_config.json")
#     logger.info("Đường dẫn tokenizer: %s", resolved_file)
    
#     logger.info("🛠️ Khởi tạo các train transform cho ảnh.")
#     train_transform = T.Compose([
#         T.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
#         T.RandomResizedCrop((224, 224), scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),  
#         T.RandomRotation(degrees=10),  
#         T.RandomHorizontalFlip(p=0.3),  
#         T.ColorJitter(brightness=0.1, contrast=0.1),  
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225]),
#         T.RandomErasing(p=0.3, scale=(0.02, 0.08), ratio=(0.3, 3.3), value='random'),
#     ])

#     logger.info("🧪 Khởi tạo validation transform cho ảnh.")
#     valid_transform = T.Compose([
#         T.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225]),
#     ])
    
#     dataset = ImageTextDataset(csv_file = "./data/train.csv", split='train', img_transform=train_transform)
#     logger.info("✅ Hoàn tất khởi tạo Dataset. Kích thước dataset: %d", len(dataset))
    
#     logger.info("📦 Tạo DataLoader cho train")
#     train_dataloader = DataLoader(
#         dataset, 
#         batch_size=64, 
#         shuffle=True, 
#         num_workers=4, 
#         pin_memory=True, 
#         collate_fn=CollateImageText(tokenizer)
#     )
    
#     logger.info("✅ Hoàn tất khởi tạo DataLoader. Tổng số batch: %d", len(train_dataloader))
        
# if __name__ == "__main__":
#     main()
    
from nano_clip.loss import ContrastiveLoss
from nano_clip.encoders import TextEncoder, ImageEncoder

def main():
    img_encoder = ImageEncoder(64, 'dinov2_vits14', 4)
    if img_encoder is None:
        print("Khong co image encoder")
    else:
        print("Co image encoder")
    
    text_encoder = TextEncoder(64, 'sentence-transformers/all-MiniLM-L6-v2', 4)
    if text_encoder is None:
        print("Khong co text encoder")
    else:
        print("Co text encoder")

if __name__ == "__main__":
    main()
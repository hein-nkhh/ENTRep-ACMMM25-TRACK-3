# from nano_clip.model import NanoCLIP
# import torch

# def main():
#     BATCH_SIZE = 2
#     IMG_SHAPE = (3, 224, 224)         # Cỡ ảnh phổ biến
#     SEQ_LEN = 16                      # Chiều dài caption
#     NB_CAPTIONS = 1                   # Số caption mỗi ảnh
#     EMBED_SIZE = 64

#     images = torch.randn(BATCH_SIZE, *IMG_SHAPE)
#     captions = torch.randint(0, 10000, (BATCH_SIZE * NB_CAPTIONS, SEQ_LEN))  # Giả sử vocab size ~10k
#     masks = torch.ones_like(captions)

#     model = NanoCLIP(
#         txt_model="sentence-transformers/all-MiniLM-L6-v2",
#         img_model='dinov2_vits14',
#         embed_size=EMBED_SIZE
#     )
    
#     with torch.no_grad():
#         img_emb, txt_emb = model(images, captions, masks)

#     print("✅ Model chạy được!")
#     print(f"Image embedding shape: {img_emb.shape}")
#     print(f"Text embedding shape: {txt_emb.shape}")

# if __name__ == "__main__":
#     main()


from nano_clip.utils.transforms import get_transforms

def main():
    train_transformers = get_transforms(mode="train")
    if train_transformers:
        print("Co train transform!")
    else:
        print("Loi train transform")
    
    valid_transformers = get_transforms(mode="valid")
    if valid_transformers:
        print("Co valid transform!")
    else:
        print("Loi valid transform")

if __name__ == "__main__":
    main()

from nano_clip.dataset import ImageTextDataset
from utils.config import load_config

cfg = load_config()
def main():
    train_dataset = ImageTextDataset(csv_file=cfg['data']['TRAIN_CSV'], split='train')
    if train_dataset is not None:
        print("NO")
    else:
        print("YES")

if __name__ == "__main__":
    main()
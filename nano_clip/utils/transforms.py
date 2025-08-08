import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from utils.config import load_config
from utils.logger import default_logger as logger

cfg = load_config()

def get_transforms(mode = "train"):
    if mode == "train":
        logger.info("🔧 Khởi tạo train transform.")
        train_transform = T.Compose([
            T.Resize(tuple(cfg['TRAIN']['resize']), interpolation=InterpolationMode.BICUBIC),
            T.RandomResizedCrop(tuple(cfg['TRAIN']['crop']), scale=tuple(cfg['TRAIN']['crop_scale']), interpolation=InterpolationMode.BICUBIC),
            T.RandomRotation(degrees=cfg['TRAIN']['rotation']),
            T.RandomHorizontalFlip(p=cfg['TRAIN']['flip_prob']),
            T.ColorJitter(**cfg['TRAIN']['color_jitter']),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            T.RandomErasing(p=cfg['TRAIN']['random_erasing']['p'],
                            scale=tuple(cfg['TRAIN']['random_erasing']['scale']),
                            ratio=tuple(cfg['TRAIN']['random_erasing']['ratio']),
                            value='random'),
        ])
        return train_transform
    
    elif mode == "valid":
        logger.info("🔧 Khởi tạo validation transform.")
        valid_transform = T.Compose([
            T.Resize(tuple(cfg['VALID']['resize']), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        return valid_transform

    else:
        raise ValueError(f"Sai transform mode: {mode}")

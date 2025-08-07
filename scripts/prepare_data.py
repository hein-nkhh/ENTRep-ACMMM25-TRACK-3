import os
import json
import csv
import random
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import load_config
cfg = load_config()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
TRAIN_IMG_DIR = cfg['TRAIN_IMG_DIR']
TEST_IMG_DIR = cfg["TEST_IMG_DIR"]
T2I_JSON = cfg["T2I_JSON"]
T2I_CSV = cfg['T2I_CSV']

OUTPUT_DIR = cfg["OUTPUT_DIR"]

VAL_RATIO = cfg["VAL_RATIO"]
random.seed(42)

os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(OUTPUT_DIR, "train.csv")
VAL_CSV = os.path.join(OUTPUT_DIR, "val.csv")
TEST_CSV = os.path.join(OUTPUT_DIR, "test.csv")

# === Load JSON ===
with open(T2I_JSON, 'r') as f:
    labels = json.load(f)

data = []
for text, filename in labels.items():
    img_path = os.path.join(TRAIN_IMG_DIR, filename)
    if os.path.exists(img_path):
        data.append((img_path, text))
    else:
        logger.warning(f"⚠️ File không tồn tại: {img_path}")

logger.info("Tổng số ảnh hợp lệ: %d", len(data))

# === Split train/val (tùy chọn) ===
# train_data, val_data = train_test_split(data, test_size=VAL_RATIO, random_state=SEED)
train_data = data
val_data = []

def write_csv(data_list, path):
    with open(path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "text"])
        for img_path, text in data_list:
            writer.writerow([img_path, text])

write_csv(train_data, TRAIN_CSV)
write_csv(val_data, VAL_CSV)
logger.info("✅ Đã tạo train.csv (%d mẫu), val.csv (%d mẫu)", len(train_data), len(val_data))

# === Chuẩn bị test.csv từ t2i.csv ===
df = pd.read_csv(T2I_CSV, header=None, names=["text"])
logger.info("Đọc test caption: %d dòng", len(df))

# Bị dupliacte
first_row = pd.DataFrame({"text": ["edema and erythema of the arytenoid cartilages"]})
df = pd.concat([first_row, df], ignore_index=True)

with open(TEST_CSV, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["image", "text"]) 
    for text in df["text"]:
        writer.writerow(["", text])

logger.info("✅ Đã tạo test.csv (%d mẫu)", len(df))

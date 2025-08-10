import gdown
import re
import zipfile
import os
from utils.logger import default_logger as logger
from utils.config import load_config

cfg = load_config()

def download_drive_folder(drive_url = "https://drive.google.com/drive/folders/1MHfq2L8q1yq9UIia6UB1zLB-iLDh76vl?hl=vi", output_path = 'downloaded_files'):
    match = re.search(r'/folders/([a-zA-Z0-9_-]+)', drive_url)
    if not match:
        logger.info("❌ Không tìm thấy folder ID trong link!")
        return
    file_id = match.group(1)

    logger.info(f"📥 Đang tải folder {file_id} ...")
    gdown.download_folder(id=file_id, output=output_path, quiet=False)
    logger.info("✅ Hoàn tất tải folder! File được lưu trong thư mục 'downloaded_files'.")

def extract_file(folder_path='downloaded_files'):
    logger.info(f"🔍 Đang tìm file ZIP trong thư mục: {folder_path}")
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".zip"):
                zip_path = os.path.join(root, file)
                extract_path = os.path.join(root, file.replace(".zip", ""))

                logger.info(f"📦 Giải nén {zip_path} → {extract_path}")
                os.makedirs(extract_path, exist_ok=True)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
    logger.info("✅ Giải nén hoàn tất!")


from utils.drive_utils import download_drive_folder, extract_file
from utils.config import load_config

cfg = load_config()

def main():
    download_drive_folder(drive_url=cfg['data']['DRIVE_URL'], output_path=cfg['data']['DOWNLOADED_FILES'])
    extract_file(folder_path=cfg['data']['DOWNLOADED_FILES'])
    
if __name__ == "__main__":
    main()
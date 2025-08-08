import logging
import os

class CustomFormatter(logging.Formatter):
    def format(self, record):
        pathname = record.pathname
        parent_dir, filename = os.path.split(pathname)
        parent = os.path.basename(parent_dir)
        record.custom_path = f"{parent}/{filename}"
        return super().format(record)

def setup_logger():
    formatter = CustomFormatter(fmt="%(asctime)s [%(levelname)s] [%(custom_path)s] %(message)s")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    logger.addHandler(handler)

    return logger

default_logger = setup_logger()
import os
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) 

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(ch)
    
class Circle_Cropper:
    def __init__(self, output_dir=None):
        """
        Parameters
        ----------
        output_dir : str, optional
            Thư mục lưu ảnh đã crop. Nếu None thì chỉ trả kết quả.
        """
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
    def crop(self, image_path):
        "Xử lý và crop ảnh đơn"
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Không tìm thấy hoặc không đọc được ảnh: {image_path}")
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)
        equalized = cv2.equalizeHist(gray_clahe)
        
        # Strong blur
        blur = cv2.GaussianBlur(equalized, (31, 31), 0)
        h, w = blur.shape[:2]
        
        # Dynamic radius & distance
        minR = round(w * 0.25)
        maxR = round(w * 0.5)
        minDis = round(w * 0.5)
        
        # Circle detection
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=minDis,
                                param1=80, param2=10, minRadius=minR, maxRadius=maxR)
            
        if circles is not None:
            x, y, r = map(int, circles[0][0])
            pad = int(r * 0.1)
            x1 = max(0, x - r - pad)
            y1 = max(0, y - r - pad)
            x2 = min(w, x + r + pad)
            y2 = min(h, y + r + pad)
            cropped = img[y1:y2, x1:x2]

            # Refine crop
            gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_crop, 10, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest = max(contours, key=cv2.contourArea)
                x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(largest)
                cropped_img = cropped[y_cnt:y_cnt + h_cnt, x_cnt:x_cnt + w_cnt]

                if self.output_dir:
                    save_path = os.path.join(self.output_dir, os.path.basename(image_path))
                    cv2.imwrite(save_path, cropped_img)
                    logger.info(f"Đã lưu ảnh crop: {save_path}")

                return cropped_img

        logger.warning(f"Không phát hiện hình tròn trong ảnh: {image_path}")
        return None
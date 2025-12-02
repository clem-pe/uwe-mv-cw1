import cv2
import os
import numpy as np

def preprocess_to_4channel(images_dir, masks_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for img_file in os.listdir(images_dir):
        if not img_file.lower().endswith('.png'): continue
        img_path = os.path.join(images_dir, img_file)
        mask_path = os.path.join(masks_dir, img_file)  # 假设文件名相同
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转RGB（YOLO期望）
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None: continue
        # Otsu二值化mask（可选；注释保持原灰度）
        _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = np.expand_dims(mask, axis=-1)  # (H,W) -> (H,W,1)
        
        four_ch = np.concatenate((img, mask), axis=-1)  # (H,W,4)
        cv2.imwrite(os.path.join(output_dir, img_file), four_ch)  # 保存4ch PNG

# 调用示例（调整路径）
preprocess_to_4channel(r"F:\study data\uwe\mv\a1\detection\train_split\images", r"F:\study data\uwe\mv\a1\detection\train_split\masks", r"F:\study data\uwe\mv\a1\detection\train_split\4ch_images")
preprocess_to_4channel(r"F:\study data\uwe\mv\a1\detection\val\images", r"F:\study data\uwe\mv\a1\detection\val\masks", r"F:\study data\uwe\mv\a1\detection\val\4ch_images")

print("4通道预处理完成。检查4ch_images文件夹，用cv2.imread确认shape[-1]==4。")
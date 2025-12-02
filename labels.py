import cv2
import os
import numpy as np

def generate_labels_from_masks(masks_dir, labels_dir):
    os.makedirs(labels_dir, exist_ok=True)
    for mask_file in os.listdir(masks_dir):
        if not mask_file.lower().endswith('.png'): continue
        mask_path = os.path.join(masks_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None: continue
        
        # Otsu二值化：处理非二值灰度mask
        _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形态学开运算：移除小噪声
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 提取contours（外部，只取苹果blob）
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 获取mask尺寸（动态，无需固定img_size）
        h, w = mask.shape
        
        label_file = os.path.join(labels_dir, mask_file.replace('.png', '.txt'))
        with open(label_file, 'w') as f:
            for cnt in contours:
                # 过滤小面积噪声（调整<10基于你的数据）
                if cv2.contourArea(cnt) < 10: continue
                x, y, bw, bh = cv2.boundingRect(cnt)
                cx = (x + bw / 2) / w  # 归一化中心x
                cy = (y + bh / 2) / h  # 中心y
                nw = bw / w            # 宽度
                nh = bh / h            # 高度
                f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

# 调用示例（调整路径）
generate_labels_from_masks(r"F:\study data\uwe\mv\a1\detection\train_split\masks", r"F:\study data\uwe\mv\a1\detection\train_split\labels")
generate_labels_from_masks(r"F:\study data\uwe\mv\a1\detection\val\masks", r"F:\study data\uwe\mv\a1\detection\val\labels")

print("Labels生成完成。检查labels文件夹，每.txt行数=苹果数。")
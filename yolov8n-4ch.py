import os
import shutil
from sklearn.model_selection import train_test_split

# 定义路径（使用r前缀避免转义问题）
root_dir = r"F:\study data\uwe\mv\a1\detection"
train_images_dir = os.path.join(root_dir, "train/images")
train_masks_dir = os.path.join(root_dir, "train/masks")

# 创建输出文件夹
train_split_images = os.path.join(root_dir, "train_split/images")
train_split_masks = os.path.join(root_dir, "train_split/masks")
val_images = os.path.join(root_dir, "val/images")
val_masks = os.path.join(root_dir, "val/masks")
os.makedirs(train_split_images, exist_ok=True)
os.makedirs(train_split_masks, exist_ok=True)
os.makedirs(val_images, exist_ok=True)
os.makedirs(val_masks, exist_ok=True)

# 获取所有图像文件名（过滤.png）
image_files = [f for f in os.listdir(train_images_dir) if f.lower().endswith('.png')]
assert len(image_files) == 670, f"预期670张图像，但找到{len(image_files)}张。检查路径或文件。"

# 随机分割：80% train_split, 20% val，可复现
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

# 复制train_split文件
for f in train_files:
    shutil.copy(os.path.join(train_images_dir, f), os.path.join(train_split_images, f))
    mask_path = os.path.join(train_masks_dir, f)  # 假设文件名相同，都是.png
    if os.path.exists(mask_path):
        shutil.copy(mask_path, os.path.join(train_split_masks, f))
    else:
        print(f"警告: 缺少mask文件 {f}")

# 复制val文件
for f in val_files:
    shutil.copy(os.path.join(train_images_dir, f), os.path.join(val_images, f))
    mask_path = os.path.join(train_masks_dir, f)
    if os.path.exists(mask_path):
        shutil.copy(mask_path, os.path.join(val_masks, f))
    else:
        print(f"警告: 缺少mask文件 {f}")

print(f"分割完成: train_split 有 {len(train_files)} 张图像/masks")
print(f"val 有 {len(val_files)} 张图像/masks")
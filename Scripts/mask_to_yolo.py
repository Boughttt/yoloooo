import os
import cv2

image_dir = "D:/github/crack/unlabeled_images"
mask_dir  = "D:/github/crack/archive/Masks"
label_dir = "D:/github/crack/archive/labels"

os.makedirs(label_dir, exist_ok=True)

# ======================
# 遍历mask
# ======================
for filename in os.listdir(mask_dir):

    if not filename.lower().endswith(".png"):
        continue

    mask_path = os.path.join(mask_dir, filename)

    # ⭐ 核心修改（处理 _label）
    image_name = filename.replace("_label.PNG", ".jpg").replace("_label.png", ".jpg")
    image_path = os.path.join(image_dir, image_name)

    mask = cv2.imread(mask_path, 0)
    image = cv2.imread(image_path)

    if mask is None:
        print(f"❌ 找不到mask: {mask_path}")
        continue

    if image is None:
        print(f"❌ 找不到image: {image_path}")
        continue

    h, w = mask.shape

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    label_lines = []

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)

        # 过滤小噪声
        if bw * bh < 50:
            continue

        x_center = (x + bw / 2) / w
        y_center = (y + bh / 2) / h
        width = bw / w
        height = bh / h

        label_lines.append(f"0 {x_center} {y_center} {width} {height}")

    label_name = image_name.replace(".jpg", ".txt")
    label_path = os.path.join(label_dir, label_name)

    with open(label_path, "w") as f:
        f.write("\n".join(label_lines))

print("🎉 转换完成！")
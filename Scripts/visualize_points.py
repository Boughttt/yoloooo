import cv2
import os
import numpy as np

ROOT = "D:/github/crack"

IMAGE_DIR = ROOT + "/unlabeled_images"
MASK_DIR = ROOT + "/masks"
POINT_DIR = ROOT + "/auto_points"
OUT_DIR = ROOT + "/visualization"

os.makedirs(OUT_DIR, exist_ok=True)

for file in os.listdir(POINT_DIR):

    base_name = os.path.splitext(file)[0]

    # ======================
    # 1. 找图片（自动匹配格式）
    # ======================
    img_path = None
    for ext in [".jpg", ".png", ".jpeg"]:
        temp = os.path.join(IMAGE_DIR, base_name + ext)
        if os.path.exists(temp):
            img_path = temp
            break

    if img_path is None:
        print(f"❌ 找不到图片: {base_name}")
        continue

    image = cv2.imdecode(
        np.fromfile(img_path, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        print(f"❌ 读取失败: {img_path}")
        continue

    # ======================
    # 2. 画点（红色）
    # ======================
    txt_path = os.path.join(POINT_DIR, file)

    with open(txt_path, "r") as f:
        for line in f:
            x, y = map(int, line.strip().split())
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    # ======================
    # 3. 用mask生成连线（核心）
    # ======================
    mask_path = os.path.join(MASK_DIR, base_name + ".jpg")
    if not os.path.exists(mask_path):
        mask_path = os.path.join(MASK_DIR, base_name + ".png")

    mask = cv2.imread(mask_path, 0)

    if mask is not None:

        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        # 🟢 连线（绿色）
        cv2.drawContours(image, contours, -1, (0, 255, 0), 1)

    else:
        print(f"⚠️ 没有mask: {base_name}")

    # ======================
    # 4. 保存
    # ======================
    out_path = os.path.join(OUT_DIR, base_name + ".jpg")
    cv2.imwrite(out_path, image)

    print(f"✅ 已生成: {out_path}")

print("🎉 完成：点 + 连线")
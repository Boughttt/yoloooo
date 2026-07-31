import os
import cv2
import numpy as np
import shutil
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

ROOT = "D:/github/crack"

IMAGE_DIR = ROOT + "/unlabeled_images"
POINT_DIR = ROOT + "/auto_points"
UNCERTAIN_DIR = ROOT + "/uncertain_samples"
MASK_DIR = ROOT + "/masks"

YOLO_MODEL = ROOT + "/models/best.pt"
SAM_MODEL = ROOT + "/models/sam_vit_b_01ec64.pth"

os.makedirs(POINT_DIR, exist_ok=True)
os.makedirs(UNCERTAIN_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

CONF_THRESHOLD = 0.5

# ======================
# 加载模型
# ======================
print("⏳ Loading YOLO...")
yolo = YOLO(YOLO_MODEL)

print("⏳ Loading SAM...")
sam = sam_model_registry["vit_b"](checkpoint=SAM_MODEL)
sam.to("cpu")
predictor = SamPredictor(sam)

print("✅ Models loaded")

# ======================
# 主流程
# ======================
for img_name in os.listdir(IMAGE_DIR):

    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    image = cv2.imread(img_path)

    if image is None:
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    results = yolo(image)[0]
    boxes = results.boxes

    if boxes is None or len(boxes) == 0:
        print(f"⚠️ No detection: {img_name}")
        continue

    predictor.set_image(image_rgb)

    final_mask = np.zeros((h, w), dtype=np.uint8)
    all_points = []

    for box in boxes:

        conf = float(box.conf)

        # 低置信度 → 人工标注池
        if conf < CONF_THRESHOLD:
            shutil.copy(img_path, os.path.join(UNCERTAIN_DIR, img_name))
            print(f"🟡 Uncertain: {img_name}")
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        input_box = np.array([[x1, y1, x2, y2]])

        masks, _, _ = predictor.predict(
            box=input_box,
            multimask_output=False
        )

        mask = masks[0]
        final_mask[mask] = 255

    # ======================
    # mask → 边缘点
    # ======================
    edges = cv2.Canny(final_mask, 50, 150)
    points = np.column_stack(np.where(edges > 0))

    # 降采样（避免点过多）
    points = points[::5]

    # ======================
    # 保存点
    # ======================
    point_path = os.path.join(
        POINT_DIR,
        img_name.replace(".jpg", ".txt").replace(".png", ".txt")
    )

    with open(point_path, "w") as f:
        for y, x in points:
            f.write(f"{x} {y}\n")

    # 保存mask
    cv2.imwrite(os.path.join(MASK_DIR, img_name), final_mask)

    print(f"✅ Processed: {img_name}")

print("🎉 Pipeline finished")
import json

INPUT = "cvat_export.json"
OUTPUT_DIR = "cvat_points"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(INPUT, "r") as f:
    data = json.load(f)

for item in data["annotations"]:
    img_id = item["image_id"]
    points = item["points"]

    with open(f"{OUTPUT_DIR}/{img_id}.txt", "w") as f:
        for x, y in points:
            f.write(f"{int(x)} {int(y)}\n")

print("✅ 转换完成")
import cv2
import numpy as np
from skimage.morphology import skeletonize

mask = cv2.imread("mask.png", 0)

# 二值化
mask = mask > 0

# 骨架提取
skeleton = skeletonize(mask)

# 转为点
points = np.column_stack(np.where(skeleton))

# 可视化
vis = np.zeros_like(mask, dtype=np.uint8)
for y, x in points:
    vis[y, x] = 255

cv2.imwrite("points.png", vis)

print("✅ 点提取完成")
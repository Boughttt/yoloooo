import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

image = cv2.imread("test.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

sam = sam_model_registry["vit_b"](checkpoint="models/sam_vit_b_01ec64.pth")
predictor = SamPredictor(sam)

predictor.set_image(image_rgb)

h, w, _ = image.shape

input_box = np.array([w//4, h//4, w*3//4, h*3//4])

masks, scores, logits = predictor.predict(
    box=input_box,
    multimask_output=False
)

mask = masks[0]
image[mask] = [0, 0, 255]

cv2.imwrite("sam_result.jpg", image)

print("SAM成功")
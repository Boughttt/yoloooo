import os
import shutil
import random

ROOT = "D:/github/crack"

image_dir = ROOT + "/unlabeled_images"
label_dir = ROOT + "/auto_labels"

output_dir = ROOT + "/dataset"

train_ratio = 0.8

images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
random.shuffle(images)

train_count = int(len(images) * train_ratio)

def copy(files, subset):
    for f in files:
        img_src = os.path.join(image_dir, f)
        lbl_src = os.path.join(label_dir, f.replace(".jpg", ".txt"))

        img_dst = os.path.join(output_dir, "images", subset, f)
        lbl_dst = os.path.join(output_dir, "labels", subset, f.replace(".jpg", ".txt"))

        os.makedirs(os.path.dirname(img_dst), exist_ok=True)
        os.makedirs(os.path.dirname(lbl_dst), exist_ok=True)

        shutil.copy(img_src, img_dst)
        shutil.copy(lbl_src, lbl_dst)

copy(images[:train_count], "train")
copy(images[train_count:], "val")

print("✅ Dataset split done")
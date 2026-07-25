import os
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np

DATASET_DIR = r"p:\project\hackothon\jnn_shivamogga\VTUAV_subset"
ANN_PATH = os.path.join(DATASET_DIR, "annotations", "val.json")
RGB_DIR = os.path.join(DATASET_DIR, "VTUAV_co")
IR_DIR = os.path.join(DATASET_DIR, "VTUAV_ir")
OUTPUT_DIR = r"p:\project\hackothon\jnn_shivamogga\output\alignment_verification"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(ANN_PATH, "r") as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]

# Group annotations by image_id
img_to_anns = {}
for ann in annotations:
    img_id = ann["image_id"]
    img_to_anns.setdefault(img_id, []).append(ann)

print(f"Visualizing RGB-Thermal alignment for 20 sample paired images from {ANN_PATH}...")

# Select 20 samples spread across the dataset
step = max(1, len(images) // 20)
selected_images = images[::step][:20]

for idx, img_info in enumerate(selected_images):
    img_id = img_info["id"]
    file_name = img_info["file_name"] # e.g. "rgb/00001.jpg" or "00001.jpg"
    
    # Extract relative filename
    base_name = os.path.basename(file_name)
    
    # Try finding RGB and IR image files
    rgb_path = os.path.join(RGB_DIR, file_name)
    if not os.path.exists(rgb_path):
        rgb_path = os.path.join(RGB_DIR, base_name)
        
    ir_path = os.path.join(IR_DIR, file_name)
    if not os.path.exists(ir_path):
        ir_path = os.path.join(IR_DIR, base_name)

    # If still not found, check subdirectories or direct search
    if not os.path.exists(rgb_path):
        # find matching basename recursively or in dir
        for root, _, files in os.walk(RGB_DIR):
            if base_name in files:
                rgb_path = os.path.join(root, base_name)
                break
                
    if not os.path.exists(ir_path):
        for root, _, files in os.walk(IR_DIR):
            if base_name in files:
                ir_path = os.path.join(root, base_name)
                break

    if not os.path.exists(rgb_path) or not os.path.exists(ir_path):
        print(f"Warning: Image pair for {base_name} missing! RGB: {os.path.exists(rgb_path)}, IR: {os.path.exists(ir_path)}")
        continue

    rgb_img = cv2.imread(rgb_path)
    ir_img = cv2.imread(ir_path)

    if rgb_img is None or ir_img is None:
        print(f"Warning: Could not read images for {base_name}")
        continue

    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    if len(ir_img.shape) == 2 or ir_img.shape[2] == 1:
        ir_img = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2RGB)
    else:
        ir_img = cv2.cvtColor(ir_img, cv2.COLOR_BGR2RGB)

    anns = img_to_anns.get(img_id, [])

    # Draw bboxes on both RGB and IR
    for ann in anns:
        bbox = ann["bbox"] # [x, y, w, h]
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # Color coding: Green for pedestrians
        color = (0, 255, 0)
        cv2.rectangle(rgb_img, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(ir_img, (x, y), (x + w, y + h), color, 2)

    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(rgb_img)
    axes[0].set_title(f"RGB Image: {base_name} ({len(anns)} pedestrians)")
    axes[0].axis("off")

    axes[1].imshow(ir_img)
    axes[1].set_title(f"Thermal IR Image: {base_name} ({len(anns)} pedestrians)")
    axes[1].axis("off")

    plt.tight_layout()
    out_file = os.path.join(OUTPUT_DIR, f"pair_{idx+1:02d}_{os.path.splitext(base_name)[0]}.png")
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved visualization [{idx+1}/20]: {out_file}")

print(f"\nCompleted alignment visualization of 20 image pairs in {OUTPUT_DIR}")

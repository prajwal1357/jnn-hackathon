import os
import json
import numpy as np

DATASET_DIR = r"p:\project\hackothon\jnn_shivamogga\VTUAV_subset"
ANNOTATIONS_DIR = os.path.join(DATASET_DIR, "annotations")

splits = {
    "train": os.path.join(ANNOTATIONS_DIR, "train.json"),
    "val": os.path.join(ANNOTATIONS_DIR, "val.json"),
    "test": os.path.join(ANNOTATIONS_DIR, "test.json")
}

summary_report = {}

print("==================================================")
print("VTUAV_subset Dataset Exploration & Analysis")
print("==================================================")

for split_name, ann_file in splits.items():
    if not os.path.exists(ann_file):
        print(f"Error: {ann_file} not found!")
        continue

    with open(ann_file, "r") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    num_images = len(images)
    num_instances = len(annotations)
    avg_per_img = num_instances / num_images if num_images > 0 else 0

    # Image Resolutions
    resolutions = set((img.get("width"), img.get("height")) for img in images)

    # Scale distribution
    small_count = 0  # area < 32^2 = 1024
    medium_count = 0 # 1024 <= area < 9216
    large_count = 0  # area >= 9216

    areas = []
    bbox_widths = []
    bbox_heights = []

    for ann in annotations:
        bbox = ann.get("bbox", []) # [x, y, w, h]
        if len(bbox) >= 4:
            w, h = bbox[2], bbox[3]
            area = ann.get("area", w * h)
            areas.append(area)
            bbox_widths.append(w)
            bbox_heights.append(h)

            if area < 32 * 32:
                small_count += 1
            elif area < 96 * 96:
                medium_count += 1
            else:
                large_count += 1

    summary_report[split_name] = {
        "num_images": num_images,
        "num_instances": num_instances,
        "avg_instances_per_image": round(avg_per_img, 2),
        "resolutions": list(resolutions),
        "scale_distribution": {
            "small (<32^2)": small_count,
            "small_pct": round(small_count / num_instances * 100, 2) if num_instances else 0,
            "medium (32^2-96^2)": medium_count,
            "medium_pct": round(medium_count / num_instances * 100, 2) if num_instances else 0,
            "large (>=96^2)": large_count,
            "large_pct": round(large_count / num_instances * 100, 2) if num_instances else 0,
        },
        "mean_area": round(float(np.mean(areas)), 2) if areas else 0,
        "median_area": round(float(np.median(areas)), 2) if areas else 0,
        "categories": categories
    }

    print(f"\n--- Split: {split_name.upper()} ---")
    print(f"Total Images: {num_images}")
    print(f"Total Pedestrian Instances: {num_instances}")
    print(f"Average Instances per Image: {avg_per_img:.2f}")
    print(f"Resolutions: {resolutions}")
    print(f"Scale Distribution:")
    print(f"  - Small  (Area < 1024 px^2):  {small_count:5d} ({small_count/num_instances*100:.2f}%)")
    print(f"  - Medium (1024 <= Area < 9216): {medium_count:5d} ({medium_count/num_instances*100:.2f}%)")
    print(f"  - Large  (Area >= 9216 px^2): {large_count:5d} ({large_count/num_instances*100:.2f}%)")

# Save summary json
output_json = os.path.join(DATASET_DIR, "dataset_analysis_summary.json")
with open(output_json, "w") as f:
    json.dump(summary_report, f, indent=2)

print(f"\nSaved analysis summary to {output_json}")

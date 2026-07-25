"""
Stage 1 Visual Analysis Script
- Alignment verification across RGB-Thermal pairs
- Cross-modal comparison (brightness, contrast analysis)
- Scale distribution tagging per image
- Challenge scenario detection (low light, occlusion, crowd, tiny pedestrians)
- Outputs annotated images + JSON summary for report
"""
import os, sys, json, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Paths
BASE = r"p:\project\hackothon\jnn_shivamogga"
VTUAV = os.path.join(BASE, "VTUAV_subset")
RGB_DIR = os.path.join(VTUAV, "VTUAV_co", "val", "images")
IR_DIR = os.path.join(VTUAV, "VTUAV_ir", "val", "images")
ANN_FILE = os.path.join(VTUAV, "annotations", "val.json")
OUT_DIR = os.path.join(BASE, "output", "stage1_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# Load annotations
with open(ANN_FILE) as f:
    coco = json.load(f)

img_id_to_info = {img["id"]: img for img in coco["images"]}
img_id_to_anns = {}
for ann in coco["annotations"]:
    img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

# Sort images by ID for reproducible sampling
all_img_ids = sorted(img_id_to_info.keys())

# Sample 20 images spread evenly across the dataset
step = max(1, len(all_img_ids) // 20)
sampled_ids = all_img_ids[::step][:20]

print(f"Analyzing {len(sampled_ids)} sampled image pairs from val set ({len(all_img_ids)} total)")

# COCO scale thresholds
SMALL_THRESH = 32 * 32  # 1024
MEDIUM_THRESH = 96 * 96  # 9216

def classify_scale(area):
    if area < SMALL_THRESH:
        return "small"
    elif area < MEDIUM_THRESH:
        return "medium"
    else:
        return "large"

def compute_brightness(img_pil):
    """Compute mean brightness of image (0-255)."""
    arr = np.array(img_pil.convert("L"))
    return float(np.mean(arr))

def compute_contrast(img_pil):
    """Compute std-dev of brightness (a proxy for contrast)."""
    arr = np.array(img_pil.convert("L"))
    return float(np.std(arr))

def compute_edge_density(img_pil, bbox):
    """Check how busy/cluttered the area around a bbox is using edge detection."""
    x, y, w, h = bbox
    # Expand region by 50%
    cx, cy = x + w/2, y + h/2
    ew, eh = w * 2, h * 2
    x1 = max(0, int(cx - ew/2))
    y1 = max(0, int(cy - eh/2))
    x2 = min(img_pil.width, int(cx + ew/2))
    y2 = min(img_pil.height, int(cy + eh/2))
    crop = img_pil.crop((x1, y1, x2, y2)).convert("L")
    edges = crop.filter(ImageFilter.FIND_EDGES)
    return float(np.mean(np.array(edges)))

def check_overlap(ann1, ann2):
    """Check IoU between two annotations (for crowd detection)."""
    x1, y1, w1, h1 = ann1["bbox"]
    x2, y2, w2, h2 = ann2["bbox"]
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1+w1, x2+w2)
    yb = min(y1+h1, y2+h2)
    inter = max(0, xb-xa) * max(0, yb-ya)
    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0

def draw_pair_with_analysis(rgb_path, ir_path, anns, pair_idx, img_info):
    """Draw annotated pair and compute metrics."""
    rgb = Image.open(rgb_path).convert("RGB")
    ir = Image.open(ir_path).convert("RGB")
    
    # Compute image-level metrics
    rgb_brightness = compute_brightness(rgb)
    ir_brightness = compute_brightness(ir)
    rgb_contrast = compute_contrast(rgb)
    ir_contrast = compute_contrast(ir)
    
    # Determine if low-light scene
    is_low_light = rgb_brightness < 80
    is_night = rgb_brightness < 50
    
    # Per-annotation metrics
    scale_counts = {"small": 0, "medium": 0, "large": 0}
    has_tiny = False
    has_crowded = False
    has_occluded = False
    has_cluttered = False
    smallest_area = float("inf")
    largest_area = 0
    
    for ann in anns:
        area = ann["area"]
        scale = classify_scale(area)
        scale_counts[scale] += 1
        if area < smallest_area:
            smallest_area = area
        if area > largest_area:
            largest_area = area
        if area < SMALL_THRESH:
            has_tiny = True
    
    # Check for crowded/overlapping pedestrians
    for i, a1 in enumerate(anns):
        for j, a2 in enumerate(anns):
            if j <= i:
                continue
            iou = check_overlap(a1, a2)
            if iou > 0.1:
                has_crowded = True
                break
    
    # Check for cluttered background around annotations
    edge_densities = []
    for ann in anns[:5]:  # sample up to 5
        ed = compute_edge_density(rgb, ann["bbox"])
        edge_densities.append(ed)
    avg_edge_density = np.mean(edge_densities) if edge_densities else 0
    has_cluttered = avg_edge_density > 25
    
    # Determine challenge tags
    challenges = []
    if is_night:
        challenges.append("night_scene")
    elif is_low_light:
        challenges.append("low_illumination")
    if has_tiny:
        challenges.append("tiny_pedestrian")
    if has_crowded:
        challenges.append("crowded")
    if has_cluttered:
        challenges.append("cluttered_background")
    if len(anns) >= 10:
        challenges.append("many_pedestrians")
    
    # --- Draw annotated images ---
    # Create side-by-side visualization
    W, H = rgb.width, rgb.height
    canvas_w = W * 2 + 20  # 20px gap
    canvas_h = H + 80  # space for title
    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
    
    # Draw boxes on copies
    rgb_draw = rgb.copy()
    ir_draw = ir.copy()
    d_rgb = ImageDraw.Draw(rgb_draw)
    d_ir = ImageDraw.Draw(ir_draw)
    
    for ann in anns:
        x, y, w, h = ann["bbox"]
        area = ann["area"]
        scale = classify_scale(area)
        
        # Color by scale
        if scale == "small":
            color = (255, 100, 100)  # Red for small
            label = "S"
        elif scale == "medium":
            color = (100, 255, 100)  # Green for medium
            label = "M"
        else:
            color = (100, 100, 255)  # Blue for large
            label = "L"
        
        # Draw on RGB
        d_rgb.rectangle([x, y, x+w, y+h], outline=color, width=2)
        d_rgb.text((x+2, y-12), f"{label}:{int(area)}", fill=color)
        
        # Draw on Thermal (same coordinates - alignment test)
        d_ir.rectangle([x, y, x+w, y+h], outline=color, width=2)
        d_ir.text((x+2, y-12), f"{label}:{int(area)}", fill=color)
    
    # Paste into canvas
    canvas.paste(rgb_draw, (0, 80))
    canvas.paste(ir_draw, (W + 20, 80))
    
    # Draw title bar
    d_canvas = ImageDraw.Draw(canvas)
    title = f"Pair {pair_idx:02d} | {img_info['file_name']} | {len(anns)} pedestrians | "
    title += f"S:{scale_counts['small']} M:{scale_counts['medium']} L:{scale_counts['large']} | "
    title += f"RGB brightness:{rgb_brightness:.0f} | Challenges: {', '.join(challenges) if challenges else 'none'}"
    d_canvas.text((10, 10), title, fill=(255, 255, 255))
    d_canvas.text((10, 35), "RGB (bboxes from GT annotations)", fill=(200, 200, 200))
    d_canvas.text((W + 30, 35), "THERMAL (same bbox coords → alignment test)", fill=(200, 200, 200))
    d_canvas.text((10, 55), f"Red=Small(<32²)  Green=Medium(32²-96²)  Blue=Large(≥96²)", fill=(180, 180, 180))
    
    out_path = os.path.join(OUT_DIR, f"analysis_{pair_idx:02d}_{img_info['file_name'].replace('.jpg','.png')}")
    canvas.save(out_path, quality=95)
    
    return {
        "pair_idx": pair_idx,
        "image_id": img_info["id"],
        "file_name": img_info["file_name"],
        "num_pedestrians": len(anns),
        "scale_counts": scale_counts,
        "smallest_area": int(smallest_area) if smallest_area != float("inf") else 0,
        "largest_area": int(largest_area),
        "rgb_brightness": round(rgb_brightness, 1),
        "ir_brightness": round(ir_brightness, 1),
        "rgb_contrast": round(rgb_contrast, 1),
        "ir_contrast": round(ir_contrast, 1),
        "is_low_light": is_low_light,
        "is_night": is_night,
        "challenges": challenges,
        "avg_edge_density": round(avg_edge_density, 1),
        "output_path": out_path,
    }

# Also generate scale distribution chart
def generate_scale_chart():
    """Generate a bar chart of scale distribution across all val annotations."""
    all_areas = [ann["area"] for ann in coco["annotations"]]
    small = sum(1 for a in all_areas if a < SMALL_THRESH)
    medium = sum(1 for a in all_areas if SMALL_THRESH <= a < MEDIUM_THRESH)
    large = sum(1 for a in all_areas if a >= MEDIUM_THRESH)
    total = len(all_areas)
    
    # Create chart as image
    chart_w, chart_h = 800, 500
    chart = Image.new("RGB", (chart_w, chart_h), (25, 25, 35))
    d = ImageDraw.Draw(chart)
    
    # Title
    d.text((chart_w//2 - 150, 15), "VTUAV Val Set - Pedestrian Scale Distribution", fill=(255,255,255))
    
    # Bar dimensions
    bar_w = 150
    max_count = max(small, medium, large)
    bar_area_y = 80
    bar_area_h = 320
    
    categories = [
        (f"Small\n(<32² px)", small, (255, 100, 100)),
        (f"Medium\n(32²-96² px)", medium, (100, 255, 100)),
        (f"Large\n(≥96² px)", large, (100, 100, 255)),
    ]
    
    for i, (label, count, color) in enumerate(categories):
        x = 120 + i * 220
        bar_h = int(bar_area_h * count / max_count)
        y_top = bar_area_y + bar_area_h - bar_h
        
        # Bar
        d.rectangle([x, y_top, x + bar_w, bar_area_y + bar_area_h], fill=color)
        
        # Count label
        pct = count / total * 100
        d.text((x + bar_w//2 - 30, y_top - 25), f"{count} ({pct:.1f}%)", fill=(255,255,255))
        
        # Category label
        d.text((x + 20, bar_area_y + bar_area_h + 15), label, fill=(200,200,200))
    
    # Axis line
    d.line([(100, bar_area_y + bar_area_h), (chart_w - 50, bar_area_y + bar_area_h)], fill=(100,100,100), width=2)
    
    # Stats
    d.text((100, chart_h - 60), f"Total annotations: {total} | Mean area: {np.mean(all_areas):.0f} px² | Median area: {np.median(all_areas):.0f} px²", fill=(180,180,180))
    
    chart_path = os.path.join(OUT_DIR, "scale_distribution_chart.png")
    chart.save(chart_path)
    print(f"Scale chart saved to {chart_path}")
    return chart_path

# Run analysis on all 20 samples
print("=" * 60)
analysis_results = []

for idx, img_id in enumerate(sampled_ids, 1):
    info = img_id_to_info[img_id]
    anns = img_id_to_anns.get(img_id, [])
    
    rgb_path = os.path.join(RGB_DIR, info["file_name"])
    ir_path = os.path.join(IR_DIR, info["file_name"])
    
    if not os.path.exists(rgb_path) or not os.path.exists(ir_path):
        print(f"  SKIP pair {idx}: missing file {info['file_name']}")
        continue
    
    print(f"[{idx:2d}/20] Analyzing {info['file_name']} ({len(anns)} pedestrians)...", flush=True)
    result = draw_pair_with_analysis(rgb_path, ir_path, anns, idx, info)
    analysis_results.append(result)

# Generate scale chart
chart_path = generate_scale_chart()

# Summary statistics
total_peds = sum(r["num_pedestrians"] for r in analysis_results)
total_small = sum(r["scale_counts"]["small"] for r in analysis_results)
total_medium = sum(r["scale_counts"]["medium"] for r in analysis_results)
total_large = sum(r["scale_counts"]["large"] for r in analysis_results)

challenge_counts = {}
for r in analysis_results:
    for c in r["challenges"]:
        challenge_counts[c] = challenge_counts.get(c, 0) + 1

low_light_pairs = [r for r in analysis_results if r["is_low_light"]]
night_pairs = [r for r in analysis_results if r["is_night"]]
tiny_pairs = [r for r in analysis_results if "tiny_pedestrian" in r["challenges"]]
crowded_pairs = [r for r in analysis_results if "crowded" in r["challenges"]]
cluttered_pairs = [r for r in analysis_results if "cluttered_background" in r["challenges"]]

summary = {
    "num_pairs_analyzed": len(analysis_results),
    "total_pedestrians_in_sample": total_peds,
    "scale_in_sample": {"small": total_small, "medium": total_medium, "large": total_large},
    "challenge_counts": challenge_counts,
    "low_light_pairs": [r["file_name"] for r in low_light_pairs],
    "night_pairs": [r["file_name"] for r in night_pairs],
    "tiny_pedestrian_pairs": [r["file_name"] for r in tiny_pairs],
    "crowded_pairs": [r["file_name"] for r in crowded_pairs],
    "cluttered_pairs": [r["file_name"] for r in cluttered_pairs],
    "brightness_stats": {
        "avg_rgb_brightness": round(np.mean([r["rgb_brightness"] for r in analysis_results]), 1),
        "avg_ir_brightness": round(np.mean([r["ir_brightness"] for r in analysis_results]), 1),
        "min_rgb_brightness": round(min(r["rgb_brightness"] for r in analysis_results), 1),
        "max_rgb_brightness": round(max(r["rgb_brightness"] for r in analysis_results), 1),
    },
    "scale_chart_path": chart_path,
    "per_pair": analysis_results,
}

summary_path = os.path.join(OUT_DIR, "stage1_analysis_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*60}")
print(f"Stage 1 Analysis Complete!")
print(f"{'='*60}")
print(f"Pairs analyzed:  {len(analysis_results)}")
print(f"Total pedestrians: {total_peds}")
print(f"Scale distribution: S={total_small} M={total_medium} L={total_large}")
print(f"Challenge counts: {challenge_counts}")
print(f"Low-light pairs: {len(low_light_pairs)}")
print(f"Night pairs:     {len(night_pairs)}")
print(f"Results saved to: {summary_path}")

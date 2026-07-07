import os
import cv2
import numpy as np
import shutil

# --- PATHS ---
INPUT_DATASET_PATH = r"/home/wot-amd/Projects/Gotilo-container/container-inspection/damage_detection/training_dataset_20260702_train_equal_valid"
OUTPUT_DATASET_PATH = r"/home/wot-amd/Projects/Gotilo-container/container-inspection/damage_detection/u2net_dataset_20260702_train_equal_valid_priority_based"
SPLITS = ["train", "valid"]

# --- MULTI-CLASS MAPPING ---
# Assuming YOLO classes are 0, 1, 2, 3. 
# We map them to U-2-Net mask values 1, 2, 3, 4 (0 is background).
# Adjust these according to your exact YOLO class definitions!
YOLO_TO_MASK_ID = {
    0: 1, # e.g., YOLO 0 -> Dent (1)
    1: 2, # e.g., YOLO 1 -> Rust (2)
    2: 3, # e.g., YOLO 2 -> Patch (3)
    3: 4  # e.g., YOLO 3 -> Scratch (4)
}

# --- OVERLAP PRIORITY ---
# Define which class draws on top when they overlap.
# Higher number = drawn last (stays on top).
# Example logic: Patches are base level, Dents on top of patches, 
# Rust inside dents, Scratches on top of everything.
DRAW_PRIORITY = {
    3: 1, # Patch (Lowest priority, drawn first)
    4: 2, # Scratch
    1: 3, # Dent 
    2: 4, # Rust (Highest priority, drawn last)
}

print(f"Starting Multi-Class conversion with overlap handling...")

for split in SPLITS:
    img_dir = os.path.join(INPUT_DATASET_PATH, split, "images")
    lbl_dir = os.path.join(INPUT_DATASET_PATH, split, "labels")
    out_img_dir = os.path.join(OUTPUT_DATASET_PATH, split, "images")
    out_msk_dir = os.path.join(OUTPUT_DATASET_PATH, split, "masks")

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_msk_dir, exist_ok=True)

    if not os.path.exists(img_dir):
        continue

    print(f"Processing '{split}' split...")

    for img_name in os.listdir(img_dir):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue

        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]

        shutil.copy(img_path, os.path.join(out_img_dir, img_name))

        # 1. Create a blank background (All 0s)
        mask = np.zeros((h, w), dtype=np.uint8)

        base_name = os.path.splitext(img_name)[0]
        lbl_path = os.path.join(lbl_dir, base_name + ".txt")

        if os.path.exists(lbl_path):
            polygons_to_draw = []
            
            # 2. Read all annotations first, don't draw yet
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 7: continue

                    yolo_class = int(parts[0])
                    if yolo_class not in YOLO_TO_MASK_ID: continue
                        
                    mask_class_id = YOLO_TO_MASK_ID[yolo_class]
                    priority = DRAW_PRIORITY.get(mask_class_id, 0)

                    coords = parts[1:]
                    pts = []
                    for i in range(0, len(coords), 2):
                        x_pixel = max(0, min(w - 1, int(round(float(coords[i]) * w))))
                        y_pixel = max(0, min(h - 1, int(round(float(coords[i+1]) * h))))
                        pts.append([x_pixel, y_pixel])
                        
                    # Store as a tuple: (priority, class_id, points)
                    polygons_to_draw.append((priority, mask_class_id, np.array(pts, dtype=np.int32)))
            
            # 3. Sort polygons by priority (lowest to highest)
            polygons_to_draw.sort(key=lambda x: x[0])
            
            # 4. Draw them in order. Overlaps are handled consistently!
            for priority, mask_class_id, pts_array in polygons_to_draw:
                # Notice we are drawing the ACTUAL CLASS ID (1, 2, 3, 4), not 255
                cv2.fillPoly(mask, [pts_array], mask_class_id)

        # 5. Save as PNG (Crucial to preserve integer values 1, 2, 3, 4)
        cv2.imwrite(os.path.join(out_msk_dir, base_name + ".png"), mask)

print("\nConversion successfully completed!")
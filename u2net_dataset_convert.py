import os
import cv2
import numpy as np
import shutil

# ---------------------------------------------------------
# UPDATE THESE PATHS TO YOUR ACTUAL DIRECTORIES
# ---------------------------------------------------------
# Use raw string (r"") for Windows paths to avoid escape character issues
INPUT_DATASET_PATH = r"/home/wot-amd/Projects/Gotilo-container/container-inspection/damage_detection/training_dataset_20260702"
OUTPUT_DATASET_PATH = r"/home/wot-amd/Projects/Gotilo-container/container-inspection/damage_detection/u2net_dataset_20260702"

# The splits to process
SPLITS = ["train", "valid"]

# If you only want to mask specific classes, add them here (e.g., ['1']). 
# Leave empty [] to mask ALL classes as foreground (white).
TARGET_CLASS_IDS = [] 

print(f"Starting conversion from YOLO to U2-Net format...")
print(f"Input: {INPUT_DATASET_PATH}")
print(f"Output: {OUTPUT_DATASET_PATH}\n")

for split in SPLITS:
    img_dir = os.path.join(INPUT_DATASET_PATH, split, "images")
    lbl_dir = os.path.join(INPUT_DATASET_PATH, split, "labels")

    # U2-Net typically expects 'images' and 'masks' (or 'gt') folders
    out_img_dir = os.path.join(OUTPUT_DATASET_PATH, split, "images")
    out_msk_dir = os.path.join(OUTPUT_DATASET_PATH, split, "masks")

    # Create output directories
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_msk_dir, exist_ok=True)

    if not os.path.exists(img_dir):
        print(f"Warning: Image directory not found, skipping: {img_dir}")
        continue

    print(f"Processing '{split}' split...")

    for img_name in os.listdir(img_dir):
        # Skip non-image files
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            continue

        img_path = os.path.join(img_dir, img_name)

        # 1. Read image to get its actual Width and Height
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}, skipping.")
            continue

        h, w = img.shape[:2]

        # Copy the original image to the output dataset folder
        # (Using shutil is faster and prevents quality loss from re-encoding)
        shutil.copy(img_path, os.path.join(out_img_dir, img_name))

        # 2. Prepare a blank black mask (0 = background)
        mask = np.zeros((h, w), dtype=np.uint8)

        # Find corresponding label file
        base_name = os.path.splitext(img_name)[0]
        lbl_path = os.path.join(lbl_dir, base_name + ".txt")

        # 3. Parse YOLO segmentation format and draw polygons
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 7:  # Need at least class_id + 3 points (6 coordinates)
                        continue

                    class_id = parts[0]

                    # Filter by class if TARGET_CLASS_IDS is specified
                    if TARGET_CLASS_IDS and class_id not in TARGET_CLASS_IDS:
                        continue

                    # Extract normalized coordinates (x1, y1, x2, y2, ...)
                    coords = parts[1:]
                    pts = []

                    # Convert normalized coordinates to absolute pixel coordinates
                    for i in range(0, len(coords), 2):
                        x_norm = float(coords[i])
                        y_norm = float(coords[i+1])

                        x_pixel = int(round(x_norm * w))
                        y_pixel = int(round(y_norm * h))

                        # Ensure coordinates are within image bounds
                        x_pixel = max(0, min(w - 1, x_pixel))
                        y_pixel = max(0, min(h - 1, y_pixel))

                        pts.append([x_pixel, y_pixel])

                    # Draw the filled polygon on the mask in white (255 = foreground)
                    pts_array = np.array(pts, dtype=np.int32)
                    cv2.fillPoly(mask, [pts_array], 255)

        # 4. Save the mask as a PNG 
        # (Crucial: Use PNG for masks! JPG compression creates artifacts like 254 or 1, 
        # which will mess up U2-Net's binary cross-entropy loss).
        mask_name = base_name + ".png"
        cv2.imwrite(os.path.join(out_msk_dir, mask_name), mask)

print("\nConversion successfully completed!")
print(f"Your U2-Net dataset is ready at: {OUTPUT_DATASET_PATH}")
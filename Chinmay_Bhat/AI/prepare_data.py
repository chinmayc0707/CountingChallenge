import cv2
import numpy as np
import os
import shutil
import glob
import random
import yaml

# Configuration
SOURCE_DIRS = [
    r"c:\pr\perspectiv\dataset\ScrewAndBolt_20240713",
    r"c:\pr\perspectiv\dataset\Screws_2024_07_15" # if this exists
]
BASE_DIR = r"c:\pr\perspectiv\Chinmay_Bhat\AI\datasets"
TRAIN_IMG = os.path.join(BASE_DIR, "train", "images")
TRAIN_LBL = os.path.join(BASE_DIR, "train", "labels")
VAL_IMG = os.path.join(BASE_DIR, "val", "images")
VAL_LBL = os.path.join(BASE_DIR, "val", "labels")

def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None: return None, []
    
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    
    polygons = []
    obj_ids = np.unique(markers)
    for obj_id in obj_ids:
        if obj_id <= 1: continue # Background/Unknown
        
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == obj_id] = 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > 500:
                # Normalize points for YOLO
                poly = []
                for point in c:
                    x_pt, y_pt = point[0]
                    poly.append(x_pt / w)
                    poly.append(y_pt / h)
                polygons.append(poly)
    
    return img, polygons

def main():
    # 1. Setup Directories
    for d in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL]:
        os.makedirs(d, exist_ok=True)
    
    # 2. Find Images
    image_paths = []
    for d in SOURCE_DIRS:
        if os.path.exists(d):
            image_paths.extend(glob.glob(os.path.join(d, "*.jpg")))
            image_paths.extend(glob.glob(os.path.join(d, "*.jpeg")))
    
    if not image_paths:
        print("No images found! Check SOURCE_DIRS in script.")
        return

    random.shuffle(image_paths)
    split_idx = int(len(image_paths) * 0.8)
    
    print(f"Processing {len(image_paths)} images...")
    
    for i, img_path in enumerate(image_paths):
        img, polygons = process_image(img_path)
        if not polygons: continue
        
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        
        is_train = i < split_idx
        save_img_dir = TRAIN_IMG if is_train else VAL_IMG
        save_lbl_dir = TRAIN_LBL if is_train else VAL_LBL
        
        # Save Image
        cv2.imwrite(os.path.join(save_img_dir, filename), img)
        
        # Save Label
        with open(os.path.join(save_lbl_dir, base_name + ".txt"), "w") as f:
            for poly in polygons:
                line = "0 " + " ".join([f"{p:.6f}" for p in poly])
                f.write(line + "\n")
                
        if i % 10 == 0: print(".", end="", flush=True)

    # 3. Create data.yaml
    yaml_content = f"""names:
  0: screw
path: {BASE_DIR}
train: train/images
val: val/images
"""
    with open(os.path.join(BASE_DIR, "data.yaml"), "w") as f:
        f.write(yaml_content)
        
    print("\nDataset generation complete!")
    print(f"Train: {len(os.listdir(TRAIN_IMG))} images")
    print(f"Val:   {len(os.listdir(VAL_IMG))} images")

if __name__ == "__main__":
    main()

import cv2
import numpy as np
import os
import glob

def process_images(input_dir):
    # Get all jpg images
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} images to process.")

    summary = []
    total = 0

    for path in sorted(image_paths):
        filename = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            continue

        # 1. Preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Binary Inverse Threshold
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 2. Morphological operations
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # 3. Contour Detection
        contours, hierarchy = cv2.findContours(sure_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 4. Filter Contours
        valid_contours = []
        min_area = 100

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                valid_contours.append(cnt)

        count = len(valid_contours)
        summary.append((filename, count))
        total += count
        # print(f"{filename}: {count}")

    print("\n" + "="*45)
    print(f"{'Image Name':<30} | {'Count':<10}")
    print("-" * 45)
    for name, count in summary:
        print(f"{name:<30} | {count:<10}")
    print("-" * 45)
    print(f"{'TOTAL':<30} | {total:<10}")
    print("="*45 + "\n")

if __name__ == "__main__":
    TEST_DIR = "Chinmay_Bhat/test_images"
    print("--- Non-AI (Simple CV) Screw Counting ---")
    process_images(TEST_DIR)

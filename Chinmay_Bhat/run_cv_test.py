import sys
import os
import glob
import cv2

# Add Non_AI directory to path
sys.path.append(os.path.abspath("Chinmay_Bhat/Non_AI"))

try:
    from solution_cv_watershed import count_items_in_image
except ImportError:
    print("Failed to import solution_cv_watershed")
    sys.exit(1)

TEST_IMAGES_DIR = "Chinmay_Bhat/test_images"

def main():
    print("--- Non-AI (CV Watershed) Screw Counting ---")
    image_paths = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg"))
    if not image_paths:
        print(f"No images found in {TEST_IMAGES_DIR}")
        return

    print(f"Found {len(image_paths)} images.")

    summary_data = []
    total_items = 0

    for path in sorted(image_paths):
        filename = os.path.basename(path)
        try:
            # count_items_in_image saves debug output to the same folder by default
            # We can disable debug_out if we want, but it's fine.
            count = count_items_in_image(path, debug_out=False)
            summary_data.append((filename, count))
            total_items += count
            print(f"{filename}: {count}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("\n" + "="*45)
    print(f"{'Image Name':<30} | {'Count':<10}")
    print("-" * 45)
    for name, count in summary_data:
        print(f"{name:<30} | {count:<10}")
    print("-" * 45)
    print(f"{'TOTAL':<30} | {total_items:<10}")
    print("="*45 + "\n")

if __name__ == "__main__":
    main()

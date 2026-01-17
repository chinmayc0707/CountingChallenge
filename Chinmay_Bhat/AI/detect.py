import cv2
import os
import glob
from ultralytics import YOLO

# ================= CONFIGURATION =================
# Path to the source images (Folder or File)
SOURCE_PATH = r"c:\pr\perspectiv\dataset\ScrewAndBolt_20240713"

# Path to the trained model
# Note: Since runs/ was deleted, you will need to retrain using train.py first.
MODEL_PATH = r"c:\pr\perspectiv\Chinmay_Bhat\AI\runs\yolo11n_fast_v3\weights\best.pt"

# Output directory for results
OUTPUT_DIR = r"c:\pr\perspectiv\Chinmay_Bhat\AI\inference_results"

# Settings
CONF_THRESH = 0.25
DEVICE = 'cpu'    # Forced to CPU (No CUDA found)
IMG_SIZE = 640    # Standard size
# =================================================

def run_inference():
    print(f"--- YOLOv11 Screw Detection ---")
    
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        print("Model file missing. Initiating training sequence automatically...")
        print("="*50)
        
        # Check if data exists
        data_yaml = r'c:\pr\perspectiv\Chinmay_Bhat\AI\datasets\data.yaml'
        if not os.path.exists(data_yaml):
            print("Dataset config not found. Running prepare_data.py first...")
            try:
                from prepare_data import main as prepare_data
                prepare_data()
            except ImportError:
                 print("Error: Could not import prepare_data.py")
                 return
        
        # Run Training
        try:
            from train import train_model
            train_model()
        except Exception as e:
            print(f"Training failed: {e}")
            return
            
        print("="*50)
        print("Training complete. Retrying inference...")
        
        # Re-check
        if not os.path.exists(MODEL_PATH):
             print(f"Error: Training finished but model still not found at {MODEL_PATH}")
             return
    
    print(f"Loading model: {os.path.basename(MODEL_PATH)}...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Find Images
    if os.path.isdir(SOURCE_PATH):
        images = []
        for root, dirs, files in os.walk(SOURCE_PATH):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    images.append(os.path.join(root, file))
    else:
        images = [SOURCE_PATH]
    
    if not images:
        print(f"No images found in {SOURCE_PATH}")
        return
        
    print(f"Found {len(images)} images in {SOURCE_PATH}")

    # 3. Process
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_data = []
    total_items = 0

    print("Processing", end="", flush=True)
    for i, img_path in enumerate(images):
        try:
            filename = os.path.basename(img_path)
            
            # Predict
            results = model(img_path, conf=CONF_THRESH, retina_masks=False, device=DEVICE, verbose=False, imgsz=IMG_SIZE)
            
            for r in results:
                count = len(r.boxes)
                total_items += count
                summary_data.append([filename, count])
                
                # Visualization
                im_array = r.plot(labels=False, conf=False)
                cv2.putText(im_array, f"Count: {count}", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
                
                save_path = os.path.join(OUTPUT_DIR, f"res_{filename}")
                cv2.imwrite(save_path, im_array)
            
            # Progress dot
            if i % 2 == 0:
                print(".", end="", flush=True)
                
        except Exception as e:
            print(f"\nError processing {filename}: {e}")

    print("\n\nDone! Results saved to:", OUTPUT_DIR)

    # 4. Summary Table
    print("\n" + "="*45)
    print(f"{'Image Name':<30} | {'Count':<10}")
    print("-" * 45)
    for name, count in summary_data:
        print(f"{name:<30} | {count:<10}")
    print("-" * 45)
    print(f"{'TOTAL':<30} | {total_items:<10}")
    print("="*45 + "\n")

if __name__ == "__main__":
    run_inference()

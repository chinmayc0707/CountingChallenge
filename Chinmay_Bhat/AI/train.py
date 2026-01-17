from ultralytics import YOLO
import os

def train_model():
    # Use Nano Segmentation model for Speed.
    model = YOLO("yolo11n-seg.pt") 

    # Note: Ensure 'datasets/data.yaml' exists. If not, run prepare_data.py
    data_path = r'c:\pr\perspectiv\Chinmay_Bhat\AI\datasets\data.yaml'

    results = model.train(
        data=data_path,
        epochs=100,           # High epochs for convergence
        imgsz=640,            # Standard resolution
        batch=4,              # Batch 4 for VRAM safety
        project=r'c:\pr\perspectiv\Chinmay_Bhat\AI\runs',
        name='yolo11n_fast_v3',
        device='cpu',         # Forced to CPU (No CUDA found)
        workers=0,            # Fix for Windows
        # Augmentation settings for robustness
        fliplr=0.5,
        flipud=0.5,
        mosaic=1.0,
    )

if __name__ == '__main__':
    train_model()

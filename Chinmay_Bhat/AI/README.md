# AI Solution: Screw & Bolt Counting (YOLOv11)

## Approach
This solution uses **YOLOv11** (Nano/Medium) to detect and segment densely packed screws, nuts, and bolts.

### 1. Data Preparation
- **Challenge:** Detecting overlapping/touching objects in dense piles.
- **Solution:** Used a custom **Watershed Algorithm** to generate high-quality polygon ground truth labels.
- **Script:** `prepare_data.py` (Generates images/labels in `datasets/`).

### 2. Model Architecture
- **Framework:** Ultralytics YOLOv11
- **Model:** `yolo11n-seg.pt` (Nano) for Speed, `yolo11m-seg.pt` (Medium) for Accuracy.
- **Image Size:** 640x640

### 3. Usage

#### Retraining (For Speed & Accuracy)
To train the faster Nano model (GPU-optimized):
```bash
python train.py
```
This is configured for `yolo11n-seg` (Nano) with batch size 4.

#### Inference
To count items (configured in `detect.py`):
```bash
python detect.py
```
*   **Speed:** fast (uses GPU).
*   **Accuracy:** Requires the training above to finish (approx 100 epochs).
This will output images with masks and a summary table of counts.

To change the source folder, edit the `SOURCE_PATH` variable at the top of `detect.py`.

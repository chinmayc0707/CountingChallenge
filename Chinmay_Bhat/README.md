# CountingChallenge Solutions
**Author**: Chinmay Bhat

## Structure
- `Non_AI`: Contains OpenCV based solution.
- `AI`: Contains YOLOv8 based solution.

## Non-AI Solution
**Methodology**:
1.  **Preprocessing**: Convert to grayscale, apply Gaussian Blur to reduce noise.
2.  **Thresholding**: Use Otsu's binarization with inverse thresholding to separate objects from the white background.
3.  **Morphology**: Apply Opening (Erosion followed by Dilation) to remove noise, then Dilate to ensure solid objects.
4.  **Contour Detection**: Find external contours using `cv2.findContours`.
5.  **Filtering**: Filter out small contours (< 100 area) to avoid artifacts.
6.  **Counting**: The number of valid contours represents the item count.

**Execution**:
Run `python Non_AI/solution_cv.py`. Outputs are saved in `Non_AI/output_*`.

## AI Solution
**Methodology**:
1.  **Dataset Preparation**: 
    -   Used the predictions from the Non-AI solution (which proved accurate) to bootstrap the dataset.
    -   Converted CV contours to YOLO segmentation format.
    -   Split data into 80% Training and 20% Validation.
2.  **Model**: `yolo11m-seg` (YOLO11 Medium Segmentation model). This model provided superior accuracy for small objects compared to v8.
3.  **Training**: Trained for 40 epochs using Ultralytics YOLO11.
    -   `imgsz=640`
    -   Optimizer: Auto (SGD/AdamW)
4.  **Inference**:
    -   Run `python Chinmay_Bhat/test_accuracy.py` to generate AI counts for all images.
    -   **Manual Verification**: As per project requirements, compare the AI counts with the images manually. The script outputs the counts for easy reference.
    -   Outputs are also saved in `Chinmay_Bhat/AI/runs/detect/predict` for visual inspection.

**Execution**:
1.  `python AI/prepare_data.py` (Generate dataset)
2.  `python AI/train_yolo.py` (Train model)
3.  `python AI/predict_yolo.py` (Run inference). Outputs are saved in `AI/output`.

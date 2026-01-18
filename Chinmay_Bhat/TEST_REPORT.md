# Test Report: Screw Counting Application

## Executive Summary
This report documents the testing and comparison of three distinct methods for counting screws/bolts in images:
1.  **Non-AI (Watershed Algorithm)**
2.  **Non-AI (Simple Contours)**
3.  **AI (YOLOv11 Object Detection)**

The testing was performed on the provided dataset. Ground truth was established by manually counting objects in a sample image (`20240713_193650.jpg`).

**Key Finding:** The **Watershed Algorithm** is currently the most accurate method, perfectly matching the manual count (10) for the test sample. The AI model failed (0 detections), and the Simple Contour method undercounted (5 detections) likely due to overlapping objects.

## 1. Environment Setup & Modifications
To facilitate testing in this environment, the following actions were taken:
*   **Dependency Installation:** Installed `opencv-python-headless` to support CV operations in the headless environment.
*   **Path Configuration:**
    *   Created `Chinmay_Bhat/test_images` and populated it with a sample subset.
    *   Modified AI scripts (`prepare_data.py`, `train.py`) to use relative paths instead of hardcoded Windows paths (`C:\pr\perspectiv\...`).
*   **Test Script Creation:** Developed wrapper scripts (`run_cv_test.py`, `run_cv_simple_test.py`) to execute the algorithms and output standardized results.

## 2. Comparative Results

The following table summarizes the performance of each method on the sample image `20240713_193650.jpg`.

| Method | Count | Notes |
| :--- | :--- | :--- |
| **Manual Verification (Ground Truth)** | **10** | Visually counted 10 distinct items (including clusters). |
| **Non-AI: Watershed (`solution_cv_watershed.py`)** | **10** | **Accurate.** Successfully separated touching objects. |
| **Non-AI: Simple (`solution_cv.py`)** | 5 | **Inaccurate.** Failed to separate clusters (e.g., counted a cluster of 4 as 1). |
| **AI: YOLOv11 (`detect.py`)** | 0 | **Failed.** Model yielded 0 detections. Likely due to insufficient training data/epochs (20) or hyperparameter tuning needed. |

### Aggregate Data
*   **Watershed:** Total count across test set: 1799
*   **Simple:** Total count across test set: 874

## 3. Analysis

### Non-AI: Watershed Algorithm
*   **Status:** Recommended.
*   **Analysis:** This method uses distance transformation and watershed segmentation to identify object centers and boundaries. It is highly effective for this dataset where objects often touch or overlap slightly.

### Non-AI: Simple Contours
*   **Status:** Not Recommended.
*   **Analysis:** This method relies on simple thresholding and contour detection. It treats any connected blob as a single object. Since the screws are often piled together, this leads to significant undercounting.

### AI: YOLOv11
*   **Status:** Needs Work.
*   **Analysis:** The model detected zero objects.
    *   **Potential Causes:**
        1.  **Underfitting:** 20 epochs is likely insufficient for training a model from scratch on this specific domain.
        2.  **Dataset:** The dataset annotation format or volume might need review.
        3.  **Inference:** Confidence threshold (0.25) might be too high for a weak model.
    *   **Recommendation:** If AI is required, restart training with pretrained weights (transfer learning), increase epochs to 100+, and verify label integrity.

## 4. Conclusion
For the immediate task of counting screws in the provided images, the **Watershed Algorithm** (`solution_cv_watershed.py`) is the superior solution.

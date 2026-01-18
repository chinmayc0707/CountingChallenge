# Non-AI Computer Vision Solutions

This directory contains Computer Vision based solutions for counting items (screws, bolts) in images using OpenCV, without the use of Deep Learning models.

## Files

### 1. `solution_cv.py`
This script uses traditional image processing techniques to detect and count objects.

**Methodology:**
1.  **Preprocessing**: Grayscale conversion and Gaussian Blurring.
2.  **Thresholding**: Otsu's binarization with inverse thresholding.
3.  **Morphology**: Opening and Dilation to clean noise and solidify objects.
4.  **Contour Detection**: Finding external contours.
5.  **Filtering**: Filtering out small contours based on area.

**Usage:**
The script currently processes directories defined in the `__main__` block. You may need to update the `dataset_screw_bolt` and `dataset_screws` variables to point to your local dataset paths.

```bash
python solution_cv.py
```
Outputs are saved in `output_screw_bolt` and `output_screws` directories relative to the output base path.

### 2. `solution_cv_watershed.py`
This script uses the Watershed algorithm, which is particularly useful for separating touching objects that might be detected as a single contour in simple thresholding methods.

**Methodology:**
1.  **Preprocessing**: Grayscale conversion and Gaussian Blurring.
2.  **Thresholding**: Otsu's binarization.
3.  **Distance Transform**: Calculates the distance to the nearest zero pixel for each pixel.
4.  **Watershed Algorithm**: Markers are created based on sure foreground and sure background, and the watershed algorithm segregates the regions.
5.  **Counting**: Counts unique markers representing objects.

**Usage:**
You can run this script on a single image.

```bash
python solution_cv_watershed.py <path_to_image>
```
If no path is provided, it defaults to a hardcoded path in the script.

## Dependencies

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy (`numpy`)

Install dependencies using:
```bash
pip install opencv-python numpy
```

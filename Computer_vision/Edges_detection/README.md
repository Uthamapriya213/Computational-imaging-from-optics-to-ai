# Edge Detection from Scratch (Python + OpenCV)

This project implements fundamental edge detection techniques from scratch using Python, NumPy, and OpenCV.

## 🔍 Implemented Algorithms

* Laplacian Edge Detection
* Laplacian of Gaussian (LoG)
* Gradient Computation (Sobel)
* Non-Maximum Suppression (NMS)
* Hysteresis Thresholding (with 8-neighborhood)
* Custom Canny Edge Detector
* OpenCV Canny (for comparison)

---

## 🧠 Key Concepts

### 1. Laplacian

Second-order derivative operator that detects rapid intensity changes.

### 2. LoG (Laplacian of Gaussian)

Applies Gaussian smoothing before Laplacian to reduce noise sensitivity.

### 3. Non-Maximum Suppression (NMS)

Thins edges by keeping only local maxima along gradient direction.

### 4. Hysteresis Thresholding

Connects weak edges to strong edges using 8-neighborhood connectivity.

### 5. Canny Edge Detector

A multi-stage algorithm:

1. Gaussian smoothing
2. Gradient computation
3. NMS
4. Double threshold
5. Edge tracking (hysteresis)

---





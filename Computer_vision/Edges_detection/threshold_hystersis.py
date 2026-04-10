# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 10:07:21 2026

@author: JHS
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Load image
# ----------------------------
img = cv2.imread("lena.jpeg", 0).astype(float)

# ----------------------------
# Gradient (Sobel)
# ----------------------------
gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

mag = np.sqrt(gx**2 + gy**2)
mag = (mag / mag.max()) * 255

# ----------------------------
# Simple Threshold
# ----------------------------
T = 50
simple_edges = mag > T

# ----------------------------
# Hysteresis Threshold
# ----------------------------
T_low = 30
T_high = 80

strong = mag > T_high
weak = (mag >= T_low) & (mag <= T_high)

# Output
hysteresis = np.zeros_like(mag)

# mark strong edges
hysteresis[strong] = 255

# connect weak edges to strong
for i in range(1, mag.shape[0]-1):
    for j in range(1, mag.shape[1]-1):
        if weak[i,j]:
            if np.any(strong[i-1:i+2, j-1:j+2]):
                hysteresis[i,j] = 255

# ----------------------------
# Visualization
# ----------------------------
plt.figure(figsize=(10,6))

plt.subplot(2,2,1); plt.imshow(mag, cmap='gray'); plt.title("Gradient Magnitude")
plt.subplot(2,2,2); plt.imshow(simple_edges, cmap='gray'); plt.title("Simple Threshold")
plt.subplot(2,2,3); plt.imshow(strong, cmap='gray'); plt.title("Strong Edges")
plt.subplot(2,2,4); plt.imshow(hysteresis, cmap='gray'); plt.title("Hysteresis")

plt.tight_layout()
plt.show()
                
                

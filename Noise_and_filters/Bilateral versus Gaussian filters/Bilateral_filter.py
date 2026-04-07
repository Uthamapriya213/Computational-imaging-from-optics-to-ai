
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:28:20 2026

@author: JHS
"""

import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim 
import time

# -----------------------------
# Utility Functions
# -----------------------------

def load_image(path):
    """Load image in grayscale"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32)


def save_image(path, image):
    """Save image to file"""
    image = np.clip(image, 0, 255)
    cv2.imwrite(path, image.astype(np.uint8))
    
def gaussian_noise(image, mean=0, sigma=10):
    """Add Gaussian noise"""
    noise = np.random.normal(mean, sigma, image.shape)
    noisy = image + noise
    return noisy


# Load image
image = cv2.imread("lena.jpeg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
# Add Gaussian noise
noise = np.random.normal(0, 10, image.shape)
noisy = image + noise


# Parameter ranges
sigma_colors = [10, 30, 50, 75, 100]
sigma_spaces = [5, 10, 20, 30, 50]


print("sigma_color | sigma_space | PSNR | SSIM | Time")
best_psnr = 0
best_image = None
for sc in sigma_colors:
    for ss in sigma_spaces:

        start = time.time()

        filtered = cv2.bilateralFilter(
            noisy.astype(np.uint8),
            d=5,
            sigmaColor=sc,
            sigmaSpace=ss
        )

        end = time.time()

        # Metrics
        image_norm=np.clip(image,0,255)/255.0
        filtered_norm = np.clip(filtered,0,255)/255.0
        psnr_val = psnr(image_norm,filtered_norm,data_range=1)
        ssim_val = ssim(image_norm, filtered_norm,data_range=1)
        
        if psnr_val>best_psnr:
            best_psnr = psnr_val
            best_image=filtered
       

        print(f"{sc:^11} | {ss:^11} | {psnr_val:.2f} | {ssim_val:.3f} | {end-start:.4f}")
print('Best PSNR',best_psnr)
save_image('results/Bilateral_filtered_image.png',best_image)       

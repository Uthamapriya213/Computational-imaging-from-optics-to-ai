
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os

# =========================
# Utility Functions
# =========================

def normalize(img):
    return img / 255.0

def gaussian_noise(signal, mean, sigma):
    noise = np.random.normal(mean, sigma, signal.shape)
    return np.clip(signal + noise, 0, 1), noise

def psnr_ssim(ref, test):
    psnr = peak_signal_noise_ratio(ref, test, data_range=1)
    ssim = structural_similarity(ref, test, data_range=1)
    return psnr, ssim

# =========================
# Blur Kernel
# =========================

def motion_kernel(size=21):
    kernel = np.eye(size)
    return kernel / kernel.sum()

# =========================
# Deconvolution Methods
# =========================

def naive_deconvolution(signal, kernel):
    H = np.fft.fft2(np.fft.ifftshift(kernel), signal.shape)
    J = np.fft.fft2(signal)

    eps = 1e-8
    I_rec = np.fft.ifft2(J / (H + eps))
    return np.clip(np.real(I_rec), 0, 1)

def wiener_deconvolution(signal, kernel, K):
    H = np.fft.fft2(np.fft.ifftshift(kernel), signal.shape)
    J = np.fft.fft2(signal)

    I_rec = np.fft.ifft2((np.conj(H) / (np.abs(H)**2 + K)) * J)
    return np.clip(np.real(I_rec), 0, 1)

# =========================
# PSD Visualization
# =========================

def plot_psd(image, title):
    F = np.fft.fftshift(np.fft.fft2(image))
    PSD = np.log(np.abs(F)**2 + 1e-8)

    plt.imshow(PSD, cmap='gray')
    plt.title(title)
    plt.colorbar()
    plt.show()

# =========================
# MAIN PIPELINE
# =========================

# Create results folder if not exists
os.makedirs("../results", exist_ok=True)

# Load image
img = cv2.imread("lena.jpeg", cv2.IMREAD_GRAYSCALE)

# Crop center 256x256 (fixed)
h, w = img.shape
img = img[h//2-200:h//2+228, w//2-228:w//2+228]

img = normalize(img)

# Blur kernel
kernel = motion_kernel(21)

# Forward model
blurred = cv2.filter2D(img, -1, kernel)

# Add noise
sigma = 0.01 * np.std(img)
noisy, noise = gaussian_noise(blurred, 0, sigma)

# =========================
# Metrics BEFORE recovery
# =========================

psnr_blur, ssim_blur = psnr_ssim(img, blurred)
psnr_noisy, ssim_noisy = psnr_ssim(img, noisy)

print("Blurred PSNR:", psnr_blur, "SSIM:", ssim_blur)
print("Noisy PSNR:", psnr_noisy, "SSIM:", ssim_noisy)

# =========================
# Deconvolution
# =========================

naive_rec = naive_deconvolution(noisy, kernel)

K_best = 0.1
wiener_rec = wiener_deconvolution(noisy, kernel, K_best)

# =========================
# Metrics AFTER recovery
# =========================

psnr_naive, ssim_naive = psnr_ssim(img, naive_rec)
psnr_wiener, ssim_wiener = psnr_ssim(img, wiener_rec)

print("Naive PSNR:", psnr_naive, "SSIM:", ssim_naive)
print("Wiener PSNR:", psnr_wiener, "SSIM:", ssim_wiener)

# =========================
# Save Results
# =========================

cv2.imwrite("../results/original.png", img*255)
cv2.imwrite("../results/blurred.png", blurred*255)
cv2.imwrite("../results/noisy.png", noisy*255)
cv2.imwrite("../results/naive.png", naive_rec*255)
cv2.imwrite("../results/wiener.png", wiener_rec*255)

# =========================
# Plot Results
# =========================

plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(2,3,2)
plt.imshow(blurred, cmap='gray')
plt.title("Blurred")

plt.subplot(2,3,3)
plt.imshow(noisy, cmap='gray')
plt.title("Noisy")

plt.subplot(2,3,4)
plt.imshow(naive_rec, cmap='gray')
plt.title("Naive")

plt.subplot(2,3,5)
plt.imshow(wiener_rec, cmap='gray')
plt.title(f"Wiener (K={K_best})")

plt.tight_layout()
plt.show()

# =========================
# PSD Analysis
# =========================

plot_psd(img, "Image PSD")
plot_psd(noise, "Noise PSD")

# =========================
# PSNR vs K Analysis
# =========================

K_values = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2]
psnr_list = []

for k in K_values:
    rec = wiener_deconvolution(noisy, kernel, k)
    psnr, _ = psnr_ssim(img, rec)
    psnr_list.append(psnr)

plt.plot(K_values, psnr_list, marker='o')
plt.xlabel("K")
plt.ylabel("PSNR")
plt.title("PSNR vs Wiener Parameter K")
plt.grid()
plt.show()

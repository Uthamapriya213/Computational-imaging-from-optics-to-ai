import numpy as np
import cv2
import matplotlib.pyplot as plt

# ---------------------------
# Add Gaussian Noise
# ---------------------------
def add_gaussian_noise(img, mean=0, sigma=20):
    noise = np.random.normal(mean, sigma, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


# ---------------------------
# Gaussian Kernel
# ---------------------------
def gaussian_kernel(size, sigma):
    ax = np.arange(-size//2 + 1., size//2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)


# ---------------------------
# Apply Gaussian Filter
# ---------------------------
def apply_gaussian(img, kernel):
    return cv2.filter2D(img, -1, kernel)


# ---------------------------
# Apply Bilateral Filter
# ---------------------------
def apply_bilateral(img, k, sigma_s, sigma_r):
    padded = np.pad(img, k//2, mode='reflect')
    output = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            window = padded[i:i+k, j:j+k]
            center = img[i, j]

            kernel = np.zeros((k, k))
            pad = k // 2

            for x in range(k):
                for y in range(k):
                    dx = x - pad
                    dy = y - pad

                    gs = np.exp(-(dx**2 + dy**2) / (2 * sigma_s**2))
                    gr = np.exp(-(window[x, y] - center)**2 / (2 * sigma_r**2))

                    kernel[x, y] = gs * gr

            kernel /= np.sum(kernel)
            output[i, j] = np.sum(window * kernel)

    return output.astype(np.uint8)


# ---------------------------
# Load Image
# ---------------------------
img = cv2.imread("lena.jpeg", cv2.IMREAD_GRAYSCALE)

# Add noise
noisy = add_gaussian_noise(img)

# Gaussian filtering
g_kernel = gaussian_kernel(7, 2)
gaussian_out = apply_gaussian(noisy, g_kernel)

# Bilateral filtering
bilateral_out = apply_bilateral(noisy, k=7, sigma_s=5, sigma_r=20)

# ---------------------------
# Plot Results
# ---------------------------
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(2,2,2)
plt.imshow(noisy, cmap='gray')
plt.title("Noisy")

plt.subplot(2,2,3)
plt.imshow(gaussian_out, cmap='gray')
plt.title("Gaussian Filter")

plt.subplot(2,2,4)
plt.imshow(bilateral_out, cmap='gray')
plt.title("Bilateral Filter (Edge Preserving)")

plt.show()

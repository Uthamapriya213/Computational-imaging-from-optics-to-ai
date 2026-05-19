import numpy as np
import matplotlib.pyplot as plt
import cv2

from utils import (
    load_image,
    create_box_kernel,
    create_coded_aperture,
    create_rectangle_image
)

from reconstruction import (
    apply_blur,
    wiener_deconvolution
)

from compressed_sensing import (
    l1_reconstruction,
    l2_reconstruction
)


# =====================================================
# LOAD IMAGE
# =====================================================

img = load_image("lena.jpeg")
# =====================================================
# BLUR KERNELS
# =====================================================

box_kernel = create_box_kernel(21)

coded_aperture = create_coded_aperture(21)


# =====================================================
# APPLY BLUR
# =====================================================


coded_aperture = create_coded_aperture(21)


# =====================================================
# APPLY BLUR
# =====================================================

normal_blur = apply_blur(img, box_kernel)

coded_blur = apply_blur(img, coded_aperture)


# =====================================================
# WIENER RECONSTRUCTION
# =====================================================

recovered = wiener_deconvolution(
    coded_blur,
    coded_aperture,
    K=0.01
)


# =====================================================
# DISPLAY RESULTS
# =====================================================

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(coded_blur)
plt.title("Coded Blur")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(recovered)
plt.title("Recovered")
plt.axis("off")

plt.show()



# =====================================================
# COMPRESSED SENSING
# =====================================================

rect = create_rectangle_image()

clipped = rect[600:640, 600:640]

N = 1600

n = 1

I_v = clipped.reshape(N, 1)

norm_img = np.linalg.norm(I_v)

measurements = np.linspace(100, N, 3).astype(np.uint16)

error_l1 = []

error_l2 = []


for M in measurements:

    D = np.random.normal(0, 1/N, (M, N))

    y = D @ I_v

    # L2 reconstruction
    x_l2 = l2_reconstruction(D, y)

    err2 = (
        np.linalg.norm(x_l2 - I_v) / norm_img
    ) * 100

    error_l2.append(err2)

    # L1 reconstruction
    x_l1 = l1_reconstruction(D, y, N)

    err1 = (
        np.linalg.norm(x_l1 - I_v) / norm_img
    ) * 100

    error_l1.append(err1)
    
    if M == measurements[n]:
        
        X_L2 = x_l2.reshape(40,40)
        X_L1 = x_l1.reshape(40,40)


# =====================================================
# ERROR PLOT
# =====================================================

plt.figure(figsize=(8,5))

plt.plot(measurements, error_l1, label='L1')

plt.plot(measurements, error_l2, label='L2')

plt.xlabel("Measurements")

plt.ylabel("Reconstruction Error (%)")

plt.title("Compressed Sensing Reconstruction")

plt.legend()

plt.grid()

plt.show()
#======================================================
# Reconstructed image
#======================================================

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

plt.imshow(X_L1)

plt.title('reconstructed from Basis Pursuit')

plt.subplot(1,2,2)

plt.imshow(X_L2)

plt.title('reconstructed from Least squares')

plt.show()

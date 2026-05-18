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

img = load_image("images/lena.jpeg")


# =====================================================
# BLUR KERNELS
# =====================================================

box_kernel = create_box_kernel(21)

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

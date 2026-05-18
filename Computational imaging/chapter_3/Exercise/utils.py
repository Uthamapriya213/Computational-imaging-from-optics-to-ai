import numpy as np
import cv2


def load_image(path):
    """Load RGB image normalized to [0,1]."""

    img = cv2.imread(path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32) / 255.0

    return img


def create_box_kernel(size=21):
    """Create normalized box blur kernel."""

    kernel = np.ones((size, size), dtype=np.float32)

    kernel /= kernel.sum()

    return kernel


def create_coded_aperture(size=21):
    """Create checkerboard coded aperture."""

    coded = np.zeros((size, size), dtype=np.float32)

    coded[::2, ::2] = 1

    coded[1::2, 1::2] = 1

    coded /= coded.sum()

    return coded


def create_rectangle_image(pixel=1024, length=5e-3):

    dx = length / pixel

    x = np.arange(-pixel, pixel) * dx

    X, Y = np.meshgrid(x, x)

    rect = np.where(
        (np.abs(X) <= 2e-3) & (np.abs(Y) <= 2e-3),
        1,
        0
    )

    return rect

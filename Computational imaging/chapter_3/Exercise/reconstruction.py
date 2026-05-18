import numpy as np
import cv2


def inverse_deconvolution(signal, blur_kernel):
    """Simple inverse filtering."""

    H = np.fft.fft2(
        np.fft.ifftshift(blur_kernel),
        signal.shape[:2]
    )

    J = np.fft.fft2(signal, axes=(0, 1))

    eps = 1e-8

    I_rec_f = J / (H[:, :, None] + eps)

    I_rec = np.fft.ifft2(I_rec_f, axes=(0, 1))

    return np.clip(np.real(I_rec), 0, 1)



def wiener_deconvolution(signal, kernel, K=0.01):
    """Wiener deconvolution."""

    H = np.fft.fft2(
        np.fft.ifftshift(kernel),
        signal.shape[:2]
    )

    G = np.fft.fft2(signal, axes=(0, 1))

    F_hat = (
        np.conj(H) / (np.abs(H)**2 + K)
    )[:, :, None] * G

    f = np.fft.ifft2(F_hat, axes=(0, 1))

    return np.clip(np.real(f), 0, 1)



def apply_blur(img, kernel):
    """Apply convolution blur."""

    return cv2.filter2D(img, -1, kernel)

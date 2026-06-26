import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Simulation Parameters
# ==========================================================

N = 2048

x = np.linspace(-1,1,N)

dx = x[1]-x[0]

fx = np.fft.fftshift(np.fft.fftfreq(N,d=dx))

# ==========================================================
# Normal Rectangular Pupil
# ==========================================================

aperture = (np.abs(x)<0.4).astype(float)

P_normal = aperture.copy()

# ==========================================================
# Edge Enhanced Pupil (Inverse Apodization)
# ==========================================================

edge_weight = (np.abs(x)/0.4)**2

P_inverse = aperture*edge_weight

# Equal transmitted energy
P_normal /= np.sqrt(np.sum(P_normal**2))

P_inverse /= np.sqrt(np.sum(P_inverse**2))

# ==========================================================
# Fourier Transform
# ==========================================================

H_normal = np.fft.fftshift(
    np.fft.fft(
        np.fft.ifftshift(P_normal)
    )
)

H_inverse = np.fft.fftshift(
    np.fft.fft(
        np.fft.ifftshift(P_inverse)
    )
)

# ==========================================================
# Point Spread Function
# ==========================================================

PSF_normal = np.abs(H_normal)**2

PSF_inverse = np.abs(H_inverse)**2

PSF_normal /= PSF_normal.max()

PSF_inverse /= PSF_inverse.max()

# ==========================================================
# Optical Transfer Function
# ==========================================================

MTF_normal = np.abs(
    np.fft.fftshift(
        np.fft.fft(
            np.fft.ifftshift(PSF_normal)
        )
    )
)

MTF_inverse = np.abs(
    np.fft.fftshift(
        np.fft.fft(
            np.fft.ifftshift(PSF_inverse)
        )
    )
)

MTF_normal /= MTF_normal.max()

MTF_inverse /= MTF_inverse.max()

# ==========================================================
# Full Width Half Maximum
# ==========================================================

half = 0.5

idx1 = np.where(PSF_normal>=half)[0]

idx2 = np.where(PSF_inverse>=half)[0]

fwhm_normal = fx[idx1[-1]]-fx[idx1[0]]

fwhm_inverse = fx[idx2[-1]]-fx[idx2[0]]

print(f"Normal FWHM : {fwhm_normal:.2f}")

print(f"Inverse FWHM: {fwhm_inverse:.2f}")

# ==========================================================
# Display
# ==========================================================

fig,ax = plt.subplots(3,2,figsize=(14,10))

# -----------------------
# Pupils
# -----------------------

ax[0,0].plot(x,P_normal)

ax[0,0].set_title("Normal Pupil")

ax[0,0].grid(True)

ax[0,1].plot(x,P_inverse)

ax[0,1].set_title("Inverse Apodized Pupil")

ax[0,1].grid(True)

# -----------------------
# PSFs
# -----------------------

eps = 1e-12

ax[1,0].semilogy(fx,PSF_normal+eps)

ax[1,0].set_title("Normal PSF")

ax[1,0].set_xlabel("Spatial Frequency")

ax[1,0].grid(True)

ax[1,1].semilogy(fx,PSF_inverse+eps)

ax[1,1].set_title("Inverse Apodized PSF")

ax[1,1].set_xlabel("Spatial Frequency")

ax[1,1].grid(True)

# -----------------------
# MTF
# -----------------------

ax[2,0].plot(fx,MTF_normal)

ax[2,0].set_title("Normal MTF")

ax[2,0].set_ylim(0,1.05)

ax[2,0].grid(True)

ax[2,1].plot(fx,MTF_inverse)

ax[2,1].set_title("Inverse Apodized MTF")

ax[2,1].set_ylim(0,1.05)

ax[2,1].grid(True)

plt.tight_layout()

plt.show()

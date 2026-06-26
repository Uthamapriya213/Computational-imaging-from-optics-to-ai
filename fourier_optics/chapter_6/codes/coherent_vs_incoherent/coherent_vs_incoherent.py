
"""
Chapter 3
Coherent vs Incoherent Imaging

Author : Uthamapriya Palani
"""

import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------------
# Grid
# -------------------------------------------------------

N = 2048
L = 1

x = np.linspace(-L/2, L/2, N)
dx = x[1]-x[0]


# -------------------------------------------------------
# Object
# -------------------------------------------------------

f0 = 25

object_field = 1 + 0.8*np.cos(2*np.pi*f0*x)

object_intensity = np.abs(object_field)**2


# -------------------------------------------------------
# Frequency Axis
# -------------------------------------------------------

fx = np.fft.fftshift(
    np.fft.fftfreq(N,dx)
)


# -------------------------------------------------------
# Circular Pupil
# -------------------------------------------------------

cutoff = 20

pupil = np.abs(fx) < cutoff


# -------------------------------------------------------
# Coherent Imaging
# -------------------------------------------------------

object_spectrum = np.fft.fftshift(
    np.fft.fft(
        np.fft.fftshift(object_field)
    )
)

image_field = np.fft.ifftshift(
    np.fft.ifft(
        np.fft.ifftshift(
            object_spectrum*pupil
        )
    )
)

coherent_image = np.abs(image_field)**2


# -------------------------------------------------------
# Amplitude PSF
# -------------------------------------------------------

amplitude_psf = np.fft.ifftshift(
    np.fft.ifft(
        np.fft.ifftshift(pupil)
    )
)


# -------------------------------------------------------
# Intensity PSF
# -------------------------------------------------------

intensity_psf = np.abs(amplitude_psf)**2


# -------------------------------------------------------
# Optical Transfer Function
# -------------------------------------------------------

OTF = np.fft.fftshift(
    np.fft.fft(
        np.fft.fftshift(intensity_psf)
    )
)

OTF = np.real(OTF)

OTF /= OTF.max()


# -------------------------------------------------------
# Incoherent Imaging
# -------------------------------------------------------

object_spectrum = np.fft.fftshift(
    np.fft.fft(
        np.fft.fftshift(object_intensity)
    )
)

image_intensity = np.real(

    np.fft.ifftshift(

        np.fft.ifft(

            np.fft.ifftshift(
                object_spectrum*OTF
            )

        )

    )

)


# -------------------------------------------------------
# Normalize
# -------------------------------------------------------

object_intensity /= object_intensity.max()

coherent_image /= coherent_image.max()

image_intensity /= image_intensity.max()


# -------------------------------------------------------
# Plot
# -------------------------------------------------------

fig,ax = plt.subplots(3,1,figsize=(10,8))

ax[0].plot(x,object_intensity)
ax[0].set_title("Object Intensity")

ax[1].plot(x,coherent_image)
ax[1].set_title("Coherent Imaging")

ax[2].plot(x,image_intensity)
ax[2].set_title("Incoherent Imaging")

plt.tight_layout()
plt.show()

# Coherent vs Incoherent Imaging

## Overview

This example demonstrates the difference between coherent and incoherent optical imaging.

Topics covered

- Fourier optics
- Imaging systems
- Pupil function
- Point Spread Function (PSF)
- Optical Transfer Function (OTF)
- Coherent transfer function
- Incoherent transfer function

## Theory

For coherent imaging

Image field

Ui(x)=F−1{P(f)T(f)}

Image intensity

Ic(x)=|Ui(x)|²

For incoherent imaging

Ii(x)=|ui(x)|²*PSF(x)

or equivalently

Ii(f)=OTF(f)O(f)

where

OTF = FFT(PSF)

PSF = |h(x)|²

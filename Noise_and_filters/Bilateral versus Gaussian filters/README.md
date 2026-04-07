## Bilateral Filter Parameter Study

We evaluated the effect of sigma_color and sigma_space on image quality.

### Observations

- Small sigma_color preserves edges but removes less noise
- Large sigma_color increases smoothing but reduces edge sharpness
- Larger sigma_space increases smoothing over larger regions
- Optimal performance occurs at moderate parameter values

### Metrics

- PSNR: Measures reconstruction quality
- SSIM: Measures structural similarity

Gaussian filtering uses a fixed kernel and performs uniform smoothing.

Bilateral filtering uses a data-dependent kernel that varies for each pixel,
combining spatial proximity and intensity similarity.

This allows bilateral filtering to preserve edges while smoothing noise.

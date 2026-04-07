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

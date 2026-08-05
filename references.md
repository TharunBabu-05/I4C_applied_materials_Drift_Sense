# References & Citations — Drift-Sense

All augmentation choices, noise models, and structural parameters are justified
with credible public references as required by the hackathon rules.

---

## 1. SEM Imaging Physics & Noise Models

### [R1] Goldstein, J.I., Newbury, D.E., et al. (2018)
**"Scanning Electron Microscopy and X-Ray Microanalysis"**
Springer, 4th Edition.
- **Used for:** SEM secondary electron contrast mechanism, edge brightening
  (topographic contrast), vignetting from column optics, electronic noise.
- **Key insight:** Secondary electron yield is highest at edges and steep
  topography, producing the characteristic bright-edge contrast in SEM images.
  This is the dominant contrast mechanism for semiconductor surface imaging.
- **Chapters cited:** Ch.2 (Electron Optics), Ch.3 (SE & BSE Contrast),
  Ch.12 (Specimen Stages & Drift).

### [R2] Foi, A., Trimeche, M., Katkovnik, V., Egiazarian, K. (2008)
**"Practical Poissonian-Gaussian Noise Modeling and Fitting for
Single-Image Raw-Data"**
IEEE Transactions on Image Processing, Vol. 17, No. 10, pp. 1737-1754.
- **Used for:** Mixed Poisson-Gaussian noise model for SEM imaging.
- **Key insight:** SEM image noise follows z = k·Poisson(x/k) + Gaussian(0,σ²),
  where Poisson component is signal-dependent (shot noise from electron
  counting) and Gaussian component is signal-independent (electronic read noise).
- **Application:** We apply Poisson noise (scale 0.85–0.99) for shot noise and
  Gaussian noise (σ=3–20) for read noise, with higher noise on the search image.

### [R3] Joy, D.C. (2002)
**"SMART – a program to measure SEM resolution and imaging performance"**
Journal of Microscopy, Vol. 208, pp. 24-34.
- **Used for:** SEM resolution limits, noise-to-signal characterization.
- **Key insight:** At lower magnifications (10x), each pixel integrates over
  a larger physical area with proportionally more noise per feature,
  justifying our higher noise levels on the search image.

---

## 2. DRAM Architecture & Semiconductor Structure

### [R4] Kim, K. & Lee, J.G. (1998)
**"DRAM Technology Perspective for Gigabit Era"**
IEEE Transactions on Electron Devices, Vol. 45, No. 3.
- **Used for:** DRAM cell architecture, 4F² layout, word-line/bit-line pitch.
- **Key insight:** DRAM cells are organized in a rectangular array with
  word-lines (horizontal, row select) and bit-lines (vertical, data carry).
  Cell area follows 4F² rule where F is the minimum feature half-pitch.
- **Application:** Our generator creates word-line/bit-line grids with
  pitches of 25–45 pixels (at 1nm/px), corresponding to real DRAM half-pitches
  of 20–44nm across technology nodes.

### [R5] ITRS (International Technology Roadmap for Semiconductors)
**"ITRS 2.0 — More Moore" (2015)**
- **Used for:** Feature size scaling, DRAM pitch dimensions.
- **Key insight:** DRAM word-line/bit-line half-pitch has scaled from ~44nm
  (at 4x nm node) to sub-15nm at advanced nodes. Contact/via dimensions
  scale proportionally.
- **Application:** Contact dot radius of 3–7 pixels (nm) in our generator.

### [R6] Keeth, B. & Baker, R.J. (2001)
**"DRAM Circuit Design: A Tutorial"**
IEEE Press / Wiley.
- **Used for:** DRAM array layout, line width-to-space ratios.
- **Key insight:** Typical metal line width is 40–50% of the pitch,
  with the remaining space serving as dielectric isolation. Contact/via
  dots connect the bit-line to the storage capacitor at each cell.
- **Application:** Our line_width_fraction = 0.35–0.50 of pitch.

---

## 3. SEM Augmentation Justification

### [R7] Reimer, L. & Kohl, H. (2008)
**"Transmission Electron Microscopy: Physics of Image Formation"**
Springer, 5th Edition.
- **Used for:** Gaussian blur model for beam spot size / defocus.
- **Key insight:** The electron probe has a finite diameter approximated
  by a Gaussian profile. Any defocus adds a convolution with a broader
  Gaussian PSF.
- **Application:** We apply Gaussian blur (σ=0.3–1.5) to simulate beam
  spot size and defocus effects.

### [R8] Postek, M.T. & Vladár, A.E. (2013)
**"Does your SEM really tell the truth? How would you know?"**
Scanning, Vol. 35, pp. 355-361.
- **Used for:** Intensity variation, drift between measurements.
- **Key insight:** SEM measurements can vary between sessions due to
  detector gain drift, beam current fluctuation, and contamination buildup.
  This causes global brightness/contrast shifts between captures.
- **Application:** Our gain_variation (0.90–1.10) and offset_variation
  (±8 gray levels) model inter-capture detector drift.

### [R9] Villarrubia, J.S., Vladár, A.E., et al. (2005)
**"Scanning electron microscope measurement of width and shape of
10nm patterned lines using a JMONSEL-modeled library"**
Ultramicroscopy, Vol. 107, pp. 1-14.
- **Used for:** Edge effect modeling in SEM of semiconductor patterns.
- **Key insight:** Edge brightening in SE images arises from increased
  secondary electron yield at topographic edges. The effect magnitude
  depends on material, beam energy, and edge geometry.
- **Application:** Our edge_brightening uses Sobel edge detection +
  additive blend (strength 0.05–0.18) to simulate this effect.

---

## 4. Computer Vision & Localization Algorithms

### [R10] Lewis, J.P. (1995)
**"Fast Normalized Cross-Correlation"**
Vision Interface, pp. 120-123.
- **Used for:** NCC algorithm for template matching / fine localization.
- **Key insight:** NCC normalizes the correlation to [-1, +1], making it
  invariant to linear brightness and contrast changes — critical for
  matching between the differently-noised reference and search images.

### [R11] Kuglin, C.D. & Hines, D.C. (1975)
**"The Phase Correlation Image Alignment Method"**
IEEE International Conference on Cybernetics and Society.
- **Used for:** Phase correlation for coarse localization.
- **Key insight:** Phase correlation uses the normalized cross-power
  spectrum (phase-only) to detect translational shifts. It produces
  sharper peaks than standard cross-correlation and is more robust to
  noise and illumination changes.

### [R12] Foroosh, H., Zerubia, J.B., Berthod, M. (2002)
**"Extension of Phase Correlation to Subpixel Registration"**
IEEE Transactions on Image Processing, Vol. 11, No. 3.
- **Used for:** Subpixel refinement of phase correlation results.
- **Key insight:** Subpixel accuracy can be achieved by fitting a
  parabola or sinc function to the phase correlation peak.

---

## Citation Summary by Augmentation

| Augmentation | Citations |
|-------------|-----------|
| Poisson (shot) noise | [R2] Foi et al. 2008, [R3] Joy 2002, [R1] Goldstein 2018 |
| Gaussian (read) noise | [R2] Foi et al. 2008, [R1] Goldstein 2018 |
| Edge brightening | [R1] Goldstein 2018, [R9] Villarrubia et al. 2005, [R7] Reimer & Kohl 2008 |
| Gaussian blur (beam) | [R7] Reimer & Kohl 2008, [R3] Joy 2002 |
| Rotation (stage drift) | [R1] Goldstein 2018, [R8] Postek & Vladár 2013 |
| Vignetting | [R1] Goldstein 2018, [R7] Reimer & Kohl 2008 |
| Intensity variation | [R8] Postek & Vladár 2013, [R1] Goldstein 2018 |
| DRAM WL/BL structure | [R4] Kim & Lee 1998, [R5] ITRS 2015, [R6] Keeth & Baker 2001 |
| Phase correlation | [R11] Kuglin & Hines 1975, [R12] Foroosh et al. 2002 |
| NCC matching | [R10] Lewis 1995 |

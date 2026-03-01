# Panoramic Image Stitching

A notebook-driven computer-vision project for stitching multiple overlapping images into a single panorama.

This repository contains an end-to-end experimentation pipeline for:
- loading image sets,
- locating overlap regions with normalized cross-correlation over Sobel edge responses,
- estimating pairwise alignment from template matches, and
- composing a panorama from the matched inputs.

The core implementation lives in a Jupyter notebook (`pano-stitch.ipynb`) with utility functions in `p2Helpers.py`.

---

## Repository Structure

- `pano-stitch.ipynb` — primary workflow and experiments for feature matching and panorama construction.
- `p2Helpers.py` — helper utilities for:
  - normalized cross-correlation (`normxcorr2`),
  - image/template loading,
  - correlation visualization,
  - template localization and overlap scoring.
- `DS/` — sample source datasets used for stitching experiments.
- `T/` — template images used for template-matching stages.

---

## Approach Overview

At a high level, the stitching workflow is:

1. **Load input images** from one of the dataset folders.
2. **Compute edge maps** (Sobel x/y) for both candidate images and templates.
3. **Run normalized cross-correlation** to identify likely overlap/control-point locations.
4. **Extract matching regions** and score alignment quality.
5. **Build image relationships** and place images into a composite panorama.

This implementation is designed for educational exploration and algorithm prototyping in notebooks.

---

## Requirements

- Python 3.9+ (recommended)
- Jupyter Notebook or JupyterLab
- Python packages:
  - `numpy`
  - `matplotlib`
  - `pandas`
  - `scipy`
  - `scikit-image`
  - `ipython`

Install dependencies with:

```bash
pip install numpy matplotlib pandas scipy scikit-image ipython notebook
```

---

## Quick Start

1. **Clone the repository**
   ```bash
   git clone <your-fork-or-this-repo-url>
   cd pano-stitch
   ```

2. **(Optional) Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter**
   ```bash
   jupyter notebook pano-stitch.ipynb
   ```

5. **Run notebook cells top-to-bottom** to reproduce the stitching pipeline.

6. **Run the CLI stitcher directly on images**
   ```bash
   python stitch_images.py DS/1/A.tif DS/1/B.tif DS/1/C.tif DS/1/D.tif -o stitched.tif
   ```
   Optional flags:
   - `--tile-factor 4` controls candidate template tiling density.
   - `--background 255` controls empty canvas fill value.

---

## Data Notes

- Example input image sets are included in `DS/` and `T/`.
- The project was originally developed around `.tif` imagery; keeping that format is recommended for easiest reuse.
- Some folders include macOS metadata files (e.g., `.DS_Store`), which are not required for execution.

---

## Known Limitations

- The current pipeline is research/prototype-oriented and tuned for the included sample data.
- Robustness can vary with major changes in scale, perspective, illumination, or very low overlap.
- The code is notebook-first rather than packaged as a production CLI/library.

---

## Authors

- **Paul Mobbs** — initial work ([pmobbs](https://github.com/pmobbs))
- **Alex Martinez Paiz** — initial work

---

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).

---

## Acknowledgments

This project makes use of images provided by Dr. Sally Wood of Santa Clara University.

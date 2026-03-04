
This ia a set of tools I am making for Ozbudak lab. They process Nikon ND2 files from multi-embryo timelapse experiments. Corrects XY and Z drift across timepoints so each embryo stays centered at the same anatomical depth throughout the movie, then exports OME-TIFFs for downstream analysis and MP4s for inspection.

## Data shape

ND2 files load as `(T, P, Z, C, Y, X)` — timepoints, embryo positions, Z-slices, channels, Y, X.

## Workflow

```
find_threshold.py  →  YAML config  →  centroid_align_xy.py  →  [centroid_align_z.py]
```

1. **`find_threshold.py`** (Streamlit) — tune Gaussian blur (`sigma`) and percentile threshold interactively on the real data until the embryo mask is clean, then save parameters to a YAML config.
2. **`centroid_align_xy.py`** — reads the YAML, detects each embryo's centroid at every timepoint by thresholding the chosen channel, and shifts the full (C, Z, Y, X) frame to keep it centered. Outputs one OME-TIFF per embryo and one MP4 per channel.
3. **`centroid_align_z.py`** — corrects focal-plane drift by tracking the Z intensity centroid of the threshold channel across time. Takes a reference timepoint (`--ref_t`) representing the Z position you want for analysis; all other frames are shifted to match it.
4. **`centroid_align_xy.py --from_z_tiffs`** — optional second XY pass on the Z-corrected TIFFs to remove any residual lateral drift.

## Other scripts

- **`movie_from_nd2.py`** — quick unaligned MP4 preview straight from the ND2, useful before committing to threshold tuning.
- **`z_diagnostic.py`** — evaluates multiple Z-position metrics (mean intensity, variance, Laplacian variance, Tenengrad) against stage Z from the ND2 event log to confirm which metric best tracks real drift.
- **`getnd2metadata.py`** / **`visualizer.py`** — exploratory tools with a hardcoded `FILE_PATH` at the top; edit before running.

## Installation

```bash
pip install nd2 numpy scipy tifffile imageio streamlit Pillow matplotlib tqdm pyyaml
```


Tools for Ozbudak lab that process Nikon ND2 files from multi-embryo timelapse experiments. Corrects XY and Z drift across timepoints so each embryo stays centered at the same anatomical depth throughout the movie, then exports OME-TIFFs for downstream analysis and MP4s for inspection.

## Data shape

ND2 files load as `(T, P, Z, C, Y, X)`, which stands for timepoints, embryo positions, Z-slices, channels, Y, and X. The OME-TIFFs that `nd2_to_tif.py` writes for each embryo have shape `(T, C, Z, Y, X)`.

## Workflow

```
nd2_to_tif.py  →  find_threshold.py  →  YAML config  →  centroid_align_xy.py  →  [centroid_align_z.py]
```

1. **`nd2_to_tif.py`** converts a folder of ND2 files into one OME-TIFF for each embryo. It splits each ND2 along the position axis and concatenates matching positions across ND2 files along the time axis.
2. **`find_threshold.py`** is a tkinter app for interactively tuning the Gaussian blur sigma and percentile threshold until the embryo mask is clean. It saves the parameters to a YAML config file. It accepts both ND2 files with multiple embryos and single-embryo TIF files.
3. **`centroid_align_xy.py`** reads the YAML config, detects each embryo's centroid at every timepoint by thresholding the chosen channel, and shifts the full (C, Z, Y, X) frame to keep it centered. It outputs one OME-TIFF and one MP4 for each channel.
4. **`centroid_align_z.py`** corrects focal-plane drift by tracking the Z intensity centroid of the threshold channel across time. It takes a reference timepoint (`--ref_t`) representing the Z position you want for analysis, and shifts all other frames to match it.

## Other scripts

- **`movie_from_nd2.py`** generates a quick unaligned MP4 preview straight from the ND2 file, useful before committing to threshold tuning.
- **`getnd2metadata.py`** and **`visualizer.py`** are exploratory tools. They have a hardcoded `FILE_PATH` variable at the top that must be edited before running.

## Installation

```bash
pip install nd2 numpy scipy tifffile imageio Pillow matplotlib tqdm pyyaml torch
```

FFmpeg must be installed separately (system binary, not a pip package) for MP4 encoding.

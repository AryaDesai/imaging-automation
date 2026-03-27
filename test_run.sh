#!/usr/bin/env bash
# test_run.sh — run the full Z-then-XY alignment pipeline with enlarge_canvas
# enabled on both steps so no image data is discarded at either stage.
#
# Step 0: Remove outputs from any previous run so the Z alignment step sees
#         only the base XY-corrected TIFFs (P0–P3.ome.tif). Without this,
#         leftover _z_xy.ome.tif files from earlier runs would also be
#         Z-corrected, doubling the number of output files and confusing the
#         subsequent XY step.
#
# Step 1: Z alignment on the XY-corrected TIFFs in aligned_nd1188/.
#         Reads nd1188_P{0-3}.ome.tif, writes nd1188_P{0-3}_z.ome.tif.
#         --enlarge_canvas pads the Z dimension asymmetrically so that no
#         slices are discarded. Because each embryo's drift is different, the
#         four output TIFFs may have different Z depths.
#
# Step 1b: Normalise Z stack sizes. centroid_align_xy.py --from_z_tiffs
#          stacks all embryos into a single array, which requires identical
#          shapes. The four _z.ome.tif files are padded to the same Z depth
#          here (zeros appended to the high end of smaller stacks) before
#          the XY step reads them.
#
# Step 2: XY alignment on the Z-corrected TIFFs just written.
#         --from_z_tiffs reads the *_z.ome.tif files instead of the raw ND2.
#         --enlarge_canvas pads XY asymmetrically for the same reason.
#         Output: nd1188_P{0-3}_z_xy.ome.tif

set -euo pipefail

YAML="nd1188_Venus_threshold.yaml"
ALIGNED_DIR="aligned_nd1188"
BASE="nd1188"

# ── Step 0: clean previous outputs ───────────────────────────────────────────
echo "=== Step 0: removing previous Z-pipeline outputs ==="
for p in 0 1 2 3; do
    rm -f "${ALIGNED_DIR}/${BASE}_P${p}_z.ome.tif"
    rm -f "${ALIGNED_DIR}/${BASE}_P${p}_z_xy.ome.tif"
    rm -f "${ALIGNED_DIR}/${BASE}_P${p}_z_xy_z.ome.tif"
done
echo "Done."

# ── Step 1: Z alignment ───────────────────────────────────────────────────────
echo ""
echo "=== Step 1: Z alignment (enlarge_canvas) ==="
python centroid_align_z.py "$YAML" --enlarge_canvas

# ── Step 1b: normalise Z stack sizes across embryos ───────────────────────────
echo ""
echo "=== Step 1b: normalising Z stack sizes ==="
python - <<'PYEOF'
# centroid_align_z.py --enlarge_canvas pads each embryo by its own max shift,
# so the four _z.ome.tif files can have different Z depths. This block pads
# all of them to the maximum Z found, appending zeros at the high end of
# smaller stacks. The high end is chosen because the enlarge_canvas padding
# also appended there for positive shifts, making the zero region contiguous.
import glob
import os
import re

import numpy as np
import tifffile
import yaml

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from useful_functions import load_nd2_metadata, save_ome_tiff

YAML = "nd1188_Venus_threshold.yaml"

with open(YAML) as f:
    cfg = yaml.safe_load(f)

nd2_path      = cfg["source"]["file"]
base          = os.path.splitext(os.path.basename(nd2_path))[0]
aligned_dir   = os.path.join(os.path.dirname(nd2_path) or ".", f"aligned_{base}")
channel_names, vox, period_s = load_nd2_metadata(nd2_path)

# Match exactly {base}_P\d+_z.ome.tif to exclude _z_xy_z.ome.tif etc.
all_paths = sorted(glob.glob(os.path.join(aligned_dir, f"{base}_P*_z.ome.tif")))
paths = [
    p for p in all_paths
    if re.fullmatch(rf".*{re.escape(base)}_P\d+_z\.ome\.tif", p)
]

vols    = [tifffile.imread(p).astype(np.float32) for p in paths]
z_sizes = [v.shape[2] for v in vols]
max_z   = max(z_sizes)

print(f"  Z sizes before normalisation: {dict(zip([os.path.basename(p) for p in paths], z_sizes))}")

if len(set(z_sizes)) == 1:
    print(f"  All equal ({max_z} slices) — nothing to do.")
else:
    print(f"  Padding all stacks to Z={max_z} ...")
    for vol, path in zip(vols, paths):
        dz = max_z - vol.shape[2]
        if dz > 0:
            vol = np.pad(vol, ((0, 0), (0, 0), (0, dz), (0, 0), (0, 0)))
        save_ome_tiff(path, vol, channel_names, vox, period_s)
        print(f"  Saved {os.path.basename(path)} (Z={vol.shape[2]})")

PYEOF

# ── Step 2: XY alignment on Z-corrected TIFFs ────────────────────────────────
echo ""
echo "=== Step 2: XY alignment on Z-corrected TIFFs (enlarge_canvas) ==="
python centroid_align_xy.py "$YAML" --from_z_tiffs --enlarge_canvas

echo ""
echo "=== Done ==="

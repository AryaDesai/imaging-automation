"""Stabilize embryo movies by centroid alignment.

Usage: python centroid_align.py nd1188_Venus_threshold.yaml
       python centroid_align.py nd1188_Venus_threshold.yaml --enlarge_canvas
"""

import argparse
import os
import sys

import imageio
import nd2
import numpy as np
import tifffile
import yaml
from scipy.ndimage import gaussian_filter, label, shift
from tqdm import tqdm


def load_nd2(file_path):
    """Load ND2 file and return full array (T,P,Z,C,Y,X) + channel names."""
    f = nd2.ND2File(file_path)
    data = f.asarray()  # (T, P, Z, C, Y, X)
    channel_names = [ch.channel.name for ch in f.metadata.channels]
    f.close()
    return data.astype(np.float32), channel_names


def find_largest_component_centroid(smoothed, percentile):
    """Threshold at percentile, return centroid (y, x) of largest connected component."""
    binary = smoothed > np.percentile(smoothed, percentile)
    labeled, _ = label(binary)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    if sizes.max() == 0:
        # No component found — return frame center
        return smoothed.shape[0] / 2, smoothed.shape[1] / 2
    mask = labeled == sizes.argmax()
    centroid = np.argwhere(mask).mean(axis=0)  # (y, x)
    return centroid[0], centroid[1]


def align_frame(frame_all_channels, sigma, percentile, ch_idx):
    """Compute centroid on max-Z of threshold channel, shift all Z slices to center it.

    Parameters
    ----------
    frame_all_channels : ndarray (C, Z, Y, X)
    sigma, percentile : threshold parameters
    ch_idx : index of channel used for centroid detection

    Returns
    -------
    shifted : ndarray (C, Z, Y, X) with black fill at edges
    """
    Y, X = frame_all_channels.shape[2], frame_all_channels.shape[3]
    data = frame_all_channels[ch_idx].max(axis=0)  # (Y, X)
    smoothed = gaussian_filter(data, sigma=sigma)
    cy, cx = find_largest_component_centroid(smoothed, percentile)
    dy, dx = Y / 2 - cy, X / 2 - cx

    shifted = shift(frame_all_channels, (0, 0, dy, dx), order=1, mode="constant", cval=0)
    return shifted


def compute_shift(frame_all_channels, sigma, percentile, ch_idx):
    """Return (dy, dx) that moves the embryo centroid to the frame centre.
    Used by --enlarge_canvas to separate the precompute pass from the apply pass.

    Parameters
    ----------
    frame_all_channels : ndarray (C, Z, Y, X)
    sigma, percentile : smoothing / threshold parameters
    ch_idx : channel index used for centroid detection
    """
    Y, X = frame_all_channels.shape[2], frame_all_channels.shape[3]
    projection = frame_all_channels[ch_idx].max(axis=0)  # collapse Z
    smoothed = gaussian_filter(projection, sigma=sigma)
    cy, cx = find_largest_component_centroid(smoothed, percentile)
    return Y / 2 - cy, X / 2 - cx


def auto_contrast(img, percentile=99.5):
    """Scale image to 0-255 uint8 using percentile clipping."""
    vmax = np.percentile(img, percentile)
    if vmax == 0:
        vmax = 1
    return np.clip(img / vmax * 255, 0, 255).astype(np.uint8)


def make_grid_frame(images, nrows=2, ncols=2):
    """Arrange up to nrows*ncols grayscale images into a grid.

    Parameters
    ----------
    images : list of 2D uint8 arrays (Y, X)

    Returns
    -------
    grid : uint8 array (nrows*Y, ncols*X)
    """
    H, W = images[0].shape
    grid = np.zeros((nrows * H, ncols * W), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, ncols)
        grid[r * H : (r + 1) * H, c * W : (c + 1) * W] = img
    return grid


def main():
    parser = argparse.ArgumentParser(description="Centroid-align embryo movies from ND2 data.")
    parser.add_argument("yaml_file", help="Threshold YAML from find_threshold.py")
    parser.add_argument("--fps", type=float, default=2, help="Frames per second (default: 2)")
    parser.add_argument("--enlarge_canvas", action="store_true",
                        help="Precompute all shifts, print them, then expand the canvas "
                             "asymmetrically before aligning so no edge data is lost.")
    args = parser.parse_args()

    # 1. Load YAML
    with open(args.yaml_file) as f:
        cfg = yaml.safe_load(f)

    sigma = cfg["parameters"]["sigma"]
    percentile = cfg["parameters"]["percentile"]
    ch_idx = cfg["parameters"]["channel_index"]
    nd2_path = cfg["source"]["file"]

    if not os.path.isfile(nd2_path):
        print(f"Error: ND2 file not found: {nd2_path}", file=sys.stderr)
        sys.exit(1)

    # 2. Load ND2
    print(f"Loading {nd2_path} ...")
    data, channel_names = load_nd2(nd2_path)
    T, P, Z, C, Y, X = data.shape
    print(f"  Shape: T={T}, P={P}, Z={Z}, C={C}, Y={Y}, X={X}")
    print(f"  Channels: {channel_names}")
    print(f"  Threshold channel: {channel_names[ch_idx]} (index {ch_idx})")
    print(f"  sigma={sigma}, percentile={percentile}")

    # Output directory
    base = os.path.splitext(os.path.basename(nd2_path))[0]
    out_dir = os.path.join(os.path.dirname(nd2_path) or ".", f"aligned_{base}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Output: {out_dir}/")

    # Both paths populate this identically; MP4 writing is shared below.
    aligned_for_mp4 = [[[] for _ in range(T)] for _ in range(C)]

    if args.enlarge_canvas:
        # --enlarge_canvas: two-pass approach.
        #
        # Pass 1 — precompute all (dy, dx) shifts without touching the canvas.
        #   Printing each value as it arrives lets the user spot directional drift
        #   patterns before we commit to any canvas expansion.
        print(f"\n--- Pass 1: precomputing shifts ---")
        shifts = np.zeros((P, T, 2))  # shifts[p, t] = [dy, dx]
        for p in range(P):
            print(f"\n  Embryo {p}/{P-1}:")
            for t in range(T):
                frame = data[t, p].transpose(1, 0, 2, 3)  # (Z,C,Y,X) -> (C,Z,Y,X)
                dy, dx = compute_shift(frame, sigma, percentile, ch_idx)
                shifts[p, t] = [dy, dx]
                print(f"    T={t:3d}:  dy={dy:+7.2f}  dx={dx:+7.2f}")

        dy_all, dx_all = shifts[:, :, 0], shifts[:, :, 1]
        print(f"\nShift summary (pixels):")
        print(f"  dy  min={dy_all.min():+.1f}  max={dy_all.max():+.1f}  mean={dy_all.mean():+.1f}")
        print(f"  dx  min={dx_all.min():+.1f}  max={dx_all.max():+.1f}  mean={dx_all.mean():+.1f}")

        # Expand the canvas asymmetrically so no original data is clipped by the shifts.
        #   scipy shift with +dy moves content downward → original bottom rows are cut → pad bottom.
        #   scipy shift with -dy moves content upward   → original top rows are cut    → pad top.
        #   Same logic applies to dx / left / right.
        pad_top    = int(np.ceil(max(0, -dy_all.min())))
        pad_bottom = int(np.ceil(max(0,  dy_all.max())))
        pad_left   = int(np.ceil(max(0, -dx_all.min())))
        pad_right  = int(np.ceil(max(0,  dx_all.max())))
        print(f"\nCanvas padding:  top={pad_top}  bottom={pad_bottom}  left={pad_left}  right={pad_right}")

        # Pad only the Y and X axes; everything else is unchanged.
        # The precomputed shifts are still valid on the larger canvas — each embryo
        # centroid lands consistently at (pad_top + Y_orig/2, pad_left + X_orig/2),
        # i.e. stabilised but not re-centred in the new frame. That is intentional.
        data = np.pad(data, ((0,0),(0,0),(0,0),(0,0),(pad_top, pad_bottom),(pad_left, pad_right)))
        T, P, Z, C, Y, X = data.shape
        print(f"  Expanded canvas: Y={Y}  X={X}")

        # Pass 2 — apply precomputed shifts on the expanded canvas.
        #   No centroid recomputation: we reuse shifts[] directly.
        print(f"\n--- Pass 2: aligning {P} embryo(s) × {T} timepoints ---")
        for p in range(P):
            print(f"\n  Embryo {p}/{P-1}")
            volume = np.zeros((T, C, Z, Y, X), dtype=np.float32)
            for t in tqdm(range(T), desc=f"    Aligning", unit="frame", leave=True):
                dy, dx = shifts[p, t]
                frame = data[t, p].transpose(1, 0, 2, 3)  # (Z,C,Y,X) -> (C,Z,Y,X)
                shifted = shift(frame, (0, 0, dy, dx), order=1, mode="constant", cval=0)
                volume[t] = shifted
                for c in range(C):
                    aligned_for_mp4[c][t].append(auto_contrast(shifted[c].max(axis=0)))

            fname = f"{base}_P{p}.ome.tif"
            fpath = os.path.join(out_dir, fname)
            print(f"    Saving OME-TIFF: {fname} ...")
            tifffile.imwrite(fpath, volume, imagej=False, photometric="minisblack",
                             metadata={"axes": "TCZYX", "Channel": {"Name": channel_names}})
            print(f"    Saved {fname}")

    else:
        # Default: single-pass alignment. Shift is computed and applied per frame.
        print(f"\nAligning {P} embryo(s) × {T} timepoints ...")
        for p in range(P):
            print(f"\n  Embryo {p}/{P-1}")
            volume = np.zeros((T, C, Z, Y, X), dtype=np.float32)
            for t in tqdm(range(T), desc=f"    Aligning timepoints", unit="frame", leave=True):
                frame = data[t, p].transpose(1, 0, 2, 3)  # (Z,C,Y,X) -> (C,Z,Y,X)
                shifted = align_frame(frame, sigma, percentile, ch_idx)
                volume[t] = shifted
                for c in range(C):
                    aligned_for_mp4[c][t].append(auto_contrast(shifted[c].max(axis=0)))

            fname = f"{base}_P{p}.ome.tif"
            fpath = os.path.join(out_dir, fname)
            print(f"    Saving OME-TIFF: {fname} ...")
            tifffile.imwrite(fpath, volume, imagej=False, photometric="minisblack",
                             metadata={"axes": "TCZYX", "Channel": {"Name": channel_names}})
            print(f"    Saved {fname}")

    # Save MP4s — one per channel, 2×2 grid over time (shared by both paths).
    print(f"\nWriting MP4s for {C} channel(s) ...")
    for c in range(C):
        ch_name = channel_names[c]
        mp4_path = os.path.join(out_dir, f"{base}_{ch_name}_aligned.mp4")
        print(f"  Channel '{ch_name}': {mp4_path}")

        writer = imageio.get_writer(mp4_path, fps=args.fps)
        for t in tqdm(range(T), desc=f"    Encoding frames", unit="frame", leave=True):
            grid = make_grid_frame(aligned_for_mp4[c][t])
            rgb = np.stack([grid, grid, grid], axis=-1)
            writer.append_data(rgb)
        writer.close()
        print(f"  Saved {os.path.basename(mp4_path)}")

    print("\nDone.")


if __name__ == "__main__":
    main()

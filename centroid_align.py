"""Stabilize embryo movies by centroid alignment.

Usage: python centroid_align.py nd1188_Venus_threshold.yaml
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

    # 3. Align and save OME-TIFFs (one per embryo, shape T,C,Y,X)
    # Also accumulate frames for MP4s: aligned[c][t] = list of P grayscale images
    aligned_for_mp4 = [[[] for _ in range(T)] for _ in range(C)]

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

    # 4. Save MP4s — one per channel, 2×2 grid over time
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

"""Make 2x2 grid MP4s from a raw ND2 file (no alignment).

Usage: python movie_from_nd2.py nd1188.nd2
"""

import argparse
import os
import sys

import imageio
import nd2
import numpy as np


def load_nd2(file_path):
    """Load ND2 file and return Z-max-projected array (T,P,C,Y,X) + channel names."""
    f = nd2.ND2File(file_path)
    data = f.asarray()  # (T, P, Z, C, Y, X)
    channel_names = [ch.channel.name for ch in f.metadata.channels]
    f.close()
    return data.max(axis=2).astype(np.float32), channel_names


def auto_contrast(img, percentile=99.5):
    """Scale image to 0-255 uint8 using percentile clipping."""
    vmax = np.percentile(img, percentile)
    if vmax == 0:
        vmax = 1
    return np.clip(img / vmax * 255, 0, 255).astype(np.uint8)


def make_grid_frame(images, nrows=2, ncols=2):
    """Arrange up to nrows*ncols grayscale images into a grid."""
    H, W = images[0].shape
    grid = np.zeros((nrows * H, ncols * W), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, ncols)
        grid[r * H : (r + 1) * H, c * W : (c + 1) * W] = img
    return grid


def main():
    parser = argparse.ArgumentParser(description="Make 2x2 grid MP4s from a raw ND2 file.")
    parser.add_argument("nd2_file", help="Path to ND2 file")
    parser.add_argument("--fps", type=float, default=2, help="Frames per second (default: 2)")
    args = parser.parse_args()

    nd2_path = args.nd2_file
    if not os.path.isfile(nd2_path):
        print(f"Error: file not found: {nd2_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {nd2_path} ...")
    max_proj, channel_names = load_nd2(nd2_path)
    T, P, C, Y, X = max_proj.shape
    print(f"  Shape: T={T}, P={P}, C={C}, Y={Y}, X={X}")
    print(f"  Channels: {channel_names}")

    base = os.path.splitext(os.path.basename(nd2_path))[0]
    out_dir = os.path.join(os.path.dirname(nd2_path) or ".", f"raw_{base}")
    os.makedirs(out_dir, exist_ok=True)

    for c in range(C):
        ch_name = channel_names[c]
        mp4_path = os.path.join(out_dir, f"{base}_{ch_name}_raw.mp4")
        print(f"  Writing {mp4_path} ...")

        writer = imageio.get_writer(mp4_path, fps=args.fps)
        for t in range(T):
            images = [auto_contrast(max_proj[t, p, c]) for p in range(P)]
            grid = make_grid_frame(images)
            rgb = np.stack([grid, grid, grid], axis=-1)
            writer.append_data(rgb)
        writer.close()

    print("Done.")


if __name__ == "__main__":
    main()

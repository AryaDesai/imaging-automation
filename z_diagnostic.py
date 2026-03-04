"""z_diagnostic.py — exploratory Z-drift diagnostic for ND2 timelapse data.

Computes and plots multiple Z-position detection metrics across all channels
and timepoints to identify which approach best tracks embryo Z drift.

Metrics computed per Z-slice for each channel:
  mean_int   — mean pixel intensity: where is fluorescent signal concentrated in Z?
  variance   — pixel variance: higher where structures are sharp or densely packed
  lap_var    — Laplacian variance: classic sharpness metric, peaks at in-focus slice
  tenengrad  — mean squared Sobel gradient: gradient-based sharpness, robust to noise

Z position estimates derived from each (T, Z) profile:
  argmax     — Z-slice with highest metric value at each timepoint (discrete, integer)
  centroid   — intensity-weighted mean Z position at each timepoint (continuous, float)
               Most physically meaningful for mean_int; included for all metrics
               so the two estimation strategies can be compared side by side.

Stage Z coordinates are extracted from the ND2 event log. Changes in stage Z
at Z-index 0 across timepoints indicate real objective drift and serve as a
reference to validate whether any image-based metric tracks the same drift.

Output per embryo position:
  z_diagnostic_output/P{p}_heatmaps.png   — (T × Z) heatmaps for all metrics
  z_diagnostic_output/P{p}_z_estimates.png — Z position over time, all metrics

Edit FILE_PATH before running.
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")   # write PNGs without opening a display window
import matplotlib.pyplot as plt
import nd2
import numpy as np
from scipy.ndimage import laplace, sobel
from tqdm import tqdm

from useful_functions import load_nd2

# ── edit this ─────────────────────────────────────────────────────────────────
FILE_PATH = "nd1188.nd2"
# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR = "z_diagnostic_output"

METRICS = ["mean_int", "variance", "lap_var", "tenengrad"]
METRIC_LABELS = {
    "mean_int":  "Mean intensity",
    "variance":  "Pixel variance",
    "lap_var":   "Laplacian variance",
    "tenengrad": "Tenengrad",
}


# ── per-slice metric helpers ───────────────────────────────────────────────────

def _lap_var(img):
    """Variance of the Laplacian of a 2-D image.

    The Laplacian accentuates rapid intensity changes (edges, fine detail).
    Its variance is low in blurry/out-of-focus slices and high where the
    image contains sharp structure.
    """
    return float(laplace(img).var())


def _tenengrad(img):
    """Mean squared Sobel gradient magnitude.

    Measures the energy of spatial gradients. Like Laplacian variance, it
    peaks at the sharpest Z-slice, but is less sensitive to noise because
    the squared gradient naturally downweights small fluctuations.
    """
    gx = sobel(img, axis=0)
    gy = sobel(img, axis=1)
    return float((gx ** 2 + gy ** 2).mean())


# ── profile computation ───────────────────────────────────────────────────────

def compute_profiles(data, p):
    """Compute all Z metrics for embryo position p.

    For every (metric, channel) pair, produces a (T, Z) matrix where
    entry [t, z] is the scalar metric value of the (Y, X) image
    data[t, p, z, c].

    Parameters
    ----------
    data : ndarray, shape (T, P, Z, C, Y, X), float32
    p    : int

    Returns
    -------
    profiles : dict
        profiles[metric_key][channel_index] = ndarray, shape (T, Z), float64
    """
    T, _, Z, C, _, _ = data.shape
    profiles = {m: {c: np.zeros((T, Z), dtype=np.float64) for c in range(C)}
                for m in METRICS}

    for t in tqdm(range(T), desc=f"  P{p}", leave=False):
        for z in range(Z):
            for c in range(C):
                img = data[t, p, z, c]                          # shape (Y, X)
                profiles["mean_int"][c][t, z]  = float(img.mean())
                profiles["variance"][c][t, z]  = float(img.var())
                profiles["lap_var"][c][t, z]   = _lap_var(img)
                profiles["tenengrad"][c][t, z] = _tenengrad(img)

    return profiles


# ── Z position estimates ──────────────────────────────────────────────────────

def z_argmax(profile_tz):
    """Z-slice index with the highest metric value at each timepoint.

    Returned as float for plotting consistency with z_centroid.
    This is a discrete estimate — it can only take integer values and
    jumps abruptly when two Z-slices have similar metric values.
    """
    return profile_tz.argmax(axis=1).astype(float)


def z_centroid(profile_tz):
    """Intensity-weighted mean Z position at each timepoint.

    The profile is first shifted so its minimum per timepoint is zero,
    removing any uniform background level before computing the centroid.
    Without this shift a flat non-zero baseline would pull the centroid
    toward Z/2 regardless of where the signal actually peaks.

    Returns float values that can fall between integer slice indices.
    Most physically meaningful for the mean_int metric; included for all
    metrics so the estimation strategy (argmax vs centroid) can be compared.
    """
    Z = profile_tz.shape[1]
    z_idx = np.arange(Z, dtype=float)

    # Subtract per-timepoint minimum so that only the peak structure
    # contributes to the centroid, not the background floor.
    shifted = profile_tz - profile_tz.min(axis=1, keepdims=True)
    total   = shifted.sum(axis=1)
    total   = np.where(total == 0, 1.0, total)   # guard against blank frames

    return (shifted * z_idx[None, :]).sum(axis=1) / total


# ── stage Z extraction ────────────────────────────────────────────────────────

def extract_stage_z(file_path):
    """Extract stage Z Coord [µm] at Z-index 0 per (T, P) from the ND2 event log.

    Z-index 0 is the start of each Z-stack acquisition and is therefore the
    consistent reference point across all timepoints. A change in this value
    between timepoints for the same embryo position means the objective moved
    in Z between acquisitions — i.e. real stage drift.

    Returns
    -------
    dict mapping (t, p) → Z Coord [µm], or None if no events are present.
    """
    f = nd2.ND2File(file_path)
    events = f.events()
    f.close()

    if events is None or len(events) == 0:
        return None

    stage_z = {}
    for ev in events:
        t     = ev.get("T Index")
        p     = ev.get("P Index")
        z_idx = ev.get("Z Index")
        coord = ev.get("Z Coord [µm]")
        if None not in (t, p, z_idx, coord) and int(z_idx) == 0:
            stage_z[(int(t), int(p))] = float(coord)

    return stage_z or None


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_heatmaps(profiles, channel_names, p, out_dir):
    """Save a grid of (T × Z) heatmaps for every metric × channel pair.

    Layout: rows = metrics (mean_int, variance, lap_var, tenengrad),
            columns = channels.

    Each heatmap is normalised to [0, 1] over its full (T, Z) range so the
    colour scale shows the relative distribution of the metric rather than
    absolute values. The argmax (peak Z) is overlaid as a white line so
    any drift in the detected Z position is immediately visible.
    """
    C = len(channel_names)
    M = len(METRICS)
    T = list(list(profiles[METRICS[0]].values())[0].shape)[0]

    fig, axes = plt.subplots(M, C, figsize=(4 * C, 3 * M), squeeze=False)
    fig.suptitle(f"Z profiles — Embryo P{p}", fontsize=13)

    t_axis = np.arange(T)

    for row, metric in enumerate(METRICS):
        for col, c in enumerate(range(C)):
            ax  = axes[row, col]
            mat = profiles[metric][c]           # shape (T, Z)

            vmin, vmax = mat.min(), mat.max()
            norm = ((mat - vmin) / (vmax - vmin)) if vmax > vmin else np.zeros_like(mat)

            ax.imshow(
                norm.T,                         # (Z, T): Z on Y-axis, T on X-axis
                aspect="auto",
                origin="lower",
                cmap="viridis",
                vmin=0, vmax=1,
            )
            ax.plot(t_axis, z_argmax(mat), color="white", lw=1.5, label="argmax")

            ax.set_title(f"{METRIC_LABELS[metric]} — {channel_names[c]}", fontsize=8)
            ax.set_xlabel("T")
            ax.set_ylabel("Z")

    plt.tight_layout()
    path = os.path.join(out_dir, f"P{p}_heatmaps.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {path}")


def plot_z_estimates(profiles, channel_names, p, stage_z, vox_z, out_dir):
    """Save a summary figure comparing all Z position estimates over time.

    Two panels:
      Top    — argmax Z for every metric × channel combination
      Bottom — centroid Z for every metric × channel combination

    Stage Z drift from the ND2 event log is expressed in slice units (by
    dividing µm drift by vox_z) and plotted as a dashed black line on both
    panels. Because the stage Z is the actual objective position, it is the
    most direct measure of drift and serves as a reference against which the
    image-based metrics can be judged.

    Lines are coloured by channel and styled (solid/dashed/etc.) by metric,
    so up to C × M lines are visible per panel.
    """
    C = len(channel_names)
    T = list(list(profiles[METRICS[0]].values())[0].shape)[0]
    t_axis = np.arange(T)

    prop_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    lstyles     = ["-", "--", "-.", ":"]

    fig, (ax_am, ax_ct) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax_am.set_title(f"Embryo P{p} — argmax Z over time")
    ax_ct.set_title(f"Embryo P{p} — centroid Z over time")

    for mi, metric in enumerate(METRICS):
        ls = lstyles[mi % len(lstyles)]
        for c in range(C):
            color = prop_colors[c % len(prop_colors)]
            mat   = profiles[metric][c]
            lbl   = f"{channel_names[c]} / {METRIC_LABELS[metric]}"
            ax_am.plot(t_axis, z_argmax(mat),   color=color, ls=ls, label=lbl, alpha=0.8)
            ax_ct.plot(t_axis, z_centroid(mat), color=color, ls=ls, label=lbl, alpha=0.8)

    # Stage Z reference: express drift in slice units relative to t=0
    # so it overlays directly on the Z-slice axis of both panels.
    if stage_z is not None:
        sz_um  = np.array([stage_z.get((int(t), int(p)), np.nan) for t in t_axis])
        if not np.all(np.isnan(sz_um)):
            ref       = sz_um[~np.isnan(sz_um)][0]
            sz_slices = (sz_um - ref) / vox_z
            for ax in (ax_am, ax_ct):
                ax.plot(t_axis, sz_slices, "k--", lw=2,
                        label="Stage Z drift (slices)", alpha=0.7)

    for ax in (ax_am, ax_ct):
        ax.set_ylabel("Z slice")
        ax.legend(fontsize=7, ncol=2, loc="best")
        ax.grid(True, alpha=0.3)
    ax_ct.set_xlabel("Timepoint")

    plt.tight_layout()
    path = os.path.join(out_dir, f"P{p}_z_estimates.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {path}")


# ── CSV export ────────────────────────────────────────────────────────────────

def save_csv(all_profiles, all_stage_z, channel_names, vox_z, out_dir):
    """Write two CSV files summarising all computed Z estimates.

    estimates.csv
        One row per (embryo, channel, metric, timepoint). Columns:
          embryo, channel, metric, t, argmax, centroid
        This is the primary output for comparing metrics and channels.

    stage_z.csv
        One row per (embryo, timepoint). Columns:
          embryo, t, stage_z_um, stage_z_drift_slices
        stage_z_drift_slices is the change from t=0 expressed in slice
        units (µm change / vox_z), so it can be compared directly with
        the argmax and centroid columns in estimates.csv.
    """
    # ── estimates.csv ──────────────────────────────────────────────────────────
    est_path = os.path.join(out_dir, "estimates.csv")
    with open(est_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["embryo", "channel", "metric", "t", "argmax", "centroid"])
        for p, profiles in all_profiles.items():
            T = list(list(profiles[METRICS[0]].values())[0].shape)[0]
            for metric in METRICS:
                for c, ch_name in enumerate(channel_names):
                    mat = profiles[metric][c]
                    am  = z_argmax(mat)
                    ct  = z_centroid(mat)
                    for t in range(T):
                        writer.writerow([p, ch_name, metric, t,
                                         f"{am[t]:.4f}", f"{ct[t]:.4f}"])
    print(f"  Saved {est_path}")

    # ── stage_z.csv ────────────────────────────────────────────────────────────
    if all_stage_z is None:
        return

    sz_path = os.path.join(out_dir, "stage_z.csv")
    with open(sz_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["embryo", "t", "stage_z_um", "stage_z_drift_slices"])
        for p in all_profiles:
            T = list(list(all_profiles[p][METRICS[0]].values())[0].shape)[0]
            sz_vals = [all_stage_z.get((t, p), float("nan")) for t in range(T)]
            ref = next((v for v in sz_vals if not np.isnan(v)), 0.0)
            for t, sz in enumerate(sz_vals):
                drift = (sz - ref) / vox_z if not np.isnan(sz) else float("nan")
                writer.writerow([p, t, f"{sz:.4f}", f"{drift:.4f}"])
    print(f"  Saved {sz_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading {FILE_PATH} ...")
    data, channel_names, vox, period_s = load_nd2(FILE_PATH)
    T, P, Z, C, Y, X = data.shape
    print(f"  Shape: T={T}, P={P}, Z={Z}, C={C}, Y={Y}, X={X}")
    print(f"  Channels: {channel_names}")
    print(f"  Voxel z: {vox.z:.3f} µm/slice")

    print("\nExtracting stage Z from event log ...")
    stage_z = extract_stage_z(FILE_PATH)
    print(f"  Found {len(stage_z) if stage_z else 0} (T, P) stage-Z entries")

    all_profiles = {}
    for p in range(P):
        print(f"\nEmbryo P{p}:")
        profiles = compute_profiles(data, p)
        all_profiles[p] = profiles
        plot_heatmaps(profiles, channel_names, p, OUT_DIR)
        plot_z_estimates(profiles, channel_names, p, stage_z, vox.z, OUT_DIR)

    print("\nSaving CSV results ...")
    save_csv(all_profiles, stage_z, channel_names, vox.z, OUT_DIR)

    print(f"\nDone. Output in {OUT_DIR}/")


if __name__ == "__main__":
    main()

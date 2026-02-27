import nd2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.ndimage import gaussian_filter, label

# ── change this to your actual file path ──────────────────────────────────────
FILE_PATH = "nd1188.nd2"
# ─────────────────────────────────────────────────────────────────────────────

f = nd2.ND2File(FILE_PATH)
data = f.asarray()  # (T, P, Z, C, Y, X)
channel_names = [ch.channel.name for ch in f.metadata.channels]
f.close()

T, P, Z, C, Y, X = data.shape
VENUS_CH = channel_names.index("Venus")

# precompute Venus max projections: shape (T, P, Y, X)
venus_max = data[:, :, :, VENUS_CH, :, :].max(axis=2).astype(np.float32)


# ── two-stage mask computation with caching ──────────────────────────────────

_smooth_cache = {}
_SMOOTH_CACHE_CAP = 40


def smooth(img_key, img, sigma):
    """Return gaussian-filtered image, cached by (img_key, sigma)."""
    key = (img_key, sigma)
    if key not in _smooth_cache:
        if len(_smooth_cache) >= _SMOOTH_CACHE_CAP:
            _smooth_cache.pop(next(iter(_smooth_cache)))
        _smooth_cache[key] = gaussian_filter(img, sigma=sigma)
    return _smooth_cache[key]


def threshold_and_label(smoothed, percentile):
    """Threshold + largest connected component. Fast, no caching needed."""
    thresh = np.percentile(smoothed, percentile)
    binary = smoothed > thresh

    labeled, n_components = label(binary)
    if n_components == 0:
        return np.zeros_like(binary, dtype=bool), None

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest = sizes.argmax()
    mask = labeled == largest

    yx = np.argwhere(mask)
    centroid = yx.mean(axis=0)  # (y, x)
    return mask, centroid


# ─────────────────────────────────────────────────────────────────────────────
# Layout: 2x2 grid of embryos, sliders at bottom for T, sigma, percentile
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(11, 10))
plt.subplots_adjust(bottom=0.22, hspace=0.35, wspace=0.3)
fig.suptitle(
    "Mask Tuner — Venus Max Projection\n"
    "Adjust sigma and percentile to optimise mask (red = mask overlay, + = centroid)",
    fontsize=11, fontweight="bold"
)

# initial parameters
INIT_T          = 0
INIT_SIGMA      = 2.0
INIT_PERCENTILE = 90.0

# store display objects per embryo
image_displays  = []   # imshow handles for the base image
mask_displays   = []   # imshow handles for the mask overlay
centroid_plots  = []   # plot handles for the centroid marker
axes_list       = []

# pre-allocate overlay buffers (reused each update to avoid repeated allocation)
overlay_buffers = []

for p in range(P):
    ax  = axes[p // 2, p % 2]
    img = venus_max[INIT_T, p]

    # base image
    im_base = ax.imshow(img, cmap="gray", origin="upper",
                        vmin=0, vmax=np.percentile(img, 99.5))

    # mask overlay (RGBA so we can make background transparent)
    smoothed = smooth((INIT_T, p), img, INIT_SIGMA)
    mask, centroid = threshold_and_label(smoothed, INIT_PERCENTILE)
    overlay = np.zeros((Y, X, 4), dtype=float)
    overlay[mask] = [1.0, 0.2, 0.2, 0.4]

    im_mask = ax.imshow(overlay, origin="upper")

    # centroid marker
    if centroid is not None:
        pt, = ax.plot(centroid[1], centroid[0], "r+",
                      markersize=14, markeredgewidth=2)
    else:
        pt, = ax.plot([], [], "r+", markersize=14, markeredgewidth=2)

    ax.set_title(f"Embryo {p}  |  T={INIT_T}")
    ax.axis("off")

    image_displays.append(im_base)
    mask_displays.append(im_mask)
    centroid_plots.append(pt)
    axes_list.append(ax)
    overlay_buffers.append(overlay)

# ── sliders (attached to fig, not plt) ────────────────────────────────────────
ax_t    = fig.add_axes([0.15, 0.14, 0.70, 0.025])
ax_sig  = fig.add_axes([0.15, 0.09, 0.70, 0.025])
ax_pct  = fig.add_axes([0.15, 0.04, 0.70, 0.025])

sl_t   = Slider(ax_t,   "T",           0,    T-1,  valinit=INIT_T,          valstep=1)
sl_sig = Slider(ax_sig, "Sigma",       1,  100.0, valinit=INIT_SIGMA,      valstep=1)
sl_pct = Slider(ax_pct, "Percentile",  10.0, 99.5, valinit=INIT_PERCENTILE, valstep=0.5)


def update(val):
    t   = int(sl_t.val)
    sig = sl_sig.val
    pct = sl_pct.val

    for p in range(P):
        img = venus_max[t, p]

        # update base image brightness scale to current timepoint
        vmax = np.percentile(img, 99.5)
        image_displays[p].set_data(img)
        image_displays[p].set_clim(vmin=0, vmax=vmax)

        # two-stage: smooth (cached) then threshold (fast)
        smoothed = smooth((t, p), img, sig)
        mask, centroid = threshold_and_label(smoothed, pct)

        # reuse pre-allocated overlay buffer
        overlay = overlay_buffers[p]
        overlay[:] = 0
        overlay[mask] = [1.0, 0.2, 0.2, 0.4]
        mask_displays[p].set_data(overlay)

        # update centroid marker
        if centroid is not None:
            centroid_plots[p].set_data([centroid[1]], [centroid[0]])
        else:
            centroid_plots[p].set_data([], [])

        axes_list[p].set_title(f"Embryo {p}  |  T={t}")

    fig.canvas.draw_idle()


sl_t.on_changed(update)
sl_sig.on_changed(update)
sl_pct.on_changed(update)

plt.show()

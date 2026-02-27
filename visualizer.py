import nd2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ── change this to your actual file path ──────────────────────────────────────
FILE_PATH = "nd1188.nd2"
# ─────────────────────────────────────────────────────────────────────────────

f = nd2.ND2File(FILE_PATH)
# shape: (T, P, Z, C, Y, X)
data = f.asarray()
channel_names = [ch.channel.name for ch in f.metadata.channels]
f.close()

T, P, Z, C, Y, X = data.shape

def normalize(img):
    """Normalize to 0-1 for display, handling flat images."""
    mn, mx = img.min(), img.max()
    if mx == mn:
        return np.zeros_like(img, dtype=float)
    return (img - mn) / (mx - mn)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Full data viewer — sliders for T, P, Z, C
# ─────────────────────────────────────────────────────────────────────────────

fig1, ax1 = plt.subplots(figsize=(7, 8))
plt.subplots_adjust(left=0.1, bottom=0.3)
fig1.suptitle("Figure 1 — Raw Data Viewer", fontsize=12, fontweight="bold")

init_img = normalize(data[0, 0, Z//2, 0])
im1 = ax1.imshow(init_img, cmap="gray", origin="upper")
title1 = ax1.set_title(
    f"T=0  |  P=0  |  Z={Z//2}  |  Channel: {channel_names[0]}"
)
ax1.axis("off")
cb1 = fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cb1.set_label("Normalised intensity")

ax_t1   = fig1.add_axes([0.15, 0.20, 0.70, 0.03])
ax_p1   = fig1.add_axes([0.15, 0.15, 0.70, 0.03])
ax_z1   = fig1.add_axes([0.15, 0.10, 0.70, 0.03])
ax_c1   = fig1.add_axes([0.15, 0.05, 0.70, 0.03])

sl_t1 = Slider(ax_t1, "T",       0, T-1, valinit=0,      valstep=1)
sl_p1 = Slider(ax_p1, "Embryo",  0, P-1, valinit=0,      valstep=1)
sl_z1 = Slider(ax_z1, "Z",       0, Z-1, valinit=Z//2,   valstep=1)
sl_c1 = Slider(ax_c1, "Channel", 0, C-1, valinit=0,      valstep=1)

def update_viewer(val):
    t = int(sl_t1.val)
    p = int(sl_p1.val)
    z = int(sl_z1.val)
    c = int(sl_c1.val)
    img = normalize(data[t, p, z, c])
    im1.set_data(img)
    ax1.set_title(
        f"T={t}  |  P={p}  |  Z={z}  |  Channel: {channel_names[c]}"
    )
    fig1.canvas.draw_idle()

sl_t1.on_changed(update_viewer)
sl_p1.on_changed(update_viewer)
sl_z1.on_changed(update_viewer)
sl_c1.on_changed(update_viewer)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: Venus max-projection across Z, 2x2 grid for all embryos, T slider
# ─────────────────────────────────────────────────────────────────────────────

VENUS_CH = channel_names.index("Venus")

# precompute all Venus max projections: shape (T, P, Y, X)
venus_max = data[:, :, :, VENUS_CH, :, :].max(axis=2)

fig2, axes2 = plt.subplots(2, 2, figsize=(9, 9))
plt.subplots_adjust(bottom=0.12, hspace=0.35, wspace=0.25)
fig2.suptitle("Figure 2 — Venus Max Projection (Z) — All Embryos", 
              fontsize=12, fontweight="bold")

ims2 = []
for p in range(P):
    ax = axes2[p // 2, p % 2]
    img = normalize(venus_max[0, p])
    im = ax.imshow(img, cmap="viridis", origin="upper")
    ax.set_title(f"Embryo {p}  |  T=0")
    ax.axis("off")
    fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Norm. intensity")
    ims2.append((im, ax))

ax_t2 = fig2.add_axes([0.2, 0.04, 0.6, 0.03])
sl_t2 = Slider(ax_t2, "T", 0, T-1, valinit=0, valstep=1)

def update_maxproj(val):
    t = int(sl_t2.val)
    for p, (im, ax) in enumerate(ims2):
        img = normalize(venus_max[t, p])
        im.set_data(img)
        ax.set_title(f"Embryo {p}  |  T={t}")
    fig2.canvas.draw_idle()

sl_t2.on_changed(update_maxproj)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: Intensity histograms — Venus max projection, all embryos, T slider
# ─────────────────────────────────────────────────────────────────────────────

NBINS   = 100
MAX_VAL = 4095   # 12-bit data stored in uint16
bin_edges = np.linspace(0, MAX_VAL, NBINS + 1)

fig3, axes3 = plt.subplots(2, 2, figsize=(10, 8))
plt.subplots_adjust(bottom=0.12, hspace=0.45, wspace=0.35)
fig3.suptitle(
    "Figure 3 — Venus Intensity Histograms (Max Projection) — All Embryos",
    fontsize=12, fontweight="bold"
)

bar_containers = []
vlines         = []

for p in range(P):
    ax = axes3[p // 2, p % 2]
    pixels = venus_max[0, p].ravel()
    counts, _ = np.histogram(pixels, bins=bin_edges)

    bars = ax.bar(
        bin_edges[:-1], counts,
        width=np.diff(bin_edges),
        color="mediumpurple", alpha=0.75, align="edge"
    )
    median_val = np.median(pixels)
    vl = ax.axvline(median_val, color="red", linewidth=1.5,
                    linestyle="--", label=f"Median: {median_val:.0f}")

    ax.set_title(f"Embryo {p}  |  T=0")
    ax.set_xlabel("Raw intensity (12-bit)")
    ax.set_ylabel("Pixel count")
    ax.legend(fontsize=8)

    bar_containers.append((bars, ax, bin_edges))
    vlines.append(vl)

ax_t3 = fig3.add_axes([0.2, 0.04, 0.6, 0.03])
sl_t3 = Slider(ax_t3, "T", 0, T-1, valinit=0, valstep=1)

def update_hist(val):
    t = int(sl_t3.val)
    for p, ((bars, ax, edges), vl) in enumerate(zip(bar_containers, vlines)):
        pixels = venus_max[t, p].ravel()
        counts, _ = np.histogram(pixels, bins=edges)
        for bar, h in zip(bars.patches, counts):
            bar.set_height(h)
        median_val = np.median(pixels)
        vl.set_xdata([median_val, median_val])
        vl.set_label(f"Median: {median_val:.0f}")
        ax.set_title(f"Embryo {p}  |  T={t}")
        ax.relim()
        ax.autoscale_view()
        ax.legend(fontsize=8)
    fig3.canvas.draw_idle()

sl_t3.on_changed(update_hist)

plt.show()
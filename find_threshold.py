import nd2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.ndimage import gaussian_filter, label

# ── change this to your actual file path ──────────────────────────────────────
FILE_PATH = "nd1188.nd2"
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_resource
def load_venus_max():
    f = nd2.ND2File(FILE_PATH)
    data = f.asarray()  # (T, P, Z, C, Y, X)
    channel_names = [ch.channel.name for ch in f.metadata.channels]
    f.close()
    T, P, Z, C, Y, X = data.shape
    VENUS_CH = channel_names.index("Venus")
    venus_max = data[:, :, :, VENUS_CH, :, :].max(axis=2).astype(np.float32)
    return venus_max


@st.cache_data(max_entries=40)
def get_smoothed(t, p, sigma):
    """Gaussian-filter the (t, p) image. Cached by (t, p, sigma)."""
    venus_max = load_venus_max()
    return gaussian_filter(venus_max[t, p], sigma=sigma)


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

st.set_page_config(page_title="Mask Tuner", layout="wide")
st.title("Mask Tuner — Venus Max Projection")
st.caption("Red overlay = mask  |  + = centroid")

venus_max = load_venus_max()
T, P = venus_max.shape[0], venus_max.shape[1]

with st.sidebar:
    st.header("Parameters")
    t          = st.slider("T",          0,    T - 1, 0,    step=1)
    sigma      = st.slider("Sigma",      1,    100,   2,    step=1)
    percentile = st.slider("Percentile", 10.0, 99.5,  90.0, step=0.5)

fig, axes = plt.subplots(2, 2, figsize=(10, 9))
fig.suptitle(
    f"T = {t}  |  sigma = {sigma}  |  percentile = {percentile}",
    fontsize=11, fontweight="bold"
)
plt.subplots_adjust(hspace=0.3, wspace=0.3)

for p in range(P):
    ax  = axes[p // 2, p % 2]
    img = venus_max[t, p]

    smoothed           = get_smoothed(t, p, sigma)
    mask, centroid     = threshold_and_label(smoothed, percentile)

    vmax = np.percentile(img, 99.5)
    ax.imshow(img, cmap="gray", origin="upper", vmin=0, vmax=vmax)

    overlay = np.zeros((*img.shape, 4), dtype=float)
    overlay[mask] = [1.0, 0.2, 0.2, 0.4]
    ax.imshow(overlay, origin="upper")

    if centroid is not None:
        ax.plot(centroid[1], centroid[0], "r+", markersize=14, markeredgewidth=2)

    ax.set_title(f"Embryo {p}")
    ax.axis("off")

st.pyplot(fig)
plt.close(fig)

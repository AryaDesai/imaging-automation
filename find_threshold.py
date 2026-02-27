import datetime
import nd2
import numpy as np
import os
import streamlit as st
import time
import yaml
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, label

# ─────────────────────────────────────────────────────────────────────────────


@st.cache_resource
def load_nd2(file_path):
    """Load ND2 file and return Z-max projection for all channels + channel names."""
    f = nd2.ND2File(file_path)
    data = f.asarray()  # (T, P, Z, C, Y, X)
    channel_names = [ch.channel.name for ch in f.metadata.channels]
    f.close()
    # Max-project over Z for every channel -> (T, P, C, Y, X)
    max_proj = data.max(axis=2).astype(np.float32)
    return max_proj, channel_names


@st.cache_data(max_entries=40)
def get_smoothed(t, p, ch_idx, sigma, _file_path):
    """Gaussian-filter the (t, p) image for a given channel. Cached by (t, p, ch_idx, sigma, file_path)."""
    max_proj, _ = load_nd2(_file_path)
    return gaussian_filter(max_proj[t, p, ch_idx], sigma=sigma)


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


def render_embryo(img, mask, centroid, vmax):
    """Composite grayscale image + red mask overlay + centroid crosshair."""
    gray = np.clip(img / vmax * 255, 0, 255).astype(np.uint8)
    base = Image.fromarray(gray, mode="L").convert("RGBA")
    overlay_arr = np.zeros((*gray.shape, 4), dtype=np.uint8)
    overlay_arr[mask] = [255, 50, 50, 100]
    overlay = Image.fromarray(overlay_arr, mode="RGBA")
    composite = Image.alpha_composite(base, overlay)
    if centroid is not None:
        draw = ImageDraw.Draw(composite)
        cy, cx = int(round(centroid[0])), int(round(centroid[1]))
        s = 25
        draw.line([(cx - s, cy), (cx + s, cy)], fill=(0, 255, 255, 255), width=7)
        draw.line([(cx, cy - s), (cx, cy + s)], fill=(0, 255, 255, 255), width=7)
    return composite


# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Find threshold for masking PSM", layout="wide")
st.title("Find threshold for masking PSM")
st.caption("Red overlay = mask  |  + = centroid")

file_path = "nd1188.nd2"
max_proj, channel_names = load_nd2(file_path)
T, P, C, Y, X = max_proj.shape

with st.sidebar:
    channel = st.selectbox("Channel", channel_names)
    ch_idx = channel_names.index(channel)
    t = st.slider("T", 0, T - 1, 0, step=1, key="T")
    sigma = st.slider("Sigma", 1, 100, 2, step=1)
    percentile = st.slider("Percentile", 10.0, 99.5, 90.0, step=0.5)

    # ── Play/pause animation ────────────────────────────────────────────────
    st.divider()
    if "playing" not in st.session_state:
        st.session_state.playing = False

    play_col, speed_col = st.columns(2)
    with play_col:
        if st.button("Start animation" if not st.session_state.playing else "Pause animation"):
            st.session_state.playing = not st.session_state.playing
            st.rerun()
    with speed_col:
        play_delay = st.number_input("Delay (s)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)

    # ── Precompute all ──────────────────────────────────────────────────────
    st.divider()
    mem_bytes = T * P * Y * X * 4  # float32 per smoothed image
    mem_gb = mem_bytes / (1024**3)
    st.info(f"Precomputing requires ~{mem_gb:.1f} GB free RAM")

    if st.button("Precompute All"):
        bar = st.progress(0, text="Precomputing smoothed images...")
        for i, ti in enumerate(range(T)):
            for pi in range(P):
                get_smoothed(ti, pi, ch_idx, sigma, file_path)
            bar.progress((i + 1) / T, text=f"Precomputing... T={ti+1}/{T}")
        bar.progress(1.0, text="Done! All frames cached.")

    # ── Save to YAML ─────────────────────────────────────────────────────────
    st.divider()
    if st.button("Save thresholds to YAML"):
        embryos_data = []
        for p in range(P):
            img = max_proj[t, p, ch_idx]
            smoothed = get_smoothed(t, p, ch_idx, sigma, file_path)
            mask, centroid = threshold_and_label(smoothed, percentile)

            mask_area = int(mask.sum())
            pct = mask_area / mask.size * 100
            mean_int = float(img[mask].mean()) if mask_area > 0 else 0.0
            embryos_data.append({
                "id": p,
                "mask_area_px": mask_area,
                "mask_area_pct": round(pct, 2),
                "mean_intensity": round(mean_int, 2),
                "centroid_x": round(float(centroid[1]), 2) if centroid is not None else None,
                "centroid_y": round(float(centroid[0]), 2) if centroid is not None else None,
            })

        now = datetime.datetime.now()
        output = {
            "file": file_path,
            "saved_at": now.isoformat(timespec="seconds"),
            "channel": channel,
            "channel_index": ch_idx,
            "sigma": sigma,
            "percentile": percentile,
            "time_point": t,
            "image_shape": {"T": T, "P": P, "C": C, "Y": Y, "X": X},
            "embryos": embryos_data,
        }

        base = os.path.splitext(os.path.basename(file_path))[0]
        fname = f"{base}_threshold_{now.strftime('%Y%m%d_%H%M%S')}.yaml"
        with open(fname, "w") as f:
            yaml.dump(output, f, default_flow_style=False, sort_keys=False)
        st.success(f"Saved → {fname}")


# ── Embryo rendering fragment ──────────────────────────────────────────────

@st.fragment
def render_grid():
    st.subheader(f"T = {t}  |  {channel}  |  sigma = {sigma}  |  percentile = {percentile}")

    for row in range(0, P, 2):
        cols = st.columns(2)
        for col_idx in range(2):
            p = row + col_idx
            if p >= P:
                break
            with cols[col_idx]:
                img = max_proj[t, p, ch_idx]
                smoothed = get_smoothed(t, p, ch_idx, sigma, file_path)
                mask, centroid = threshold_and_label(smoothed, percentile)

                vmax = np.percentile(img, 99.5)
                composite = render_embryo(img, mask, centroid, vmax)
                st.image(composite, caption=f"Embryo {p}")

                # ── Mask statistics ──────────────────────────────────
                mask_area = int(mask.sum())
                total_px = mask.size
                pct = mask_area / total_px * 100
                mean_int = float(img[mask].mean()) if mask_area > 0 else 0.0
                cx_str = f"{centroid[1]:.1f}" if centroid is not None else "—"
                cy_str = f"{centroid[0]:.1f}" if centroid is not None else "—"
                st.caption(
                    f"Area: {mask_area} px ({pct:.1f}%)  |  "
                    f"Centroid: ({cx_str}, {cy_str})  |  "
                    f"Mean intensity: {mean_int:.1f}"
                )


render_grid()

# ── Auto-play logic ─────────────────────────────────────────────────────────

if st.session_state.playing:
    time.sleep(play_delay)
    next_t = (t + 1) % T
    # Update the slider value via session state key
    st.session_state["T"] = next_t
    st.rerun()

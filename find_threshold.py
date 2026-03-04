import datetime
import numpy as np
import os
import streamlit as st
import time
import subprocess
import yaml
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from useful_functions import find_largest_mask_xy, load_nd2, max_project_z


@st.cache_resource
def load_nd2_projected(file_path):
    """Load an ND2 file and return a Z-max-projected array (T,P,C,Y,X) + channel names.

    This is a Streamlit-specific cached wrapper around embryo_tools.load_nd2.
    The @st.cache_resource decorator means the file is loaded only once per
    Streamlit session; subsequent widget interactions (slider moves, parameter
    changes) reuse the cached array rather than re-reading from disk.

    We project to (T, P, C, Y, X) here because find_threshold.py never needs
    the full Z stack — it works on 2-D images for display and thresholding.
    Discarding Z early reduces the memory held by the Streamlit cache.
    """
    data, channel_names, _, _ = load_nd2(file_path)
    return max_project_z(data), channel_names


@st.cache_data()
def get_smoothed(t, p, ch_idx, sigma, _file_path):
    """Gaussian-filter a single frame and cache the result.

    The @st.cache_data decorator caches the output keyed on (t, p, ch_idx,
    sigma, file_path). This means the Gaussian filter is only computed once
    per unique combination of parameters; re-running the Streamlit script
    after a slider move reuses cached results for previously seen combinations.

    The _file_path argument is prefixed with _ to tell Streamlit's cache not
    to hash it (hashing a long path string on every rerun is wasteful; the
    path is effectively constant within a session).
    """
    max_proj, _ = load_nd2_projected(_file_path)
    return gaussian_filter(max_proj[t, p, ch_idx], sigma=sigma)


def render_embryo(img, mask, centroid):
    """Grayscale image + red mask overlay + centroid crosshair -> RGBA PIL image."""
    vmax = np.percentile(img, 99.5)
    gray = np.clip(img / vmax * 255, 0, 255).astype(np.uint8)
    base = Image.fromarray(gray, mode="L").convert("RGBA")

    overlay_arr = np.zeros((*gray.shape, 4), dtype=np.uint8)
    overlay_arr[mask] = [255, 50, 50, 100]
    composite = Image.alpha_composite(base, Image.fromarray(overlay_arr, mode="RGBA"))

    draw = ImageDraw.Draw(composite)
    cy, cx = int(round(centroid[0])), int(round(centroid[1]))
    s = 25
    draw.line([(cx - s, cy), (cx + s, cy)], fill=(0, 255, 255, 255), width=7)
    draw.line([(cx, cy - s), (cx, cy + s)], fill=(0, 255, 255, 255), width=7)
    return composite


def pick_nd2_file():
    """Open a native macOS file dialog filtered to .nd2 files."""
    script = '''
    set f to POSIX path of (choose file of type {"nd2"} with prompt "Select ND2 file")
    return f
    '''
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


# ── Page setup ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Find threshold for masking PSM", layout="wide")
st.title("Find threshold for masking PSM")
st.caption("Red overlay = mask  |  + = centroid")

if "playing" not in st.session_state:
    st.session_state.playing = False

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    if st.button("Browse for ND2 file"):
        path = pick_nd2_file()
        if path:
            st.session_state["nd2_path"] = path

    file_path = st.session_state.get("nd2_path", "")
    if file_path:
        st.text(os.path.basename(file_path))

if not file_path:
    st.info("Click **Browse for ND2 file** in the sidebar to get started.")
    st.stop()

max_proj, channel_names = load_nd2_projected(file_path)
T, P, C, Y, X = max_proj.shape

with st.sidebar:
    channel = st.selectbox("Channel", channel_names)
    ch_idx = channel_names.index(channel)
    t = st.slider("T", 0, T - 1, 0, step=1, key="T")
    sigma = st.slider("Sigma", 1, 100, 2, step=1)
    percentile = st.slider("Percentile", 10.0, 99.5, 90.0, step=0.5)

    # Animation
    st.divider()
    play_col, speed_col = st.columns(2)
    with play_col:
        if st.button("Pause" if st.session_state.playing else "Play"):
            st.session_state.playing = not st.session_state.playing
            st.rerun()
    with speed_col:
        play_delay = st.number_input("Delay (s)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)

    # Precompute
    st.divider()
    mem_gb = T * P * Y * X * 4 / (1024**3)
    st.info(f"Precomputing requires ~{mem_gb:.1f} GB free RAM")

    if st.button("Precompute All"):
        bar = st.progress(0, text="Precomputing smoothed images...")
        for ti in range(T):
            for pi in range(P):
                get_smoothed(ti, pi, ch_idx, sigma, file_path)
            bar.progress((ti + 1) / T, text=f"Precomputing... T={ti+1}/{T}")
        bar.progress(1.0, text="Done!")

    # Save to YAML
    st.divider()
    if st.button("Save thresholds to YAML"):
        embryos_data = []
        for pi in range(P):
            img = max_proj[0, pi, ch_idx]
            smoothed = get_smoothed(0, pi, ch_idx, sigma, file_path)
            mask, centroid = find_largest_mask_xy(smoothed, percentile)
            area = int(mask.sum())
            embryos_data.append({
                "id": pi,
                "mask_area_px": area,
                "mask_area_pct": round(area / mask.size * 100, 2),
                "mean_intensity": round(float(img[mask].mean()), 2),
                "centroid": [round(float(centroid[1]), 2), round(float(centroid[0]), 2)],
            })

        output = {
            "parameters": {
                "channel": channel,
                "channel_index": ch_idx,
                "sigma": sigma,
                "percentile": percentile,
            },
            "source": {
                "file": file_path,
                "image_shape": {"T": T, "P": P, "Y": Y, "X": X},
            },
            "diagnostics": {
                "time_point": 0,
                "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "embryos": embryos_data,
            },
        }

        base = os.path.splitext(os.path.basename(file_path))[0]
        fname = f"{base}_{channel}_threshold.yaml"
        with open(fname, "w") as fout:
            yaml.dump(output, fout, default_flow_style=False, sort_keys=False)
        st.success(f"Saved → {fname}")

# ── Main grid ────────────────────────────────────────────────────────────────


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
                mask, centroid = find_largest_mask_xy(smoothed, percentile)

                st.image(render_embryo(img, mask, centroid), caption=f"Embryo {p}")

                area = int(mask.sum())
                cx, cy = centroid[1], centroid[0]
                st.caption(
                    f"Area: {area} px ({area / mask.size * 100:.1f}%)  |  "
                    f"Centroid: ({cx:.1f}, {cy:.1f})  |  "
                    f"Mean intensity: {img[mask].mean():.1f}"
                )


render_grid()

if st.session_state.playing:
    time.sleep(play_delay)
    st.session_state["T"] = (t + 1) % T
    st.rerun()

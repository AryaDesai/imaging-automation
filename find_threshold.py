"""find_threshold.py — interactive threshold tuning for embryo masking.

Tkinter desktop application for tuning Gaussian blur (sigma) and percentile
threshold parameters on microscopy data. Displays one embryo at a time with
a red mask overlay and cyan centroid crosshair so the user can visually
confirm the mask quality before saving parameters to a YAML config.

Supports two input formats:
  - ND2 files (multi-embryo timelapse): loads the full (T, P, Z, C, Y, X)
    volume via useful_functions.load_nd2, max-projects Z, and lets the user
    navigate between embryo positions with Prev/Next buttons.
  - OME-TIFF / TIFF files (single-embryo): loads a (T, C, Z, Y, X) volume
    via tifffile.imread, max-projects Z, and treats it as a single position.
    Embryo navigation is hidden in this case.

Each embryo's parameters are saved to its own YAML file so that different
embryos can have different sigma/percentile values. The YAML schema matches
the format expected by centroid_align_xy.py.

Usage:
    python find_threshold.py
"""

import datetime
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import tifffile
import yaml
from PIL import Image, ImageDraw, ImageTk
from scipy.ndimage import gaussian_filter

from useful_functions import find_largest_mask_xy, load_nd2, load_tif_metadata, max_project_z


# ── Data loading ──────────────────────────────────────────────────────────────


class LazyTifAccessor:
    """Lazy accessor for TIF files that loads one timepoint at a time.

    Provides array-like indexing `[t, p, c]` to retrieve a 2-D (Y, X) image,
    but only loads and max-projects the requested timepoint from disk. This
    prevents out-of-memory errors when working with large concatenated TIFs.

    The accessor mimics the shape (T, P, C, Y, X) that the app expects, with
    P always equal to 1 for single-embryo TIFs.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.tif = tifffile.TiffFile(file_path)
        self.series = self.tif.series[0]
        # Expected shape from save_ome_tiff: (T, C, Z, Y, X)
        self.T, self.C, self.Z, self.Y, self.X = self.series.shape
        self.P = 1  # Single embryo per TIF
        self.shape = (self.T, self.P, self.C, self.Y, self.X)
        # Load channel names from OME metadata (e.g. "Venus", "mCherry").
        self.channel_names, _, _ = load_tif_metadata(file_path)
        # Cache the most recently loaded timepoint to avoid redundant reads
        # when the user switches channels without changing T.
        self._cache_t = None
        self._cache_data = None  # Shape: (C, Y, X) — max-projected

    def __getitem__(self, idx):
        """Return a 2-D (Y, X) image for the given (t, p, c) index."""
        t, p, c = idx
        if t != self._cache_t:
            # Load all C*Z pages for this timepoint and max-project Z.
            pages_per_t = self.C * self.Z
            start = t * pages_per_t
            frame_czyx = np.stack(
                [self.series.pages[start + i].asarray() for i in range(pages_per_t)]
            ).reshape(self.C, self.Z, self.Y, self.X).astype(np.float32)
            # Max-project Z → (C, Y, X)
            self._cache_data = frame_czyx.max(axis=1)
            self._cache_t = t
        return self._cache_data[c]

    def close(self):
        """Close the underlying TIF file handle."""
        self.tif.close()


def load_file(file_path):
    """Load an ND2 or TIF file and return a Z-max-projected accessor + metadata.

    Both file types are normalised to the same output shape (T, P, C, Y, X)
    so that the rest of the application does not need to distinguish between
    them. ND2 files naturally have a P (position) axis; TIF files are assumed
    to be single-embryo and get a P axis of size 1 inserted.

    Parameters
    ----------
    file_path : str
        Path to an .nd2 or .tif/.ome.tif file.

    Returns
    -------
    max_proj : ndarray, shape (T, P, C, Y, X), float32
        Z-max-projected image data.
    channel_names : list of str
        Channel names. For ND2 files these come from the microscope metadata;
        for TIF files they default to "Ch0", "Ch1", etc. because TIF metadata
        does not reliably carry channel names.
    is_nd2 : bool
        True if the file was an ND2, False if TIF. Used downstream to decide
        the YAML filename pattern and whether to show embryo navigation.
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".nd2":
        data, channel_names, _, _ = load_nd2(file_path)
        # data is (T, P, Z, C, Y, X). Max-project Z to get (T, P, C, Y, X).
        max_proj = max_project_z(data)
        return max_proj, channel_names, True

    # TIF / OME-TIFF path.
    # tifffile.imread returns the raw array. The expected axis order for
    # single-embryo stacks from this pipeline is (T, C, Z, Y, X).
    raw = tifffile.imread(file_path).astype(np.float32)

    # Max-project Z (axis 2) to collapse to (T, C, Y, X).
    projected = raw.max(axis=2)

    # Insert a P axis at position 1 so the shape becomes (T, 1, C, Y, X),
    # matching the ND2 convention. This lets all downstream code treat
    # single-embryo TIFs identically to multi-embryo ND2 files.
    max_proj = np.expand_dims(projected, axis=1)

    C = max_proj.shape[2]
    channel_names = [f"Ch{i}" for i in range(C)]

    return max_proj, channel_names, False


# ── Rendering ─────────────────────────────────────────────────────────────────


def render_embryo(img, mask, centroid):
    """Grayscale image + red mask overlay + centroid crosshair -> RGBA PIL image.

    The red semi-transparent overlay shows which pixels the thresholding
    selected as foreground. The cyan crosshair marks the centroid of the
    largest connected component — this is the point that centroid_align_xy.py
    will use to centre the embryo.

    Parameters
    ----------
    img : ndarray, shape (Y, X), float32
        Raw (unsmoothed) max-projected image for display.
    mask : ndarray of bool, shape (Y, X)
        Binary mask from find_largest_mask_xy.
    centroid : ndarray, shape (2,)
        [cy, cx] centroid coordinates from find_largest_mask_xy.

    Returns
    -------
    composite : PIL.Image, mode RGBA
    """
    # Auto-contrast normalisation: clip at the 99.5th percentile to avoid
    # hot pixels dominating the display range. Same logic as auto_contrast
    # in useful_functions.py but producing an RGBA composite rather than
    # a bare uint8 array.
    vmax = np.percentile(img, 99.5)
    if vmax == 0:
        vmax = 1
    gray = np.clip(img / vmax * 255, 0, 255).astype(np.uint8)
    base = Image.fromarray(gray, mode="L").convert("RGBA")

    # Red semi-transparent overlay on masked pixels. Alpha=100 (out of 255)
    # lets the underlying grayscale detail show through while clearly marking
    # the mask boundary.
    overlay_arr = np.zeros((*gray.shape, 4), dtype=np.uint8)
    overlay_arr[mask] = [255, 50, 50, 100]
    composite = Image.alpha_composite(base, Image.fromarray(overlay_arr, mode="RGBA"))

    # Cyan crosshair at the centroid. The arm length (s=25 pixels) and line
    # width (7 pixels) are chosen to be visible on 1024×1024 embryo images
    # without obscuring nearby anatomy.
    draw = ImageDraw.Draw(composite)
    cy, cx = int(round(centroid[0])), int(round(centroid[1]))
    s = 25
    draw.line([(cx - s, cy), (cx + s, cy)], fill=(0, 255, 255, 255), width=7)
    draw.line([(cx, cy - s), (cx, cy + s)], fill=(0, 255, 255, 255), width=7)

    return composite


# ── Application ───────────────────────────────────────────────────────────────


class ThresholdApp:
    """Tkinter application for interactive threshold parameter tuning.

    The user loads an ND2 or TIF file, selects a channel, and adjusts sigma
    and percentile sliders while watching the mask overlay update in real time.
    When satisfied, they save the parameters to a YAML file that
    centroid_align_xy.py will read.

    One embryo is displayed at a time. For multi-embryo ND2 files, Prev/Next
    buttons navigate between positions. Each embryo's parameters are saved
    independently to its own YAML file so different embryos can have different
    thresholds.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Find threshold for masking PSM")

        # ── Data state ────────────────────────────────────────────────────
        # These are populated by _load_file and remain None until a file is
        # loaded. All rendering code checks for None before proceeding.
        self.file_path = None
        self.max_proj = None       # (T, P, C, Y, X) float32
        self.channel_names = []
        self.is_nd2 = False
        self.T = self.P = self.C = self.Y = self.X = 0

        # ── Animation state ───────────────────────────────────────────────
        self.playing = False
        # Stores the id returned by root.after() so the scheduled callback
        # can be cancelled when the user presses Pause.
        self.after_id = None

        # ── Display scaling ───────────────────────────────────────────────
        # Each embryo image is scaled to this size in pixels for display.
        # The raw images are typically 1024×1024; displaying them at full
        # resolution would make the window too large on most screens.
        self.display_size = 512

        # PhotoImage references must be kept alive for the duration of
        # display. Tkinter does not hold a reference internally, so without
        # this the garbage collector would free the image and the label
        # would go blank.
        self.photo_ref = None

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        # Left panel: all controls. Using a fixed-width frame keeps the
        # controls from reflowing when the image panel resizes.
        ctrl = ttk.Frame(self.root, padding=10)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(ctrl, text="Find threshold for masking PSM",
                  font=("Helvetica", 14, "bold")).pack(anchor=tk.W, pady=(0, 5))
        ttk.Label(ctrl, text="Red overlay = mask  |  + = centroid",
                  foreground="gray").pack(anchor=tk.W, pady=(0, 10))

        # ── File picker ───────────────────────────────────────────────────
        # Browse opens a native file dialog filtered to ND2 and TIF files.
        # The text entry allows pasting a path directly (useful when the
        # file is on a remote mount that the dialog cannot browse).
        ttk.Button(ctrl, text="Browse", command=self._browse).pack(
            fill=tk.X, pady=(0, 5))
        self.path_var = tk.StringVar()
        ttk.Entry(ctrl, textvariable=self.path_var, width=40).pack(
            fill=tk.X, pady=(0, 2))
        ttk.Button(ctrl, text="Load", command=self._load_file).pack(
            fill=tk.X, pady=(0, 10))

        # ── Channel selector ──────────────────────────────────────────────
        ttk.Label(ctrl, text="Channel").pack(anchor=tk.W)
        self.channel_var = tk.StringVar()
        self.channel_combo = ttk.Combobox(ctrl, textvariable=self.channel_var,
                                          state="readonly", width=20)
        self.channel_combo.pack(fill=tk.X, pady=(0, 10))
        self.channel_combo.bind("<<ComboboxSelected>>", lambda _: self._update())

        # ── Embryo navigation ─────────────────────────────────────────────
        # Prev/Next buttons and a label showing the current position index.
        # This frame is hidden when a single-embryo TIF is loaded because
        # there is only one position to display.
        self.nav_frame = ttk.Frame(ctrl)
        self.nav_frame.pack(fill=tk.X, pady=(0, 10))
        self.embryo_idx = 0
        ttk.Button(self.nav_frame, text="Prev", command=self._prev_embryo).pack(
            side=tk.LEFT)
        self.embryo_label_var = tk.StringVar(value="Embryo 0 / 0")
        ttk.Label(self.nav_frame, textvariable=self.embryo_label_var).pack(
            side=tk.LEFT, padx=10)
        ttk.Button(self.nav_frame, text="Next", command=self._next_embryo).pack(
            side=tk.LEFT)

        # ── T slider ─────────────────────────────────────────────────────
        ttk.Label(ctrl, text="T").pack(anchor=tk.W)
        self.t_var = tk.IntVar(value=0)
        self.t_slider = tk.Scale(ctrl, from_=0, to=0, orient=tk.HORIZONTAL,
                                 variable=self.t_var, command=lambda _: self._update())
        self.t_slider.pack(fill=tk.X, pady=(0, 10))

        # ── Sigma slider ──────────────────────────────────────────────────
        # Gaussian blur radius in pixels. Higher values smooth over noise
        # but reduce sensitivity to fine embryo boundary details. Typical
        # values for PSM imaging are 15–40.
        ttk.Label(ctrl, text="Sigma").pack(anchor=tk.W)
        self.sigma_var = tk.IntVar(value=2)
        tk.Scale(ctrl, from_=1, to=100, orient=tk.HORIZONTAL,
                 variable=self.sigma_var,
                 command=lambda _: self._update()).pack(fill=tk.X, pady=(0, 10))

        # ── Percentile slider ─────────────────────────────────────────────
        # Pixels above this percentile of the smoothed image are included
        # in the binary mask. Lower values include more background; higher
        # values restrict the mask to only the brightest regions.
        ttk.Label(ctrl, text="Percentile").pack(anchor=tk.W)
        self.pct_var = tk.DoubleVar(value=90.0)
        tk.Scale(ctrl, from_=10.0, to=99.5, orient=tk.HORIZONTAL,
                 resolution=0.5, variable=self.pct_var,
                 command=lambda _: self._update()).pack(fill=tk.X, pady=(0, 10))

        # ── Animation controls ────────────────────────────────────────────
        ttk.Separator(ctrl).pack(fill=tk.X, pady=5)
        anim_frame = ttk.Frame(ctrl)
        anim_frame.pack(fill=tk.X, pady=(0, 5))
        self.play_btn = ttk.Button(anim_frame, text="Play",
                                   command=self._toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(anim_frame, text="Delay (s):").pack(side=tk.LEFT)
        self.delay_var = tk.DoubleVar(value=0.5)
        ttk.Spinbox(anim_frame, from_=0.1, to=5.0, increment=0.1,
                     textvariable=self.delay_var, width=5).pack(side=tk.LEFT)

        # ── Save YAML ────────────────────────────────────────────────────
        ttk.Separator(ctrl).pack(fill=tk.X, pady=5)
        ttk.Button(ctrl, text="Save thresholds to YAML",
                   command=self._save_yaml).pack(fill=tk.X, pady=(5, 0))

        # ── Status label ──────────────────────────────────────────────────
        # Displays load confirmation, save confirmation, and error messages.
        self.status_var = tk.StringVar()
        ttk.Label(ctrl, textvariable=self.status_var, foreground="green",
                  wraplength=250).pack(anchor=tk.W, pady=(5, 0))

        # ── Right panel: image display ────────────────────────────────────
        self.display_frame = ttk.Frame(self.root, padding=10)
        self.display_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Header shows current parameter values so the user can see at a
        # glance what combination produced the displayed mask.
        self.header_var = tk.StringVar()
        ttk.Label(self.display_frame, textvariable=self.header_var,
                  font=("Helvetica", 12)).pack(anchor=tk.W, pady=(0, 5))

        # The image label displays the rendered embryo composite. It starts
        # empty and is populated after the first file load.
        self.img_label = ttk.Label(self.display_frame)
        self.img_label.pack(pady=5)

        # Caption shows per-embryo quantitative diagnostics: mask area,
        # centroid coordinates, and mean intensity within the mask.
        self.cap_var = tk.StringVar()
        ttk.Label(self.display_frame, textvariable=self.cap_var,
                  foreground="gray").pack(anchor=tk.W, pady=(0, 5))

    # ── File loading ──────────────────────────────────────────────────────

    def _browse(self):
        """Open a native file dialog filtered to ND2 and TIF files."""
        path = filedialog.askopenfilename(
            title="Select ND2 or TIF file",
            filetypes=[
                ("ND2 files", "*.nd2"),
                ("TIFF files", "*.tif"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.path_var.set(path)
            self._load_file()

    def _load_file(self):
        """Load the file specified in the path entry and update all controls."""
        path = self.path_var.get().strip()
        if not path:
            return
        if not Path(path).is_file():
            messagebox.showerror("Error", f"File not found:\n{path}")
            return

        self.status_var.set("Loading...")
        self.root.update_idletasks()

        self.file_path = path
        self.max_proj, self.channel_names, self.is_nd2 = load_file(path)
        self.T, self.P, self.C, self.Y, self.X = self.max_proj.shape

        # Reset embryo index to the first position.
        self.embryo_idx = 0

        # Populate the channel dropdown with the names read from the file.
        self.channel_combo["values"] = self.channel_names
        self.channel_combo.current(0)

        # Set the T slider range to match the number of timepoints.
        self.t_slider.config(to=max(0, self.T - 1))
        self.t_var.set(0)

        # Show embryo navigation only for multi-position ND2 files.
        # Single-embryo TIFs have P=1 so navigation is meaningless.
        if self.P > 1:
            self.nav_frame.pack(fill=tk.X, pady=(0, 10))
        else:
            self.nav_frame.pack_forget()
        self._update_embryo_label()

        self.status_var.set(
            f"Loaded: T={self.T}, P={self.P}, C={self.C}, "
            f"Y={self.Y}, X={self.X}")
        self._update()

    # ── Embryo navigation ─────────────────────────────────────────────────

    def _prev_embryo(self):
        """Navigate to the previous embryo position, wrapping around."""
        if self.P <= 1:
            return
        self.embryo_idx = (self.embryo_idx - 1) % self.P
        self._update_embryo_label()
        self._update()

    def _next_embryo(self):
        """Navigate to the next embryo position, wrapping around."""
        if self.P <= 1:
            return
        self.embryo_idx = (self.embryo_idx + 1) % self.P
        self._update_embryo_label()
        self._update()

    def _update_embryo_label(self):
        """Update the navigation label to show the current position index."""
        self.embryo_label_var.set(f"Embryo {self.embryo_idx} / {self.P - 1}")

    # ── Rendering ─────────────────────────────────────────────────────────

    def _update(self):
        """Recompute the mask and redraw the embryo image.

        Called whenever any parameter changes: T slider, sigma, percentile,
        channel selection, or embryo navigation. Each call applies the
        Gaussian filter and connected-component detection from scratch.
        Tkinter updates are event-driven (not polling), so the filter only
        runs when a widget value actually changes.
        """
        if self.max_proj is None:
            return

        t = self.t_var.get()
        p = self.embryo_idx
        ch_idx = self.channel_combo.current()
        if ch_idx < 0:
            ch_idx = 0
        sigma = self.sigma_var.get()
        percentile = self.pct_var.get()
        channel = self.channel_names[ch_idx]

        self.header_var.set(
            f"T = {t}  |  {channel}  |  sigma = {sigma}  |  "
            f"percentile = {percentile}")

        # Extract the 2-D image for the current (t, p, ch) combination.
        img = self.max_proj[t, p, ch_idx]

        # Gaussian blur suppresses noise and isolated bright spots before
        # thresholding, matching the processing in compute_shift_xy.
        smoothed = gaussian_filter(img, sigma=sigma)

        # Detect the largest connected component above the percentile
        # threshold. This is the same function used by centroid_align_xy.py
        # so that the mask the user sees here is exactly what the alignment
        # script will detect.
        mask, centroid = find_largest_mask_xy(smoothed, percentile)

        # Render the composite image and scale it for display.
        pil_img = render_embryo(img, mask, centroid)
        pil_img = pil_img.resize(
            (self.display_size, self.display_size), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil_img)

        # Store the PhotoImage reference to prevent garbage collection.
        # Tkinter labels do not increment the Python reference count on
        # the image object, so without this assignment the image would be
        # collected and the label would display a blank rectangle.
        self.photo_ref = photo
        self.img_label.config(image=photo)

        # Update the caption with quantitative diagnostics.
        area = int(mask.sum())
        cx, cy = centroid[1], centroid[0]
        mean_val = float(img[mask].mean()) if mask.any() else 0.0
        self.cap_var.set(
            f"Embryo {p}  |  Area: {area} px ({area / mask.size * 100:.1f}%)  |  "
            f"Centroid: ({cx:.1f}, {cy:.1f})  |  "
            f"Mean intensity: {mean_val:.1f}")

    # ── Animation ─────────────────────────────────────────────────────────

    def _toggle_play(self):
        """Toggle between playing and paused states."""
        self.playing = not self.playing
        self.play_btn.config(text="Pause" if self.playing else "Play")
        if self.playing:
            self._animate()
        elif self.after_id is not None:
            # Cancel the pending callback so the animation actually stops.
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def _animate(self):
        """Advance T by one frame and schedule the next advance.

        Uses root.after() rather than time.sleep() so the tkinter event
        loop remains responsive during animation. The T slider wraps
        around from T-1 back to 0.
        """
        if not self.playing or self.max_proj is None:
            return
        t = (self.t_var.get() + 1) % self.T
        self.t_var.set(t)
        self.t_slider.set(t)
        self._update()
        delay_ms = int(self.delay_var.get() * 1000)
        self.after_id = self.root.after(delay_ms, self._animate)

    # ── Save YAML ─────────────────────────────────────────────────────────

    def _save_yaml(self):
        """Save the current embryo's threshold parameters to a YAML file.

        Each embryo gets its own YAML file so that different embryos can
        have different sigma and percentile values. The YAML schema matches
        what centroid_align_xy.py expects: parameters (channel, channel_index,
        sigma, percentile), source (file path, image shape), and diagnostics
        (mask area, centroid, mean intensity at t=0).

        Filename pattern:
          ND2 input:  {base}_P{n}_{channel}_threshold.yaml
          TIF input:  {base}_{channel}_threshold.yaml
        """
        if self.max_proj is None:
            messagebox.showwarning("No data", "Load a file first.")
            return

        p = self.embryo_idx
        ch_idx = self.channel_combo.current()
        channel = self.channel_names[ch_idx]
        sigma = self.sigma_var.get()
        percentile = self.pct_var.get()

        t = self.t_var.get()
        img = self.max_proj[t, p, ch_idx]
        smoothed = gaussian_filter(img, sigma=sigma)
        mask, centroid = find_largest_mask_xy(smoothed, percentile)
        area = int(mask.sum())

        output = {
            "parameters": {
                "channel": channel,
                "channel_index": ch_idx,
                "sigma": sigma,
                "percentile": percentile,
            },
            "source": {
                "file": str(Path(self.file_path)),
                "image_shape": {
                    "T": self.T, "P": self.P,
                    "Y": self.Y, "X": self.X,
                },
            },
            "diagnostics": {
                "time_point": t,
                "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "embryos": [{
                    "id": p,
                    "mask_area_px": area,
                    "mask_area_pct": round(area / mask.size * 100, 2),
                    "mean_intensity": round(float(img[mask].mean()), 2) if mask.any() else 0.0,
                    "centroid": [round(float(centroid[1]), 2),
                                 round(float(centroid[0]), 2)],
                }],
            },
        }

        base = Path(self.file_path).stem
        if self.is_nd2:
            fname = f"{base}_P{p}_{channel}_threshold.yaml"
        else:
            # TIF files are single-embryo so no position index is needed.
            # The .ome suffix (if present) is already removed by Path.stem,
            # but the base might still end with ".ome" if the file was named
            # e.g. "nd1188_P0.ome.tif". Strip it for a cleaner filename.
            if base.endswith(".ome"):
                base = base[:-4]
            fname = f"{base}_{channel}_threshold.yaml"

        with open(fname, "w") as fout:
            yaml.dump(output, fout, default_flow_style=False, sort_keys=False)

        self.status_var.set(f"Saved \u2192 {fname}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    ThresholdApp(root)
    root.mainloop()

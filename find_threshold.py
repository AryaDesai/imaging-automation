
""

 Displays one embryo at a time with
a  mask overlay and centroid crosshair so the user can visually
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

from useful_functions import (
    MAX_OTSU_LEVELS,
    find_largest_mask_xy,
    load_nd2,
    load_tif_metadata,
    max_project_z,
    multiotsu_class_map,
)


# Data loading


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
    max_proj : ndarray or LazyTifAccessor, shape (T, P, C, Y, X)
        Z-max-projected image data. For ND2 files this is a float32 ndarray
        loaded into memory. For TIF files this is a LazyTifAccessor that
        loads one timepoint at a time to avoid out-of-memory errors.
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

    # TIF / OME-TIFF path: use lazy accessor to avoid loading entire file.
    accessor = LazyTifAccessor(file_path)

    return accessor, accessor.channel_names, False


# Rendering


# Overlay colour per intensity class, dark class first. Distinct hues rather
# than a brightness ramp so that several checked classes stay tellable apart,
# which a single-colour overlay could not do.
LEVEL_COLORS = [
    (255, 50, 50),    # red
    (50, 200, 255),   # blue
    (120, 255, 80),   # green
    (255, 220, 40),   # yellow
    (220, 100, 255),  # purple
]


def render_embryo(img, mask, centroid, class_map=None, selected_levels=None):
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

    # Semi-transparent overlay on masked pixels. Alpha=100 (out of 255) lets
    # the underlying grayscale detail show through while clearly marking the
    # mask boundary.
    overlay_arr = np.zeros((*gray.shape, 4), dtype=np.uint8)
    if class_map is None:
        overlay_arr[mask] = [255, 50, 50, 100]
    else:
        # Multi-level Otsu: colour each checked class separately so the user
        # can see which class contributed which region. This shows every
        # selected pixel, whereas mask holds only the largest connected
        # component, so the two deliberately differ when a selection is
        # fragmented.
        for i in selected_levels or []:
            overlay_arr[class_map == i] = [*LEVEL_COLORS[i % len(LEVEL_COLORS)], 100]
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


# Applicatioj


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

        # ------------------------ Data state ------------------------------------------------------------------------------------------------------------─
        # These are populated by _load_file and remain None until a file is
        # loaded. All rendering code checks for None before proceeding.
        self.file_path = None
        self.max_proj = None       # (T, P, C, Y, X) float32
        self.channel_names = []
        self.is_nd2 = False
        self.T = self.P = self.C = self.Y = self.X = 0
        self.custom_timepoint_params = {}

        # ------------------------ Multi-level Otsu cache ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # The class map depends only on the image and the level count, not on
        # which classes are checked, so it is kept between redraws. Toggling a
        # checkbox would otherwise repeat a threshold search that takes about
        # two seconds at five levels and would freeze the window each time.
        self._class_map = None
        self._class_map_key = None

        # ------------------------ Animation state ------------------------------------------------------------
        self.playing = False
        # Stores the id returned by root.after() so the scheduled callback
        # can be cancelled when the user presses Pause.
        self.after_id = None

        # ------------------------ Display scaling ------------------------------------------------------------
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

    # -- UI construction --

    def _build_ui(self):
        # Left panel: all controls. Using a fixed-width frame keeps the
        # controls from reflowing when the image panel resizes.
        ctrl = ttk.Frame(self.root, padding=10)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(ctrl, text="Find threshold for masking PSM",
                  font=("Helvetica", 14, "bold")).pack(anchor=tk.W, pady=(0, 5))
        ttk.Label(ctrl, text="Red overlay = mask  |  + = centroid",
                  foreground="gray").pack(anchor=tk.W, pady=(0, 10))

        # -- File picker--
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

        #--------- Channel selector------------------------
        ttk.Label(ctrl, text="Channel").pack(anchor=tk.W)
        self.channel_var = tk.StringVar()
        self.channel_combo = ttk.Combobox(ctrl, textvariable=self.channel_var,
                                          state="readonly", width=20)
        self.channel_combo.pack(fill=tk.X, pady=(0, 10))
        self.channel_combo.bind("<<ComboboxSelected>>", lambda _: self._update())

        # --------- Embryo navigation -------------------------------
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

        #--------------------- T slider-----------------------
        ttk.Label(ctrl, text="T").pack(anchor=tk.W)
        self.t_var = tk.IntVar(value=0)
        self.t_slider = tk.Scale(ctrl, from_=0, to=0, orient=tk.HORIZONTAL,
                                 variable=self.t_var, command=lambda _: self._update())
        self.t_slider.pack(fill=tk.X, pady=(0, 10))

        # ---------------------------Sigma slider--------------------------------------------
        # Gaussian blur radius in pixels. Higher values smooth over noise
        # but reduce sensitivity to fine embryo boundary details. Typical
        # values for PSM imaging are 15–40.
        ttk.Label(ctrl, text="Sigma").pack(anchor=tk.W)
        self.sigma_var = tk.IntVar(value=2)
        tk.Scale(ctrl, from_=1, to=100, orient=tk.HORIZONTAL,
                 variable=self.sigma_var,
                 command=lambda _: self._update()).pack(fill=tk.X, pady=(0, 10))

        # ---------------------------- Method selector--------------------------------------
        ttk.Label(ctrl, text="Threshold Method").pack(anchor=tk.W)
        self.method_var = tk.StringVar(value="Percentile")
        self.method_combo = ttk.Combobox(ctrl, textvariable=self.method_var,
                                         state="readonly", width=34)
        self.method_combo["values"] = [
            "Percentile",
            "Multi-Level Otsu",
            "Percentile -> Otsu ROI",
            "Local Otsu (Block-Interpolated)", 
            "Local Otsu (Pixel-by-Pixel)"
        ]
        self.method_combo.pack(fill=tk.X, pady=(0, 5))
        self.method_combo.bind("<<ComboboxSelected>>", lambda _: self._on_method_change())

        # ------------------------ Block Size selector ------------------------------------
        self.block_label = ttk.Label(ctrl, text="Block Size (Local Otsu only)", state=tk.DISABLED)
        self.block_label.pack(anchor=tk.W)
        self.block_var = tk.IntVar(value=64)
        self.block_combo = ttk.Combobox(ctrl, textvariable=self.block_var,
                                         state=tk.DISABLED, width=20)
        self.block_combo["values"] = [16, 32, 64, 128]
        self.block_combo.pack(fill=tk.X, pady=(0, 10))
        self.block_combo.bind("<<ComboboxSelected>>", lambda _: self._update())

        # ------------------------ Multi-level Otsu levels ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------─
        # Number of intensity classes the image is split into. Capped at
        # MAX_OTSU_LEVELS because the threshold search cost grows steeply
        # with each added level.
        self.levels_frame = ttk.Frame(ctrl)
        self.levels_label = ttk.Label(self.levels_frame, text="Levels")
        self.levels_label.pack(side=tk.LEFT)
        self.levels_var = tk.IntVar(value=3)
        self.levels_spin = ttk.Spinbox(
            self.levels_frame, from_=2, to=MAX_OTSU_LEVELS, width=5,
            textvariable=self.levels_var, command=self._on_levels_change,
        )
        self.levels_spin.pack(side=tk.LEFT, padx=(6, 0))

        # ------------------------ Level selection checkboxes ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # One checkbox per class, rebuilt whenever the level count changes.
        # Any combination can be selected; the mask is the union of the
        # checked classes. Populated by _rebuild_level_checkboxes.
        self.levels_check_frame = ttk.Frame(ctrl)
        self.level_vars = []

        # ------------------------ Percentile slider ------------------------------------
        # Pixels above this percentile of the smoothed image are included
        # in the binary mask. Lower values include more background; higher
        # values restrict the mask to only the brightest regions.
        # Kept as an attribute so the multi-level Otsu controls can be packed
        # directly above it; pack() alone would append them to the bottom of
        # the panel.
        self.pct_label = ttk.Label(ctrl, text="Percentile")
        self.pct_label.pack(anchor=tk.W)
        self.pct_var = tk.DoubleVar(value=90.0)
        self.pct_slider = tk.Scale(ctrl, from_=10.0, to=99.5, orient=tk.HORIZONTAL,
                                   resolution=0.5, variable=self.pct_var,
                                   command=lambda _: self._update())
        self.pct_slider.pack(fill=tk.X, pady=(0, 10))
        
        # ------------------------ Invert selector ------------------------------------------------------------
        self.invert_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Invert Mask", variable=self.invert_var,
                        command=self._update).pack(anchor=tk.W, pady=(0, 10))

        # ------------------------ Per-timepoint parameter overrides ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------─
        ttk.Separator(ctrl).pack(fill=tk.X, pady=5)
        override_frame = ttk.Frame(ctrl)
        override_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(override_frame, text="Apply to current T",
                   command=self._apply_to_current_t).pack(fill=tk.X, pady=(0, 4))
        range_frame = ttk.Frame(override_frame)
        range_frame.pack(fill=tk.X)
        self.range_start_var = tk.IntVar(value=0)
        self.range_end_var = tk.IntVar(value=0)
        ttk.Label(range_frame, text="Range").pack(side=tk.LEFT)
        self.range_start_spin = ttk.Spinbox(
            range_frame, from_=0, to=0, textvariable=self.range_start_var, width=5
        )
        self.range_start_spin.pack(side=tk.LEFT, padx=(6, 2))
        self.range_end_spin = ttk.Spinbox(
            range_frame, from_=0, to=0, textvariable=self.range_end_var, width=5
        )
        self.range_end_spin.pack(side=tk.LEFT, padx=(2, 6))
        ttk.Button(range_frame, text="Apply",
                   command=self._apply_to_range).pack(side=tk.LEFT)
        self.override_status_var = tk.StringVar(value="Customized: 0 / 0 T")
        ttk.Label(override_frame, textvariable=self.override_status_var,
                  foreground="gray").pack(anchor=tk.W, pady=(4, 0))

        # ------------------------ Animation controls ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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

        # ------------------------ Save YAML ------------------------------------------------------------------------------------------------------------─
        ttk.Separator(ctrl).pack(fill=tk.X, pady=5)
        ttk.Button(ctrl, text="Save thresholds to YAML",
                   command=self._save_yaml).pack(fill=tk.X, pady=(5, 0))

        # ------------------------ Status label ------------------------------------------------------------------------------------─
        # Displays load confirmation, save confirmation, and error messages.
        self.status_var = tk.StringVar()
        ttk.Label(ctrl, textvariable=self.status_var, foreground="green",
                  wraplength=250).pack(anchor=tk.W, pady=(5, 0))

        # ------------------------ Right panel: image display ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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

    # ------------------------ File loading ------------------------------------------------------------------------------------------------------------------------------------─

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

        # Close the previous TIF accessor if one exists (releases file handle).
        if hasattr(self.max_proj, 'close'):
            self.max_proj.close()

        self.file_path = path
        self.max_proj, self.channel_names, self.is_nd2 = load_file(path)
        self.T, self.P, self.C, self.Y, self.X = self.max_proj.shape
        self.custom_timepoint_params = {}

        # Reset embryo index to the first position.
        self.embryo_idx = 0

        # Populate the channel dropdown with the names read from the file.
        self.channel_combo["values"] = self.channel_names
        self.channel_combo.current(0)

        # Set the T slider range to match the number of timepoints.
        self.t_slider.config(to=max(0, self.T - 1))
        self.t_var.set(0)
        self.range_start_var.set(0)
        self.range_end_var.set(0)
        self.range_start_spin.config(to=max(0, self.T - 1))
        self.range_end_spin.config(to=max(0, self.T - 1))

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
        self._update_override_status()
        self._update()

    # ------------------------ Embryo navigation ------------------------------------------------------------------------------------

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

    # ------------------------ Rendering ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def _on_method_change(self):
        """Enable/disable controls based on method and trigger update."""
        method = self.method_var.get()
        if method == "Multi-Level Otsu":
            self.pct_slider.config(state=tk.DISABLED)
        else:
            self.pct_slider.config(state=tk.NORMAL)

        # The level spinbox and class checkboxes are packed and unpacked
        # rather than greyed out because the checkbox count varies, so an
        # inactive row of them would be misleading clutter.
        if method == "Multi-Level Otsu":
            self.levels_frame.pack(anchor=tk.W, pady=(0, 4), before=self.pct_label)
            self.levels_check_frame.pack(anchor=tk.W, pady=(0, 10), before=self.pct_label)
            if not self.level_vars:
                self._rebuild_level_checkboxes()
        else:
            self.levels_frame.pack_forget()
            self.levels_check_frame.pack_forget()


        if "Local Otsu" in method:
            self.block_label.config(state=tk.NORMAL)
            self.block_combo.config(state="readonly")
        else:
            self.block_label.config(state=tk.DISABLED)
            self.block_combo.config(state=tk.DISABLED)
            
        self._update()

    def _rebuild_level_checkboxes(self):
        """Recreate one checkbox per intensity class for the current level count.

        The checkboxes are destroyed and rebuilt rather than hidden because
        the number of classes changes with the Levels spinbox. Selection is
        reset to the brightest class instead of being carried over: class
        indices are relative to the level count, so class 2 of 3 covers a
        different intensity band than class 2 of 4 and preserving the old
        checks would silently change which pixels are selected.
        """
        for widget in self.levels_check_frame.winfo_children():
            widget.destroy()
        self.level_vars = []

        levels = self.levels_var.get()
        for i in range(levels):
            # Default to the brightest class alone, which reproduces a plain
            # global Otsu threshold and is the usual starting point.
            var = tk.BooleanVar(value=(i == levels - 1))
            self.level_vars.append(var)
            ttk.Checkbutton(self.levels_check_frame, text=str(i), variable=var,
                            command=self._update).pack(side=tk.LEFT)

    def _on_levels_change(self):
        """Rebuild the class checkboxes after the level count changes."""
        self._rebuild_level_checkboxes()
        self._update()

    def _selected_levels(self):
        """Return the indices of the currently checked intensity classes."""
        return [i for i, var in enumerate(self.level_vars) if var.get()]

    def _method_code(self):
        """Return the internal method string for the current UI selection."""
        method_map = {
            "Percentile": "percentile",
            "Multi-Level Otsu": "multiotsu",
            "Percentile -> Otsu ROI": "percentile_otsu_roi",
            "Local Otsu (Block-Interpolated)": "local_otsu_interp",
            "Local Otsu (Pixel-by-Pixel)": "local_otsu_pixel",
        }
        return method_map.get(self.method_var.get(), "percentile")

    def _current_params(self):
        """Return threshold parameters represented by the current controls."""
        ch_idx = self.channel_combo.current()
        if ch_idx < 0:
            ch_idx = 0
        channel = self.channel_names[ch_idx] if self.channel_names else f"Ch{ch_idx}"
        method = self._method_code()
        params = {
            "channel": channel,
            "channel_index": ch_idx,
            "sigma": self.sigma_var.get(),
            "percentile": self.pct_var.get(),
            "method": method,
            "invert": self.invert_var.get(),
        }
        if "local_otsu" in method:
            params["block_size"] = self.block_var.get()
        if method == "multiotsu":
            params["levels"] = self.levels_var.get()
            params["selected_levels"] = self._selected_levels()
        return params

    def _apply_to_current_t(self):
        """Store current controls for the currently selected timepoint."""
        if self.max_proj is None:
            return
        t = int(self.t_var.get())
        self.custom_timepoint_params[t] = dict(self._current_params())
        self._update_override_status()

    def _apply_to_range(self):
        """Store current controls for every timepoint in an inclusive range."""
        if self.max_proj is None:
            return
        start = int(self.range_start_var.get())
        end = int(self.range_end_var.get())
        if start > end:
            start, end = end, start
        start = max(0, min(start, self.T - 1))
        end = max(0, min(end, self.T - 1))
        params = dict(self._current_params())
        for t in range(start, end + 1):
            self.custom_timepoint_params[t] = dict(params)
        self.range_start_var.set(start)
        self.range_end_var.set(end)
        self._update_override_status()

    def _update_override_status(self):
        total = self.T if self.max_proj is not None else 0
        customized = len(self.custom_timepoint_params)
        self.override_status_var.set(f"Customized: {customized} / {total} T")

    def _materialize_timepoint_params(self, default_params):
        """Build the full indexed parameter list written to YAML."""
        params_by_t = [dict(default_params) for _ in range(self.T)]
        for t, params in self.custom_timepoint_params.items():
            if 0 <= t < self.T:
                params_by_t[t] = dict(params)
        return params_by_t

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
        
        method = self._method_code()
        block_size = self.block_var.get()
        invert = self.invert_var.get()
        levels = self.levels_var.get()
        selected_levels = self._selected_levels()

        h_str = (
            f"T = {t}  |  {channel}  |  sigma = {sigma}  |  "
            f"method = {method}  |  percentile = {percentile}"
        )
        if "local_otsu" in method:
            h_str += f"  |  block = {block_size}"
        if method == "multiotsu":
            shown = ",".join(str(i) for i in selected_levels) or "none"
            h_str += f"  |  levels = {levels}  |  selected = {shown}"
        if invert:
            h_str += "  |  INVERTED"
        self.header_var.set(h_str)

        # Extract the 2-D image for the current (t, p, ch) combination.
        img = self.max_proj[t, p, ch_idx]

        # Gaussian blur suppresses noise and isolated bright spots before
        # thresholding, matching the processing in compute_shift_xy.
        smoothed = gaussian_filter(img, sigma=sigma)

        # Detect the largest connected component above the percentile
        # threshold. This is the same function used by centroid_align_xy.py
        # so that the mask the user sees here is exactly what the alignment
        # script will detect.
        #
        # The class map is reused for both the mask and the per-class overlay,
        # and is only recomputed when the image or the level count changes.
        class_map = None
        if method == "multiotsu":
            key = (t, p, ch_idx, sigma, levels)
            if key != self._class_map_key:
                self._class_map = multiotsu_class_map(smoothed, levels)
                self._class_map_key = key
            class_map = self._class_map

        mask, centroid = find_largest_mask_xy(smoothed, percentile, method=method, block_size=block_size, invert=invert,
                                              levels=levels, selected_levels=selected_levels, class_map=class_map)

        # Render the composite image and scale it for display.
        pil_img = render_embryo(img, mask, centroid,
                                class_map=class_map, selected_levels=selected_levels)
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
        cap = (
            f"Embryo {p}  |  Area: {area} px ({area / mask.size * 100:.1f}%)  |  "
            f"Centroid: ({cx:.1f}, {cy:.1f})  |  "
            f"Mean intensity: {mean_val:.1f}")

        # Report how much of the coloured overlay survives into the mask. The
        # mask keeps only the largest connected component, so a selection whose
        # classes do not touch loses everything outside the biggest piece.
        # Anything below 100% means the crosshair may sit on a different
        # structure than the overlay suggests.
        if class_map is not None and selected_levels:
            selected_px = int(np.isin(class_map, selected_levels).sum())
            if selected_px:
                cap += f"  |  Largest component: {area / selected_px * 100:.0f}% of selected"
        self.cap_var.set(cap)

    # ------------------------ Animation ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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

    # ------------------------ Save YAML ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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
        if ch_idx < 0:
            ch_idx = 0
        params = self._current_params()
        channel = params["channel"]
        sigma = params["sigma"]
        percentile = params["percentile"]
        method = params["method"]
        block_size = params.get("block_size", self.block_var.get())

        t = self.t_var.get()
        invert = params.get("invert", False)
        
        img = self.max_proj[t, p, ch_idx]
        smoothed = gaussian_filter(img, sigma=sigma)
        mask, centroid = find_largest_mask_xy(
            smoothed, percentile, method=method, block_size=block_size, invert=invert,
            levels=params.get("levels", self.levels_var.get()),
            selected_levels=params.get("selected_levels"))
        area = int(mask.sum())
        
        output = {
            "parameters": params,
            "timepoint_parameters": self._materialize_timepoint_params(params),
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

if __name__ == "__main__":
    root = tk.Tk()
    ThresholdApp(root)
    root.mainloop()

find_threshold.py
Displaying find_threshold.py.

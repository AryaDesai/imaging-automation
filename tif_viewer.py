"""tif_viewer.py -- lazy OME-TIFF and ND2 viewer with per-slice and max-projected display.

PyQt6 desktop application for browsing OME-TIFF and Nikon ND2 files.
Loads one timepoint at a time via LazyTifReader or LazyNd2Reader so
arbitrarily large files can be viewed without running out of memory.

Controls:
  - Position selector: switch between embryo positions (ND2 files only).
  - Channel selector: switch between fluorescent channels.
  - T slider: scrub through timepoints, with Play/Pause animation.
  - Z slider: view individual Z slices.
  - Max Project toggle: switch between single-slice and max-projected view.
  - Encode MP4: encode the current view (single Z or max-projected) across
    all timepoints to an MP4. Supports 8-bit and 10-bit HEVC encoding.

Usage:
    python tif_viewer.py
"""

import sys
import subprocess
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QComboBox, QSlider, QCheckBox,
    QDoubleSpinBox, QFrame, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont

from useful_functions import LazyTifReader, LazyNd2Reader, encode_mp4, load_tif_metadata


# ── Display ──────────────────────────────────────────────────────────────────

def auto_contrast_uint8(img):
    """Scale a uint16 2-D image to uint8 with 99.5th percentile clipping."""
    vmax = np.percentile(img, 99.5)
    if vmax == 0:
        vmax = 1
    return np.clip(img / vmax * 255, 0, 255).astype(np.uint8)

def auto_contrast_uint16(img):
    """Scale a 2-D image to uint16 with 99.5th percentile clipping."""
    vmax = np.percentile(img, 99.5)
    if vmax == 0:
        vmax = 1
    return np.clip((img / vmax) * 65535, 0, 65535).astype(np.uint16)
# ── 10-bit HEVC encoding ────────────────────────────────────────────────────

def encode_mp4_10bit(frame_generator_func, output_path, width, height, fps=2):
    """Encode uint16 grayscale frames to 10-bit HEVC MP4."""
    cmd_base = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "gray16le",
        "-r", str(fps),
        "-i", "-",
    ]

    # Hardware encoders require even dimensions. This filter pads odd dimensions.
    vf_pad = "pad=ceil(iw/2)*2:ceil(ih/2)*2"

    encoders_to_try = [
        (
            ["-c:v", "hevc_videotoolbox", "-b:v", "10M",
             "-tag:v", "hvc1", "-pix_fmt", "p010le", "-vf", vf_pad],
            "Hardware (VideoToolbox)"
        ),
        (
            ["-c:v", "libx265", "-crf", "18", "-preset", "medium",
             "-tag:v", "hvc1", "-pix_fmt", "yuv420p10le",
             "-x265-params", "profile=main10", "-vf", vf_pad],
            "Software (libx265)"
        ),
    ]

    for encoder_args, encoder_name in encoders_to_try:
        cmd = cmd_base + encoder_args + [str(output_path)]
        process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
        )

        try:
            # Call the function to get a fresh generator for this attempt
            for frame in frame_generator_func():
                process.stdin.write(frame.tobytes())

            process.stdin.close()
            process.wait()

            if process.returncode == 0:
                return
            else:
                raise RuntimeError(f"FFmpeg exited with code {process.returncode}")

        except (BrokenPipeError, RuntimeError):
            print(f"[!] {encoder_name} encoding failed. Attempting fallback...")
            if encoder_name == encoders_to_try[-1][1]:
                print(f"Error: All encoding methods failed for {output_path.name}")

# ── Application ──────────────────────────────────────────────────────────────

class TifViewer(QMainWindow):
    """PyQt6 application for lazy OME-TIFF viewing."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("OME-TIFF Viewer")

        # ── Data state ────────────────────────────────────────────────────
        self.reader = None
        self.channel_names = []
        self.T = self.P = self.C = self.Z = self.Y = self.X = 0

        # ── Animation state ───────────────────────────────────────────────
        self.playing = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._animate)

        # ── Display scaling ───────────────────────────────────────────────
        self.display_size = 512

        # ── Max project state ─────────────────────────────────────────────
        self.max_projected = False

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # ── Left Panel (Controls) ─────────────────────────────────────────
        ctrl_layout = QVBoxLayout()
        ctrl_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        title = QLabel("OME-TIFF Viewer")
        title.setFont(QFont("Helvetica", 14, QFont.Weight.Bold))
        ctrl_layout.addWidget(title)

        # File picker
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse)
        ctrl_layout.addWidget(browse_btn)
        
        self.path_input = QLineEdit()
        ctrl_layout.addWidget(self.path_input)
        
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._load_file)
        ctrl_layout.addWidget(load_btn)

        # Position selector (ND2 only, hidden for TIFs)
        self.pos_label = QLabel("Position")
        ctrl_layout.addWidget(self.pos_label)
        self.pos_combo = QComboBox()
        self.pos_combo.currentIndexChanged.connect(self._update_view)
        ctrl_layout.addWidget(self.pos_combo)
        self.pos_label.hide()
        self.pos_combo.hide()

        # Channel selector
        ctrl_layout.addWidget(QLabel("Channel"))
        self.channel_combo = QComboBox()
        self.channel_combo.currentIndexChanged.connect(self._update_view)
        ctrl_layout.addWidget(self.channel_combo)

        # T slider
        ctrl_layout.addWidget(QLabel("T"))
        self.t_slider = QSlider(Qt.Orientation.Horizontal)
        self.t_slider.valueChanged.connect(self._update_view)
        ctrl_layout.addWidget(self.t_slider)

        # Z slider
        ctrl_layout.addWidget(QLabel("Z"))
        self.z_slider = QSlider(Qt.Orientation.Horizontal)
        self.z_slider.valueChanged.connect(self._update_view)
        ctrl_layout.addWidget(self.z_slider)

        # Max project toggle
        self.max_proj_cb = QCheckBox("Max Z Project")
        self.max_proj_cb.toggled.connect(self._toggle_max_proj)
        ctrl_layout.addWidget(self.max_proj_cb)

        # Animation controls
        line1 = QFrame()
        line1.setFrameShape(QFrame.Shape.HLine)
        ctrl_layout.addWidget(line1)
        
        anim_layout = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        anim_layout.addWidget(self.play_btn)
        
        anim_layout.addWidget(QLabel("Delay (s):"))
        self.delay_spin = QDoubleSpinBox()
        self.delay_spin.setRange(0.1, 5.0)
        self.delay_spin.setSingleStep(0.1)
        self.delay_spin.setValue(0.5)
        anim_layout.addWidget(self.delay_spin)
        ctrl_layout.addLayout(anim_layout)

        # Encode MP4
        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.HLine)
        ctrl_layout.addWidget(line2)
        
        enc8_btn = QPushButton("Encode MP4 (8-bit)")
        enc8_btn.clicked.connect(lambda: self._encode_mp4(ten_bit=False))
        ctrl_layout.addWidget(enc8_btn)
        
        # enc10_btn = QPushButton("Encode MP4 (10-bit)")
        # enc10_btn.clicked.connect(lambda: self._encode_mp4(ten_bit=True))
        # ctrl_layout.addWidget(enc10_btn)

        # Status label
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: green;")
        self.status_label.setWordWrap(True)
        ctrl_layout.addWidget(self.status_label)

        # ── Right Panel (Image Display) ───────────────────────────────────
        disp_layout = QVBoxLayout()
        disp_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.header_label = QLabel()
        self.header_label.setFont(QFont("Helvetica", 12))
        disp_layout.addWidget(self.header_label)

        self.img_label = QLabel()
        disp_layout.addWidget(self.img_label)

        # Add layouts to main
        main_layout.addLayout(ctrl_layout)
        main_layout.addLayout(disp_layout, stretch=1)

    # ── File loading ──────────────────────────────────────────────────────

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select image file", "",
            "Image files (*.tif *.nd2);;TIFF files (*.tif);;ND2 files (*.nd2);;All files (*.*)"
        )
        if path:
            self.path_input.setText(path)
            self._load_file()

    def _load_file(self):
        path = self.path_input.text().strip()
        if not path:
            return
        if not Path(path).is_file():
            QMessageBox.critical(self, "Error", f"File not found:\n{path}")
            return

        self.status_label.setText("Loading...")
        QApplication.processEvents()

        if self.reader is not None:
            self.reader.close()

        ext = Path(path).suffix.lower()
        if ext == ".nd2":
            self.reader = LazyNd2Reader(path)
            self.channel_names = self.reader.channel_names
            self.P = self.reader.P
        else:
            self.reader = LazyTifReader(path)
            self.channel_names, _, _ = load_tif_metadata(path)
            self.P = 1

        self.T = self.reader.T
        self.C = self.reader.C
        self.Z = self.reader.Z
        self.Y = self.reader.Y
        self.X = self.reader.X

        # Position selector: show only for multi-position files
        self.pos_combo.blockSignals(True)
        self.pos_combo.clear()
        if self.P > 1:
            self.pos_combo.addItems([str(p) for p in range(self.P)])
            self.pos_label.show()
            self.pos_combo.show()
        else:
            self.pos_label.hide()
            self.pos_combo.hide()
        self.pos_combo.blockSignals(False)

        self.channel_combo.clear()
        self.channel_combo.addItems(self.channel_names)

        self.t_slider.setRange(0, max(0, self.T - 1))
        self.t_slider.setValue(0)

        self.z_slider.setRange(0, max(0, self.Z - 1))
        self.z_slider.setValue(0)

        self.max_proj_cb.setChecked(False)
        self.max_projected = False

        self.status_label.setText(
            f"Loaded: T={self.T}, C={self.C}, Z={self.Z}, Y={self.Y}, X={self.X}"
        )
        self._update_view()

    # ── Max project toggle ────────────────────────────────────────────────

    def _toggle_max_proj(self, checked):
        self.max_projected = checked
        self.z_slider.setEnabled(not self.max_projected)
        self._update_view()

    # ── Rendering ─────────────────────────────────────────────────────────

    def _get_slice(self, t, ch_idx, z=None):
        """Return a 2-D (Y, X) image for the given t, channel, and z."""
        p = max(0, self.pos_combo.currentIndex()) if self.P > 1 else 0
        if self.P > 1:
            frame = self.reader.read_frame(t, p)  # (C, Z, Y, X)
        else:
            frame = self.reader.read_frame(t)  # (C, Z, Y, X)
        if z is None:
            return frame[ch_idx].max(axis=0)
        return frame[ch_idx, z]

    def _update_view(self):
        if self.reader is None:
            return

        t = self.t_slider.value()
        ch_idx = max(0, self.channel_combo.currentIndex())
        z = None if self.max_projected else self.z_slider.value()
        
        # Guard against empty channels on load
        if not self.channel_names:
            return
        channel = self.channel_names[ch_idx]

        z_label = "max" if self.max_projected else str(z)
        header = f"T={t}  |  {channel}  |  Z={z_label}"
        if self.P > 1:
            header = f"P={self.pos_combo.currentIndex()}  |  " + header
        self.header_label.setText(header)

        img = self._get_slice(t, ch_idx, z)
        gray = auto_contrast_uint8(img)

        # Convert numpy array to QImage then QPixmap
        h, w = gray.shape
        q_img = QImage(gray.data, w, h, w, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img).scaled(
            self.display_size, self.display_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.img_label.setPixmap(pixmap)

    # ── Animation ─────────────────────────────────────────────────────────

    def _toggle_play(self):
        if self.reader is None:
            return
        self.playing = not self.playing
        self.play_btn.setText("Pause" if self.playing else "Play")
        
        if self.playing:
            delay_ms = int(self.delay_spin.value() * 1000)
            self.timer.start(delay_ms)
        else:
            self.timer.stop()

    def _animate(self):
        if self.reader is None:
            return
        t = (self.t_slider.value() + 1) % self.T
        self.t_slider.setValue(t)
        
        # Update timer interval dynamically in case user changed it while playing
        delay_ms = int(self.delay_spin.value() * 1000)
        if self.timer.interval() != delay_ms:
            self.timer.setInterval(delay_ms)

    # ── MP4 encoding ──────────────────────────────────────────────────────

    def _encode_mp4(self, ten_bit=False):
        if self.reader is None:
            QMessageBox.warning(self, "No data", "Load a file first.")
            return

        ch_idx = max(0, self.channel_combo.currentIndex())
        channel = self.channel_names[ch_idx]
        z = None if self.max_projected else self.z_slider.value()

        tif_path = Path(self.path_input.text())
        stem = tif_path.stem
        if stem.endswith(".ome"):
            stem = stem[:-4]
        z_tag = "maxZ" if z is None else f"z{z}"
        bit_tag = "10bit" if ten_bit else "8bit"
        out_path = tif_path.parent / f"{stem}_{channel}_{z_tag}_{bit_tag}.mp4"

        self.status_label.setText(f"Encoding {out_path.name} ...")
        QApplication.processEvents()

        if ten_bit:
            def frames():
                for t in range(self.T):
                    img = self._get_slice(t, ch_idx, z)
                    yield auto_contrast_uint16(img)

            # Pass the function reference `frames`, NOT the execution `frames()`
            encode_mp4_10bit(frames, out_path, self.X, self.Y)
        else:
            def frames():
                for t in range(self.T):
                    gray = auto_contrast_uint8(self._get_slice(t, ch_idx, z))
                    yield np.stack([gray, gray, gray], axis=-1)

            encode_mp4(frames(), out_path)

        self.status_label.setText(f"Saved {out_path.name}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = TifViewer()
    viewer.show()
    sys.exit(app.exec())
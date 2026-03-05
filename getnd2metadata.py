"""getnd2metadata.py — print all metadata from an ND2 file to stdout.

Exploratory script for inspecting a new dataset before running any pipeline
steps. Prints physical calibration via load_nd2_metadata (the same path used
by the pipeline so values are guaranteed to match), then opens the ND2 file
directly for fields that go beyond what the pipeline extracts: raw axis
sizes, dtype, excitation/emission wavelengths, the full event log, and the
raw OME metadata and experiment-loop structures.

Edit FILE_PATH before running.
"""

import nd2
from pathlib import Path

from useful_functions import load_nd2_metadata

# ── edit this ──────────────────────────────────────────────────────────────────
FILE_PATH = Path("nd1188.nd2")
# ──────────────────────────────────────────────────────────────────────────────

# ── pipeline metadata ──────────────────────────────────────────────────────────
# load_nd2_metadata extracts the three values the pipeline cares about without
# loading any image data. Printing them here confirms that what the pipeline
# will see matches what the microscope recorded.
channel_names, vox, period_s = load_nd2_metadata(FILE_PATH)

print("=" * 60)
print("PIPELINE METADATA  (via load_nd2_metadata)")
print("=" * 60)
print(f"  Channels:    {channel_names}")
print(f"  Voxel x:     {vox.x:.4f} µm/px")
print(f"  Voxel y:     {vox.y:.4f} µm/px")
print(f"  Voxel z:     {vox.z:.4f} µm/slice")
print(f"  Period:      {period_s} s" if period_s is not None else "  Period:      not a timelapse")

# ── raw ND2 exploration ────────────────────────────────────────────────────────
# The sections below open the ND2 file directly to access fields that the
# pipeline does not need but that are useful when inspecting a new dataset:
# total file size, per-axis dimension sizes, per-channel wavelengths, the
# event log showing per-frame stage positions and laser settings, and the
# raw metadata and experiment-loop structures from the ND2 SDK.
#
# try/except is intentionally absent here. If any section fails it is more
# useful to see the full traceback than a suppressed error string, because
# a failure usually means the ND2 SDK version or file format has changed and
# the whole pipeline needs attention.
with nd2.ND2File(FILE_PATH) as f:

    print("\n" + "=" * 60)
    print("BASIC INFO")
    print("=" * 60)
    print(f"  Path:      {f.path}")
    print(f"  Shape:     {f.shape}")
    print(f"  Axes:      {list(f.sizes.keys())}")
    print(f"  Dtype:     {f.dtype}")
    # nbytes is the uncompressed in-memory size, which is what matters for
    # deciding whether the full array will fit in RAM before calling load_nd2.
    print(f"  Size (GB): {f.nbytes / 1e9:.2f}")

    print("\n" + "=" * 60)
    print("DIMENSIONS")
    print("=" * 60)
    for ax, size in f.sizes.items():
        print(f"  {ax}: {size}")

    print("\n" + "=" * 60)
    print("CHANNEL DETAIL  (excitation / emission wavelengths)")
    print("=" * 60)
    for i, ch in enumerate(f.metadata.channels):
        print(
            f"  Channel {i}: {ch.channel.name}"
            f"  |  ex: {ch.channel.excitationLambdaNm} nm"
            f"  |  em: {ch.channel.emissionLambdaNm} nm"
        )

    print("\n" + "=" * 60)
    print("EVENT LOG  (first and last frame)")
    print("=" * 60)
    events = f.events()
    if events is not None and len(events) > 0:
        print(f"  Total events: {len(events)}")
        print(f"  First: {events[0]}")
        print(f"  Last:  {events[-1]}")
    else:
        print("  No events found")

    print("\n" + "=" * 60)
    print("FULL METADATA (raw ND2 SDK output)")
    print("=" * 60)
    print(f.metadata)

    print("\n" + "=" * 60)
    print("EXPERIMENT / LOOP INFO")
    print("=" * 60)
    print(f.experiment)

import nd2
import numpy as np

# ── change this to your actual file path ──────────────────────────────────────
FILE_PATH = "nd1188.nd2"
# ─────────────────────────────────────────────────────────────────────────────

with nd2.ND2File(FILE_PATH) as f:
    print("=" * 60)
    print("BASIC INFO")
    print("=" * 60)
    print(f"Path:         {f.path}")
    print(f"Shape:        {f.shape}")
    print(f"Axes:         {list(f.sizes.keys())}")
    print(f"Dtype:        {f.dtype}")
    print(f"Size (GB):    {f.nbytes / 1e9:.2f}")

    print("\n" + "=" * 60)
    print("DIMENSIONS")
    print("=" * 60)
    for ax, size in f.sizes.items():
        print(f"  {ax}: {size}")

    print("\n" + "=" * 60)
    print("CHANNEL INFO")
    print("=" * 60)
    try:
        for i, ch in enumerate(f.metadata.channels):
            print(f"  Channel {i}: {ch.channel.name}  |  excitation: {ch.channel.excitationLambdaNm} nm  |  emission: {ch.channel.emissionLambdaNm} nm")
    except Exception as e:
        print(f"  Could not parse channel info: {e}")

    print("\n" + "=" * 60)
    print("PIXEL CALIBRATION")
    print("=" * 60)
    try:
        vox = f.voxel_size()
        print(f"  x: {vox.x:.4f} um/px")
        print(f"  y: {vox.y:.4f} um/px")
        print(f"  z: {vox.z:.4f} um/px")
    except Exception as e:
        print(f"  Could not parse voxel size: {e}")

    print("\n" + "=" * 60)
    print("TIME INFO")
    print("=" * 60)
    try:
        times = f.events()
        if times is not None and len(times) > 0:
            print(f"  Number of time events: {len(times)}")
            print(f"  First event: {times[0]}")
            print(f"  Last event:  {times[-1]}")
        else:
            print("  No time events found")
    except Exception as e:
        print(f"  Could not parse time events: {e}")

    print("\n" + "=" * 60)
    print("FULL METADATA (raw)")
    print("=" * 60)
    try:
        print(f.metadata)
    except Exception as e:
        print(f"  Could not print metadata: {e}")

    print("\n" + "=" * 60)
    print("EXPERIMENT / LOOP INFO")
    print("=" * 60)
    try:
        print(f.experiment)
    except Exception as e:
        print(f"  Could not print experiment info: {e}")

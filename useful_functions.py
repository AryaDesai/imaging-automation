"""embryo_tools.py — shared library for the Ozbudak Lab ND2 imaging pipeline.

All functions needed by more than one script live here so that the logic is
defined and maintained in one place. The individual run-scripts
(centroid_align_xy.py, movie_from_nd2.py, find_threshold.py) are thin entry
points that handle argument parsing and I/O orchestration only.

A single flat file is used rather than a sub-package because the toolkit is
small enough that one file is easier to audit, share, and extend. If the
library grows substantially, splitting into sub-modules is straightforward
once the API has stabilised.
"""

import nd2
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, label, shift
from tqdm import tqdm


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_nd2(file_path):
    """Load a Nikon ND2 file and return the full raw array plus acquisition metadata.

    Returns the array as (T, P, Z, C, Y, X) float32. We deliberately keep all
    six axes rather than collapsing Z here because different consumers have
    different needs: centroid_align_xy.py works on the full 3-D volume per
    timepoint, while find_threshold.py and movie_from_nd2.py only need a
    max-projection. Callers that need a 2-D representation should pass the
    result to max_project_z() below.

    We cast to float32 immediately so that all arithmetic in downstream
    functions operates on a consistent numeric type. The raw ND2 values are
    typically uint16, but float32 avoids integer overflow when we do blurring
    or intensity scaling, and is still compact enough for large files (half the
    memory of float64).

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the .nd2 file.

    Returns
    -------
    data : ndarray, shape (T, P, Z, C, Y, X), dtype float32
    channel_names : list of str
        Human-readable names from the microscope metadata (e.g. "Venus",
        "Brightfield"). Preserved so output filenames and OME-TIFF metadata
        are readable without looking up index numbers.
    vox : VoxelSize
        Physical pixel size in µm (.x, .y, .z). Written into OME-TIFF output
        so downstream tools (Fiji, napari) can display images at correct scale.
    period_s : float or None
        Time between frames in seconds. None if the file has no TimeLoop block
        (i.e. it is not a timelapse acquisition).
    """
    f = nd2.ND2File(file_path)

    # asarray() loads the entire file into memory as a NumPy array.
    # The axis order (T, P, Z, C, Y, X) is the nd2 library's default for
    # multi-position timelapse experiments and is the convention assumed
    # throughout this pipeline.
    data = f.asarray()

    channel_names = [ch.channel.name for ch in f.metadata.channels]

    vox = f.voxel_size()

    # Extract acquisition period from the TimeLoop experiment block if present.
    # We convert from ms to s here so callers always work in SI units.
    # next(..., None) avoids a StopIteration exception when the block is absent.
    period_s = next(
        (loop.parameters.periodMs / 1000.0
         for loop in f.experiment if loop.type == "TimeLoop"),
        None,
    )

    # Close the file handle now that the array is in memory.
    # Leaving it open would hold a file lock for the duration of the script.
    f.close()

    return data.astype(np.float32), channel_names, vox, period_s


def max_project_z(data):
    """Collapse the Z axis of a (T, P, Z, C, Y, X) array by taking the maximum.

    Max-projection is the standard way to reduce a z-stack to a 2-D image for
    display and thresholding. The brightest voxel along Z represents the plane
    where the fluorescent signal is sharpest, which is what we want for both
    centroid detection and movie generation. Alternative projections (mean,
    sum) would dilute bright structures with out-of-focus background.

    We project over axis=2 because the axis order is (T, P, Z, C, Y, X) and Z
    is at position 2. The result has shape (T, P, C, Y, X) — Z is gone.

    Parameters
    ----------
    data : ndarray, shape (T, P, Z, C, Y, X)

    Returns
    -------
    projected : ndarray, shape (T, P, C, Y, X)
    """
    # np.max along axis=2 collapses Z while keeping all other axes intact.
    return data.max(axis=2)


# ── XY centroid detection ─────────────────────────────────────────────────────
#
# All functions in this section operate on 2-D (Y, X) images and return
# quantities in pixel coordinates (y, x). "XY" in the function names is
# intentional: we are detecting and correcting lateral drift only. Z-axis
# (focus) correction is a separate problem — it requires comparing sharpness
# metrics across Z-slices rather than intensity centroids — and is not
# addressed here.

def find_largest_mask_xy(smoothed, percentile):
    """Threshold a 2-D image and return the mask and centroid of the largest blob.

    This is the core detection step shared by find_threshold.py (for overlay
    visualisation) and centroid_align_xy.py (for computing shifts). Defining
    it once ensures that what the user sees in the Streamlit preview is exactly
    what the alignment script will detect.

    We use a percentile threshold rather than a fixed intensity value because
    ND2 files from different experiments have very different intensity ranges.
    The percentile is tuned interactively in find_threshold.py and saved to
    the YAML config.

    We take the *largest* connected component rather than all components above
    threshold because in a multi-embryo field there can be small bright
    artifacts or reflections. The embryo body is almost always the largest
    object in the frame.

    Both the mask and the centroid are returned because find_threshold.py needs
    the mask for the red overlay visualisation, while centroid_align_xy.py only
    needs the centroid coordinates. Returning both avoids running the
    connected-component labelling twice when both are needed.

    Parameters
    ----------
    smoothed : ndarray, shape (Y, X)
        Gaussian-blurred 2-D image. The caller is responsible for blurring
        before calling this function so that the blur radius can be tuned
        independently from the threshold percentile.
    percentile : float
        Pixels above this percentile of the image are included in the binary
        mask. Typical values are 85–95 depending on how bright the embryo is
        relative to background.

    Returns
    -------
    mask : ndarray of bool, shape (Y, X)
        True where the pixel belongs to the largest connected component.
    centroid : ndarray, shape (2,)
        [cy, cx] — row, column coordinates of the centroid. If no component
        is found (e.g. a blank frame or threshold too high), returns the frame
        centre so that the downstream shift is zero and the frame is unchanged.
    """
    # Binarise by thresholding at the given percentile of the smoothed image.
    # Using the image's own percentile makes the threshold adaptive to
    # per-frame intensity variation rather than requiring a fixed absolute value.
    binary = smoothed > np.percentile(smoothed, percentile)

    # Label connected components. Default 4-connectivity is sufficient for
    # detecting embryo blobs; 8-connectivity would merge diagonally adjacent
    # objects but is not needed here.
    labeled, _ = label(binary)

    # Count pixels per component. Component 0 is background; zeroing it out
    # ensures argmax never selects the background as the "largest component".
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0

    if sizes.max() == 0:
        # No foreground component found — can happen in very early/late frames
        # if the embryo is out of focus or if the threshold is too aggressive.
        # Returning the frame centre means the downstream shift will be (0, 0),
        # leaving the frame unchanged rather than crashing or producing garbage.
        cy, cx = smoothed.shape[0] / 2, smoothed.shape[1] / 2
        mask = np.zeros(smoothed.shape, dtype=bool)
        return mask, np.array([cy, cx])

    # Build a boolean mask containing only the largest component.
    mask = labeled == sizes.argmax()

    # Centroid = mean (row, col) position of all True pixels.
    # np.argwhere returns the indices of True elements as an (N, 2) array,
    # so .mean(axis=0) gives [mean_row, mean_col] = [cy, cx].
    centroid = np.argwhere(mask).mean(axis=0)

    return mask, centroid


def compute_shift_xy(frame, sigma, percentile, ch_idx):
    """Return the (dy, dx) translation that moves the embryo centroid to the frame centre.

    This function answers: "by how many pixels must we shift this frame so
    that the embryo ends up centred?" It does not apply the shift — that is
    done by align_frame_xy(), or directly with scipy.ndimage.shift in the
    --enlarge_canvas two-pass workflow.

    Separating shift computation from application is important for the
    --enlarge_canvas workflow in centroid_align_xy.py, where all shifts across
    all timepoints must be known before the canvas can be padded. Fusing
    computation and application would require either a second pass over the
    data or storing large intermediate arrays.

    Parameters
    ----------
    frame : ndarray, shape (C, Z, Y, X)
        A single timepoint for a single embryo, with all channels and Z-slices.
    sigma : float
        Gaussian blur radius in pixels. Higher values smooth over noise but
        reduce sensitivity to fine embryo boundary details. Typical values
        are 15–40 for PSM imaging.
    percentile : float
        Threshold percentile passed to find_largest_mask_xy.
    ch_idx : int
        Index of the channel used for centroid detection (e.g. Venus = 0).
        We use a single channel for detection so that the computed shift is
        consistent across all channels; using different channels per frame
        would produce inconsistent shifts.

    Returns
    -------
    dy : float
        Pixels to shift in Y. Positive = move image content downward
        (scipy convention: positive shift moves the array values in the
        positive-index direction).
    dx : float
        Pixels to shift in X. Positive = move image content rightward.
    """
    Y, X = frame.shape[2], frame.shape[3]

    # Max-project the detection channel over Z to collapse to a 2-D image.
    # We project only the one detection channel here rather than the whole
    # frame to keep this function fast; the other channels are not needed
    # for centroid detection.
    projection = frame[ch_idx].max(axis=0)  # shape (Y, X)

    # Gaussian blur suppresses noise and isolated bright spots that could
    # pull the centroid away from the embryo body.
    smoothed = gaussian_filter(projection, sigma=sigma)

    # Detect the largest blob and get its centroid in pixel coordinates.
    # The mask is discarded here; only the centroid coordinates are needed.
    _, centroid = find_largest_mask_xy(smoothed, percentile)
    cy, cx = centroid[0], centroid[1]

    # The target position is the frame centre (Y/2, X/2).
    # dy > 0 when the embryo is above centre → we shift the image downward.
    # dy < 0 when the embryo is below centre → we shift the image upward.
    dy = Y / 2 - cy
    dx = X / 2 - cx

    # Always print the computed shift so any caller — whether a batch script,
    # a real-time autofocus loop, or an interactive notebook — automatically
    # gets a record of what the centroid detector decided, without needing to
    # add logging at every call site.
    # tqdm.write is used instead of print so that this output does not
    # visually corrupt any tqdm progress bar that may be running in the caller.
    # If no tqdm bar is active, tqdm.write behaves identically to print.
    # Format: +7.2f means always show the sign (+ or -), right-pad to 7
    # characters wide so columns line up when many lines are printed, and
    # show 2 decimal places.
    tqdm.write(f"dy={dy:+7.2f}  dx={dx:+7.2f}")

    return dy, dx


def align_frame_xy(frame, sigma, percentile, ch_idx):
    """Compute the centring shift and apply it to all channels and Z-slices.

    This is the single-pass alignment function used in the default (no
    --enlarge_canvas) workflow. It wraps compute_shift_xy and the scipy shift
    call into one step so callers do not have to manage the shift value
    separately when they do not need to inspect it before applying it.

    All channels and Z-slices are shifted by the same (dy, dx) because lateral
    embryo position is the same in every channel and Z-slice within one
    timepoint. Shifting them together keeps channels in registration with each
    other, which matters for any downstream co-localisation analysis.

    We use order=1 (bilinear) interpolation. order=0 (nearest-neighbour) is
    faster but creates staircase artefacts on diagonal edges. order=3 (cubic)
    is smoother but noticeably slower, and the improvement is not visible at
    typical display resolutions or after MP4 compression.

    Parameters
    ----------
    frame : ndarray, shape (C, Z, Y, X)
    sigma, percentile, ch_idx : see compute_shift_xy

    Returns
    -------
    shifted : ndarray, shape (C, Z, Y, X)
        Frame shifted so the embryo centroid is at the frame centre.
        Pixels that shift outside the original canvas boundary are filled
        with 0 (black), which is visually unambiguous and does not introduce
        false fluorescence signal.
    dy, dx : float
        The shift that was applied. Returned so the caller can log it.
    """
    dy, dx = compute_shift_xy(frame, sigma, percentile, ch_idx)

    # Apply the same (dy, dx) shift to every channel and Z-slice simultaneously.
    # The (0, 0) entries for the C and Z axes ensure those axes are untouched.
    # mode="constant", cval=0 fills newly exposed border pixels with black.
    shifted = shift(frame, (0, 0, dy, dx), order=1, mode="constant", cval=0)

    return shifted, dy, dx


# ── Format conversion ─────────────────────────────────────────────────────────

def save_ome_tiff(filepath, volume, channel_names, vox, period_s):
    """Write a (T, C, Z, Y, X) float32 volume as an OME-TIFF with full physical metadata.

    OME-TIFF is chosen as the output format because it embeds physical pixel
    size, channel names, and acquisition timing in a standardised XML block
    that Fiji (Bio-Formats) and napari can read without any manual calibration.
    A plain TIFF would require the user to enter these values by hand every
    time they open the file.

    The function takes a filepath rather than a directory + naming components
    so that it is not coupled to any particular filename convention. The caller
    decides the name; this function only handles the writing.

    Parameters
    ----------
    filepath : str
        Full destination path including filename, e.g.
        "/data/aligned_nd1188/nd1188_P0.ome.tif".
    volume : ndarray, shape (T, C, Z, Y, X), dtype float32
        The aligned image stack for one embryo position.
    channel_names : list of str
        Channel names in the same order as the C axis (e.g. ["Venus", "BF"]).
        Written into the OME metadata so channels are labelled correctly in
        Fiji's channel manager.
    vox : VoxelSize
        Physical voxel size from load_nd2 (.x, .y, .z in µm). Written so
        downstream tools display images at the correct physical scale rather
        than in arbitrary pixel units.
    period_s : float or None
        Time between frames in seconds from load_nd2. Written as TimeIncrement
        so the time axis is correctly calibrated. None is safe to pass — Fiji
        will simply leave the time axis uncalibrated.
    """
    # photometric="minisblack" tells readers that low values = black (dark),
    # high values = bright signal. This is the correct convention for
    # fluorescence microscopy; the alternative "miniswhite" would invert the LUT.
    tifffile.imwrite(
        filepath,
        volume,
        photometric="minisblack",
        metadata={
            # "axes" tells Bio-Formats the axis order of the array so it does
            # not have to guess. TCZYX is the standard OME axis order.
            "axes": "TCZYX",
            "PhysicalSizeX": vox.x, "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": vox.y, "PhysicalSizeYUnit": "µm",
            "PhysicalSizeZ": vox.z, "PhysicalSizeZUnit": "µm",
            "TimeIncrement": period_s, "TimeIncrementUnit": "s",
            # Channel names are passed as a dict so tifffile formats them
            # into the OME-XML <Channel Name="..."> attribute.
            "Channel": {"Name": channel_names},
        },
    )


# ── Display utilities ─────────────────────────────────────────────────────────

def auto_contrast(img, percentile=99.5):
    """Scale a float32 image to uint8 by clipping at a high percentile.

    A fixed 0-to-max normalisation would make dim frames appear very dark
    whenever a single hot pixel or camera artefact inflates the maximum.
    Clipping at the 99.5th percentile means at most 0.5% of pixels are
    saturated, which preserves perceptual contrast across frames with varying
    peak intensities.

    We use 99.5 rather than 100 (the true maximum) because cosmic ray hits
    and readout noise can create single-pixel outliers that would otherwise
    dominate the normalisation. 99.5 is a conventional value in fluorescence
    microscopy display pipelines.

    Parameters
    ----------
    img : ndarray, shape (Y, X), dtype float32
    percentile : float, default 99.5

    Returns
    -------
    ndarray, shape (Y, X), dtype uint8
    """
    vmax = np.percentile(img, percentile)

    # Guard against all-zero frames (blank channels, failed acquisition, or
    # frames that are entirely background after alignment padding).
    # Division by zero would produce NaN; setting vmax=1 gives a black frame,
    # which is the correct representation of "no signal present".
    if vmax == 0:
        vmax = 1

    # Clip to [0, 255] before casting to uint8. Without the clip, values
    # slightly above vmax would wrap around to 0 in uint8 arithmetic, creating
    # spurious black pixels at the brightest spots.
    return np.clip(img / vmax * 255, 0, 255).astype(np.uint8)


def make_grid_frame(images, nrows=2, ncols=2):
    """Tile a list of 2-D uint8 images into a single nrows×ncols grid image.

    We use a 2×2 grid because the microscope captures 4 embryo positions
    arranged in a 2×2 physical layout on the dish. Preserving this spatial
    arrangement in the output movie makes it straightforward to correlate
    features in the movie with their physical position.

    Images are placed in row-major order (left-to-right, top-to-bottom).
    If fewer than nrows*ncols images are provided (e.g. a run with only 3
    embryos), the remaining cells are left black (zero), which is unambiguous
    and avoids index-out-of-range errors.

    Parameters
    ----------
    images : list of ndarray, each shape (Y, X), dtype uint8
        All images must have the same (Y, X) dimensions.
    nrows, ncols : int, default 2
        Grid dimensions. Change these if the experiment uses a different
        number of embryo positions (e.g. a 1×4 strip or a 3×3 array).

    Returns
    -------
    grid : ndarray, shape (nrows*Y, ncols*X), dtype uint8
    """
    H, W = images[0].shape

    # Pre-allocate with zeros so unfilled cells appear black rather than
    # containing uninitialised memory values.
    grid = np.zeros((nrows * H, ncols * W), dtype=np.uint8)

    for i, img in enumerate(images):
        # divmod maps the flat index i to (row, col) in the grid,
        # placing images in row-major (reading) order.
        r, c = divmod(i, ncols)
        grid[r * H : (r + 1) * H, c * W : (c + 1) * W] = img

    return grid

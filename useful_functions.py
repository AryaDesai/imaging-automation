"""

All functions needed by more than one script live here so that the logic is
defined and maintained in one place. The individual run-scripts
(centroid_align_xy.py, movie_from_nd2.py, find_threshold.py) are thin entry
points that handle argument parsing and I/O only.

A single flat file is used rather than a sub-package because the toolkit is
small enough that one file is easier to audit, share, and extend. If the
library grows substantially, splitting into sub-modules is straightforward
once the API has stabilised.
"""

from pathlib import Path
import subprocess
import nd2
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, label, shift
from tqdm import tqdm


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_nd2(file_path):
    """Load a Nikon ND2 file and return the full raw array plus acquisition metadata.

    Returns the array as (T, P, Z, C, Y, X) in the file's native dtype (usually uint16).
    We deliberately keep all six axes rather than collapsing Z here because
    different consumers have different needs: centroid_align_xy.py works on
    the full 3-D volume per timepoint, while find_threshold.py and
    movie_from_nd2.py only need a max-projection. Callers that need a 2-D
    representation should pass the result to max_project_z() below.

    We return the native dtype to save memory and avoid premature casting.
    Callers should cast to float32 if they need to perform floating-point
    arithmetic (e.g. for Gaussian smoothing). The raw ND2 values are typically
    uint16.

    Parameters
    ----------
    file_path : str or Path
        Absolute or relative path to the .nd2 file.

    Returns
    -------
    data : ndarray, shape (T, P, Z, C, Y, X), dtype (native, usually uint16)
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

    return data, channel_names, vox, period_s


def load_nd2_metadata(file_path):
    """Load channel names, voxel size, and acquisition period from an ND2 file
    without reading any image data into memory.

    load_nd2 calls f.asarray() which loads the entire image volume — up to
    several gigabytes for a typical timelapse. Scripts that only need metadata
    (channel names, physical voxel size, time interval) should call this
    function instead to avoid that cost. The z-alignment script, for example,
    reads image data from already-written OME-TIFFs and only needs the ND2
    for the physical calibration values that were stored there at acquisition
    time.

    The return signature is a subset of load_nd2 (minus the data array) so
    that callers can switch between the two functions without restructuring
    their unpacking.

    Parameters
    ----------
    file_path : str or Path
        Path to the .nd2 file.

    Returns
    -------
    channel_names : list of str
    vox           : VoxelSize  (.x, .y, .z in µm)
    period_s      : float or None
    """
    f = nd2.ND2File(file_path)

    channel_names = [ch.channel.name for ch in f.metadata.channels]

    vox = f.voxel_size()

    # Same TimeLoop extraction as load_nd2 — see that function for rationale.
    period_s = next(
        (loop.parameters.periodMs / 1000.0
         for loop in f.experiment if loop.type == "TimeLoop"),
        None,
    )

    # Close immediately — we never called asarray() so no image data was
    # loaded, but the file handle must still be released.
    f.close()

    return channel_names, vox, period_s


def load_tif_metadata(file_path):
    """Load channel names, voxel size, and acquisition period from an
    OME-TIFF written by save_ome_tiff, without reading any image data.

    Reads the JSON stored in the ImageDescription tag by save_ome_tiff.
    Returns the same (channel_names, vox, period_s) signature as
    load_nd2_metadata so callers can switch between the two without
    restructuring their unpacking.

    Parameters
    ----------
    file_path : str or Path

    Returns
    -------
    channel_names : list of str
    vox           : SimpleNamespace  (.x, .y, .z in µm)
    period_s      : float or None
    """
    import json
    import xml.etree.ElementTree as ET
    from types import SimpleNamespace

    with tifffile.TiffFile(file_path) as tif:
        C = tif.series[0].shape[1]  # TCZYX — C is axis 1
        try:
            # save_ome_tiff stores the metadata dict as JSON in the ImageDescription
            # tag of the first page. tif.pages[0].description decodes that tag as a string.
            meta = json.loads(tif.pages[0].description)
            channel_names = meta["Channel"]["Name"]
            # SimpleNamespace provides the .x, .y, .z attribute interface that save_ome_tiff expects.
            vox = SimpleNamespace(
                x=meta["PhysicalSizeX"],
                y=meta["PhysicalSizeY"],
                z=meta["PhysicalSizeZ"],
            )
            period_s = meta.get("TimeIncrement")
        except Exception as e:
            print(f"  Could not read save_ome_tiff JSON metadata ({e}), trying OME-XML ...")
            try:
                # tif.ome_metadata returns the raw OME-XML string embedded by external
                # software (e.g. Fiji, Imaris). Parse it with the stdlib XML parser.
                root = ET.fromstring(tif.ome_metadata)
                # OME-XML tags are namespace-qualified, e.g.
                # {http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels.
                # Extract the namespace URI from the root tag so XPath queries work
                # regardless of which OME schema version the file uses.
                ns = root.tag.split('}')[0].lstrip('{') if '}' in root.tag else ''
                # Find the Pixels element anywhere in the tree using a namespace-aware path.
                p = root.find(f'.//{{{ns}}}Pixels')
                channel_names = [
                    ch.get('Name', f'ch{i}')
                    for i, ch in enumerate(p.findall(f'{{{ns}}}Channel'))
                ]
                # SimpleNamespace provides the .x, .y, .z attribute interface that save_ome_tiff expects.
                vox = SimpleNamespace(
                    x=float(p.get('PhysicalSizeX')),
                    y=float(p.get('PhysicalSizeY')),
                    z=float(p.get('PhysicalSizeZ')),
                )
                # TimeIncrement is optional in OME-XML; absent means no timelapse.
                period_s = float(p.get('TimeIncrement')) if p.get('TimeIncrement') else None
            except Exception as e2:
                print(f"  Could not read OME-XML metadata ({e2}), using dummy values.")
                channel_names = [f"ch{i}" for i in range(C)]
                vox = SimpleNamespace(x=1.0, y=1.0, z=1.0)
                period_s = None

    return channel_names, vox, period_s


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


def compute_shift_xy_gpu(frame, sigma, percentile, ch_idx):
    """GPU-accelerated version of compute_shift_xy.

    Gaussian blur runs on GPU; connected-component labeling stays on CPU.
    Parameters are identical to compute_shift_xy.
    """
    Y, X = frame.shape[2], frame.shape[3]

    # Move the entire frame to GPU once at the start.
    t_frame = torch.from_numpy(frame).to("mps")  # (C, Z, Y, X)

    # Extract the detection channel and max-project over Z to get a 2-D image.
    projection = t_frame[ch_idx].max(dim=0).values  # (Y, X)

    # Gaussian blur suppresses noise and isolated bright spots that could
    # pull the centroid away from the embryo body. conv2d expects (N, C_in, H, W).
    kernel = _gaussian_kernel_2d(sigma, device="mps")
    padding = kernel.shape[-1] // 2
    proj_4d = projection.reshape(1, 1, Y, X)
    smoothed_gpu = F.conv2d(proj_4d, kernel, padding=padding)

    # Squeeze to (Y, X) and pull to CPU for connected-component labeling.
    smoothed = smoothed_gpu.squeeze().cpu().numpy()

    _, centroid = find_largest_mask_xy(smoothed, percentile)
    cy, cx = centroid[0], centroid[1]

    # dy > 0 when the embryo is above centre → shift image downward.
    # dy < 0 when the embryo is below centre → shift image upward.
    dy = Y / 2 - cy
    dx = X / 2 - cx
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


def _gaussian_kernel_2d(sigma, device="mps"):
    """Build a normalised 2D Gaussian kernel as a torch tensor.

    Kernel radius is 4*sigma (rounded up), matching scipy's default truncate=4.
    """
    radius = int(np.ceil(sigma * 4))
    size = 2 * radius + 1
    coords = torch.arange(size, dtype=torch.float32, device=device) - radius
    g1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    kernel = g1d[:, None] * g1d[None, :]
    kernel /= kernel.sum()
    # conv2d expects (out_channels, in_channels, height, width).
    # Single input and output channel since we blur one grayscale image.
    return kernel.reshape(1, 1, size, size)


def align_frame_xy_gpu(frame, sigma, percentile, ch_idx):
    """GPU-accelerated version of align_frame_xy.

    Gaussian blur and integer-pixel shift run on GPU. Connected-component
    labeling (find_largest_mask_xy) stays on CPU because scipy.ndimage.label
    has no torch equivalent.

    Returns the shifted frame as a torch tensor on the GPU device.
    dy and dx are the raw centroid offsets before rounding.

    Parameters are identical to align_frame_xy.
    """
    C, Z, Y, X = frame.shape

    # Move entire frame to GPU once; it stays there until the final result.
    t_frame = torch.from_numpy(frame).to("mps")  # (C, Z, Y, X)

    # Max-project the detection channel over Z to get a 2D image.
    projection = t_frame[ch_idx].max(dim=0).values  # (Y, X)

    # Gaussian blur on GPU. conv2d expects (N, C_in, H, W).
    kernel = _gaussian_kernel_2d(sigma, device="mps")
    padding = kernel.shape[-1] // 2  # same-size output
    proj_4d = projection.reshape(1, 1, Y, X)
    smoothed_gpu = F.conv2d(proj_4d, kernel, padding=padding)

    # Pull the small blurred 2D image to CPU for connected-component labeling.
    smoothed = smoothed_gpu.squeeze().cpu().numpy()  # (Y, X)

    _, centroid = find_largest_mask_xy(smoothed, percentile)
    cy, cx = centroid[0], centroid[1]

    dy = Y / 2 - cy
    dx = X / 2 - cx
    tqdm.write(f"dy={dy:+7.2f}  dx={dx:+7.2f}")

    # Round to integer pixels — sub-pixel interpolation is not needed here.
    idy = int(round(dy))
    idx = int(round(dx))

    # Pad zeros on the side content shifts away from, then slice back to
    # original size. Exposed edges become zero, no wrapping occurs.
    padded = F.pad(t_frame, (max(idx, 0), max(-idx, 0), max(idy, 0), max(-idy, 0)))
    shifted_gpu = padded[:, :, max(-idy, 0):max(-idy, 0)+Y, max(-idx, 0):max(-idx, 0)+X]

    return shifted_gpu, dy, dx


def apply_shift_xy_gpu(frame, dy, dx):
    """Apply a precomputed (dy, dx) shift to a frame on GPU.

    Used in the --enlarge_canvas two-pass workflow, where shifts are
    precomputed in Pass 1 and applied in Pass 2.

    The shift is rounded to integer pixels, consistent with align_frame_xy_gpu.

    Parameters
    ----------
    frame : ndarray, shape (C, Z, Y, X)
        A single timepoint for a single embryo.
    dy : float
        Shift in Y (positive = move content downward).
    dx : float
        Shift in X (positive = move content rightward).

    Returns
    -------
    shifted_gpu : torch.Tensor, shape (C, Z, Y, X), on GPU
        Caller is responsible for moving to CPU when done.
    """
    C, Z, Y, X = frame.shape

    t_frame = torch.from_numpy(frame).to("mps")  # (C, Z, Y, X)

    idy = int(round(dy))
    idx = int(round(dx))

    # Pad zeros on the side content shifts away from, then slice back to
    # original size. Mirrors the same operation in align_frame_xy_gpu.
    padded = F.pad(t_frame, (max(idx, 0), max(-idx, 0), max(idy, 0), max(-idy, 0)))
    shifted_gpu = padded[:, :, max(-idy, 0):max(-idy, 0)+Y, max(-idx, 0):max(-idx, 0)+X]

    return shifted_gpu


# ── Z centroid detection ──────────────────────────────────────────────────────
#
# These functions detect and correct drift along the optical (Z) axis.
# The approach mirrors the XY centroid pipeline: compute a scalar position
# estimate from the signal, compare it to a reference, and shift the data
# to compensate.
#
# Unlike XY drift, which is corrected by a continuous sub-pixel shift, Z
# correction uses an integer slice shift because the Z axis is discretely
# sampled (one confocal plane per slice) and sub-slice interpolation along Z
# would resample across planes that were physically acquired independently.
# The integer shift is rounded from a float centroid, so the detection
# precision is still sub-slice even though the applied correction is not.
#
# The float centroid is also converted to physical units (µm) and returned
# so that callers driving automated stage control can command the objective
# to move by the exact physical amount rather than a rounded slice count.
#
# Detection channel: the diagnostic in z_diagnostic.py established that the
# Venus channel (clock gene) gives the most reliable Z centroid across all
# four embryo positions. mCherry (nuclear marker) signal is concentrated at
# the top of the Z range and shows no useful Z-dependent variation. TD
# (transmission) metrics are noisier than Venus. Venus mean intensity per
# Z-slice is therefore the detection signal used throughout this section.

def compute_z_profile(frame, ch_idx):
    """Return the mean intensity per Z-slice for the chosen channel.

    Averaging over all pixels in each (Y, X) plane produces a 1-D profile
    of signal strength versus Z position. This profile is the input to
    compute_centroid_z and captures how the fluorescent signal is distributed
    along the optical axis.

    We use mean rather than sum so that the profile values are independent
    of image size and can be compared across experiments with different
    field-of-view dimensions.

    Parameters
    ----------
    frame : ndarray, shape (C, Z, Y, X), float32
        A single timepoint for a single embryo position.
    ch_idx : int
        Index of the channel to use for Z detection. The z_diagnostic.py
        analysis identified Venus (index 0) as the most reliable choice.

    Returns
    -------
    profile : ndarray, shape (Z,), float64
        Mean pixel intensity at each Z-slice.
    """
    # Select the detection channel and average over the spatial axes (Y, X).
    # axis=(1, 2) collapses Y and X simultaneously, leaving one value per
    # Z-slice. We compute over the full (Y, X) frame rather than a subregion
    # because the Venus signal fills the embryo body and using the full frame
    # gives a more stable average than any manually chosen subregion.
    return frame[ch_idx].mean(axis=(1, 2))


def compute_centroid_z(profile):
    """Return the intensity-weighted mean Z position (centroid) of a Z profile.

    The centroid is the continuous analogue of argmax: instead of returning
    the single slice with the highest intensity, it returns the weighted
    average Z position, which is more stable when the peak is broad or when
    two adjacent slices have similar intensities.

    Before computing the weighted average, the minimum value of the profile
    is subtracted from every element. This removes any uniform background
    floor — signal that is present at the same level across all Z-slices and
    therefore carries no information about the embryo's Z position. Without
    this subtraction, a high background would pull the centroid toward Z/2
    regardless of where the actual signal peak is.

    If the background-subtracted profile is all zeros (blank frame or
    signal entirely below background level), the centroid falls back to
    the midpoint of the Z range. This keeps the downstream shift at zero
    rather than producing a NaN or an extreme value that would corrupt the
    alignment.

    Parameters
    ----------
    profile : ndarray, shape (Z,)
        Mean intensity per Z-slice, as returned by compute_z_profile.

    Returns
    -------
    centroid : float
        Intensity-weighted mean Z position in slice units. Can be fractional.
    """
    # Subtract the minimum to remove background before computing the centroid.
    # The minimum is the baseline signal present even in out-of-signal slices;
    # only the excess above this baseline reflects actual Z-localised signal.
    above_background = profile - profile.min()

    total = above_background.sum()

    if total == 0:
        # All slices are equally bright (or all zero) — the profile carries
        # no Z position information. Returning the midpoint means the computed
        # shift will be zero, leaving the frame unchanged rather than crashing.
        return len(profile) / 2.0

    # np.arange gives the slice index for each element of the profile.
    # The weighted sum (index * weight) / total_weight is the standard
    # formula for centre of mass, applied here along the Z axis.
    z_indices = np.arange(len(profile), dtype=float)
    return float((above_background * z_indices).sum() / total)


def compute_shift_z(frame, ch_idx, reference_centroid, vox_z):
    """Return the integer Z shift and its physical equivalent in micrometres.

    Computes the current Z centroid of the embryo, compares it to the
    reference centroid established at t=0, and returns the correction needed
    to restore alignment.

    Two values are returned because they serve different purposes:
      dz_slices — the integer shift applied to the image data in post-
                  processing. Rounded from the float centroid difference so
                  that the correction is always a whole number of slices.
      dz_um     — the exact physical correction in micrometres, derived from
                  the un-rounded float centroid difference multiplied by the
                  Z voxel size. Used by automated acquisition pipelines to
                  command the microscope stage by the precise physical amount
                  rather than the nearest slice boundary.

    The sign convention matches scipy.ndimage.shift: a positive dz_slices
    moves image content toward higher Z indices (upward in the stack), and
    a negative dz_slices moves it toward lower Z indices.

    Parameters
    ----------
    frame : ndarray, shape (C, Z, Y, X), float32
        Current timepoint for one embryo position.
    ch_idx : int
        Channel index for Z detection (Venus = 0).
    reference_centroid : float
        Z centroid at t=0, in slice units. Stored by the calling script
        on the first timepoint and passed in on all subsequent timepoints.
    vox_z : float
        Physical Z voxel size in µm/slice, from load_nd2 vox.z. Used to
        convert the slice-unit centroid difference to micrometres.

    Returns
    -------
    dz_slices : int
        Integer number of slices to shift. Positive = toward higher Z.
    dz_um : float
        Physical correction in µm (un-rounded, for stage control).
    """
    profile          = compute_z_profile(frame, ch_idx)
    current_centroid = compute_centroid_z(profile)

    # The raw float difference gives the exact drift in slice units.
    # Rounding to the nearest integer gives the shift we can apply to
    # the discrete Z stack without interpolation.
    dz_float  = reference_centroid - current_centroid
    dz_slices = int(round(dz_float))

    # Multiply the un-rounded float by vox_z so the physical correction
    # preserves sub-slice precision for stage control. Using dz_float
    # rather than dz_slices here avoids accumulating rounding error when
    # this value is fed to a stage controller over many timepoints.
    dz_um = dz_float * vox_z

    # tqdm.write is used instead of print so this line does not visually
    # corrupt any active tqdm progress bar in the calling script.
    tqdm.write(f"dz={dz_slices:+d} slices  ({dz_um:+.2f} µm)")

    return dz_slices, dz_um


def align_frame_z(frame, dz_slices):
    """Shift a (C, Z, Y, X) frame along the Z axis by an integer number of slices.

    Moves all channels together by the same dz_slices so that every channel
    remains in Z-registration with the others after correction — the same
    reason all channels are shifted together in align_frame_xy.

    The shift is implemented with array slicing rather than scipy.ndimage.shift
    because Z correction is always an integer number of slices. scipy shift
    would apply interpolation along Z even with an integer shift value, which
    would mix signal from adjacent confocal planes that were physically
    acquired independently. Array slicing moves whole planes without any
    resampling.

    Slices that shift outside the original Z range are replaced with zeros.
    This is the programmatic equivalent of the blank frames inserted manually
    when scrolling through Z to track drifting cells.

    Parameters
    ----------
    frame : ndarray, shape (C, Z, Y, X), float32
        Single timepoint for one embryo position.
    dz_slices : int
        Number of slices to shift. Positive = move content toward higher Z
        indices (slices at the low end become zero). Negative = move content
        toward lower Z indices (slices at the high end become zero).

    Returns
    -------
    shifted : ndarray, shape (C, Z, Y, X), float32
        Z-shifted copy of the input frame. Input is not modified.
    """
    if dz_slices == 0:
        # No shift needed — return a copy for consistency with the non-zero
        # case so callers can always treat the return value as a new array.
        return frame.copy()

    Z = frame.shape[1]

    # Pre-fill with zeros so that any slice position not written by the
    # copy below is automatically zero-padded rather than uninitialised.
    shifted = np.zeros_like(frame)

    # Clamp the shift magnitude to Z so that an over-large shift (more slices
    # than the stack has) produces an all-zero output rather than an index
    # error. In normal use dz_slices << Z because the user monitors the
    # acquisition, but this guard prevents a crash if a corrupt centroid
    # estimate produces an extreme value.
    dz = min(abs(dz_slices), Z)

    if dz_slices > 0:
        # Positive shift: content moves toward higher Z indices.
        # frame[:, 0:Z-dz, :, :] (the portion that stays in frame)
        # is placed at shifted[:, dz:Z, :, :].
        # The first dz slices of shifted remain zero (the newly exposed
        # low end of the stack).
        shifted[:, dz:, :, :] = frame[:, :Z - dz, :, :]
    else:
        # Negative shift: content moves toward lower Z indices.
        # frame[:, dz:Z, :, :] is placed at shifted[:, 0:Z-dz, :, :].
        # The last dz slices of shifted remain zero (the newly exposed
        # high end of the stack).
        shifted[:, :Z - dz, :, :] = frame[:, dz:, :, :]

    return shifted


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
    filepath : str or Path
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



def concatenate_tifs(tif_paths, output_path, axis=0):
    """Concatenate multiple OME-TIFFs along a chosen axis.

    Joins a set of OME-TIFF files by concatenating along the specified
    axis (default 0, which is T in TCZYX volumes). Input files are sorted
    by filename before concatenation — ND2 files are named sequentially
    by the microscope (nd1188, nd1189, ...), so alphabetical order
    corresponds to acquisition time order.

    All dimensions except the concatenation axis must match across files.
    A mismatch is raised as an error.

    OME metadata is compared across files. Inconsistencies are printed as
    warnings but do not prevent concatenation. The OME-XML from the first
    file is passed through to the output.

    Data is streamed to the output one input file at a time using
    tifffile's append mode, so only one input file's data is in memory
    at any point.

    Parameters
    ----------
    tif_paths : list of str or Path
        Paths to OME-TIFF files.
    output_path : str or Path
        Destination path for the concatenated OME-TIFF.
    axis : int, default 0
        Axis along which to concatenate. 0=T, 1=C, 2=Z, 3=Y, 4=X for
        TCZYX volumes.
    """
    # Sort by filename so that files from sequentially named ND2s
    # (nd1188, nd1189, ...) are concatenated in acquisition order.
    # Alphabetical sorting works because the microscope assigns ascending
    # numeric suffixes within each session.
    sorted_paths = sorted(tif_paths, key=lambda p: Path(p).name)

    # Collect array shapes and OME-XML metadata from each file before
    # loading any pixel data. This pass is used to verify that all files
    # have compatible dimensions and consistent metadata before committing
    # to the concatenation.
    shapes = []
    ome_xmls = []
    for path in sorted_paths:
        # TiffFile opens the file and parses the TIFF header and any
        # OME-XML in the ImageDescription tag. series[0].shape gives the
        # full array shape without reading pixel data into memory.
        # ome_metadata returns the OME-XML as a string if present, or
        # None for plain TIFFs that have no OME block.
        f = tifffile.TiffFile(path)
        shapes.append(f.series[0].shape)
        ome_xmls.append(f.ome_metadata)
        f.close()

    # Verify that all dimensions except the concatenation axis match.
    # For example, when concatenating along axis 0 (T) with shape
    # (T, C, Z, Y, X), the C, Z, Y, X values must be identical across
    # all files. Files with different spatial dimensions or channel counts
    # cannot be concatenated into a single coherent volume.
    ref_shape = shapes[0]
    for i, shape in enumerate(shapes[1:], start=1):
        for dim in range(len(ref_shape)):
            if dim == axis:
                continue
            if shape[dim] != ref_shape[dim]:
                raise ValueError(
                    f"Dimension mismatch on axis {dim}: "
                    f"{Path(sorted_paths[0]).name} has {ref_shape[dim]} "
                    f"but {Path(sorted_paths[i]).name} has {shape[dim]}."
                )

    # Compare OME metadata across files. The OME-XML string encodes
    # voxel sizes, channel names, and time calibration. Files from the
    # same imaging session should have identical metadata, but slight
    # differences (e.g. floating-point rounding in voxel size between
    # acquisition runs) can occur without affecting the image data.
    # These are printed as warnings rather than raised as errors so the
    # user is aware but concatenation still proceeds.
    ref_ome = ome_xmls[0]
    if ref_ome is not None:
        for i, ome in enumerate(ome_xmls[1:], start=1):
            if ome is None:
                tqdm.write(
                    f"Warning: {Path(sorted_paths[i]).name} has no OME metadata"
                )
            elif ome != ref_ome:
                tqdm.write(
                    f"Warning: OME metadata in {Path(sorted_paths[i]).name} "
                    f"differs from {Path(sorted_paths[0]).name}"
                )

    total_along_axis = sum(s[axis] for s in shapes)
    tqdm.write(f"Concatenating {len(sorted_paths)} files -> {total_along_axis} total along axis {axis}")

    # Parse the first file's OME-XML to extract physical calibration
    # (voxel sizes, time interval, channel names). These values are
    # passed to TiffWriter as a metadata dict so that tifffile generates
    # a fresh OME-XML header whose dimension counts (SizeT, SizeC, SizeZ)
    # match the actual number of pages in the concatenated output.
    # Passing the raw XML string directly would preserve the original
    # SizeT from a single ND2 file, causing Fiji to misread the axes.
    import xml.etree.ElementTree as ET

    ome_metadata = {"axes": "TCZYX"}
    if ref_ome is not None:
        root = ET.fromstring(ref_ome)
        # OME-XML wraps every tag in its namespace, e.g.
        # {http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels.
        # Extracting the namespace from the root tag lets us find
        # child elements without assuming a specific schema version.
        ns = root.tag.split("}")[0] + "}" if "}" in root.tag else ""
        pixels = root.find(f".//{ns}Pixels")
        if pixels is not None:
            for attr in ["PhysicalSizeX", "PhysicalSizeY", "PhysicalSizeZ", "TimeIncrement"]:
                val = pixels.get(attr)
                if val is not None:
                    ome_metadata[attr] = float(val)
            for attr in ["PhysicalSizeXUnit", "PhysicalSizeYUnit", "PhysicalSizeZUnit", "TimeIncrementUnit"]:
                val = pixels.get(attr)
                if val is not None:
                    ome_metadata[attr] = val

        # Channel names (Venus, mCherry, TD, etc.) so Fiji labels them
        # correctly in the channel manager instead of showing "C=0".
        channels = root.findall(f".//{ns}Channel")
        if channels:
            names = [ch.get("Name") for ch in channels if ch.get("Name")]
            if names:
                ome_metadata["Channel"] = {"Name": names}

    # Build the full output shape by summing along the concatenation
    # axis. The remaining dimensions (C, Z, Y, X) come from the first
    # file since they were already verified to match across all inputs.
    out_shape = list(ref_shape)
    out_shape[axis] = total_along_axis
    out_shape = tuple(out_shape)

    def page_generator():
        """Yield 2D (Y, X) pages from each input file in order.

        Each file's 5D array (T, C, Z, Y, X) is flattened to (T*C*Z, Y, X)
        so that every confocal plane becomes one TIFF page. Reading and
        flattening one file at a time keeps peak memory at the size of a
        single input file rather than the full concatenated volume.
        """
        for path in tqdm(sorted_paths, desc="Concatenating", unit="file"):
            data = tifffile.imread(path).astype(np.float32)
            yield from data.reshape(-1, data.shape[-2], data.shape[-1])
            del data

    # imwrite with a generator and an explicit shape lets tifffile
    # stream pages to disk while knowing the full 5D dimensions
    # upfront. It generates a single OME Image element with the
    # correct SizeT, SizeC, SizeZ, voxel sizes, and channel names.
    tifffile.imwrite(
        output_path,
        page_generator(),
        shape=out_shape,
        dtype=np.float32,
        bigtiff=True,
        photometric="minisblack",
        metadata=ome_metadata,
    )

    tqdm.write(f"Done. Output: {output_path}")


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


def encode_mp4(frame_generator, output_path, fps=2):
    """
    Consumes a generator of RGB image frames and encodes them to an HEVC (H.265) MP4.
    Uses Apple's hardware encoder (hevc_videotoolbox) for speed, falling back to 
    CPU (libx265) if the hardware encoder is unavailable.
    """
    # Pull the first frame to get the video dimensions.
    # If the generator is empty, we just exit cleanly.
    try:
        first_frame = next(frame_generator)
    except StopIteration:
        tqdm.write(f"Warning: No frames generated for {output_path.name}")
        return

    height, width, _ = first_frame.shape

    # Base FFmpeg command setup for raw RGB24 input via a pipe.
    # -y overwrites the output if it already exists.
    # -f rawvideo and -pix_fmt rgb24 tell FFmpeg the exact format of our numpy arrays.
    cmd_base = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-", # Instructs FFmpeg to read video data from standard input
    ]

    # Try hardware acceleration first, then fallback to CPU software encoding.
    # -tag:v hvc1 ensures the resulting HEVC file is playable in QuickTime/macOS.
    encoders_to_try = [
        (["-c:v", "hevc_videotoolbox", "-b:v", "5M", "-tag:v", "hvc1"], "Hardware (VideoToolbox)"),
        (["-c:v", "libx265", "-crf", "23", "-preset", "medium", "-tag:v", "hvc1"], "Software (libx265)")
    ]

    for encoder_args, encoder_name in encoders_to_try:
        cmd = cmd_base + encoder_args + [str(output_path)]
        
        # Spin up the FFmpeg subprocess. stderr=subprocess.DEVNULL keeps the console clean,
        # otherwise FFmpeg prints a massive amount of output per frame.
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        
        try:
            # Write the first frame we grabbed earlier to the FFmpeg pipe.
            process.stdin.write(first_frame.tobytes())
            
            # Pipe the rest of the frames directly from the generator to FFmpeg.
            for frame in frame_generator:
                process.stdin.write(frame.tobytes())
                
            # Close stdin to signal FFmpeg that the video stream is done, then wait for it to finish.
            process.stdin.close()
            process.wait()
            
            if process.returncode == 0:
                # Success! Break out of the fallback loop.
                return
            else:
                raise RuntimeError(f"FFmpeg exited with code {process.returncode}")
                
        except (BrokenPipeError, RuntimeError):
            # If the pipe breaks immediately (e.g., codec not found), the hardware encoder failed.
            tqdm.write(f"\n[!] {encoder_name} encoding failed. Attempting fallback...")
            
            # Since codec failures happen on initialization (the first frame), we only 
            # reach this if we haven't consumed the rest of the generator yet.
            if encoder_name == encoders_to_try[-1][1]:
                tqdm.write(f"Error: All encoding methods failed for {output_path.name}")

def tiff_to_mp4(tif_path, output_path, channel_index, fps=2):
    """
    Reads a standard (T, C, Z, Y, X) OME-TIFF from disk, max-projects the 
    specified channel, and encodes it directly to an MP4 using encode_mp4.
    """
    # Read the metadata from the file to get the channel name
    channel_names, _, _ = load_tif_metadata(tif_path)
    ch_name = channel_names[channel_index]

    with tifffile.TiffFile(tif_path) as tif:
        series = tif.series[0] # .series is a method of the TiffFile object that returns the first series of the file. 
        T, C, Z, Y, X = series.shape # shape is written by save_ome_tiff as (T, C, Z, Y, X). 

        def generate_frames():
            for t in tqdm(range(T), desc=f"    Encoding {ch_name}", unit="frame", leave=False): # leave=False means don't show the progress bar at the end.
                frame = np.stack([series.pages[t*C*Z + i].asarray() for i in range(C*Z)]).reshape(C, Z, Y, X).astype(np.float32)
                proj = auto_contrast(frame[channel_index].max(axis=0))
                yield np.stack([proj, proj, proj], axis=-1)

        tqdm.write(f"\nStarting encode to: {output_path.name}")
        encode_mp4(generate_frames(), output_path, fps)
        tqdm.write(f"Done! Video saved at: {output_path}")
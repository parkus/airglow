import numpy as np


def midpts(ary, axis=None):
    """Computes the midpoints between points in a vector.

    Output has length len(vec)-1.
    """
    if type(ary) != np.ndarray:
        ary = np.array(ary)
    if axis is None:
        return (ary[1:] + ary[:-1]) / 2.0
    hi = np.split(ary, [1], axis=axis)[1]
    lo = np.split(ary, [-1], axis=axis)[0]
    return (hi + lo) / 2.0


def mids2edges(mids):
    """
    Reconstructs bin edges given only the midpoints.

    Parameters
    ----------
    mids : 1-D array-like
        A 1-D array or list of the midpoints from which bin edges are to be
        inferred.

    Result
    ------
    edges : np.array
        The inferred bin edges.

    Could be accelerated with a cython implementation.
    """
    edges = midpts(mids)
    d0 = edges[0] - mids[0]
    d1 = mids[-1] - edges[-1]
    return np.insert(edges, [0, len(edges)], [mids[0] - d0, mids[-1] + d1])


def cumulative_trapz(y, x, zero_start=False):
    result = np.cumsum(midpts(y) * np.diff(x))
    if zero_start:
        result = np.insert(result, 0, 0)
    return result


def bin_average(x_bin_edges, x, y, left=None, right=None):
    """Compute average of xin,yin within supplied bins.

    This funtion is similar to interpolation, but averages the curve repesented
    by xin,yin over the supplied bins to produce the output, yout.

    This is particularly useful, for example, for a spectrum of narrow emission
    incident on a detector with broad pixels. Each pixel averages out or
    "dilutes" the lines that fall within its range. However, simply
    interpolating at the pixel midpoints is a mistake as these points will
    often land between lines and predict no flux in a pixel where narrow but
    strong lines will actually produce significant flux.

    left and right have the same definition as in np.interp
    """
    I = cumulative_trapz(y, x, True)
    Iedges = np.interp(x_bin_edges, x, I, left=left, right=right)
    y_bin_avg = np.diff(Iedges) / np.diff(x_bin_edges)
    return y_bin_avg


def boxcars_to_bins(midpoints, widths, heights, bin_edges):
    """
    Distribute the areas of multiple 1D boxcar functions into bins.

    Parameters
    ----------
    midpoints : array-like, shape (M,)
        Midpoints of the M boxcar functions.
    widths : array-like, shape (M,)
        Widths of the M boxcar functions.
    heights : array-like, shape (M,)
        Heights of the M boxcar functions.
    bin_edges : array-like, shape (N+1,)
        Bin edges defining N bins.

    Returns
    -------
    bin_areas : ndarray, shape (M, N)
        Amount of boxcar area from each boxcar in each bin.
    """
    midpoints = np.asarray(midpoints)[:, None]  # shape (M,1)
    widths = np.asarray(widths)[:, None]
    heights = np.asarray(heights)[:, None]
    bin_edges = np.asarray(bin_edges)  # shape (N+1,)

    lefts = midpoints - widths / 2  # (M,1)
    rights = midpoints + widths / 2  # (M,1)

    bin_lefts = bin_edges[:-1][None, :]  # shape (1, N)
    bin_rights = bin_edges[1:][None, :]  # shape (1, N)

    # Compute overlap region per boxcar and bin
    overlap_lefts = np.maximum(lefts, bin_lefts)  # shape (M,N)
    overlap_rights = np.minimum(rights, bin_rights)
    overlap_widths = np.maximum(0.0, overlap_rights - overlap_lefts)

    bin_areas = overlap_widths * heights  # shape (M, N)

    return bin_areas


def fit_poly_surface_2d(data, roi, order=2, mask=None, weights=None):
    """
    Fit a 2D polynomial surface to an ROI.

    Parameters
    ----------
    data : array-like
        2D image data.
    roi : tuple
        ROI bounds as (y0, y1, x0, x1) or ((y0, y1), (x0, x1)).
    order : int, optional
        Polynomial order (total degree). Default is 2.
    mask : array-like, optional
        Boolean mask for data; True values are excluded from the fit.
    weights : array-like, optional
        Weights per pixel (same shape as data). Applied as WLS.

    Returns
    -------
    result : dict
        Dictionary with keys: coeffs, powers, model, center, roi, keep.
    """
    data = np.asarray(data, dtype=float)
    if len(roi) == 2 and len(roi[0]) == 2 and len(roi[1]) == 2:
        (y0, y1), (x0, x1) = roi
    else:
        y0, y1, x0, x1 = roi

    roi_data = data[y0:y1, x0:x1]
    y_grid, x_grid = np.indices(roi_data.shape, dtype=float)
    y_center = np.nanmean(y_grid)
    x_center = np.nanmean(x_grid)
    y = y_grid - y_center
    x = x_grid - x_center

    keep = np.isfinite(roi_data)
    if mask is not None:
        roi_mask = np.asarray(mask, dtype=bool)[y0:y1, x0:x1]
        keep &= ~roi_mask

    if weights is not None:
        roi_weights = np.asarray(weights, dtype=float)[y0:y1, x0:x1]
        keep &= np.isfinite(roi_weights) & (roi_weights > 0)

    if not np.any(keep):
        raise ValueError("No valid pixels to fit in the requested ROI.")

    powers = []
    terms = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            powers.append((i, j))
            terms.append((x ** i) * (y ** j))

    A = np.stack([term[keep] for term in terms], axis=1)
    b = roi_data[keep]

    if weights is not None:
        w = roi_weights[keep]
        A = A * w[:, None]
        b = b * w

    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    model = np.zeros_like(roi_data)
    for coeff, (i, j) in zip(coeffs, powers):
        model += coeff * (x ** i) * (y ** j)

    return {
        "coeffs": coeffs,
        "powers": np.array(powers, dtype=int),
        "model": model,
        "center": (y_center, x_center),
        "roi": (y0, y1, x0, x1),
        "keep": keep,
    }


def infer_flat_field(flt, raw):
    
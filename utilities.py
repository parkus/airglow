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

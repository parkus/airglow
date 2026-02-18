from __future__ import annotations

import os
from copy import copy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from astropy.io import fits
from IPython.display import display
import ipywidgets as widgets
import matplotlib.pyplot as plt
import stistools as stis

SELECTED_TRACE_Y: float | None = None
DEFAULT_TRACE_Y: float | None = None

x_center = 512.5

def find_test_data_dir() -> Path:
    """Locate the test-data directory. Checks AIRGLOW_TEST_DATA env var, then cwd-relative paths."""
    env_path = os.environ.get("AIRGLOW_TEST_DATA")
    if env_path:
        td = Path(env_path).resolve()
        if td.exists():
            return td
    cwd = Path.cwd().resolve()
    candidates = [cwd, cwd.parent, cwd.parent.parent]
    for base in candidates:
        td = base / "test-data"
        if td.exists():
            return td
    raise FileNotFoundError(
        "Could not locate test-data directory. Set AIRGLOW_TEST_DATA environment variable "
        "or run from the repository root (or notebooks/) where test-data exists."
    )


default_x1d_params = dict(
    maxsrch=0.01, 
    extrsize=19,
    bk1offst=-30,
    bk2offst=30,
    bk1size=20,
    bk2size=20,
)


def get_x1d_trace_params(params: dict | None = None) -> dict:
    """Return a copy of x1d parameters for trace extraction. For notebook compatibility."""
    return copy(default_x1d_params) if params is None else copy(params)


def ensure_x1d(fltfile: Path, force: bool = False, x1d_params: dict = default_x1d_params) -> Path:
    x1dfile = fltfile.with_name(fltfile.name.replace("_flt", "_x1d"))
    if x1dfile.exists() and not force:
        return x1dfile
    stis.x1d.x1d(str(fltfile), str(x1dfile), **x1d_params)
    return x1dfile


def get_default_traceloc(fltfile: Path) -> float | None:
    x1dfile = ensure_x1d(fltfile, force=False)
    if not x1dfile.exists():
        return None

    x1d = fits.getdata(x1dfile, 1)
    tracelocs = x1d["a2center"]

    if np.ndim(tracelocs):
        return float(tracelocs[0])
    return float(tracelocs)


def _get_column(data, name: str):
    if data is None or not hasattr(data, "names") or data.names is None:
        raise ValueError("data is None or does not have a names attribute.")
    name_lower = name.lower()
    for col in data.names:
        if col.lower() == name_lower:
            return data[col]
    return None


def _get_trace_profile(x1d):
    trace = _get_column(x1d, "etraclocy")
    if trace is None:
        trace = _get_column(x1d, "extrlocy")
    if trace is None:
        return None
    if np.ndim(trace) > 1:
        trace = trace[0]
    return np.asarray(trace, dtype=float)


def _correct_traceloc(
    x_pixels: np.ndarray,
    trace_y: np.ndarray,
    x_click: float,
    y_click: float,
) -> float:
    y_trace_click = np.interp(x_click, x_pixels, trace_y)
    y_trace_center = np.interp(x_center, x_pixels, trace_y)
    return float(y_click + (y_trace_center - y_trace_click))


def plot_trace_image(fltfile: Path):
    x1dfile = ensure_x1d(fltfile, force=False)
    data = fits.getdata(fltfile, 1)

    fig, ax = plt.subplots()
    ax.set_title(fltfile.name)
    ax.imshow(np.cbrt(data), aspect="auto")

    if x1dfile.exists():
        x1d = fits.getdata(x1dfile, 1)
        x = np.arange(data.shape[1]) + 0.5
        y = _get_column(x1d, "extrlocy")
        if np.ndim(y) > 1:
            ax.plot(x, y.T, color="r", lw=0.5, alpha=0.5, label="pipeline trace")
        else:
            ax.plot(x, y, color="r", lw=0.5, alpha=0.5, label="pipeline trace")

    ax.text(
        0.02,
        0.98,
        "Click the trace. Use the buttons to keep/clear.",
        transform=ax.transAxes,
        va="top",
        color="w",
        fontsize="small",
    )

    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right")

    return fig, ax


def setup_trace_selector(fltfile: Path):
    global SELECTED_TRACE_Y, DEFAULT_TRACE_Y

    SELECTED_TRACE_Y = None
    DEFAULT_TRACE_Y = get_default_traceloc(fltfile)

    fig, ax = plot_trace_image(fltfile)
    x1dfile = ensure_x1d(fltfile, force=False)
    x1d = fits.getdata(x1dfile, 1) if x1dfile.exists() else None
    trace_y = _get_trace_profile(x1d) if x1d is not None else None
    data = fits.getdata(fltfile, 1)
    x_pixels = np.arange(data.shape[1]) + 0.5

    hline = ax.axhline(0, color="y", lw=1, alpha=0.7)
    hline.set_visible(False)

    status = widgets.Label(value="Click on the image to select the trace location.")
    use_default_btn = widgets.Button(description="Use default")
    clear_btn = widgets.Button(description="Clear selection")

    def set_selection(y: float, msg: str) -> None:
        global SELECTED_TRACE_Y
        SELECTED_TRACE_Y = float(y)
        hline.set_ydata([y, y])
        hline.set_visible(True)
        fig.canvas.draw_idle()
        status.value = msg

    def on_click(event):
        if event.inaxes != ax or event.ydata is None:
            return
        if trace_y is None or event.xdata is None:
            set_selection(event.ydata, f"Selected y: {event.ydata:.2f}")
            return
        corrected_y = _correct_traceloc(
            x_pixels=x_pixels,
            trace_y=trace_y,
            x_click=event.xdata,
            y_click=event.ydata,
        )
        delta = corrected_y - event.ydata
        set_selection(
            corrected_y,
            f"Selected y after correction: {corrected_y:.2f} (Δ{delta:+.2f})",
        )

    def use_default(_):
        if DEFAULT_TRACE_Y is None:
            status.value = "Default trace not available."
            return
        set_selection(DEFAULT_TRACE_Y, f"Using default y: {DEFAULT_TRACE_Y:.2f}")

    def clear(_):
        global SELECTED_TRACE_Y
        SELECTED_TRACE_Y = None
        hline.set_visible(False)
        fig.canvas.draw_idle()
        status.value = "Selection cleared."

    fig.canvas.mpl_connect("button_press_event", on_click)
    use_default_btn.on_click(use_default)
    clear_btn.on_click(clear)

    display(widgets.HBox([use_default_btn, clear_btn]), status)
    return fig, ax


def extract_standard_x1d(
    fltfile: Path, 
    overwrite: bool = False, 
    manual_traceloc: float | None = None,
    x1d_params: dict = default_x1d_params
) -> None:
    out = fltfile.with_name(fltfile.name.replace("_flt", "_x1d"))
    if out.exists() and overwrite:
        out.unlink()
    if out.exists() and not overwrite:
        return out
    stis.x1d.x1d(str(fltfile), str(out), a2center=manual_traceloc, **x1d_params)
    return out


def extract_background_traces(
    fltfile: Path,
    overwrite: bool = False,
    manual_traceloc: float | None = None,
    x1d_params: dict = default_x1d_params,
) -> None:
    header = fits.getheader(fltfile, 0)
    grating = str(header.get("opt_elem", "")).lower()

    x1dfile = ensure_x1d(fltfile, force=False)
    tracelocs = fits.getdata(x1dfile, 1)["a2center"]

    labels = ["x1dbk1", "x1dtrace", "x1dbk2"]
    mod_params = copy(x1d_params)
    mod_params["bk1size"] = mod_params["bk2size"] = 0
    mod_params["bk1offst"] = mod_params["bk2offst"] = 0
    mod_params.pop("extrsize", None)

    traceloc = (
        manual_traceloc
        if manual_traceloc is not None
        else (tracelocs[0] if np.ndim(tracelocs) else tracelocs)
    )
    y1 = traceloc + x1d_params["bk1offst"]
    yt = traceloc
    y2 = traceloc + x1d_params["bk2offst"]
    sz1 = x1d_params["bk1size"]
    szt = x1d_params["extrsize"]
    sz2 = x1d_params["bk2size"]
    sets = ((y1, sz1, labels[0]), (yt, szt, labels[1]), (y2, sz2, labels[2]))

    for y, sz, lbl in sets:
        out = fltfile.with_name(fltfile.name.replace("_flt", f"_{lbl}"))
        if out.exists() and overwrite:
            out.unlink()
        if out.exists() and not overwrite:
            continue
        stis.x1d.x1d(str(fltfile), str(out), a2center=y, extrsize=sz, **mod_params)


def plot_extraction_locations(
    fltfile: Path,
    suffixes: tuple[str, ...] = ("x1d", "x1dbk1", "x1dtrace", "x1dbk2"),
):
    data = fits.getdata(fltfile, 1)
    x = np.arange(data.shape[1]) + 0.5

    fig, ax = plt.subplots()
    ax.set_title(fltfile.name)
    ax.imshow(np.cbrt(data), aspect="auto")

    for suffix in suffixes:
        x1dfile = fltfile.with_name(fltfile.name.replace("_flt", f"_{suffix}"))
        if not x1dfile.exists():
            continue
        x1d = fits.getdata(x1dfile, 1)
        y = _get_column(x1d, "extrlocy")
        if y is None:
            continue
        if np.ndim(y) > 1:
            y = y[0]
        ax.plot(x, y, lw=0.8, label=x1dfile.name)

    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right")

    return fig, ax


def get_flt_y_positions(
    fltfile: Path,
    step: float,
    y_min: float | None = None,
    y_max: float | None = None,
) -> np.ndarray:
    data = fits.getdata(fltfile, 1)
    nrows = data.shape[0]
    y_min = 0.5 if y_min is None else y_min
    y_max = (nrows - 0.5) if y_max is None else y_max
    if step <= 0:
        raise ValueError("step must be positive.")
    return np.arange(y_min, y_max + 1e-6, step, dtype=float)


def _read_spectrum_from_x1d(x1dfile: Path, column: str = "flux") -> np.ndarray:
    x1d = fits.getdata(x1dfile, 1)
    spectrum = _get_column(x1d, column)
    if spectrum is None:
        raise KeyError(f"Column '{column}' not found in {x1dfile.name}.")
    if np.ndim(spectrum) > 1:
        spectrum = spectrum[0]
    return np.asarray(spectrum, dtype=float)


def _read_wavelength_from_x1d(x1dfile: Path) -> np.ndarray:
    x1d = fits.getdata(x1dfile, 1)
    wavelength = _get_column(x1d, "wavelength")
    if wavelength is None:
        raise KeyError(f"Column 'wavelength' not found in {x1dfile.name}.")
    if np.ndim(wavelength) > 1:
        wavelength = wavelength[0]
    return np.asarray(wavelength, dtype=float)


def _get_sci_table(hdul: fits.HDUList) -> fits.BinTableHDU:
    if "SCI" in hdul:
        return hdul["SCI"]
    return hdul[1]


def extract_x1d_grid(
    fltfile: Path,
    y_positions: np.ndarray,
    output_dir: Path,
    x1d_params: dict = default_x1d_params,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    x1d_files = []
    for idx, y in enumerate(y_positions):
        out = output_dir / f"{fltfile.stem}_y{idx:04d}_x1d.fits"
        stis.x1d.x1d(str(fltfile), str(out), a2center=float(y), **x1d_params)
        x1d_files.append(out)
    return x1d_files


def collate_trace_spectra(
    x1d_files: list[Path],
    column: str = "flux",
) -> tuple[np.ndarray, np.ndarray]:
    if not x1d_files:
        raise ValueError("x1d_files is empty.")
    wavelength = _read_wavelength_from_x1d(x1d_files[0])
    spectra = []
    for x1dfile in x1d_files:
        spectrum = _read_spectrum_from_x1d(x1dfile, column=column)
        if spectrum.shape != wavelength.shape:
            raise ValueError(f"Spectrum shape mismatch for {x1dfile.name}.")
        spectra.append(spectrum)
    return np.asarray(spectra, dtype=float), wavelength


def collate_trace_table(x1d_files: list[Path]) -> fits.BinTableHDU:
    if not x1d_files:
        raise ValueError("x1d_files is empty.")

    with fits.open(x1d_files[0]) as hdul:
        base_table = _get_sci_table(hdul)
        columns = base_table.columns
        header = base_table.header.copy()

    data_per_col: dict[str, list[np.ndarray]] = {col.name: [] for col in columns}
    for x1dfile in x1d_files:
        with fits.open(x1dfile) as hdul:
            table = _get_sci_table(hdul).data
            for col in columns:
                values = table[col.name]
                if np.ndim(values) > 1:
                    values = values[0]
                data_per_col[col.name].append(values)

    coldefs = []
    for col in columns:
        array = np.asarray(data_per_col[col.name])
        coldefs.append(
            fits.Column(
                name=col.name,
                format=col.format,
                dim=col.dim,
                unit=col.unit,
                array=array,
            )
        )

    header["EXTNAME"] = "TRACES"
    return fits.BinTableHDU.from_columns(fits.ColDefs(coldefs), header=header, name="TRACES")


def _robust_sigma_from_mad(values: np.ndarray, axis: int = 0) -> np.ndarray:
    median = np.nanmedian(values, axis=axis)
    mad = np.nanmedian(np.abs(values - median), axis=axis)
    return 1.4826 * mad


def _weighted_centroid(values: np.ndarray, x: np.ndarray) -> float:
    valid = np.isfinite(values)
    if not np.any(valid):
        return np.nan
    baseline = np.nanmedian(values[valid])
    weights = values - baseline
    weights = np.where(weights > 0, weights, 0.0)
    weight_sum = np.nansum(weights)
    if weight_sum <= 0:
        weight_sum = np.nansum(values[valid])
        if weight_sum <= 0:
            return np.nan
        weights = values
    return float(np.nansum(x * weights) / weight_sum)


def estimate_background_sigma_from_traces(
    trace_stack: np.ndarray,
    wavelength_stack: np.ndarray | None = None,
    y_positions: np.ndarray | None = None,
    y_windows: list[tuple[float, float]] | None = None,
    centroid_stack: np.ndarray | None = None,
    align_centroids: bool = False,
) -> dict:
    trace_stack = np.asarray(trace_stack, dtype=float)
    if trace_stack.ndim != 2:
        raise ValueError("trace_stack must be 2D (n_traces, n_pixels).")

    n_traces, n_pix = trace_stack.shape
    if wavelength_stack is None:
        wavelength_stack = np.tile(np.arange(n_pix, dtype=float), (n_traces, 1))
    else:
        wavelength_stack = np.asarray(wavelength_stack, dtype=float)
        if wavelength_stack.shape != trace_stack.shape:
            raise ValueError("wavelength_stack must match trace_stack shape.")

    if y_windows:
        if y_positions is None:
            y_positions = np.arange(n_traces, dtype=float)
        y_positions = np.asarray(y_positions, dtype=float)
        if y_positions.shape[0] != n_traces:
            raise ValueError("y_positions must match trace_stack length.")
        windows = [(min(y0, y1), max(y0, y1)) for y0, y1 in y_windows]
        selected = np.zeros(n_traces, dtype=bool)
        for y0, y1 in windows:
            selected |= (y_positions >= y0) & (y_positions <= y1)
    else:
        selected = np.ones(n_traces, dtype=bool)

    if not np.any(selected):
        raise ValueError("No traces selected with the provided y_windows.")

    traces = trace_stack[selected]
    wavelengths = wavelength_stack[selected]
    centroid_source = traces
    if centroid_stack is not None:
        centroid_stack = np.asarray(centroid_stack, dtype=float)
        if centroid_stack.shape != trace_stack.shape:
            raise ValueError("centroid_stack must match trace_stack shape.")
        centroid_source = centroid_stack[selected]

    centroids = np.array(
        [_weighted_centroid(values, wave) for values, wave in zip(centroid_source, wavelengths)],
        dtype=float,
    )
    target_centroid = np.nanmedian(centroids)
    shifts = centroids - target_centroid

    if align_centroids:
        shifted_wavelengths = wavelengths - shifts[:, None]
    else:
        shifted_wavelengths = wavelengths

    common_grid = shifted_wavelengths[len(shifted_wavelengths) // 2]
    aligned = np.vstack(
        [
            np.interp(common_grid, wave, trace, left=np.nan, right=np.nan)
            for wave, trace in zip(shifted_wavelengths, traces)
        ]
    )

    median = np.nanmedian(aligned, axis=0)
    sigma = _robust_sigma_from_mad(aligned, axis=0)

    return {
        "aligned_traces": aligned,
        "aligned_median": median,
        "aligned_sigma": sigma,
        "centroids": centroids,
        "target_centroid": target_centroid,
        "shifts": shifts,
        "selected_mask": selected,
        "common_wavelength": common_grid,
    }


def _get_trace_y_positions(hdul: fits.HDUList) -> np.ndarray | None:
    if "TRACES" in hdul:
        return np.asarray(hdul["TRACES"].data["a2center"], dtype=float)
    raise KeyError("Column 'a2center' not found in traces extension.")


def _build_revised_error_column(table_hdu: fits.BinTableHDU, sigma: np.ndarray, name: str = "ERROR_EMPIRICAL") -> fits.ColDefs:
    error_col = None
    for col in table_hdu.columns:
        if col.name.lower() == "error":
            error_col = col
            break

    if error_col is None:
        raise KeyError("Column 'ERROR' not found in x1d table.")

    error_data = table_hdu.data[error_col.name]
    if np.ndim(error_data) > 1:
        n_rows = error_data.shape[0]
        revised = np.tile(sigma, (n_rows, 1)).astype(np.float32)
    else:
        revised = np.asarray(sigma, dtype=np.float32)

    new_col = fits.Column(
        name=name,
        format=error_col.format,
        dim=error_col.dim,
        array=revised,
    )

    # Filter out any pre-existing column with the same name (idempotent).
    existing_cols = fits.ColDefs(
        [c for c in table_hdu.columns if c.name.upper() != name.upper()]
    )
    return existing_cols + new_col


def revise_background_error_in_x1d(
    x1dfile: Path,
    output_x1d: Path | None = None,
    y_windows: tuple[int, int] | None = None,
    align_centroids: bool = False,
) -> dict:

    with fits.open(x1dfile) as hdul:
        if "TRACES" not in hdul:
            raise KeyError("TRACES extension not found in x1d file.")

        trace_table = hdul["TRACES"].data
        if trace_table is None:
            raise ValueError("TRACES extension has no data.")

        trace_background = None
        if trace_table.dtype.fields:
            trace_background = _get_column(trace_table, "background")

        trace_data = _get_column(trace_table, "flux")
        if trace_data is None:
            raise KeyError("TRACES table missing 'BACKGROUND' and 'FLUX' columns.")
        trace_data = np.asarray(trace_data, dtype=float)
        if trace_data.ndim != 2:
            raise ValueError(f"Expected 2D traces array, got shape {trace_data.shape}.")

        wavelength_stack = _get_column(trace_table, "wavelength")
        if wavelength_stack is not None:
            wavelength_stack = np.asarray(wavelength_stack, dtype=float)
            if wavelength_stack.shape != trace_data.shape:
                raise ValueError("TRACES wavelength shape does not match traces.")

        n_traces = trace_data.shape[0]
        y_positions = _get_trace_y_positions(hdul)
        if y_positions is None:
            y_positions = np.arange(n_traces, dtype=float)
        else:
            y_positions = np.asarray(y_positions, dtype=float)
            if y_positions.shape[0] != n_traces:
                raise ValueError("TRACE_Y length does not match trace stack.")

        science_data = hdul[1].data
        science_center = _get_column(science_data, "a2center")
        if science_center is not None and np.ndim(science_center):
            science_center = float(science_center[0])
        elif science_center is not None:
            science_center = float(science_center)

        result = estimate_background_sigma_from_traces(
            trace_stack=trace_data,
            wavelength_stack=wavelength_stack,
            y_positions=y_positions,
            y_windows=y_windows,
            centroid_stack=trace_background,
            align_centroids=align_centroids,
        )

        sigma = result["aligned_sigma"]
        common_wavelength = result["common_wavelength"]

        sci_wavelength = _get_column(science_data, "wavelength")
        if sci_wavelength is not None:
            sci_wave = np.asarray(sci_wavelength, dtype=float)
            if sci_wave.ndim > 1:
                sci_wave = sci_wave[0]
            sci_wave = np.atleast_1d(sci_wave)
            sigma = np.interp(
                sci_wave,
                common_wavelength,
                sigma,
                left=np.nan,
                right=np.nan,
            ).astype(np.float32)

        sci_hdu = hdul["SCI"] if "SCI" in hdul else hdul[1]
        cols = _build_revised_error_column(sci_hdu, sigma, name="ERROR_EMPIRICAL")
        new_table_hdu = fits.BinTableHDU.from_columns(cols, header=sci_hdu.header, name=sci_hdu.name)

        new_hdul = fits.HDUList([hdu.copy() for hdu in hdul])
        new_hdul[1] = new_table_hdu

    if output_x1d is None:
        output_x1d = x1dfile.with_name(x1dfile.stem + "_revised_error" + x1dfile.suffix)
    new_hdul.writeto(output_x1d, overwrite=True)

    return output_x1d


def _compute_trace_overlap_mask(
    trace_centers: np.ndarray,
    trace_extrsize: float,
    science_center: float,
    science_extrsize: float,
) -> np.ndarray:
    half_trace = trace_extrsize / 2.0
    half_science = science_extrsize / 2.0
    trace_min = trace_centers - half_trace
    trace_max = trace_centers + half_trace
    science_min = science_center - half_science
    science_max = science_center + half_science
    overlaps = (trace_min <= science_max) & (trace_max >= science_min)
    return overlaps.astype(np.uint8)[:, None]


def add_traces_to_x1d(
    science_x1d: Path,
    output_x1d: Path,
    trace_table: fits.BinTableHDU,
    trace_mask: np.ndarray,
) -> Path:
    with fits.open(science_x1d) as hdul:
        new_hdul = fits.HDUList([hdu.copy() for hdu in hdul])

    for name in ("TRACES", "TRACE_MASK"):
        if name in new_hdul:
            idx = new_hdul.index_of(name)
            new_hdul.pop(idx)

    mask_col = fits.Column(
        name="TRACE_MASK",
        format="1B",
        array=np.asarray(trace_mask, dtype=np.uint8),
    )
    trace_table = fits.BinTableHDU.from_columns(
        trace_table.columns + mask_col,
        header=trace_table.header,
        name="TRACES",
    )
    new_hdul.append(trace_table)
    new_hdul.writeto(output_x1d, overwrite=True)
    return output_x1d


def build_trace_stack_x1d(
    fltfile: Path,
    output_x1d: Path,
    step: float,
    x1d_params: dict = default_x1d_params,
    spectrum_column: str = "flux",
    y_min: float | None = None,
    y_max: float | None = None,
) -> dict:
    y_positions = get_flt_y_positions(fltfile, step=step, y_min=y_min, y_max=y_max)
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        x1d_files = extract_x1d_grid(
            fltfile=fltfile,
            y_positions=y_positions,
            output_dir=tmp_path,
            x1d_params=x1d_params,
        )
        trace_table = collate_trace_table(x1d_files)

    science_x1d = ensure_x1d(fltfile, force=False, x1d_params=x1d_params)
    science_data = fits.getdata(science_x1d, 1)
    science_center = _get_column(science_data, "a2center")
    if science_center is None:
        raise KeyError("Column 'a2center' not found in science x1d.")
    if np.ndim(science_center):
        science_center = float(science_center[0])
    else:
        science_center = float(science_center)

    science_extrsize = x1d_params.get("extrsize", None)
    if science_extrsize is None:
        raise KeyError("extrsize not found in x1d_params.")

    trace_mask = _compute_trace_overlap_mask(
        trace_centers=y_positions,
        trace_extrsize=float(x1d_params["extrsize"]),
        science_center=science_center,
        science_extrsize=float(science_extrsize),
    )
    add_traces_to_x1d(
        science_x1d=science_x1d,
        output_x1d=output_x1d,
        trace_table=trace_table,
        trace_mask=trace_mask,
    )
    trace_spectra = _get_column(trace_table.data, spectrum_column)
    wavelength = _get_column(trace_table.data, "wavelength")
    return {
        "output_x1d": output_x1d,
        "y_positions": y_positions,
        "wavelength": wavelength,
        "trace_spectra": trace_spectra,
        "trace_mask": trace_mask,
    }

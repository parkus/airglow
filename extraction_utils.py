from __future__ import annotations

from copy import copy
from pathlib import Path
from re import X

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
    cwd = Path.cwd().resolve()
    candidates = [cwd, cwd.parent, cwd.parent.parent]
    for base in candidates:
        td = base / "test-data"
        if td.exists():
            return td
    raise FileNotFoundError("Could not locate test-data directory from current working dir")


default_x1d_params = dict(
    maxsrch=0.01, 
    extrsize=19,
    bk1offst=-30,
    bk2offst=30,
    bk1size=20,
    bk2size=20,
)


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
        return None
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

import io
import numpy as np
import pandas as pd
import porespy as ps
import pyvista as pv
from scipy.ndimage import label
from skimage.graph import route_through_array
from skimage.morphology import remove_small_objects
import re
import matplotlib

# AutoPore saves figures to files and displays them in its own Tk window.
# A non-interactive backend also permits plotting from workflow worker threads.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from pathlib import Path
import shutil
import os
import stat
import time
import gc
import imageio.v3 as iio
import glob
import platform
import sys
import argparse
import imageio.v2 as imageio
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from matplotlib.ticker import MaxNLocator, FixedLocator

PLOT_BASE_FONT_SIZE = 12
PLOT_TITLE_FONT_SIZE = 12
PLOT_LEGEND_FONT_SIZE = 10
PLOT_SMALL_LEGEND_FONT_SIZE = 10
PLOT_3D_TEXT_FONT_SIZE = 12

plt.rcParams.update(
    {
        "font.size": PLOT_BASE_FONT_SIZE,
        "axes.labelsize": PLOT_BASE_FONT_SIZE,
        "axes.titlesize": PLOT_TITLE_FONT_SIZE,
        "xtick.labelsize": PLOT_BASE_FONT_SIZE,
        "ytick.labelsize": PLOT_BASE_FONT_SIZE,
        "legend.fontsize": PLOT_BASE_FONT_SIZE,
        "figure.titlesize": PLOT_TITLE_FONT_SIZE,
    }
)

def _finite_axis_values(data) -> np.ndarray:
    if isinstance(data, (list, tuple)):
        finite_parts = []
        for value in data:
            arr = np.asarray(value, dtype=float).ravel()
            arr = arr[np.isfinite(arr)]
            if arr.size:
                finite_parts.append(arr)
        if finite_parts:
            return np.concatenate(finite_parts)
        return np.array([], dtype=float)

    arr = np.asarray(data, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def set_axis_with_extra_top_gap(ax, data, axis="y", nbins=5, too_far_frac=0.5):
    """
    Set axis limits and ticks using MaxNLocator.

    If the highest tick is too far above the actual data maximum,
    extend the axis upper limit by one tick interval plus the bottom gap.

    Parameters
    ----------
    ax : matplotlib axis
        Usually plt.gca() or the axis object from fig, ax = plt.subplots().
    data : array-like
        Data used to determine min/max.
    axis : {"x", "y"}
        Which axis to modify.
    nbins : int
        Approximate number of tick intervals.
    too_far_frac : float
        If top_gap > too_far_frac * tick_interval, extend upper limit.
    """

    data = _finite_axis_values(data)
    if data.size == 0:
        return

    dmin = np.nanmin(data)
    dmax = np.nanmax(data)

    # Handle constant data
    if np.isclose(dmin, dmax):
        pad = abs(dmin) * 0.1 if dmin != 0 else 1.0
        dmin -= pad
        dmax += pad

    locator = MaxNLocator(nbins=nbins)
    ticks = locator.tick_values(dmin, dmax)
    ticks = ticks[np.isfinite(ticks)]

    if len(ticks) < 2:
        return

    min_tick = ticks[0]
    max_tick = ticks[-1]
    tick_interval = ticks[-1] - ticks[-2]

    bottom_gap = dmin - min_tick
    top_gap = max_tick - dmax

    # Default upper limit
    upper_lim = max_tick
    final_ticks = ticks.copy()

    # If the highest tick is too far from actual data maximum,
    # extend one more tick interval, plus bottom gap
    if top_gap > too_far_frac * tick_interval:
        new_top_tick = max_tick + tick_interval
        upper_lim = new_top_tick + bottom_gap

        final_ticks = np.arange(
            min_tick,
            new_top_tick + 0.5 * tick_interval,
            tick_interval
        )

    lower_lim = min_tick

    if axis == "y":
        ax.set_ylim(lower_lim - 0.05*(upper_lim - lower_lim), upper_lim + 0.05*(upper_lim - lower_lim))
        ax.yaxis.set_major_locator(FixedLocator(final_ticks))
    elif axis == "x":
        ax.set_xlim(lower_lim, upper_lim)
        ax.xaxis.set_major_locator(FixedLocator(final_ticks))
    else:
        raise ValueError("axis must be 'x' or 'y'")


def swap_xz_and_scale_domain(input_vti, output_vti, scale=0.5):
    """
    Convert VTI coordinates from (x,y,z) -> (z,y,x),
    then scale the whole physical domain by 'scale'.

    Keeps the minimum corner fixed as the origin.
    """
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(input_vti)
    reader.Update()
    img = reader.GetOutput()

    old_dims = img.GetDimensions()   # (nx, ny, nz)
    old_spacing = img.GetSpacing()   # (sx, sy, sz)
    old_origin = img.GetOrigin()     # (ox, oy, oz)

    nx, ny, nz = old_dims
    sx, sy, sz = old_spacing
    ox, oy, oz = old_origin

    # swap x and z, then scale whole domain
    new_dims = (nz, ny, nx)
    new_spacing = (sz * scale, sy * scale, sx * scale)

    # keep minimum corner fixed after swap
    new_origin = (oz, oy, ox)

    out_img = vtk.vtkImageData()
    out_img.SetDimensions(new_dims)
    out_img.SetSpacing(new_spacing)
    out_img.SetOrigin(new_origin)
    out_img.GetPointData().SetScalars(None)

    # -------- point data --------
    in_pd = img.GetPointData()
    out_pd = out_img.GetPointData()

    for i in range(in_pd.GetNumberOfArrays()):
        arr = in_pd.GetArray(i)
        if arr is None:
            continue

        name = arr.GetName()
        ncomp = arr.GetNumberOfComponents()
        data = vtk_to_numpy(arr)

        if ncomp == 1:
            data_reshaped = data.reshape((nz, ny, nx))
            data_swapped = np.transpose(data_reshaped, (2, 1, 0))
            data_flat = data_swapped.ravel(order="C")
        else:
            data_reshaped = data.reshape((nz, ny, nx, ncomp))
            data_swapped = np.transpose(data_reshaped, (2, 1, 0, 3))
            data_flat = data_swapped.reshape(-1, ncomp, order="C")

        vtk_out = numpy_to_vtk(data_flat, deep=True)
        vtk_out.SetName(name)
        vtk_out.SetNumberOfComponents(ncomp)
        out_pd.AddArray(vtk_out)

        if i == 0:
            out_pd.SetScalars(vtk_out)

    # -------- cell data --------
    in_cd = img.GetCellData()
    out_cd = out_img.GetCellData()

    cx, cy, cz = max(nx - 1, 1), max(ny - 1, 1), max(nz - 1, 1)

    for i in range(in_cd.GetNumberOfArrays()):
        arr = in_cd.GetArray(i)
        if arr is None:
            continue

        name = arr.GetName()
        ncomp = arr.GetNumberOfComponents()
        data = vtk_to_numpy(arr)

        if ncomp == 1:
            data_reshaped = data.reshape((cz, cy, cx))
            data_swapped = np.transpose(data_reshaped, (2, 1, 0))
            data_flat = data_swapped.ravel(order="C")
        else:
            data_reshaped = data.reshape((cz, cy, cx, ncomp))
            data_swapped = np.transpose(data_reshaped, (2, 1, 0, 3))
            data_flat = data_swapped.reshape(-1, ncomp, order="C")

        vtk_out = numpy_to_vtk(data_flat, deep=True)
        vtk_out.SetName(name)
        vtk_out.SetNumberOfComponents(ncomp)
        out_cd.AddArray(vtk_out)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(output_vti)
    writer.SetInputData(out_img)
    writer.Write()

    print(f"Saved: {output_vti}")
    print("Old dims   :", old_dims)
    print("New dims   :", new_dims)
    print("Old spacing:", old_spacing)
    print("New spacing:", new_spacing)
    print("Old origin :", old_origin)
    print("New origin :", new_origin)


def avi_to_gif(input_avi, output_gif, fps=None, loop=0):
    reader = imageio.get_reader(input_avi)
    meta = reader.get_meta_data()

    if fps is None:
        fps = meta.get("fps", 10)

    frames = []
    for frame in reader:
        frames.append(frame)

    reader.close()
    imageio.mimsave(output_gif, frames, fps=fps, loop = loop)

def find_pvbatch(THIS_DIR):
    # 1) check PATH
    for name in ("pvbatch.exe", "pvbatch"):
        p = shutil.which(name)
        if p:
            return Path(p).resolve()

    # 2) check paraview_path.txt next to this script (pvpython -> same bin folder)
    paraview_txt = THIS_DIR / "paraview_path.txt"
    try:
        if paraview_txt.exists():
            txt = paraview_txt.read_text(encoding="utf-8").strip()
            if txt:
                pvpython = Path(txt)
                pvbin = pvpython.parent
                candidate = pvbin / ("pvbatch.exe" if platform.system() == "Windows" else "pvbatch")
                if candidate.exists():
                    return candidate.resolve()
    except Exception:
        pass

    # 3) check common Program Files locations (Windows)
    if platform.system() == "Windows":
        candidates = []
        prog = os.environ.get("ProgramFiles")
        prog_x86 = os.environ.get("ProgramFiles(x86)")
        for base in (prog, prog_x86):
            if not base:
                continue
            # look for ParaView installations under Program Files
            pattern = os.path.join(base, "ParaView*", "bin", "pvbatch.exe")
            candidates.extend(glob.glob(pattern))

        # also check default 64-bit path if env vars missing
        candidates.extend(glob.glob(r"C:\\Program Files\\ParaView*\\bin\\pvbatch.exe"))
        candidates.extend(glob.glob(r"C:\\Program Files (x86)\\ParaView*\\bin\\pvbatch.exe"))

        if candidates:
            # return the newest ParaView (highest version) if multiple
            candidates = sorted(set(candidates))
            return Path(candidates[-1]).resolve()

    # 4) last resort: attempt a targeted recursive search under C:\ (may be slow)
    if platform.system() == "Windows":
        root = Path("C:\\")
        try:
            for p in root.rglob("pvbatch.exe"):
                return p.resolve()
        except Exception:
            pass

    # 5) Unix-like systems: check common installation paths
    if platform.system() != "Windows":
        unix_candidates = ["/usr/local/bin/pvbatch", "/usr/bin/pvbatch", "/opt/paraview/bin/pvbatch"]
        for c in unix_candidates:
            if Path(c).exists():
                return Path(c).resolve()

    return None

def show_image(filepath):
    """Open an image file non-blockingly in the default viewer."""
    filepath = str(filepath)
    if os.path.isfile(filepath):
        os.startfile(filepath)

def _mathtext(label):
    """Wrap tokens containing _ or ^ in $...$ so matplotlib renders sub/superscripts."""
    def convert(token):
        if '_' not in token and '^' not in token:
            return token
        return '$' + token + '$'
    return ' '.join(convert(t) for t in label.split(' '))


def save_plot(plot_path, x, y, case_name = 'save_plot.png', xlab = '', ylab = '', title = '', labels=None):
    plt.figure(figsize=(5, 4))
    # For each column, remove duplicate max-value points keeping only the first.
    # Plot each column with its own filtered x/y so decreasing tails are preserved.
    x_all, y_all = [], []
    for i in range(y.shape[1]):
        col = y[:, i]
        col_max = np.max(col)
        is_max = col == col_max
        # Keep: all non-max points + only the first max point
        first_max = np.argmax(is_max)
        keep = ~is_max
        keep[first_max] = True
        x_all.append(x[keep])
        y_all.append(col[keep])
        label = labels[i] if labels is not None and i < len(labels) else None
        plt.plot(x[keep], col[keep], marker="o", markersize=2, label=label)
    all_x = np.concatenate(x_all)
    all_y = np.concatenate(y_all)
    x_lo, x_hi = np.floor(np.min(all_x)), np.ceil(np.max(all_x))
    y_lo, y_hi = np.floor(np.min(all_y)), np.ceil(np.max(all_y))
    
    x_pad = (x_hi - x_lo) * 0.05
    y_pad = (y_hi - y_lo) * 0.05
    plt.xlim([x_lo - x_pad, x_hi + x_pad])
    plt.ylim([y_lo - y_pad, y_hi + y_pad])
    plt.tick_params(axis='both', direction='in')
    ax = plt.gca()
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 2))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
    set_axis_with_extra_top_gap(ax, all_y, axis="y", nbins=5)

    plt.xlabel(_mathtext(xlab))
    plt.ylabel(_mathtext(ylab))
    plt.title(title)
    if labels is not None:
        plt.legend(fontsize=PLOT_LEGEND_FONT_SIZE, frameon=False)
    plt.tight_layout()
    fig_path = Path(plot_path) / case_name
    plt.savefig(fig_path, dpi=300)
    plt.close()
    return fig_path


def save_time_series_plot(
    plot_path,
    x,
    y,
    case_name="time_series.png",
    xlab="",
    ylab="",
    title="",
    normalize_x=False,
):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        print(f"No valid data for plot: {case_name}")
        return None

    x = x[valid]
    y = y[valid]
    if normalize_x:
        max_x = np.max(x)
        if not np.isfinite(max_x) or max_x == 0:
            print(f"Cannot normalize x axis for plot: {case_name}")
            return None
        x = x / max_x

    fig_path = Path(plot_path) / case_name
    plt.figure(figsize=(5, 4))
    plt.plot(x, y, marker="o", markersize=2, linewidth=1.2)
    plt.xlabel(_mathtext(xlab))
    plt.ylabel(_mathtext(ylab))
    plt.title(title)
    plt.tick_params(axis="both", direction="in")
    ax = plt.gca()
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 2))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
    set_axis_with_extra_top_gap(ax, y, axis="y", nbins=5)
    plt.subplots_adjust(left=0.17, right=0.96, bottom=0.17, top=0.88)
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved plot to: {fig_path}")
    return fig_path


def save_pore_size_distribution_plot(
    plot_path,
    pore_diameters,
    throat_diameters=None,
    case_name="pore_size_distribution.png",
):
    plot_path = Path(plot_path)
    plot_path.mkdir(parents=True, exist_ok=True)
    fig_path = plot_path / case_name

    pore_values = _finite_axis_values(pore_diameters)
    throat_values = _finite_axis_values(throat_diameters) if throat_diameters is not None else np.array([], dtype=float)
    if pore_values.size == 0 and throat_values.size == 0:
        print(f"No valid diameter data for plot: {case_name}")
        return None

    plt.figure(figsize=(5, 4))
    bins_source = np.concatenate([arr for arr in (pore_values, throat_values) if arr.size])
    bin_count = min(30, max(8, int(np.sqrt(max(1, bins_source.size)))))
    if pore_values.size:
        plt.hist(
            pore_values,
            bins=bin_count,
            alpha=0.65,
            label="Pores",
            color="#4f83cc",
            edgecolor="white",
            linewidth=0.5,
        )
    if throat_values.size:
        plt.hist(
            throat_values,
            bins=bin_count,
            alpha=0.55,
            label="Throats",
            color="#e66b5b",
            edgecolor="white",
            linewidth=0.5,
        )
    plt.xlabel("Diameter (um)")
    plt.ylabel("Count")
    plt.title("Pore network size distribution")
    plt.tick_params(axis="both", direction="in")
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved plot to: {fig_path}")
    return fig_path


def _canonical_time_series_name(name) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _find_time_series_column(columns, aliases) -> str | None:
    alias_keys = {_canonical_time_series_name(alias) for alias in aliases}
    for column in columns:
        if _canonical_time_series_name(column) in alias_keys:
            return column
    return None


def _first_data_field_count(time_series_file: Path) -> int | None:
    try:
        with open(time_series_file, "r", encoding="utf-8") as handle:
            next(handle, "")
            for line in handle:
                stripped = line.strip()
                if stripped and not stripped.startswith(";"):
                    return len(stripped.split())
    except Exception:
        return None
    return None


def _read_dynamic_time_series_by_position(time_series_file: Path) -> pd.DataFrame | None:
    try:
        with open(time_series_file, "r", encoding="utf-8") as handle:
            header_tokens = handle.readline().split()
    except Exception as exc:
        print(f"Could not read dynamic time-series header: {time_series_file}: {exc}")
        return None

    try:
        data = pd.read_csv(
            time_series_file,
            sep=r"\s+",
            engine="python",
            header=None,
            skiprows=1,
            comment=";",
        )
    except Exception as exc:
        print(f"Could not read dynamic time-series data: {time_series_file}: {exc}")
        return None

    if data.empty:
        print(f"Dynamic time-series file has no data rows: {time_series_file}")
        return None

    canonical_header = {
        _canonical_time_series_name(token): index
        for index, token in enumerate(header_tokens)
    }
    time_idx = canonical_header.get("productiontime", 0)
    pcap_header_idx = canonical_header.get("avepcap")
    aff_bulk_header_idx = canonical_header.get(
        "aveaffb", canonical_header.get("aveaff")
    )
    aff_bulk_film_header_idx = canonical_header.get("aveaffbf")
    sg_header_idx = canonical_header.get("sgall", canonical_header.get("sg"))

    if pcap_header_idx is None:
        pcap_header_idx = 1
    if sg_header_idx is None:
        sg_header_idx = 17

    extra_columns = max(0, data.shape[1] - len(header_tokens))

    def shifted_index(header_idx: int) -> int:
        if header_idx >= pcap_header_idx:
            return header_idx + extra_columns
        return header_idx

    pcap_idx = shifted_index(pcap_header_idx)
    aff_bulk_idx = (
        shifted_index(aff_bulk_header_idx)
        if aff_bulk_header_idx is not None
        else None
    )
    aff_bulk_film_idx = (
        shifted_index(aff_bulk_film_header_idx)
        if aff_bulk_film_header_idx is not None
        else None
    )
    sg_idx = shifted_index(sg_header_idx)
    required_indexes = {
        "Production_Time": time_idx,
        "ave_pcap": pcap_idx,
        "Sg_all": sg_idx,
    }
    missing = [
        f"{name} at column {index}"
        for name, index in required_indexes.items()
        if index >= data.shape[1]
    ]
    if missing:
        print(f"Dynamic time-series columns not found by position: {missing}")
        return None

    columns = {
        "Production_Time": pd.to_numeric(data.iloc[:, time_idx], errors="coerce"),
        "ave_pcap": pd.to_numeric(data.iloc[:, pcap_idx], errors="coerce"),
        "Sg_all": pd.to_numeric(data.iloc[:, sg_idx], errors="coerce"),
    }
    if aff_bulk_idx is not None and aff_bulk_idx < data.shape[1]:
        bulk_area = pd.to_numeric(data.iloc[:, aff_bulk_idx], errors="coerce")
        columns["ave_aff_b"] = bulk_area
        columns["ave_aff"] = bulk_area
    if aff_bulk_film_idx is not None and aff_bulk_film_idx < data.shape[1]:
        columns["ave_aff_bf"] = pd.to_numeric(
            data.iloc[:, aff_bulk_film_idx], errors="coerce"
        )

    return pd.DataFrame(columns)


def read_dynamic_time_series(time_series_file) -> pd.DataFrame | None:
    time_series_file = Path(time_series_file)

    try:
        with open(time_series_file, "r", encoding="utf-8") as handle:
            header_count = len(handle.readline().split())
    except Exception:
        header_count = None

    # If data rows have more fields than the header, new columns were likely
    # inserted before ave_pcap without matching old header names.
    data_field_count = _first_data_field_count(time_series_file)
    if (
        header_count is not None
        and data_field_count is not None
        and data_field_count != header_count
    ):
        return _read_dynamic_time_series_by_position(time_series_file)

    try:
        df = pd.read_csv(time_series_file, sep=r"\s+", engine="python")
    except Exception as exc:
        print(f"Could not read dynamic time-series file: {time_series_file}: {exc}")
        return _read_dynamic_time_series_by_position(time_series_file)

    if df.empty:
        print(f"Dynamic time-series file has no data rows: {time_series_file}")
        return None

    time_col = _find_time_series_column(df.columns, ["Production_Time", "production_time"])
    pcap_col = _find_time_series_column(df.columns, ["ave_pcap", "average_pcap", "avg_pcap"])
    aff_bulk_col = _find_time_series_column(
        df.columns,
        [
            "ave_aff_b",
            "ave_aff",
            "average_aff",
            "avg_aff",
            "bulk_fluid_fluid_interfacial_area",
        ],
    )
    aff_bulk_film_col = _find_time_series_column(
        df.columns,
        [
            "ave_aff_bf",
            "average_aff_bf",
            "avg_aff_bf",
            "bulk_plus_film_fluid_fluid_interfacial_area",
        ],
    )
    sg_col = _find_time_series_column(
        df.columns,
        ["Sg_all", "Sg", "gas_saturation", "gas_saturation_all"],
    )

    if time_col is None or pcap_col is None or sg_col is None:
        return _read_dynamic_time_series_by_position(time_series_file)

    columns = {
        "Production_Time": pd.to_numeric(df[time_col], errors="coerce"),
        "ave_pcap": pd.to_numeric(df[pcap_col], errors="coerce"),
        "Sg_all": pd.to_numeric(df[sg_col], errors="coerce"),
    }
    if aff_bulk_col is not None:
        bulk_area = pd.to_numeric(df[aff_bulk_col], errors="coerce")
        columns["ave_aff_b"] = bulk_area
        columns["ave_aff"] = bulk_area
    if aff_bulk_film_col is not None:
        columns["ave_aff_bf"] = pd.to_numeric(
            df[aff_bulk_film_col], errors="coerce"
        )

    return pd.DataFrame(columns)


def _uq_results_root(pnm_mode: str, base_dir) -> Path:
    base_dir = Path(base_dir)
    if pnm_mode == "dynamic":
        return base_dir / "uq_dyn_pnm"
    return base_dir / "uq_qs_pnm"


def _case_folders(results_root: Path) -> list[Path]:
    if not results_root.exists():
        return []
    return sorted(
        [
            child for child in results_root.iterdir()
            if child.is_dir() and child.name.startswith("case_")
        ],
        key=lambda path: path.name,
    )


def _axis_limits(values: list[np.ndarray]) -> tuple[float, float] | None:
    finite_parts = []
    for value in values:
        arr = np.asarray(value, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            finite_parts.append(arr)
    if not finite_parts:
        return None

    all_values = np.concatenate(finite_parts)
    lo = float(np.min(all_values))
    hi = float(np.max(all_values))
    if lo == hi:
        pad = abs(lo) * 0.05 if lo != 0 else 1.0
    else:
        pad = (hi - lo) * 0.05
    return lo - pad, hi + pad


def _save_uq_overlay_plot(
    output_folder: Path,
    series: list[tuple[str, np.ndarray, np.ndarray]],
    filename: str,
    xlab: str,
    ylab: str,
    title: str,
) -> Path | None:
    output_folder.mkdir(parents=True, exist_ok=True)

    clean_series = []
    for label, x, y in series:
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        n = min(x.size, y.size)
        if n == 0:
            continue
        x = x[:n]
        y = y[:n]
        valid = np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            continue
        clean_series.append((label, x[valid], y[valid]))

    if not clean_series:
        print(f"No valid data for UQ plot: {filename}")
        return None

    fig_path = output_folder / filename
    plt.figure(figsize=(5, 4))
    for label, x, y in clean_series:
        plt.plot(x, y, marker="o", markersize=2, linewidth=1.0, alpha=0.85, label=label)

    xlim = _axis_limits([x for _, x, _ in clean_series])
    y_values = [y for _, _, y in clean_series]
    if xlim is not None:
        plt.xlim(xlim)

    plt.tick_params(axis="both", direction="in")
    ax = plt.gca()
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 2))
    set_axis_with_extra_top_gap(ax, y_values, axis="y", nbins=5)
    plt.xlabel(_mathtext(xlab))
    plt.ylabel(_mathtext(ylab))
    plt.title(title)
    if len(clean_series) <= 16:
        plt.legend(fontsize=PLOT_LEGEND_FONT_SIZE, frameon=False)
    plt.subplots_adjust(left=0.17, right=0.96, bottom=0.17, top=0.88)
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved UQ plot to: {fig_path}")
    return fig_path


def _save_uq_kr_overlay_plot(
    output_folder: Path,
    series: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    filename: str,
    xlab: str,
    ylab: str,
    title: str,
) -> Path | None:
    output_folder.mkdir(parents=True, exist_ok=True)

    clean_series = []
    for label, x, krw, krg in series:
        x = np.asarray(x, dtype=float).ravel()
        krw = np.asarray(krw, dtype=float).ravel()
        krg = np.asarray(krg, dtype=float).ravel()
        n = min(x.size, krw.size, krg.size)
        if n == 0:
            continue
        x = x[:n]
        krw = krw[:n]
        krg = krg[:n]
        valid = np.isfinite(x) & (np.isfinite(krw) | np.isfinite(krg))
        if not np.any(valid):
            continue
        clean_series.append((label, x[valid], krw[valid], krg[valid]))

    if not clean_series:
        print(f"No valid data for UQ plot: {filename}")
        return None

    fig_path = output_folder / filename
    plt.figure(figsize=(5, 4))
    ax = plt.gca()
    legend_specs = []
    for label, x, krw, krg in clean_series:
        line = ax.plot(
            x,
            krw,
            marker="o",
            markersize=2,
            linewidth=1.0,
            alpha=0.85,
            label=label,
        )[0]
        ax.plot(
            x,
            krg,
            marker="o",
            markersize=2,
            linewidth=1.0,
            alpha=0.85,
            color=line.get_color(),
            label="_nolegend_",
        )
        legend_specs.append((label, line.get_color(), line.get_color(), 1.0))

    xlim = _axis_limits([x for _, x, _, _ in clean_series])
    y_values = (
        [krw for _, _, krw, _ in clean_series]
        + [krg for _, _, _, krg in clean_series]
    )
    if xlim is not None:
        ax.set_xlim(xlim)

    ax.tick_params(axis="both", direction="in")
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 2))
    set_axis_with_extra_top_gap(ax, y_values, axis="y", nbins=5)
    ax.set_xlabel(_mathtext(xlab))
    ax.set_ylabel(_mathtext(ylab))
    ax.set_title(title)
    if len(legend_specs) <= 16:
        _add_two_column_line_legend(
            ax,
            legend_specs,
            "Wetting",
            "Nonwetting",
            position="center_top",
            box_x_override=0.08,
            background_alpha=0.0,
        )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved UQ plot to: {fig_path}")
    return fig_path


def _save_uq_area_overlay_plot(
    output_folder: Path,
    series: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    filename: str,
    xlab: str,
    ylab: str,
    title: str,
) -> Path | None:
    output_folder.mkdir(parents=True, exist_ok=True)

    clean_series = []
    for label, x, bulk, bulk_film in series:
        x = np.asarray(x, dtype=float).ravel()
        bulk = np.asarray(bulk, dtype=float).ravel()
        bulk_film = np.asarray(bulk_film, dtype=float).ravel()
        n = min(x.size, bulk.size, bulk_film.size)
        if n == 0:
            continue
        x = x[:n]
        bulk = bulk[:n]
        bulk_film = bulk_film[:n]
        valid = np.isfinite(x) & (np.isfinite(bulk) | np.isfinite(bulk_film))
        if not np.any(valid):
            continue
        clean_series.append((label, x[valid], bulk[valid], bulk_film[valid]))

    if not clean_series:
        print(f"No valid data for UQ plot: {filename}")
        return None

    fig_path = output_folder / filename
    plt.figure(figsize=(5, 4))
    ax = plt.gca()
    legend_specs = []
    for label, x, bulk, bulk_film in clean_series:
        line = ax.plot(
            x,
            bulk,
            marker="o",
            markersize=2,
            linewidth=1.0,
            alpha=0.85,
            linestyle="-",
            label=label,
        )[0]
        ax.plot(
            x,
            bulk_film,
            marker="o",
            markersize=2,
            linewidth=1.0,
            alpha=0.85,
            linestyle="--",
            color=line.get_color(),
            label="_nolegend_",
        )
        legend_specs.append((label, line.get_color(), line.get_color(), 1.0, "-", "--"))

    xlim = _axis_limits([x for _, x, _, _ in clean_series])
    y_values = (
        [bulk for _, _, bulk, _ in clean_series]
        + [bulk_film for _, _, _, bulk_film in clean_series]
    )
    if xlim is not None:
        ax.set_xlim(xlim)

    ax.tick_params(axis="both", direction="in")
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 2))
    set_axis_with_extra_top_gap(ax, y_values, axis="y", nbins=5)
    ax.set_xlabel(_mathtext(xlab))
    ax.set_ylabel(_mathtext(ylab))
    ax.set_title(title)

    if len(legend_specs) <= 16:
        _add_two_column_line_legend(ax, legend_specs, "Bulk", "Bulk+Film")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved UQ plot to: {fig_path}")
    return fig_path


def _prepare_percentile_series(x, y) -> tuple[np.ndarray, np.ndarray] | None:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = min(x.size, y.size)
    if n < 2:
        return None

    x = x[:n]
    y = y[:n]
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 2:
        return None

    x = x[valid]
    y = y[valid]
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    unique_x, inverse = np.unique(x, return_inverse=True)
    if unique_x.size < 2:
        return None

    unique_y = np.zeros(unique_x.size, dtype=float)
    for index in range(unique_x.size):
        unique_y[index] = np.nanmean(y[inverse == index])
    return unique_x, unique_y


def _interpolate_percentile_matrix(
    series: list[tuple[str, np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray] | None:
    prepared = []
    all_x_values = []
    common_min = -np.inf
    common_max = np.inf

    for label, x, y in series:
        prepared_series = _prepare_percentile_series(x, y)
        if prepared_series is None:
            continue

        x_sorted, y_sorted = prepared_series
        prepared.append((label, x_sorted, y_sorted))
        all_x_values.append(x_sorted)
        common_min = max(common_min, float(x_sorted[0]))
        common_max = min(common_max, float(x_sorted[-1]))

    if not prepared or common_min >= common_max:
        return None

    ranked_x = np.unique(np.concatenate(all_x_values))
    ranked_x = ranked_x[(ranked_x >= common_min) & (ranked_x <= common_max)]
    if ranked_x.size < 2:
        return None

    x_mid = 0.5 * (ranked_x[:-1] + ranked_x[1:])
    x_mid = x_mid[(x_mid >= common_min) & (x_mid <= common_max)]
    if x_mid.size == 0:
        return None

    y_vectors = []
    for _, x_sorted, y_sorted in prepared:
        y_vectors.append(np.interp(x_mid, x_sorted, y_sorted))

    return x_mid, np.vstack(y_vectors)


def _save_uq_percentile_plot(
    output_folder: Path,
    series: list[tuple[str, np.ndarray, np.ndarray]],
    filename: str,
    xlab: str,
    ylab: str,
    title: str,
    outer_color: str = "#fcae91",
    inner_color: str = "#fb6a4a",
    median_color: str = "black",
) -> Path | None:
    interpolated = _interpolate_percentile_matrix(series)
    if interpolated is None:
        print(f"No valid data for UQ percentile plot: {filename}")
        return None

    x_mid, y_matrix = interpolated
    pct0, pct25, pct50, pct75, pct100 = np.nanpercentile(
        y_matrix,
        [0, 25, 50, 75, 100],
        axis=0,
    )

    output_folder.mkdir(parents=True, exist_ok=True)
    fig_path = output_folder / filename

    plt.figure(figsize=(5, 4))
    plt.fill_between(x_mid, pct0, pct100, color=outer_color, alpha=0.25, linewidth=0)
    plt.fill_between(x_mid, pct25, pct75, color=inner_color, alpha=0.50, linewidth=0)
    plt.plot(x_mid, pct0, color=outer_color, linewidth=0.7, linestyle="-", label="0%")
    plt.plot(x_mid, pct25, color=inner_color, linewidth=0.8, linestyle="-", label="25%")
    plt.plot(x_mid, pct50, color=median_color, linewidth=1.0, linestyle="-", label="50%")
    plt.plot(x_mid, pct75, color=inner_color, linewidth=0.8, linestyle="-", label="75%")
    plt.plot(x_mid, pct100, color=outer_color, linewidth=0.7, linestyle="-", label="100%")
    plt.tick_params(axis="both", direction="in")
    ax = plt.gca()
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 2))
    set_axis_with_extra_top_gap(
        ax,
        [pct0, pct25, pct50, pct75, pct100],
        axis="y",
        nbins=5,
    )
    plt.xlabel(_mathtext(xlab))
    plt.ylabel(_mathtext(ylab))
    plt.title(title)
    plt.legend(fontsize=PLOT_LEGEND_FONT_SIZE, frameon=False)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved UQ percentile plot to: {fig_path}")
    return fig_path


def _save_uq_kr_percentile_plot(
    output_folder: Path,
    krw_series: list[tuple[str, np.ndarray, np.ndarray]],
    krg_series: list[tuple[str, np.ndarray, np.ndarray]],
    filename: str,
    xlab: str,
    ylab: str,
    title: str,
) -> Path | None:
    krw_interpolated = _interpolate_percentile_matrix(krw_series)
    krg_interpolated = _interpolate_percentile_matrix(krg_series)
    if krw_interpolated is None or krg_interpolated is None:
        print(f"No valid data for UQ percentile plot: {filename}")
        return None

    krw_x, krw_matrix = krw_interpolated
    krg_x, krg_matrix = krg_interpolated
    krw_pct0, krw_pct25, krw_pct50, krw_pct75, krw_pct100 = np.nanpercentile(
        krw_matrix,
        [0, 25, 50, 75, 100],
        axis=0,
    )
    krg_pct0, krg_pct25, krg_pct50, krg_pct75, krg_pct100 = np.nanpercentile(
        krg_matrix,
        [0, 25, 50, 75, 100],
        axis=0,
    )

    output_folder.mkdir(parents=True, exist_ok=True)
    fig_path = output_folder / filename

    krw_outer = "#9ecae1"
    krw_inner = "#4292c6"
    krw_median = "#08519c"
    krg_outer = "#fcae91"
    krg_inner = "#fb6a4a"
    krg_median = "#cb181d"

    plt.figure(figsize=(5, 4))

    plt.fill_between(krw_x, krw_pct0, krw_pct100, color=krw_outer, alpha=0.25, linewidth=0)
    plt.fill_between(krw_x, krw_pct25, krw_pct75, color=krw_inner, alpha=0.50, linewidth=0)
    plt.plot(krw_x, krw_pct0, color=krw_outer, linewidth=0.7, linestyle="-", label="_nolegend_")
    plt.plot(krw_x, krw_pct25, color=krw_inner, linewidth=0.8, linestyle="-", label="_nolegend_")
    plt.plot(krw_x, krw_pct50, color=krw_median, linewidth=1.0, linestyle="-", label="_nolegend_")
    plt.plot(krw_x, krw_pct75, color=krw_inner, linewidth=0.8, linestyle="-", label="_nolegend_")
    plt.plot(krw_x, krw_pct100, color=krw_outer, linewidth=0.7, linestyle="-", label="_nolegend_")

    plt.fill_between(krg_x, krg_pct0, krg_pct100, color=krg_outer, alpha=0.25, linewidth=0)
    plt.fill_between(krg_x, krg_pct25, krg_pct75, color=krg_inner, alpha=0.50, linewidth=0)
    plt.plot(krg_x, krg_pct0, color=krg_outer, linewidth=0.7, linestyle="-", label="_nolegend_")
    plt.plot(krg_x, krg_pct25, color=krg_inner, linewidth=0.8, linestyle="-", label="_nolegend_")
    plt.plot(krg_x, krg_pct50, color=krg_median, linewidth=1.0, linestyle="-", label="_nolegend_")
    plt.plot(krg_x, krg_pct75, color=krg_inner, linewidth=0.8, linestyle="-", label="_nolegend_")
    plt.plot(krg_x, krg_pct100, color=krg_outer, linewidth=0.7, linestyle="-", label="_nolegend_")

    plt.tick_params(axis="both", direction="in")
    ax = plt.gca()
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 2))
    set_axis_with_extra_top_gap(
        ax,
        [
            krw_pct0, krw_pct25, krw_pct50, krw_pct75, krw_pct100,
            krg_pct0, krg_pct25, krg_pct50, krg_pct75, krg_pct100,
        ],
        axis="y",
        nbins=5,
    )
    plt.xlabel(_mathtext(xlab))
    plt.ylabel(_mathtext(ylab))
    plt.title(title)
    legend_specs = [
        ("0%", krw_outer, krg_outer, 0.7),
        ("25%", krw_inner, krg_inner, 0.8),
        ("50%", krw_median, krg_median, 1.0),
        ("75%", krw_inner, krg_inner, 0.8),
        ("100%", krw_outer, krg_outer, 0.7),
    ]
    _add_two_column_line_legend(
        ax,
        legend_specs,
        "Wetting",
        "Nonwetting",
        position="center_top",
        line_width_scale=1.25,
        box_x_override=0.08,
        background_alpha=0.0,
    )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved UQ percentile plot to: {fig_path}")
    return fig_path


def _add_two_column_line_legend(
    ax,
    legend_specs: list[tuple],
    left_title: str,
    right_title: str,
    position: str = "upper_right",
    line_width_scale: float = 1.0,
    box_x_override: float | None = None,
    background_alpha: float = 0.80,
) -> None:
    from matplotlib.patches import Rectangle

    if not legend_specs:
        return

    row_count = len(legend_specs)
    row_gap = min(0.065, 0.42 / max(row_count, 1))
    font_size = PLOT_LEGEND_FONT_SIZE if row_count <= 8 else PLOT_SMALL_LEGEND_FONT_SIZE
    line_length = 0.055
    if position == "center_top":
        box_x = 0.20
        box_width = 0.56
        left_center = 0.32
        right_center = 0.52
        label_x = 0.70
        header_y = 0.94
    else:
        box_x = 0.54
        box_width = 0.43
        left_center = 0.60
        right_center = 0.75
        label_x = 0.90
        header_y = 0.94
    if box_x_override is not None:
        shift = box_x_override - box_x
        box_x = box_x_override
        left_center += shift
        right_center += shift
        label_x += shift
    left_line = (left_center - line_length / 2, left_center + line_length / 2)
    right_line = (right_center - line_length / 2, right_center + line_length / 2)
    bottom_y = header_y - row_gap * (row_count + 0.75)

    box = Rectangle(
        (box_x, bottom_y),
        box_width,
        header_y - bottom_y + 0.02,
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="none",
        alpha=background_alpha,
        zorder=3,
    )
    ax.add_patch(box)

    ax.text(
        left_center,
        header_y,
        left_title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=font_size,
        clip_on=True,
        zorder=4,
    )
    ax.text(
        right_center,
        header_y,
        right_title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=font_size,
        clip_on=True,
        zorder=4,
    )

    for row, spec in enumerate(legend_specs, start=1):
        if len(spec) == 4:
            row_label, left_color, right_color, line_width = spec
            left_style = right_style = "-"
        else:
            row_label, left_color, right_color, line_width, left_style, right_style = spec
        line_width *= line_width_scale
        y = header_y - row * row_gap
        ax.plot(
            left_line,
            (y, y),
            color=left_color,
            linewidth=line_width,
            linestyle=left_style,
            transform=ax.transAxes,
            clip_on=True,
            zorder=4,
        )
        ax.plot(
            right_line,
            (y, y),
            color=right_color,
            linewidth=line_width,
            linestyle=right_style,
            transform=ax.transAxes,
            clip_on=True,
            zorder=4,
        )
        ax.text(
            label_x,
            y,
            row_label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=font_size,
            clip_on=True,
            zorder=4,
        )


def _save_uq_area_percentile_plot(
    output_folder: Path,
    bulk_series: list[tuple[str, np.ndarray, np.ndarray]],
    bulk_film_series: list[tuple[str, np.ndarray, np.ndarray]],
    filename: str,
    xlab: str,
    ylab: str,
    title: str,
) -> Path | None:
    bulk_interpolated = _interpolate_percentile_matrix(bulk_series)
    bulk_film_interpolated = _interpolate_percentile_matrix(bulk_film_series)
    if bulk_interpolated is None or bulk_film_interpolated is None:
        print(f"No valid data for UQ percentile plot: {filename}")
        return None

    bulk_x, bulk_matrix = bulk_interpolated
    bulk_film_x, bulk_film_matrix = bulk_film_interpolated
    bulk_pct0, bulk_pct25, bulk_pct50, bulk_pct75, bulk_pct100 = np.nanpercentile(
        bulk_matrix,
        [0, 25, 50, 75, 100],
        axis=0,
    )
    (
        bulk_film_pct0,
        bulk_film_pct25,
        bulk_film_pct50,
        bulk_film_pct75,
        bulk_film_pct100,
    ) = np.nanpercentile(
        bulk_film_matrix,
        [0, 25, 50, 75, 100],
        axis=0,
    )

    output_folder.mkdir(parents=True, exist_ok=True)
    fig_path = output_folder / filename

    bulk_outer = "#9ecae1"
    bulk_inner = "#4292c6"
    bulk_median = "#08519c"
    bulk_film_outer = "#fcae91"
    bulk_film_inner = "#fb6a4a"
    bulk_film_median = "#cb181d"

    plt.figure(figsize=(5, 4))

    plt.fill_between(bulk_x, bulk_pct0, bulk_pct100, color=bulk_outer, alpha=0.25, linewidth=0)
    plt.fill_between(bulk_x, bulk_pct25, bulk_pct75, color=bulk_inner, alpha=0.50, linewidth=0)
    plt.plot(bulk_x, bulk_pct0, color=bulk_outer, linewidth=0.7, linestyle="-", label="_nolegend_")
    plt.plot(bulk_x, bulk_pct25, color=bulk_inner, linewidth=0.8, linestyle="-", label="_nolegend_")
    plt.plot(bulk_x, bulk_pct50, color=bulk_median, linewidth=1.0, linestyle="-", label="_nolegend_")
    plt.plot(bulk_x, bulk_pct75, color=bulk_inner, linewidth=0.8, linestyle="-", label="_nolegend_")
    plt.plot(bulk_x, bulk_pct100, color=bulk_outer, linewidth=0.7, linestyle="-", label="_nolegend_")

    plt.fill_between(
        bulk_film_x,
        bulk_film_pct0,
        bulk_film_pct100,
        color=bulk_film_outer,
        alpha=0.25,
        linewidth=0,
    )
    plt.fill_between(
        bulk_film_x,
        bulk_film_pct25,
        bulk_film_pct75,
        color=bulk_film_inner,
        alpha=0.50,
        linewidth=0,
    )
    plt.plot(
        bulk_film_x,
        bulk_film_pct0,
        color=bulk_film_outer,
        linewidth=0.7,
        linestyle="-",
        label="_nolegend_",
    )
    plt.plot(
        bulk_film_x,
        bulk_film_pct25,
        color=bulk_film_inner,
        linewidth=0.8,
        linestyle="-",
        label="_nolegend_",
    )
    plt.plot(
        bulk_film_x,
        bulk_film_pct50,
        color=bulk_film_median,
        linewidth=1.0,
        linestyle="-",
        label="_nolegend_",
    )
    plt.plot(
        bulk_film_x,
        bulk_film_pct75,
        color=bulk_film_inner,
        linewidth=0.8,
        linestyle="-",
        label="_nolegend_",
    )
    plt.plot(
        bulk_film_x,
        bulk_film_pct100,
        color=bulk_film_outer,
        linewidth=0.7,
        linestyle="-",
        label="_nolegend_",
    )

    plt.tick_params(axis="both", direction="in")
    ax = plt.gca()
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 2))
    set_axis_with_extra_top_gap(
        ax,
        [
            bulk_pct0, bulk_pct25, bulk_pct50, bulk_pct75, bulk_pct100,
            bulk_film_pct0, bulk_film_pct25, bulk_film_pct50,
            bulk_film_pct75, bulk_film_pct100,
        ],
        axis="y",
        nbins=5,
    )
    plt.xlabel(_mathtext(xlab))
    plt.ylabel(_mathtext(ylab))
    plt.title(title)
    legend_specs = [
        ("0%", bulk_outer, bulk_film_outer, 0.7),
        ("25%", bulk_inner, bulk_film_inner, 0.8),
        ("50%", bulk_median, bulk_film_median, 1.0),
        ("75%", bulk_inner, bulk_film_inner, 0.8),
        ("100%", bulk_outer, bulk_film_outer, 0.7),
    ]
    _add_two_column_line_legend(
        ax,
        legend_specs,
        "Bulk",
        "Bulk+Film",
        line_width_scale=1.25,
    )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved UQ percentile plot to: {fig_path}")
    return fig_path


def _read_qs_case_results(case_folder: Path) -> pd.DataFrame | None:
    result_file = case_folder / "qs_results.txt"
    if not result_file.exists():
        print(f"Skipping {case_folder.name}: missing {result_file.name}")
        return None

    try:
        df = pd.read_csv(result_file, sep=r"\s+", engine="python")
    except Exception as exc:
        print(f"Skipping {case_folder.name}: could not read {result_file.name}: {exc}")
        return None

    required = ["Pc", "Sw", "Krw", "Krg", "Aaw"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        print(f"Skipping {case_folder.name}: missing columns {missing}")
        return None

    numeric_columns = required + [column for column in ("Aaw_b",) if column in df.columns]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _save_uq_quasi_static_plots(results_root: Path) -> list[str]:
    cases = _case_folders(results_root)
    if not cases:
        raise FileNotFoundError(f"No UQ quasi-static case folders found in: {results_root}")

    pc_sw_series = []
    kr_sw_series = []
    area_sw_series = []
    area_b_sw_series = []
    area_pair_sw_series = []
    pc_percentile_series = []
    krw_percentile_series = []
    krg_percentile_series = []
    area_percentile_series = []
    area_b_percentile_series = []

    for case_folder in cases:
        df = _read_qs_case_results(case_folder)
        if df is None:
            continue

        case_label = case_folder.name
        sw = df["Sw"].to_numpy()
        pc = df["Pc"].to_numpy() / 1e3
        krw = df["Krw"].to_numpy()
        krg = df["Krg"].to_numpy()
        aaw = df["Aaw"].to_numpy() / 1e2
        aaw_b = df["Aaw_b"].to_numpy() / 1e2 if "Aaw_b" in df.columns else None

        pc_sw_series.append((case_label, sw, pc))
        kr_sw_series.append((
            case_label,
            sw,
            krw,
            krg,
        ))
        area_sw_series.append((case_label, sw, aaw))
        if aaw_b is not None:
            area_b_sw_series.append((case_label, sw, aaw_b))
            area_pair_sw_series.append((case_label, sw, aaw_b, aaw))
        pc_percentile_series.append((case_label, sw, pc))
        krw_percentile_series.append((case_label, sw, krw))
        krg_percentile_series.append((case_label, sw, krg))
        area_percentile_series.append((case_label, sw, aaw))
        if aaw_b is not None:
            area_b_percentile_series.append((case_label, sw, aaw_b))

    saved = []
    for stale_name in (
        "UQ A_w_nw_b-Sw curve.png",
        "UQ Percentile A_w_nw_b-Sw curve.png",
    ):
        stale_path = results_root / stale_name
        if stale_path.exists():
            stale_path.unlink()

    fig_path = _save_uq_overlay_plot(
        results_root,
        pc_sw_series,
        "UQ Pc-Sw curve.png",
        "S_{w} (-)",
        "P^{c} (kPa)",
        "Capillary pressure curve",
    )
    if fig_path is not None:
        saved.append(str(fig_path))

    fig_path = _save_uq_kr_overlay_plot(
        results_root,
        kr_sw_series,
        "UQ Kr-Sw curve.png",
        "S_{w} (-)",
        "k_{r} (-)",
        "Relative permeability curve",
    )
    if fig_path is not None:
        saved.append(str(fig_path))

    if area_pair_sw_series:
        fig_path = _save_uq_area_overlay_plot(
            results_root,
            area_pair_sw_series,
            "UQ A_w_nw-Sw curve.png",
            "S_{w} (-)",
            "A_{wn} (cm^{-1})",
            "Fluid-fluid interfacial area curve",
        )
        if fig_path is not None:
            saved.append(str(fig_path))
    else:
        fig_path = _save_uq_overlay_plot(
            results_root,
            area_sw_series,
            "UQ A_w_nw-Sw curve.png",
            "S_{w} (-)",
            "A_{wn} (cm^{-1})",
            "Fluid-fluid interfacial area curve (Bulk + Film)",
        )
        if fig_path is not None:
            saved.append(str(fig_path))

    percentile_specs = [
        (
            pc_percentile_series,
            "UQ Percentile Pc-Sw curve.png",
            "S_{w} (-)",
            "P^{c} (kPa)",
            "Capillary pressure curves",
        ),
    ]

    for series, filename, xlab, ylab, title in percentile_specs:
        fig_path = _save_uq_percentile_plot(
            results_root,
            series,
            filename,
            xlab,
            ylab,
            title,
            outer_color="#9ecae1",
            inner_color="#4292c6",
            median_color="#08519c",
        )
        if fig_path is not None:
            saved.append(str(fig_path))

    if area_b_percentile_series:
        fig_path = _save_uq_area_percentile_plot(
            results_root,
            area_b_percentile_series,
            area_percentile_series,
            "UQ Percentile A_w_nw-Sw curve.png",
            "S_{w} (-)",
            "A_{wn} (cm^{-1})",
            "Fluid-fluid interfacial area curves",
        )
        if fig_path is not None:
            saved.append(str(fig_path))
    else:
        fig_path = _save_uq_percentile_plot(
            results_root,
            area_percentile_series,
            "UQ Percentile A_w_nw-Sw curve.png",
            "S_{w} (-)",
            "A_{wn} (cm^{-1})",
            "Fluid-fluid interfacial area curves (Bulk + Film)",
        )
        if fig_path is not None:
            saved.append(str(fig_path))

    fig_path = _save_uq_kr_percentile_plot(
        results_root,
        krw_percentile_series,
        krg_percentile_series,
        "UQ Percentile Kr-Sw curve.png",
        "S_{w} (-)",
        "k_{r} (-)",
        "Relative permeability curves",
    )
    if fig_path is not None:
        saved.append(str(fig_path))

    if not saved:
        raise ValueError(f"No valid quasi-static UQ results were found in: {results_root}")
    return saved


def _read_dynamic_case_results(case_folder: Path) -> pd.DataFrame | None:
    result_file = case_folder / "time_series.txt"
    if not result_file.exists():
        print(f"Skipping {case_folder.name}: missing {result_file.name}")
        return None

    df = read_dynamic_time_series(result_file)
    if df is None:
        print(f"Skipping {case_folder.name}: dynamic time-series columns could not be resolved")
        return None

    return df


def _save_uq_dynamic_plots(results_root: Path) -> list[str]:
    cases = _case_folders(results_root)
    if not cases:
        raise FileNotFoundError(f"No UQ dynamic case folders found in: {results_root}")

    case_results = []
    for case_folder in cases:
        df = _read_dynamic_case_results(case_folder)
        if df is None:
            continue

        time_values = df["Production_Time"].to_numpy(dtype=float)
        finite_time_indices = np.flatnonzero(np.isfinite(time_values))
        if finite_time_indices.size == 0:
            print(f"Skipping {case_folder.name}: no valid Production_Time values")
            continue

        case_last_time = float(time_values[finite_time_indices[-1]])
        if not np.isfinite(case_last_time) or case_last_time == 0:
            print(f"Skipping {case_folder.name}: invalid final Production_Time value")
            continue

        case_results.append((case_folder.name, df, case_last_time))

    if not case_results:
        raise ValueError(f"No valid dynamic UQ time values were found in: {results_root}")

    pcap_series = []
    sg_series = []
    aff_series = []
    aff_bulk_film_series = []
    aff_pair_series = []
    pcap_saturation_series = []
    aff_saturation_series = []
    aff_bulk_film_saturation_series = []
    aff_pair_saturation_series = []
    pcap_percentile_series = []
    sg_percentile_series = []

    for case_label, df, case_last_time in case_results:
        x = df["Production_Time"].to_numpy(dtype=float)
        x = x / case_last_time
        ave_pcap = df["ave_pcap"].to_numpy()
        sg_all = df["Sg_all"].to_numpy()
        sw_all = 1.0 - sg_all
        pcap_series.append((case_label, x, ave_pcap))
        sg_series.append((case_label, x, sg_all))
        pcap_saturation_series.append((case_label, sw_all, ave_pcap))
        bulk_column = "ave_aff_b" if "ave_aff_b" in df.columns else "ave_aff"
        if bulk_column in df.columns:
            ave_aff = df[bulk_column].to_numpy()
            aff_series.append((case_label, x, ave_aff))
            aff_saturation_series.append((case_label, sw_all, ave_aff))
        if "ave_aff_bf" in df.columns:
            # The updated solver reports the combined Bulk + Film value directly.
            # Do not add ave_aff_b again here.
            ave_aff_bf = df["ave_aff_bf"].to_numpy()
            aff_bulk_film_series.append((case_label, x, ave_aff_bf))
            aff_bulk_film_saturation_series.append((case_label, sw_all, ave_aff_bf))
            if bulk_column in df.columns:
                aff_pair_series.append((case_label, x, ave_aff, ave_aff_bf))
                aff_pair_saturation_series.append(
                    (case_label, sw_all, ave_aff, ave_aff_bf)
                )
        pcap_percentile_series.append((case_label, x, ave_pcap))
        sg_percentile_series.append((case_label, x, sg_all))

    for stale_name in (
        "UQ Bulk fluid-fluid interfacial area-Time curve.png",
        "UQ Bulk plus film fluid-fluid interfacial area-Time curve.png",
        "UQ Bulk fluid-fluid interfacial area-Saturation curve.png",
        "UQ Bulk plus film fluid-fluid interfacial area-Saturation curve.png",
    ):
        stale_path = results_root / stale_name
        if stale_path.exists():
            stale_path.unlink()

    saved = []
    plot_specs = [
        (
            pcap_series,
            "UQ Average capillary pressure-Time curve.png",
            "Normalized time",
            "Average P^{c} (Pa)",
            "Average capillary pressure over time",
        ),
        (
            sg_series,
            "UQ Gas saturation-Time curve.png",
            "Normalized time",
            "S_{g} (-)",
            "Gas saturation over time",
        ),
    ]
    if aff_pair_series:
        fig_path = _save_uq_area_overlay_plot(
            results_root,
            aff_pair_series,
            "UQ Fluid-fluid interfacial area-Time curve.png",
            "Normalized time",
            "A_{wn}",
            "Fluid-fluid interfacial area over time",
        )
        if fig_path is not None:
            saved.append(str(fig_path))
    elif aff_series:
        plot_specs.append(
            (
                aff_series,
                "UQ Bulk fluid-fluid interfacial area-Time curve.png",
                "Normalized time",
                "A_{wn,b}",
                "Bulk fluid-fluid interfacial area over time",
            )
        )
    elif aff_bulk_film_series:
        plot_specs.append(
            (
                aff_bulk_film_series,
                "UQ Bulk plus film fluid-fluid interfacial area-Time curve.png",
                "Normalized time",
                "A_{wn,bf}",
                "Bulk plus film fluid-fluid interfacial area over time",
            )
        )
    plot_specs.append(
        (
            pcap_saturation_series,
            "UQ Average capillary pressure-Saturation curve.png",
            "S_{w} (-)",
            "Average P^{c} (Pa)",
            "Average capillary pressure over wetting saturation",
        )
    )
    if aff_pair_saturation_series:
        fig_path = _save_uq_area_overlay_plot(
            results_root,
            aff_pair_saturation_series,
            "UQ Fluid-fluid interfacial area-Saturation curve.png",
            "S_{w} (-)",
            "A_{wn}",
            "Fluid-fluid interfacial area over wetting saturation",
        )
        if fig_path is not None:
            saved.append(str(fig_path))
    elif aff_saturation_series:
        plot_specs.append(
            (
                aff_saturation_series,
                "UQ Bulk fluid-fluid interfacial area-Saturation curve.png",
                "S_{w} (-)",
                "A_{wn,b}",
                "Bulk fluid-fluid interfacial area over wetting saturation",
            )
        )
    elif aff_bulk_film_saturation_series:
        plot_specs.append(
            (
                aff_bulk_film_saturation_series,
                "UQ Bulk plus film fluid-fluid interfacial area-Saturation curve.png",
                "S_{w} (-)",
                "A_{wn,bf}",
                "Bulk plus film fluid-fluid interfacial area over wetting saturation",
            )
        )

    for series, filename, xlab, ylab, title in plot_specs:
        fig_path = _save_uq_overlay_plot(results_root, series, filename, xlab, ylab, title)
        if fig_path is not None:
            saved.append(str(fig_path))

    percentile_specs = [
        (
            pcap_percentile_series,
            "UQ Percentile Average capillary pressure-Time curve.png",
            "Normalized time",
            "Average P^{c} (Pa)",
            "Average capillary pressure curves",
        ),
        (
            sg_percentile_series,
            "UQ Percentile Gas saturation-Time curve.png",
            "Normalized time",
            "S_{g} (-)",
            "Gas saturation curves",
        ),
    ]

    for series, filename, xlab, ylab, title in percentile_specs:
        fig_path = _save_uq_percentile_plot(results_root, series, filename, xlab, ylab, title)
        if fig_path is not None:
            saved.append(str(fig_path))

    if aff_pair_series:
        fig_path = _save_uq_area_percentile_plot(
            results_root,
            aff_series,
            aff_bulk_film_series,
            "UQ Percentile Fluid-fluid interfacial area-Time curve.png",
            "Normalized time",
            "A_{wn}",
            "Fluid-fluid interfacial area curves",
        )
        if fig_path is not None:
            saved.append(str(fig_path))

    if aff_pair_saturation_series:
        fig_path = _save_uq_area_percentile_plot(
            results_root,
            aff_saturation_series,
            aff_bulk_film_saturation_series,
            "UQ Percentile Fluid-fluid interfacial area-Saturation curve.png",
            "S_{w} (-)",
            "A_{wn}",
            "Fluid-fluid interfacial area curves",
        )
        if fig_path is not None:
            saved.append(str(fig_path))

    if not saved:
        raise ValueError(f"No valid dynamic UQ results were found in: {results_root}")
    return saved


def save_uq_results_plots(pnm_mode: str, base_dir) -> list[str]:
    results_root = _uq_results_root(pnm_mode, base_dir)
    if pnm_mode == "dynamic":
        return _save_uq_dynamic_plots(results_root)
    return _save_uq_quasi_static_plots(results_root)


def make_gif_from_folder(
    image_folder,
    save_folder,
    output_gif=None,
    duration=0.5,
    loop=0,
    extensions=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
):
    """
    Create a GIF from all images in a folder.

    Parameters
    ----------
    image_folder : str or Path
        Path to the folder containing images.
    output_gif : str or Path, optional
        Output GIF path. If None, saves as 'animation.gif' in the same folder.
    duration : float, optional
        Time per frame in seconds.
    loop : int, optional
        Number of GIF loops. 0 means infinite loop.
    extensions : tuple, optional
        Allowed image file extensions.

    Returns
    -------
    Path
        Path to the saved GIF.
    """
    image_folder = Path(image_folder)
    save_folder = Path(save_folder)
    if output_gif is None:
        output_gif = save_folder / "animation.gif"
    else:
        output_gif = save_folder / output_gif

    image_files = sorted(
        [f for f in image_folder.iterdir() if f.suffix.lower() in extensions]
    )

    if not image_files:
        print(f"No image files found in: {image_folder}")
    else:
        frames = [iio.imread(img) for img in image_files]
        iio.imwrite(output_gif, frames, duration=duration, loop=loop)

    return output_gif

def _handle_remove_readonly(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def clear_folder(folder, retries=5, delay=0.5):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    for item in folder.iterdir():
        for i in range(retries):
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item, onerror=_handle_remove_readonly)
                break
            except PermissionError:
                gc.collect()
                time.sleep(delay)
        else:
            raise PermissionError(f"Could not remove: {item}")

def plot_3d_binary_image(
    im,
    filename="3d_image.png",
    voxel_size=1.0,
    color="#7aa6d8",
    opacity=1.0,
    background="white",
    window_size=(1400, 1000),
    camera_position=None,
    camera_azimuth=180,
    camera_elevation=20,
    camera_roll=0,
):
    im = np.asarray(im, dtype=bool)
    im = np.transpose(im, (2, 1, 0))   # swap x and z

    if im.ndim != 3:
        raise ValueError("im must be a 3D array")

    # hard cleanup of old windows
    pv.close_all()

    grid = pv.ImageData()
    grid.dimensions = np.array(im.shape) + 1
    grid.spacing = (voxel_size, voxel_size, voxel_size)
    grid.origin = (0.0, 0.0, 0.0)
    grid.cell_data["phase"] = im.astype(np.uint8).ravel(order="F")

    surf = grid.threshold(0.5, scalars="phase")

    # IMPORTANT: off_screen=True and DO NOT call show()
    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.set_background(background)

    p.add_mesh(
        surf,
        color=color,
        opacity=opacity,
        show_edges=False,
        smooth_shading=True,
        ambient=0.15,
        diffuse=0.75,
        specular=0.40,
        specular_power=25,
    )

    p.enable_anti_aliasing()
    p.reset_camera()

    if camera_position is not None:
        p.camera_position = camera_position
    else:
        p.camera.Azimuth(camera_azimuth)
        p.camera.Elevation(camera_elevation)
        p.camera.Roll(camera_roll)
    p.add_axes()
    p.render()
    p.screenshot(filename)

    # extra cleanup
    p.close()
    pv.close_all()
    del p

    print(f"Saved image to: {filename}")


def extract_scientific_number(name):
    match = re.search(
        r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?',
        name,
        re.IGNORECASE
    )
    return float(match.group()) if match else None


def enforce_dual_percolation(
    im, axis=0, connectivity=3, penalty=1000.0,
    keep_spanning_only=True,          # removes islands (best for avoiding isolated PN pores)
    ensure_face_contact=True,         # makes sure each phase actually touches both faces
    max_iters=5
):
    """
    Make pore(True) and solid(False) both:
      (a) span from one face to the other along `axis` (percolate), AND
      (b) optionally remove islands so extraction doesn't create isolated pores.

    This does NOT guarantee zero isolated pores in the extracted PNM (segmentation can still do it),
    but it dramatically reduces them.

    Steps:
      1) If a phase doesn't span, carve a minimal shortest-path bridge.
      2) Optionally keep only spanning components for each phase (removes islands).
      3) Repeat a few times since enforcing one phase can break the other.
    """

    im2 = im.copy()

    # --- connectivity structuring element for labeling ---
    if im2.ndim == 2:
        structure = np.ones((3, 3), bool) if connectivity > 1 else None
    else:
        structure = np.ones((3, 3, 3), bool) if connectivity > 1 else None

    def spanning_labels(mask):
        L, _ = label(mask, structure=structure)
        inlet  = np.take(L, 0, axis=axis)
        outlet = np.take(L, -1, axis=axis)
        labs = np.intersect1d(np.unique(inlet[inlet > 0]), np.unique(outlet[outlet > 0]))
        return L, labs

    def spanning_exists(mask):
        _, labs = spanning_labels(mask)
        return labs.size > 0

    def endpoints_from_faces(target_mask):
        # choose endpoints that minimize distance (more stable than "first voxel")
        sl_in  = [slice(None)] * target_mask.ndim
        sl_out = [slice(None)] * target_mask.ndim
        sl_in[axis] = 0
        sl_out[axis] = -1

        Ain = np.argwhere(target_mask[tuple(sl_in)])
        Aout = np.argwhere(target_mask[tuple(sl_out)])
        if Ain.size == 0 or Aout.size == 0:
            return None, None

        # pick pair with minimal squared distance in the face plane (cheap heuristic)
        # (avoids a long, unnecessary bridge)
        a0 = Ain[np.argmin(np.sum((Ain - Ain.mean(axis=0))**2, axis=1))]
        b0 = Aout[np.argmin(np.sum((Aout - a0)**2, axis=1))]

        a = a0.tolist(); a.insert(axis, 0)
        b = b0.tolist(); b.insert(axis, target_mask.shape[axis] - 1)
        return tuple(a), tuple(b)

    def add_minimal_bridge(im_bool, target_is_pore=True):
        target = im_bool if target_is_pore else ~im_bool

        start, end = endpoints_from_faces(target)
        if start is None:
            # If the target phase doesn't touch a face at all, force a seed voxel
            # at each face at the center line, then bridge between them.
            center = [s//2 for s in target.shape]
            a = center.copy(); a[axis] = 0
            b = center.copy(); b[axis] = target.shape[axis]-1
            start, end = tuple(a), tuple(b)

        # cost: very cheap in target, expensive in opposite phase
        cost = np.where(target, 1.0, penalty).astype(float)

        path, _ = route_through_array(
            cost, start, end,
            fully_connected=(connectivity > 1),
            geometric=True
        )
        path = np.asarray(path, dtype=int)

        if target_is_pore:
            im_bool[tuple(path.T)] = True
        else:
            im_bool[tuple(path.T)] = False
        return im_bool

    def keep_only_spanning(mask):
        L, labs = spanning_labels(mask)
        if labs.size == 0:
            return mask  # nothing spans, leave as-is
        return np.isin(L, labs)

    def force_face_contact(im_bool):
        # ensures each phase has at least one voxel on each face;
        # reduces cases where PN inlet/outlet marking creates "disconnected" subnetworks
        shp = im_bool.shape
        center = [s//2 for s in shp]

        a = tuple([0 if d == axis else center[d] for d in range(im_bool.ndim)])
        b = tuple([shp[axis]-1 if d == axis else center[d] for d in range(im_bool.ndim)])

        # make sure pore touches both faces at least somewhere (single voxel)
        im_bool[a] = True
        im_bool[b] = True
        # and solid as well (neighbor voxel if possible)
        a2 = list(a); a2[axis] = min(1, shp[axis]-1)
        b2 = list(b); b2[axis] = max(shp[axis]-2, 0)
        im_bool[tuple(a2)] = False
        im_bool[tuple(b2)] = False
        return im_bool

    # ---- Iterate: bridging one phase can break the other ----
    for _ in range(max_iters):

        if ensure_face_contact:
            im2 = force_face_contact(im2)

        # Ensure pore spans
        if not spanning_exists(im2):
            im2 = add_minimal_bridge(im2, target_is_pore=True)

        # Ensure solid spans
        if not spanning_exists(~im2):
            im2 = add_minimal_bridge(im2, target_is_pore=False)

        if keep_spanning_only:
            # Remove islands in each phase -> fewer isolated pores/solids after extraction
            im2 = keep_only_spanning(im2)          # keep spanning pores, remove pore islands
            im2 = ~keep_only_spanning(~im2)        # keep spanning solid, remove solid islands

        # If both span now, we can stop early
        if spanning_exists(im2) and spanning_exists(~im2):
            break

    return im2

def keep_percolating_pore(im, axis=0, connectivity=3):
    """
    Return image where only pores connected from inlet to outlet are kept.
    Everything else becomes solid.

    Parameters
    ----------
    im : bool array
        True = pore, False = solid

    axis : int
        Flow direction (0=x, 1=y, 2=z)

    connectivity : int
        1=5, 2=18, 3=26 neighbor

    Returns
    -------
    im_clean : bool array
        True=pore connected through domain, others solid
    """

    # label pore space
    L = label(im, connectivity=connectivity)

    # labels touching inlet and outlet
    inlet  = L.take(indices=0, axis=axis)
    outlet = L.take(indices=-1, axis=axis)

    lab_in  = np.unique(inlet[inlet > 0])
    lab_out = np.unique(outlet[outlet > 0])

    good = np.intersect1d(lab_in, lab_out)

    im_clean = np.isin(L, good)

    return im_clean


def clean_floating_phases(im,
                          frac_solid=0.001,
                          frac_pore=0.001,
                          connectivity=3):
    """
    Remove tiny floating grains (solid) and tiny isolated voids (pores)
    after segmentation.

    Parameters
    ----------
    im : ndarray (bool)
        3D binary image, True = pore, False = solid

    frac_solid : float
        Fraction of total solid voxels used as min size for solids

    frac_pore : float
        Fraction of total pore voxels used as min size for pores

    connectivity : int
        1 -> 6-neighbor, 2 -> 18-neighbor, 3 -> 26-neighbor

    Returns
    -------
    im_clean : ndarray (bool)
        Cleaned image (True = pore, False = solid)

    info : dict
        Thresholds and voxel counts used
    """

    im_clean = im.copy()

    # ----- 1. Remove tiny SOLID grains -----
    solid = ~im_clean
    n_solid = int(solid.sum())

    if n_solid > 0:
        min_solid = max(1, int(frac_solid * n_solid))

        solid = remove_small_objects(
            solid,
            min_size=min_solid,
            connectivity=connectivity
        )

        im_clean = ~solid
    else:
        min_solid = 0

    # ----- 2. Remove tiny PORE islands -----
    pore = im_clean
    n_pore = int(pore.sum())

    if n_pore > 0:
        min_pore = max(1, int(frac_pore * n_pore))

        pore = remove_small_objects(
            pore,
            min_size=min_pore,
            connectivity=connectivity
        )

        im_clean = pore
    else:
        min_pore = 0

    info = {
        "n_solid": n_solid,
        "n_pore": n_pore,
        "min_solid": min_solid,
        "min_pore": min_pore
    }

    return im_clean



def remove_isolated_solids_fraction(im,
                                    frac=0.001,
                                    connectivity=3,
                                    remove_voids=False,
                                    frac_void=None):
    """
    Convert isolated solid clusters to pore using a fraction of total solid voxels
    as the minimum size threshold. Optionally also remove isolated voids.

    Parameters
    ----------
    im : ndarray (bool)
        3D binary image, True = pore, False = solid

    frac : float
        Fraction of total solid voxels used as min_size for solids

    connectivity : int
        1 -> 6-neighbor, 2 -> 18-neighbor, 3 -> 26-neighbor

    remove_voids : bool
        If True, also remove isolated pore clusters

    frac_void : float or None
        Fraction of total pore voxels used as min_size for void removal.
        If None, the same 'frac' value is used.

    Returns
    -------
    im_clean : ndarray (bool)
        Cleaned image (True = pore, False = solid)

    info : dict
        Dictionary with thresholds used
    """

    # ---------- Remove isolated SOLIDS ----------
    solid = ~im
    n_solid = int(solid.sum())

    if n_solid == 0:
        return im.copy(), {"min_solid": 0, "min_void": 0}

    min_solid = max(1, int(frac * n_solid))

    solid_clean = remove_small_objects(
        solid.astype(bool),
        min_size=min_solid,
        connectivity=connectivity
    )

    im_clean = ~solid_clean

    # ---------- Optionally remove isolated VOIDS ----------
    min_void = 0

    if remove_voids:
        pore = im_clean
        n_pore = int(pore.sum())

        if n_pore > 0:
            f_void = frac if frac_void is None else frac_void
            min_void = max(1, int(f_void * n_pore))

            pore_clean = remove_small_objects(
                pore.astype(bool),
                min_size=min_void,
                connectivity=connectivity
            )

            im_clean = pore_clean

    return im_clean




def coarsen_image_majority(im, factor, threshold=0.5):
    """
    Coarsen a 3D binary image using block averaging and majority rule.

    Parameters
    ----------
    im : ndarray (bool or 0/1)
        3D binary image where True = pore, False = solid
    factor : int
        Coarsening factor (voxel size multiplier)
    threshold : float, optional
        Porosity threshold for pore classification (default = 0.5)

    Returns
    -------
    im_coarse : ndarray (bool)
        Coarsened binary image (True = pore, False = solid)
    phi : ndarray (float)
        Block-averaged porosity field
    """

    im = im.astype(float)
    sx, sy, sz = im.shape

    # Trim so dimensions are divisible by factor
    sx2 = (sx // factor) * factor
    sy2 = (sy // factor) * factor
    sz2 = (sz // factor) * factor

    im_trim = im[:sx2, :sy2, :sz2]

    # Block averaging → porosity
    phi = im_trim.reshape(
        sx2 // factor, factor,
        sy2 // factor, factor,
        sz2 // factor, factor
    ).mean(axis=(1, 3, 5))

    # Majority rule threshold
    im_coarse = phi > threshold

    return im_coarse, phi




def _count_face_contacts_between_masks_6n(A, B):
    """Count 6-neighbor face adjacencies between boolean masks A and B."""
    if (not A.any()) or (not B.any()):
        return 0
    cnt = 0
    cnt += np.count_nonzero(A[:-1, :, :] & B[1:, :, :]) + np.count_nonzero(B[:-1, :, :] & A[1:, :, :])
    cnt += np.count_nonzero(A[:, :-1, :] & B[:, 1:, :]) + np.count_nonzero(B[:, :-1, :] & A[:, 1:, :])
    cnt += np.count_nonzero(A[:, :, :-1] & B[:, :, 1:]) + np.count_nonzero(B[:, :, :-1] & A[:, :, 1:])
    return int(cnt)


def _safe_extend_array(arr, new_len):
    """Extend per-pore or per-throat array along axis 0, preserving dtype and shape."""
    arr = np.asarray(arr)
    old_len = arr.shape[0]
    if old_len == new_len:
        return arr
    new_shape = (new_len,) + arr.shape[1:]
    dt = arr.dtype

    if dt == np.bool_:
        out = np.zeros(new_shape, dtype=dt)
    elif np.issubdtype(dt, np.integer):
        out = np.zeros(new_shape, dtype=dt)
    elif np.issubdtype(dt, np.floating):
        out = np.full(new_shape, np.nan, dtype=dt)
    else:
        out = np.empty(new_shape, dtype=dt)
        out[:] = None

    out[:old_len, ...] = arr
    return out



def _bbox_slices_from_mask(mask):
    """Return tight (z,y,x) bbox slices for a 3D boolean mask; None if empty."""
    idx = np.where(mask)
    if idx[0].size == 0:
        return None
    z0, z1 = int(idx[0].min()), int(idx[0].max()) + 1
    y0, y1 = int(idx[1].min()), int(idx[1].max()) + 1
    x0, x1 = int(idx[2].min()), int(idx[2].max()) + 1
    return (slice(z0, z1), slice(y0, y1), slice(x0, x1))



def safe_boundary_pores_with_area(pn, regions, args, axis=0):
    """
    Returns:
      left_pore_ids,  left_area_vox,
      right_pore_ids, right_area_vox

    Area definition:
      For each chosen boundary pore, its "area" is the number of voxels on the
      boundary face belonging to that pore's REGION (from `regions`).

    Uses:
      - pn['pore.region_label'] (region id for each pore)
      - pn['pore.coords'] (pore centroid coords)
      - regions (3D int label image; 0 = background/solid)
      - args['min_boundary_face_area'] (min face voxels per region)
    """
    min_face_area = 1#int(float(args.get('min_boundary_face_area', 1)))

    coords = pn['pore.coords'].astype(float)
    reg_labels = pn['pore.region_label'].astype(np.int64)
    axc = coords[:, axis]

    def one_face(face_index, pick_min=True):
        # 2D region labels on the face
        slicer = [slice(None)] * 3
        slicer[axis] = face_index
        face_map = regions[tuple(slicer)]

        # region IDs present on this face + their voxel counts (area in voxels)
        face_regs, counts = np.unique(face_map, return_counts=True)

        # keep only real pore regions (exclude 0) and those with enough area
        valid = (face_regs > 0) & (counts >= min_face_area)
        face_regs = face_regs[valid]
        counts = counts[valid]

        if face_regs.size == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        # For each face region, pick ONE pore (most extreme along axis)
        chosen_pores = []
        chosen_areas = []

        for reg, area_vox in zip(face_regs, counts):
            pores_in_reg = np.where(reg_labels == reg)[0]
            if pores_in_reg.size == 0:
                continue

            if pick_min:
                pore = pores_in_reg[np.argmin(axc[pores_in_reg])]
            else:
                pore = pores_in_reg[np.argmax(axc[pores_in_reg])]

            chosen_pores.append(pore)
            chosen_areas.append(int(area_vox))

        if len(chosen_pores) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        chosen_pores = np.asarray(chosen_pores, dtype=np.int64)
        chosen_areas = np.asarray(chosen_areas, dtype=np.int64)

        # If multiple regions ended up choosing the same pore, sum areas
        uniq_pores, inv = np.unique(chosen_pores, return_inverse=True)
        area_sum = np.zeros_like(uniq_pores, dtype=np.int64)
        np.add.at(area_sum, inv, chosen_areas)

        return uniq_pores, area_sum

    left_pore_ids,  left_area_vox  = one_face(face_index=0,  pick_min=True)   # xmin face
    right_pore_ids, right_area_vox = one_face(face_index=-1, pick_min=False)  # xmax face

    return left_pore_ids, left_area_vox, right_pore_ids, right_area_vox

def pore_y_to_im(
    y, im, pn2, net2,
    fill_value=np.nan,
    filename="output_without_naming"
):
    """
    Map per-pore values y back to a voxel image with same shape as im
    using region labels.

    Parameters
    ----------
    y : (Np,) array
        Per-pore values (pn2 ordering)
    im : (nx, ny, nz) bool array
        Reference image (only used for shape)
    pn2 : OpenPNM Network
        Must contain pn2['pore.region_label']
    net2 : PoreSpy snow2 output
        Must contain net2['regions']
    fill_value : float
        Value for voxels not mapped to any pore region (e.g. solids/background)

    Returns
    -------
    out : (nx, ny, nz) ndarray
        Voxel image with region-wise constant values
    """
    shape = im.shape
    y = np.asarray(y)

    regions = np.asarray(net2['regions'])
    if regions.shape != shape:
        raise ValueError("net2['regions'] and im must have the same shape")

    if y.size != pn2.Np:
        raise ValueError("y must have length pn2.Np")

    # --- initialize output ---
    out = np.full(shape, fill_value, dtype=np.float32)

    # --- build LUT: region_label -> pore value ---
    pore_labels = np.asarray(pn2['pore.region_label']).astype(regions.dtype)
    max_lab = int(regions.max())

    lut = np.full(max_lab + 1, fill_value, dtype=np.float32)
    lut[pore_labels] = y

    # --- broadcast back to voxels ---
    out[:] = lut[regions]

    ps.io.to_vtk(out, filename=filename)



def format_general_params(params):
    """Formats the simple key-value parameters at the start of the file."""
    output = []
    for key, value in params.items():
        output.append(f"[{key.upper()}]\n{value}")
    return "\n".join(output)

def format_table(section_title, comments, data, int_col = 4, col_separator='\t'):
    """
    Formats a data table for a specific section (e.g., PORE_BODY).
    
    Args:
        section_title (str): The section title, e.g., "[PORE_BODY]".
        comments (str): The multi-line string containing the header comments.
        data (list of list): A list where each inner list represents a row of data.
        col_separator (str): The character to use for separating columns.
    
    Returns:
        str: The fully formatted section as a string.
    """
    with io.StringIO() as buffer:
        buffer.write(f"{section_title}\n")
        buffer.write(f"{comments.strip()}\n")
        
        # CORRECTED LINE: The loop now iterates over the 'data' list.
        for row in data:
            # Convert all items in the row to strings and join them
            str_row = [str(int(item)) if i < int_col else str(item) for i, item in enumerate(row)]
            buffer.write(f"{col_separator.join(str_row)}\n")
        buffer.write(f";\n")
        return buffer.getvalue()


def add_ghost_layer(
    pore_body_data,
    pore_throat_data,
    boundary_data,
    coordination_data,
    boundary_type_for_ghost=0,
    pore_radius_col=2,
    pore_length_col=3,
    coord_shift_axis=None,
    coord_shift_sign=1,
    ghost_throat_length=None,
):
    """
    Add one ghost pore body for each boundary pore whose boundary type equals
    boundary_type_for_ghost, usually outlet boundary type = 0.

    What this does:
    1. Finds boundary rows like:
           [BoundaryPoreBodyID, BoundaryType, Value]
       where BoundaryType == 0.

    2. For each such boundary pore:
       - creates a ghost pore body by copying the original pore-body properties
       - assigns a new pore-body ID
       - creates a ghost coordinate row by copying the original coordinate row
       - creates a new pore throat connecting original pore -> ghost pore
       - replaces the boundary pore ID in boundary_data with the ghost pore ID

    3. The new throat length is set to ghost_throat_length when supplied.
       Otherwise, it falls back to the maximum pore radius among all selected
       boundary pores.

    Assumed column structure:
        pore_body_data:
            [ID, Shape, Radius, Length, ...]

        pore_throat_data:
            [ID, Inlet, Outlet, Shape, Radius, Length, ...]

        boundary_data:
            [PoreBodyID, BoundaryType, BoundaryValue]

        coordination_data:
            [PoreBodyID, X, Y, Z]
    """

    def as_int(x):
        return int(float(x))

    def as_float(x):
        return float(x)

    # Make copies so the original input lists are not modified in place
    new_pore_body_data = [list(row) for row in pore_body_data]
    new_pore_throat_data = [list(row) for row in pore_throat_data]
    new_boundary_data = [list(row) for row in boundary_data]
    new_coordination_data = [list(row) for row in coordination_data]

    # Fast lookup dictionaries
    pore_body_by_id = {
        as_int(row[0]): row for row in new_pore_body_data
    }

    coord_by_id = {
        as_int(row[0]): row for row in new_coordination_data
    }

    # Find boundary rows where the second column is boundary_type_for_ghost
    target_boundary_rows = [
        row for row in new_boundary_data
        if as_int(row[1]) == boundary_type_for_ghost
    ]

    if not target_boundary_rows:
        print(f"No boundary rows with type {boundary_type_for_ghost} were found.")
        return (
            new_pore_body_data,
            new_pore_throat_data,
            new_boundary_data,
            new_coordination_data,
        )

    target_pore_ids = [as_int(row[0]) for row in target_boundary_rows]

    # Check that all target boundary pore IDs exist in pore_body_data and coordination_data
    missing_pore_body = [pid for pid in target_pore_ids if pid not in pore_body_by_id]
    missing_coord = [pid for pid in target_pore_ids if pid not in coord_by_id]

    if missing_pore_body:
        raise ValueError(
            f"These boundary pore IDs are missing from pore_body_data: {missing_pore_body[:20]}"
        )

    if missing_coord:
        raise ValueError(
            f"These boundary pore IDs are missing from coordination_data: {missing_coord[:20]}"
        )

    # Default ghost throat length preserves the previous behavior unless the
    # caller supplies a domain-based length.
    default_ghost_throat_length = max(
        as_float(pore_body_by_id[pid][pore_radius_col])
        for pid in target_pore_ids
    )
    if ghost_throat_length is None:
        ghost_throat_length_value = default_ghost_throat_length
    else:
        ghost_throat_length_value = float(ghost_throat_length)
        if not np.isfinite(ghost_throat_length_value) or ghost_throat_length_value <= 0:
            raise ValueError("ghost_throat_length must be a positive finite value.")

    # Next available IDs
    next_pore_id = max(as_int(row[0]) for row in new_pore_body_data) + 1
    next_throat_id = max(as_int(row[0]) for row in new_pore_throat_data) + 1

    # Map original boundary pore ID -> ghost pore ID
    ghost_id_map = {}

    for original_pore_id in target_pore_ids:
        ghost_pore_id = next_pore_id
        next_pore_id += 1

        ghost_id_map[original_pore_id] = ghost_pore_id

        original_pore_body = list(pore_body_by_id[original_pore_id])
        original_coord = list(coord_by_id[original_pore_id])

        # ------------------------------------------------------------
        # 1. Add ghost pore body
        # Copy original pore body properties, only replace ID
        # ------------------------------------------------------------
        ghost_pore_body = list(original_pore_body)
        ghost_pore_body[0] = ghost_pore_id
        new_pore_body_data.append(ghost_pore_body)

        # ------------------------------------------------------------
        # 2. Add ghost coordinate
        # By default, exact same coordinate.
        # Optional: shift along x/y/z if coord_shift_axis is set.
        # ------------------------------------------------------------
        ghost_coord = list(original_coord)
        ghost_coord[0] = ghost_pore_id

        if coord_shift_axis is not None:
            # coord columns: 0 = ID, 1 = X, 2 = Y, 3 = Z
            axis_to_col = {
                "x": 1,
                "y": 2,
                "z": 3,
                0: 1,
                1: 2,
                2: 3,
            }

            if coord_shift_axis not in axis_to_col:
                raise ValueError("coord_shift_axis must be one of: 'x', 'y', 'z', 0, 1, 2, or None")

            shift_col = axis_to_col[coord_shift_axis]
            ghost_coord[shift_col] = (
                as_float(ghost_coord[shift_col])
                + coord_shift_sign * ghost_throat_length_value
            )

        new_coordination_data.append(ghost_coord)

        # ------------------------------------------------------------
        # 3. Add ghost throat
        # Format:
        # [ThroatID, Inlet, Outlet, Shape, Radius, Length, ...]
        #
        # Shape/radius/angles are copied from the original pore body.
        # Length is the requested ghost throat length.
        # ------------------------------------------------------------
        pore_shape = original_pore_body[1]
        pore_radius = original_pore_body[pore_radius_col]

        # Any columns after pore Length are copied as angle-like geometry columns
        extra_geometry_cols = original_pore_body[pore_length_col + 1:]

        ghost_throat = [
            next_throat_id,
            original_pore_id,
            ghost_pore_id,
            pore_shape,
            pore_radius,
            ghost_throat_length_value,
        ] + extra_geometry_cols

        next_throat_id += 1
        new_pore_throat_data.append(ghost_throat)

    # ------------------------------------------------------------
    # 4. Replace boundary pore IDs with ghost pore IDs
    # Only for boundary rows where BoundaryType == 0
    # ------------------------------------------------------------
    for row in new_boundary_data:
        if as_int(row[1]) == boundary_type_for_ghost:
            old_id = as_int(row[0])
            row[0] = ghost_id_map[old_id]

    print(f"Added {len(ghost_id_map)} ghost pore bodies.")
    print(f"Added {len(ghost_id_map)} ghost pore throats.")
    print(f"Ghost throat length = {ghost_throat_length_value}")

    return (
        new_pore_body_data,
        new_pore_throat_data,
        new_boundary_data,
        new_coordination_data,
    )


def generate_input_file(
    pore_body_data,
    pore_throat_data,
    boundary_data,
    coordination_data,
    output_filename="ExtractedPNM",
    add_ghost=True,
    ghost_throat_length=None,
):
    """
    Assembles all formatted sections into the final input file string.
    """

    if add_ghost:
        (
            pore_body_data,
            pore_throat_data,
            boundary_data,
            coordination_data,
        ) = add_ghost_layer(
            pore_body_data=pore_body_data,
            pore_throat_data=pore_throat_data,
            boundary_data=boundary_data,
            coordination_data=coordination_data,
            boundary_type_for_ghost=0,
            coord_shift_axis=None,   # exact same coordinate as boundary pore
            ghost_throat_length=ghost_throat_length,
        )

    # General simulation parameters
    general_params = {
        "PHASES": 2,
        "COMPONENTS": 2,
        "START_TIME": 0,
        "END_TIME": 1000000,
        "TIME_STEP": 1,
        "REPORT_STRAT_TIME": 0,
        "REPORT_END_TIME": 50,
        "REPORT_TIME_STEP": 10,
        "MIN_PRESSURE": 105,
        "MAX_PRESSURE": 105,
        "MIN_TEMP": 311,
        "MAX_TEMP": 311,
        "DELTA_PRES": 0.1,
        "DELTA_TEMP": 0.2,
    }

    # --- Define Comment Blocks for each Table ---
    pore_body_comments = """
;------------------------------------------------------------------------------------------------------------
;ID     Shape    Radius     Length   Corner_Angle 1     Corner_Angle 2      Corner_Angle 2
;      (0,1,2)  (Micro)     (Micro)      (Degree)          (Degree)          (Degree)
;------------------------------------------------------------------------------------------------------------
"""

    pore_throat_comments = """
;------------------------------------------------------------------------------------------------------------
;ID   Inlet      Outlet      Shape      Radius     Length     Corner_Angle01     Corner_Angle02  Corner_Angle03
;                           (0,1,2)    (Micro)     (Micro)    (Degree)          (Degree)         (Degree)
;------------------------------------------------------------------------------------------------------------
"""

    boundary_comments = """
;------------------------------------------------------------------------------------------------------------
;BoundaryPoreBodyID  BoundaryType     FixedPressure(MPa)/FixedFlowRate   
;                    (1, 0)
;------------------------------------------------------------------------------------------------------------
"""

    coordination_comments = """
;------------------------------------------------------------------------------------------------------------
; PoreBody_ID    X             Y                 Z
;------------------------------------------------------------------------------------------------------------
"""

    file_content = []

    file_content.append(format_general_params(general_params))

    file_content.append(
        format_table("[PORE_BODY]", pore_body_comments, pore_body_data, int_col=2).rstrip()
    )

    file_content.append(
        format_table("[PORE_THROAT]", pore_throat_comments, pore_throat_data, int_col=4).rstrip()
    )

    file_content.append(
        format_table("[BOUNDARY]", boundary_comments, boundary_data, int_col=3).rstrip()
    )

    file_content.append(
        format_table("[COORDINATION]", coordination_comments, coordination_data, int_col=1).rstrip()
    )

    notes = """
;Notes:
;
;******* Pore body ******* 
; Shape: 0 - cylindrical; 1 - Square;2 - Triangular.
;*************************
;
;******* Pore throat ******* 
; Shape: 0 - cylindrical; 1 - Square;2 - Triangular.
;*************************** 
;
;******* Boundary ******* 
; Shape: 0 - cylindrical; 1 - Square;2 - Triangular.
; BoundaryType: -1 - inlet; 0 - outlet.
;************************ 
"""
    file_content.append(notes.strip())

    output_string = "\n".join(file_content)

    with open(output_filename, "w") as f:
        f.write(output_string)

    print(f"\n--- File successfully saved as '{output_filename}' ---")
    print("Finish generating PNM file!!!")
    
def _get_net_array(net, key):
    try:
        return np.asarray(net[key])
    except Exception as e:
        raise KeyError(f"Network does not contain '{key}'") from e


def _estimate_default_cube_size(coords, conns, fraction=0.25):
    if conns.size > 0:
        lengths = np.linalg.norm(coords[conns[:, 1]] - coords[conns[:, 0]], axis=1)
        lengths = lengths[np.isfinite(lengths) & (lengths > 0)]
        if lengths.size > 0:
            return fraction * np.median(lengths)

    span = coords.max(axis=0) - coords.min(axis=0)
    span = span[span > 0]
    if span.size > 0:
        return fraction * np.min(span)

    return 1.0


def _safe_sizes(values, fallback):
    values = np.asarray(values, dtype=float).ravel()
    good = np.isfinite(values) & (values > 0)
    if np.any(good):
        med = np.nanmedian(values[good])
        if np.isfinite(med) and med > 0:
            fallback = med
    return np.where(good, values, fallback)


def _right_vertical_scalar_bar_args(title):
    return {
        "title": title,
        "title_font_size": PLOT_3D_TEXT_FONT_SIZE,
        "label_font_size": PLOT_BASE_FONT_SIZE,
        "vertical": True,
        "position_x": 0.88,
        "position_y": 0.10,
        "width": 0.08,
        "height": 0.80,
    }


def _mark_modified(mesh):
    if hasattr(mesh, "modified"):
        mesh.modified()
    elif hasattr(mesh, "Modified"):
        mesh.Modified()


def _apply_camera(
    plotter,
    camera_position=None,
    camera_azimuth=0.0,
    camera_elevation=0.0,
    camera_roll=0.0,
):
    """
    Apply either an explicit camera_position or incremental azimuth/elevation/roll.
    Assumes reset_camera() has already been called.
    """
    if camera_position is not None:
        plotter.camera_position = camera_position
    else:
        if camera_azimuth:
            plotter.camera.Azimuth(camera_azimuth)
        if camera_elevation:
            plotter.camera.Elevation(camera_elevation)
        if camera_roll:
            plotter.camera.Roll(camera_roll)


def _get_per_pore_sizes(net, coords, conns, pore_size, size_factor=1.0):
    npores = coords.shape[0]

    if pore_size is None:
        sizes = np.full(npores, _estimate_default_cube_size(coords, conns), dtype=float)
    elif isinstance(pore_size, str):
        try:
            sizes = np.asarray(net[pore_size], dtype=float).ravel()
        except Exception as e:
            raise KeyError(
                f"Could not find pore size property '{pore_size}'. "
                f"Try pore_size=None or another property name."
            ) from e
    else:
        sizes = np.asarray(pore_size, dtype=float).ravel()

    if sizes.size != npores:
        raise ValueError(f"pore_size must have one value per pore ({npores}), got {sizes.size}")

    fallback = _estimate_default_cube_size(coords, conns)
    sizes = _safe_sizes(sizes, fallback)
    return size_factor * sizes


def _get_per_throat_sizes(net, coords, conns, throat_size, size_factor=1.0):
    nthroats = conns.shape[0]

    if nthroats == 0:
        return np.array([], dtype=float)

    if throat_size is None:
        fallback = 0.4 * _estimate_default_cube_size(coords, conns)
        sizes = np.full(nthroats, fallback, dtype=float)
    elif isinstance(throat_size, str):
        try:
            sizes = np.asarray(net[throat_size], dtype=float).ravel()
        except Exception as e:
            raise KeyError(
                f"Could not find throat size property '{throat_size}'. "
                f"Try throat_size=None or another property name."
            ) from e
    else:
        sizes = np.asarray(throat_size, dtype=float).ravel()

    if sizes.size != nthroats:
        raise ValueError(
            f"throat_size must have one value per throat ({nthroats}), got {sizes.size}"
        )

    fallback = 0.4 * _estimate_default_cube_size(coords, conns)
    sizes = _safe_sizes(sizes, fallback)
    return size_factor * sizes


def _segment_basis(p0, p1):
    v = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    L = np.linalg.norm(v)
    if not np.isfinite(L) or L <= 0:
        raise ValueError("Encountered a zero-length throat")

    ex = v / L

    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ex, helper)) > 0.95:
        helper = np.array([0.0, 1.0, 0.0])

    ey = np.cross(helper, ex)
    ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    ez /= np.linalg.norm(ez)

    return ex, ey, ez, L


def _make_square_throat_prism(p0, p1, width):
    ex, ey, ez, L = _segment_basis(p0, p1)
    center = 0.5 * (np.asarray(p0) + np.asarray(p1))

    base = pv.Cube(center=(0.0, 0.0, 0.0), x_length=1.0, y_length=1.0, z_length=1.0)

    M = np.eye(4)
    M[:3, 0] = ex * L
    M[:3, 1] = ey * width
    M[:3, 2] = ez * width
    M[:3, 3] = center

    return base.transform(M, inplace=False)


def _build_square_throat_mesh(
    coords,
    conns,
    throat_sizes,
    throat_scalars=None,
    scalar_name="throat_values",
):
    if conns.size == 0:
        return pv.PolyData()

    if throat_scalars is not None:
        throat_scalars = np.asarray(throat_scalars, dtype=float).ravel()
        if throat_scalars.size != conns.shape[0]:
            raise ValueError(
                f"throat_scalars must have one value per throat ({conns.shape[0]}), "
                f"got {throat_scalars.size}"
            )

    pieces = []
    for i, (a, b) in enumerate(conns):
        p0 = coords[a]
        p1 = coords[b]
        w = float(throat_sizes[i])

        if not np.isfinite(w) or w <= 0:
            continue

        try:
            prism = _make_square_throat_prism(p0, p1, w)
            if throat_scalars is not None:
                val = float(throat_scalars[i])
                prism.point_data[scalar_name] = np.full(prism.n_points, val, dtype=float)
            pieces.append(prism)
        except ValueError:
            continue

    if len(pieces) == 0:
        return pv.PolyData()

    merged = pieces[0]
    for piece in pieces[1:]:
        merged = merged.merge(piece, merge_points=False)

    return merged


def _build_throat_polydata(coords, conns, throat_scalars=None, scalar_name="throat_values"):
    pts = coords[conns].reshape(-1, 3)
    nt = conns.shape[0]

    lines = np.empty((nt, 3), dtype=np.int64)
    lines[:, 0] = 2
    lines[:, 1] = np.arange(0, 2 * nt, 2, dtype=np.int64)
    lines[:, 2] = np.arange(1, 2 * nt, 2, dtype=np.int64)

    poly = pv.PolyData(pts, lines=lines.ravel())

    if throat_scalars is not None:
        throat_scalars = np.asarray(throat_scalars, dtype=float).ravel()
        if throat_scalars.size != nt:
            raise ValueError(f"throat_scalars must have length {nt}, got {throat_scalars.size}")

        poly.cell_data[scalar_name] = throat_scalars
        poly.point_data[scalar_name] = np.repeat(throat_scalars, 2)

    return poly


def _compute_throat_values_from_pores(conns, pore_values, mode="min"):
    pore_values = np.asarray(pore_values, dtype=float).ravel()
    a = pore_values[conns[:, 0]]
    b = pore_values[conns[:, 1]]

    mode = str(mode).lower()
    if mode in ("min", "smaller", "lower"):
        return np.minimum(a, b)
    elif mode in ("max", "larger", "higher"):
        return np.maximum(a, b)
    elif mode in ("mean", "average", "avg"):
        return 0.5 * (a + b)
    else:
        raise ValueError("mode must be 'min', 'max', or 'mean'")


def _compute_frame_clim(vals):
    vals = np.asarray(vals, dtype=float).ravel()
    finite = np.isfinite(vals)
    if not np.any(finite):
        raise ValueError("Frame contains no finite pore values")

    vmin = float(np.nanmin(vals[finite]))
    vmax = float(np.nanmax(vals[finite]))

    if vmin == vmax:
        pad = 1e-12 if vmin == 0 else 1e-12 * abs(vmin)
        vmin -= pad
        vmax += pad

    return (vmin, vmax)


def plot_network_with_cubes(
    net,
    pore_size="pore.equivalent_diameter",
    throat_size="throat.equivalent_diameter",
    pore_size_factor=1.0,
    throat_size_factor=1.0,
    pore_color="royalblue",
    throat_color="black",
    pore_values=None,
    time_value=None,
    pore_values_name="pore_values",
    cmap="viridis",
    clim=None,
    show_scalar_bar=True,
    show_time_text=True,
    time_fmt="",
    show_throats=True,
    throat_style="square",          # "square", "tube", or "line"
    throat_radius=None,
    throat_line_width=1.0,
    throat_values=None,
    throat_color_mode=None,         # "min", "max", or "mean"
    throat_values_name="throat_values",
    color_throats_with_scalars=True,
    background="white",
    window_size=(1400, 1000),
    show_edges=True,
    opacity=1.0,
    notebook=False,
    show_axes_widget=True,
    show_grid=False,
    camera_position=None,
    camera_azimuth=0.0,
    camera_elevation=0.0,
    camera_roll=0.0,
    filename= 'saved_result.png'
):
    coords = _get_net_array(net, "pore.coords").astype(float)
    conns = _get_net_array(net, "throat.conns").astype(int)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("'pore.coords' must have shape (Np, 3)")
    if conns.ndim != 2 or conns.shape[1] != 2:
        raise ValueError("'throat.conns' must have shape (Nt, 2)")

    npores = coords.shape[0]

    pore_sizes = _get_per_pore_sizes(
        net, coords, conns, pore_size, size_factor=pore_size_factor
    )

    cloud = pv.PolyData(coords)
    cloud["cube_size"] = pore_sizes

    pore_scalar_name = None
    solid_pore_color = pore_color
    pore_scalar_values = None

    if pore_values is not None:
        pore_values = np.asarray(pore_values, dtype=float).ravel()
        if pore_values.size != npores:
            raise ValueError(
                f"pore_values must have one value per pore ({npores}), got {pore_values.size}"
            )
        cloud[pore_values_name] = pore_values
        pore_scalar_name = pore_values_name
        pore_scalar_values = pore_values
        solid_pore_color = None

    elif isinstance(pore_color, str):
        try:
            maybe_scalars = np.asarray(net[pore_color], dtype=float).ravel()
            if maybe_scalars.size == npores:
                cloud[pore_color] = maybe_scalars
                pore_scalar_name = pore_color
                pore_scalar_values = maybe_scalars
                solid_pore_color = None
        except Exception:
            pass

    cube = pv.Cube(center=(0.0, 0.0, 0.0), x_length=1.0, y_length=1.0, z_length=1.0)
    pore_mesh = cloud.glyph(scale="cube_size", geom=cube, orient=False)

    throat_scalar_values = None
    if throat_values is not None:
        throat_scalar_values = np.asarray(throat_values, dtype=float).ravel()
        if throat_scalar_values.size != conns.shape[0]:
            raise ValueError(
                f"throat_values must have one value per throat ({conns.shape[0]}), "
                f"got {throat_scalar_values.size}"
            )
    elif throat_color_mode is not None:
        if pore_scalar_values is None:
            raise ValueError(
                "throat_color_mode requires pore_values "
                "(or pore_color mapped to a pore scalar field)"
            )
        throat_scalar_values = _compute_throat_values_from_pores(
            conns, pore_scalar_values, mode=throat_color_mode
        )

    shared_clim = clim
    if shared_clim is None and pore_scalar_values is not None:
        shared_clim = _compute_frame_clim(pore_scalar_values)

    p = pv.Plotter(off_screen=True, notebook=notebook, window_size=window_size)
    p.set_background(background)

    if show_throats and conns.size > 0:
        use_throat_scalars = color_throats_with_scalars and (throat_scalar_values is not None)

        if throat_style == "square":
            throat_sizes = _get_per_throat_sizes(
                net, coords, conns, throat_size, size_factor=throat_size_factor
            )
            throat_mesh = _build_square_throat_mesh(
                coords,
                conns,
                throat_sizes,
                throat_scalars=throat_scalar_values if use_throat_scalars else None,
                scalar_name=throat_values_name,
            )

            if throat_mesh.n_cells > 0:
                if use_throat_scalars:
                    p.add_mesh(
                        throat_mesh,
                        scalars=throat_values_name,
                        cmap=cmap,
                        clim=shared_clim,
                        opacity=opacity,
                        show_edges=show_edges,
                        show_scalar_bar=False,
                    )
                else:
                    p.add_mesh(
                        throat_mesh,
                        color=throat_color,
                        opacity=opacity,
                        show_edges=show_edges,
                    )

        elif throat_style == "tube":
            throat_lines = _build_throat_polydata(
                coords,
                conns,
                throat_scalars=throat_scalar_values if use_throat_scalars else None,
                scalar_name=throat_values_name,
            )

            if throat_radius is None:
                throat_sizes = _get_per_throat_sizes(
                    net, coords, conns, throat_size, size_factor=throat_size_factor
                )
                throat_radius = 0.5 * np.nanmedian(throat_sizes)

            tubes = throat_lines.tube(radius=float(throat_radius))

            if use_throat_scalars:
                p.add_mesh(
                    tubes,
                    scalars=throat_values_name,
                    cmap=cmap,
                    clim=shared_clim,
                    opacity=0.6,
                    show_scalar_bar=False,
                )
            else:
                p.add_mesh(tubes, color=throat_color, opacity=0.6)

        elif throat_style == "line":
            throat_lines = _build_throat_polydata(
                coords,
                conns,
                throat_scalars=throat_scalar_values if use_throat_scalars else None,
                scalar_name=throat_values_name,
            )

            if use_throat_scalars:
                p.add_mesh(
                    throat_lines,
                    scalars=throat_values_name,
                    cmap=cmap,
                    clim=shared_clim,
                    line_width=throat_line_width,
                    render_lines_as_tubes=False,
                    opacity=opacity,
                    show_scalar_bar=False,
                )
            else:
                p.add_mesh(
                    throat_lines,
                    color=throat_color,
                    line_width=throat_line_width,
                    render_lines_as_tubes=False,
                    opacity=opacity,
                )
        else:
            raise ValueError("throat_style must be 'square', 'tube', or 'line'")

    if pore_scalar_name is None:
        p.add_mesh(
            pore_mesh,
            color=solid_pore_color,
            opacity=opacity,
            show_edges=show_edges,
        )
    else:
        p.add_mesh(
            pore_mesh,
            scalars=pore_scalar_name,
            cmap=cmap,
            clim=shared_clim,
            opacity=opacity,
            show_edges=show_edges,
            scalar_bar_args=_right_vertical_scalar_bar_args(pore_scalar_name) if show_scalar_bar else None,
            show_scalar_bar=show_scalar_bar,
        )

    if time_value is not None and show_time_text:
        p.add_text(
            time_fmt,
            position="upper_left",
            font_size=PLOT_3D_TEXT_FONT_SIZE,
            color="black",
        )

    if show_axes_widget:
        p.add_axes()
    if show_grid:
        p.show_grid()

    p.reset_camera()
    _apply_camera(
        p,
        camera_position=camera_position,
        camera_azimuth=camera_azimuth,
        camera_elevation=camera_elevation,
        camera_roll=camera_roll,
    )
    
    p.show(screenshot=filename)
    p.close()
    return pore_mesh


class NetworkVideoWriter:
    """
    Stable video writer:
    - builds the scene once
    - keeps pore actor + scalar bar alive
    - updates pore scalars in place
    - updates line throats in place
    - rebuilds square/tube throats each frame for correct throat coloring
    - updates time text by name
    - never uses clear_actors()
    """

    def __init__(
        self,
        net,
        filename="network_timeseries.mp4",
        interactive=True,
        pore_size="pore.equivalent_diameter",
        throat_size="throat.equivalent_diameter",
        pore_size_factor=1.0,
        throat_size_factor=1.0,
        throat_color="black",
        throat_color_mode=None,
        pore_values_name="pore_values",
        throat_values_name="throat_values",
        cmap="viridis",
        clim=None,
        show_scalar_bar=True,
        time_fmt="",
        fps=10,
        show_throats=True,
        throat_style="square",
        throat_radius=None,
        throat_line_width=1.0,
        background="white",
        window_size=(1400, 1000),
        show_edges=True,
        opacity=1.0,
        show_axes_widget=False,
        show_grid=False,
        camera_position=None,
        camera_azimuth=0.0,
        camera_elevation=0.0,
        camera_roll=0.0,
    ):
        self.interactive = interactive
        self.net = net
        self.filename = filename
        self.throat_color = throat_color
        self.throat_color_mode = throat_color_mode
        self.pore_values_name = pore_values_name
        self.throat_values_name = throat_values_name
        self.cmap = cmap
        self.clim = clim
        self._auto_clim = clim is None
        self.show_scalar_bar = show_scalar_bar
        self.time_fmt = time_fmt
        self.fps = fps
        self.show_throats = show_throats
        self.throat_style = throat_style
        self.throat_radius = throat_radius
        self.throat_line_width = throat_line_width
        self.background = background
        self.window_size = window_size
        self.show_edges = show_edges
        self.opacity = opacity
        self.show_axes_widget = show_axes_widget
        self.show_grid = show_grid
        self.camera_position = camera_position
        self.camera_azimuth = camera_azimuth
        self.camera_elevation = camera_elevation
        self.camera_roll = camera_roll

        self.coords = _get_net_array(net, "pore.coords").astype(float)
        self.conns = _get_net_array(net, "throat.conns").astype(int)

        if self.coords.ndim != 2 or self.coords.shape[1] != 3:
            raise ValueError("'pore.coords' must have shape (Np, 3)")
        if self.conns.ndim != 2 or self.conns.shape[1] != 2:
            raise ValueError("'throat.conns' must have shape (Nt, 2)")

        self.npores = self.coords.shape[0]
        self.nthroats = self.conns.shape[0]

        self.pore_sizes = _get_per_pore_sizes(
            net, self.coords, self.conns, pore_size, size_factor=pore_size_factor
        )

        if self.show_throats and self.nthroats > 0 and self.throat_style in ("square", "tube"):
            self.throat_sizes = _get_per_throat_sizes(
                net, self.coords, self.conns, throat_size, size_factor=throat_size_factor
            )
        else:
            self.throat_sizes = None

        self.cube = pv.Cube(center=(0.0, 0.0, 0.0), x_length=1.0, y_length=1.0, z_length=1.0)
        self._pore_points_per_glyph = self.cube.n_points

        self.plotter = pv.Plotter(
            off_screen=not self.interactive,
            window_size=self.window_size,
        )
        self.plotter.set_background(self.background)

        ext = self.filename.lower().split(".")[-1]
        if ext == "gif":
            self.plotter.open_gif(self.filename)
        elif ext in ("mp4", "avi", "mov"):
            self.plotter.open_movie(self.filename, framerate=self.fps)
        else:
            raise ValueError("Use .mp4, .avi, .mov, or .gif")

        self._initialized_scene = False
        self._camera_initialized = False
        self._nframes_written = 0

        self.pore_mesh = None
        self.pore_actor = None

        self.throat_mesh = None
        self.throat_actor = None

        self.throat_line_source = None

        if self.interactive:
            self.plotter.show(auto_close=False, interactive=True, interactive_update=True)

    def _expand_pore_values_to_glyph_points(self, vals):
        return np.repeat(np.asarray(vals, dtype=float).ravel(), self._pore_points_per_glyph)

    def _expand_line_throat_values_to_points(self, vals):
        return np.repeat(np.asarray(vals, dtype=float).ravel(), 2)

    def _compute_throat_scalars(self, vals):
        if not (self.show_throats and self.nthroats > 0 and self.throat_color_mode is not None):
            return None
        return _compute_throat_values_from_pores(self.conns, vals, mode=self.throat_color_mode)

    def _build_initial_scene(self, vals):
        if self.clim is None:
            self.clim = _compute_frame_clim(vals)

        # pores
        cloud = pv.PolyData(self.coords)
        cloud["cube_size"] = self.pore_sizes
        self.pore_mesh = cloud.glyph(scale="cube_size", geom=self.cube, orient=False)
        self.pore_mesh.point_data[self.pore_values_name] = self._expand_pore_values_to_glyph_points(vals)

        self.pore_actor = self.plotter.add_mesh(
            self.pore_mesh,
            scalars=self.pore_values_name,
            cmap=self.cmap,
            clim=self.clim,
            opacity=self.opacity,
            show_edges=self.show_edges,
            scalar_bar_args=_right_vertical_scalar_bar_args(self.pore_values_name) if self.show_scalar_bar else None,
            show_scalar_bar=self.show_scalar_bar,
            name="pores",
        )

        # throats
        throat_scalars = self._compute_throat_scalars(vals)

        if self.show_throats and self.nthroats > 0:
            if self.throat_style == "square":
                self.throat_mesh = _build_square_throat_mesh(
                    self.coords,
                    self.conns,
                    self.throat_sizes,
                    throat_scalars=throat_scalars,
                    scalar_name=self.throat_values_name,
                )

                if self.throat_mesh.n_cells > 0:
                    if throat_scalars is not None:
                        self.throat_actor = self.plotter.add_mesh(
                            self.throat_mesh,
                            scalars=self.throat_values_name,
                            cmap=self.cmap,
                            clim=self.clim,
                            opacity=self.opacity,
                            show_edges=self.show_edges,
                            show_scalar_bar=False,
                            name="throats",
                        )
                    else:
                        self.throat_actor = self.plotter.add_mesh(
                            self.throat_mesh,
                            color=self.throat_color,
                            opacity=self.opacity,
                            show_edges=self.show_edges,
                            name="throats",
                        )

            elif self.throat_style == "line":
                self.throat_mesh = _build_throat_polydata(
                    self.coords,
                    self.conns,
                    throat_scalars=throat_scalars,
                    scalar_name=self.throat_values_name,
                )

                if self.throat_mesh.n_cells > 0:
                    if throat_scalars is not None:
                        self.throat_actor = self.plotter.add_mesh(
                            self.throat_mesh,
                            scalars=self.throat_values_name,
                            cmap=self.cmap,
                            clim=self.clim,
                            line_width=self.throat_line_width,
                            render_lines_as_tubes=False,
                            opacity=self.opacity,
                            show_scalar_bar=False,
                            name="throats",
                        )
                    else:
                        self.throat_actor = self.plotter.add_mesh(
                            self.throat_mesh,
                            color=self.throat_color,
                            line_width=self.throat_line_width,
                            render_lines_as_tubes=False,
                            opacity=self.opacity,
                            name="throats",
                        )

            elif self.throat_style == "tube":
                self.throat_line_source = _build_throat_polydata(
                    self.coords,
                    self.conns,
                    throat_scalars=throat_scalars,
                    scalar_name=self.throat_values_name,
                )

                rr = self.throat_radius
                if rr is None:
                    rr = 0.5 * np.nanmedian(self.throat_sizes)

                tube_mesh = self.throat_line_source.tube(radius=float(rr))

                if tube_mesh.n_cells > 0:
                    if throat_scalars is not None:
                        self.throat_actor = self.plotter.add_mesh(
                            tube_mesh,
                            scalars=self.throat_values_name,
                            cmap=self.cmap,
                            clim=self.clim,
                            opacity=0.6,
                            show_scalar_bar=False,
                            name="throats",
                        )
                    else:
                        self.throat_actor = self.plotter.add_mesh(
                            tube_mesh,
                            color=self.throat_color,
                            opacity=0.6,
                            name="throats",
                        )
            else:
                raise ValueError("throat_style must be 'square', 'tube', or 'line'")

        if self.show_axes_widget:
            self.plotter.add_axes()
        if self.show_grid:
            self.plotter.show_grid()

        self.plotter.add_text(
            self.time_fmt,
            position="upper_left",
            font_size=PLOT_3D_TEXT_FONT_SIZE,
            color="black",
            name="time_label",
        )

        self._initialized_scene = True

    def _update_pores_in_place(self, vals):
        self.pore_mesh.point_data[self.pore_values_name] = self._expand_pore_values_to_glyph_points(vals)
        _mark_modified(self.pore_mesh)

    def _update_throats(self, throat_scalars):
        if not (self.show_throats and self.nthroats > 0):
            return

        if self.throat_style == "square":
            new_throat_mesh = _build_square_throat_mesh(
                self.coords,
                self.conns,
                self.throat_sizes,
                throat_scalars=throat_scalars,
                scalar_name=self.throat_values_name,
            )

            if self.throat_actor is not None:
                self.plotter.remove_actor(self.throat_actor, render=False)

            self.throat_mesh = new_throat_mesh

            if self.throat_mesh.n_cells > 0:
                if throat_scalars is not None:
                    self.throat_actor = self.plotter.add_mesh(
                        self.throat_mesh,
                        scalars=self.throat_values_name,
                        cmap=self.cmap,
                        clim=self.clim,
                        opacity=self.opacity,
                        show_edges=self.show_edges,
                        show_scalar_bar=False,
                        name="throats",
                    )
                else:
                    self.throat_actor = self.plotter.add_mesh(
                        self.throat_mesh,
                        color=self.throat_color,
                        opacity=self.opacity,
                        show_edges=self.show_edges,
                        name="throats",
                    )

        elif self.throat_style == "line":
            if self.throat_mesh is None or throat_scalars is None:
                return

            self.throat_mesh.point_data[self.throat_values_name] = (
                self._expand_line_throat_values_to_points(throat_scalars)
            )
            self.throat_mesh.cell_data[self.throat_values_name] = throat_scalars
            _mark_modified(self.throat_mesh)

            if self.throat_actor is not None and hasattr(self.throat_actor, "mapper"):
                self.throat_actor.mapper.scalar_range = self.clim

        elif self.throat_style == "tube":
            rr = self.throat_radius
            if rr is None:
                rr = 0.5 * np.nanmedian(self.throat_sizes)

            if throat_scalars is not None:
                self.throat_line_source.point_data[self.throat_values_name] = (
                    self._expand_line_throat_values_to_points(throat_scalars)
                )
                self.throat_line_source.cell_data[self.throat_values_name] = throat_scalars
            elif self.throat_values_name in self.throat_line_source.point_data:
                del self.throat_line_source.point_data[self.throat_values_name]
                if self.throat_values_name in self.throat_line_source.cell_data:
                    del self.throat_line_source.cell_data[self.throat_values_name]

            tube_mesh = self.throat_line_source.tube(radius=float(rr))

            if self.throat_actor is not None:
                self.plotter.remove_actor(self.throat_actor, render=False)

            if throat_scalars is not None:
                self.throat_actor = self.plotter.add_mesh(
                    tube_mesh,
                    scalars=self.throat_values_name,
                    cmap=self.cmap,
                    clim=self.clim,
                    opacity=0.6,
                    show_scalar_bar=False,
                    name="throats",
                )
            else:
                self.throat_actor = self.plotter.add_mesh(
                    tube_mesh,
                    color=self.throat_color,
                    opacity=0.6,
                    name="throats",
                )

    def add_frame(self, time_value, pore_values):
        vals = np.asarray(pore_values, dtype=float).ravel()
        if vals.size != self.npores:
            raise ValueError(
                f"pore_values must have one value per pore ({self.npores}), got {vals.size}"
            )

        if not self._initialized_scene:
            self._build_initial_scene(vals)

        if self._auto_clim:
            self.clim = _compute_frame_clim(vals)

        self._update_pores_in_place(vals)
        throat_scalars = self._compute_throat_scalars(vals)
        self._update_throats(throat_scalars)

        if self.pore_actor is not None and hasattr(self.pore_actor, "mapper"):
            self.pore_actor.mapper.scalar_range = self.clim

        self.plotter.add_text(
            self.time_fmt,
            position="upper_left",
            font_size=PLOT_3D_TEXT_FONT_SIZE,
            color="black",
            name="time_label",
            render=False,
        )

        if self.show_scalar_bar:
            self.plotter.update_scalar_bar_range(self.clim, name=self.pore_values_name)

        if not self._camera_initialized:
            self.plotter.reset_camera()
            _apply_camera(
                self.plotter,
                camera_position=self.camera_position,
                camera_azimuth=self.camera_azimuth,
                camera_elevation=self.camera_elevation,
                camera_roll=self.camera_roll,
            )
            self._camera_initialized = True

        self.plotter.render()
        if self.interactive:
            self.plotter.update()

        self.plotter.write_frame()
        self._nframes_written += 1

    def close(self):
        self.plotter.close()
        return self.filename

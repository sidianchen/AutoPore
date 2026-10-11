"""
Plot dynamic PNM time-series results from multiple single-run folders.

Default behavior:
    - Search for folders matching single_run_dyn_pnm_* under the current
      folder and pore_network_model/.
    - Read time_series.txt from each folder.
    - Save overlay plots with one curve per folder.

Examples:
    python -m pore_network_model.autopore.compare_dynamic
    python -m pore_network_model.autopore.compare_dynamic --suffixes a b c --labels "case A" "case B" "case C"
    python -m pore_network_model.autopore.compare_dynamic --case a="case A" --case b="case B" --case c="case C"
    python -m pore_network_model.autopore.compare_dynamic --root pore_network_model --raw-time
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


PLOT_FONT_SIZE = 8
PLOT_LINE_WIDTH = 1.2
PLOT_MARKER_SIZE = 2.0
DYNAMIC_FOLDER_PREFIX = "single_run_dyn_pnm"


# ---------------------------------------------------------------------------
# User-editable placeholders
# ---------------------------------------------------------------------------
# Option 1: provide suffixes and labels.
# These suffixes map to folders named single_run_dyn_pnm_<suffix>.
CUSTOM_SUFFIXES = [
    # "low",
    # "mid",
    # "high",
]
CUSTOM_LABELS = [
    # "Low capillary number",
    # "Middle capillary number",
    # "High capillary number",
]

# Option 2: provide suffix=label pairs. If this list is non-empty, it is used
# before CUSTOM_SUFFIXES/CUSTOM_LABELS.
CUSTOM_CASES = [
    # "low=Low capillary number",
    # "mid=Middle capillary number",
    # "high=High capillary number",
]

# Option 3: provide exact folders instead of suffixes.
CUSTOM_FOLDERS = [
    # "single_run_dyn_pnm_low",
    # "single_run_dyn_pnm_mid",
    # "single_run_dyn_pnm_high",
]

# General plotting settings.
CUSTOM_ROOT = "."
CUSTOM_PATTERN = "single_run_dyn_pnm_*"
CUSTOM_OUTPUT = "dynamic_time_comparison_figs"
CUSTOM_RAW_TIME = False

plt.rcParams.update(
    {
        "font.size": PLOT_FONT_SIZE,
        "axes.labelsize": PLOT_FONT_SIZE,
        "axes.titlesize": PLOT_FONT_SIZE,
        "xtick.labelsize": PLOT_FONT_SIZE,
        "ytick.labelsize": PLOT_FONT_SIZE,
        "legend.fontsize": PLOT_FONT_SIZE,
        "figure.titlesize": PLOT_FONT_SIZE,
    }
)


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
    """Fallback reader for files whose data rows have extra inserted columns."""
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


def read_dynamic_time_series(time_series_file: Path) -> pd.DataFrame | None:
    time_series_file = Path(time_series_file)

    try:
        with open(time_series_file, "r", encoding="utf-8") as handle:
            header_count = len(handle.readline().split())
    except Exception:
        header_count = None

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


def _case_label(folder: Path) -> str:
    name = folder.name
    if name.startswith(DYNAMIC_FOLDER_PREFIX + "_"):
        return name[len(DYNAMIC_FOLDER_PREFIX) + 1 :]
    return name


def _unique_sorted_paths(paths) -> list[Path]:
    seen = set()
    result = []
    for path in sorted(Path(p).resolve() for p in paths):
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _unique_paths_preserve_order(paths) -> list[Path]:
    seen = set()
    result = []
    for path in (Path(p).resolve() for p in paths):
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _search_roots(root: Path) -> list[Path]:
    roots = [root]
    model_root = root / "pore_network_model"
    if model_root.exists():
        roots.append(model_root)
    return roots


def discover_dynamic_folders(root: Path, pattern: str) -> list[Path]:
    matches = []
    for search_root in _search_roots(root):
        matches.extend(
            path for path in search_root.glob(pattern)
            if path.is_dir() and (path / "time_series.txt").exists()
        )
    return _unique_sorted_paths(matches)


def resolve_folder_args(folder_args: list[str]) -> list[Path]:
    matches = []
    for item in folder_args:
        path = Path(item)
        if any(char in item for char in "*?[]"):
            matches.extend(Path(match) for match in glob.glob(item))
        elif path.is_dir():
            matches.append(path)
        else:
            print(f"Skipping missing folder: {path}")
    return _unique_sorted_paths(
        path for path in matches
        if path.is_dir() and (path / "time_series.txt").exists()
    )


def _folder_name_from_suffix(suffix: str) -> str:
    suffix = str(suffix).strip()
    if suffix.startswith(DYNAMIC_FOLDER_PREFIX):
        return suffix
    return f"{DYNAMIC_FOLDER_PREFIX}_{suffix.lstrip('_')}"


def _resolve_one_suffix(root: Path, suffix: str) -> Path | None:
    direct_path = Path(suffix)
    if direct_path.is_dir() and (direct_path / "time_series.txt").exists():
        return direct_path

    folder_name = _folder_name_from_suffix(suffix)
    for search_root in _search_roots(root):
        candidate = search_root / folder_name
        if candidate.is_dir() and (candidate / "time_series.txt").exists():
            return candidate

    print(f"Skipping missing dynamic suffix: {suffix} (expected folder {folder_name})")
    return None


def resolve_suffix_args(root: Path, suffixes: list[str]) -> list[Path]:
    matches = []
    for suffix in suffixes:
        folder = _resolve_one_suffix(root, suffix)
        if folder is not None:
            matches.append(folder)
    return _unique_paths_preserve_order(matches)


def split_case_spec(case_spec: str) -> tuple[str, str]:
    if "=" in case_spec:
        suffix, label = case_spec.split("=", 1)
    else:
        suffix, label = case_spec, case_spec
    suffix = suffix.strip()
    label = label.strip().strip("\"'")
    return suffix, label or suffix


def resolve_case_specs(root: Path, case_specs: list[str]) -> tuple[list[Path], list[str]]:
    folders = []
    labels = []
    for case_spec in case_specs:
        suffix, label = split_case_spec(case_spec)
        folder = _resolve_one_suffix(root, suffix)
        if folder is None:
            continue
        folders.append(folder)
        labels.append(label)
    unique_folders = []
    unique_labels = []
    seen = set()
    for folder, label in zip(folders, labels):
        key = str(folder.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        unique_folders.append(folder.resolve())
        unique_labels.append(label)
    return unique_folders, unique_labels


def _set_axis_with_top_gap(ax, values) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    ymin = float(np.min(values))
    ymax = float(np.max(values))
    if ymin == ymax:
        pad = 1.0 if ymin == 0 else abs(ymin) * 0.05
        ax.set_ylim(ymin - pad, ymax + pad)
        return
    pad = 0.08 * (ymax - ymin)
    ax.set_ylim(ymin - 0.03 * (ymax - ymin), ymax + pad)


def save_overlay_plot(
    case_results: list[tuple[str, Path, pd.DataFrame]],
    column: str,
    ylabel: str,
    title: str,
    filename: str,
    output_dir: Path,
    normalize_time: bool,
) -> Path | None:
    fig, ax = plt.subplots(figsize=(5, 4))
    all_y = []
    plotted = 0

    for label, folder, df in case_results:
        if column not in df.columns:
            print(f"Skipping {folder.name} for {filename}: missing column {column}")
            continue

        x = pd.to_numeric(df["Production_Time"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            print(f"Skipping {folder.name} for {filename}: no valid data")
            continue

        x = x[valid]
        y = y[valid]
        if normalize_time:
            finite_time_indices = np.flatnonzero(np.isfinite(x))
            final_time = float(x[finite_time_indices[-1]]) if finite_time_indices.size else np.nan
            if not np.isfinite(final_time) or final_time == 0:
                print(f"Skipping {folder.name} for {filename}: cannot normalize time")
                continue
            x = x / final_time

        ax.plot(
            x,
            y,
            marker="o",
            markersize=PLOT_MARKER_SIZE,
            linewidth=PLOT_LINE_WIDTH,
            label=label,
        )
        all_y.append(y)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    ax.set_xlabel("Normalized time" if normalize_time else "Production time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="both", direction="in")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    try:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 2))
    except Exception:
        pass
    if all_y:
        _set_axis_with_top_gap(ax, np.concatenate(all_y))
    ax.legend(frameon=False)
    fig.subplots_adjust(left=0.17, right=0.96, bottom=0.17, top=0.88)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / filename
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved plot: {fig_path}")
    return fig_path


def save_area_overlay_plot(
    case_results: list[tuple[str, Path, pd.DataFrame]],
    filename: str,
    output_dir: Path,
    normalize_time: bool,
) -> Path | None:
    fig, ax = plt.subplots(figsize=(5, 4))
    all_y = []
    case_handles = []

    for label, folder, df in case_results:
        bulk_column = "ave_aff_b" if "ave_aff_b" in df.columns else "ave_aff"
        if bulk_column not in df.columns:
            print(f"Skipping {folder.name} for {filename}: missing bulk area")
            continue

        x = pd.to_numeric(df["Production_Time"], errors="coerce").to_numpy(dtype=float)
        bulk = pd.to_numeric(df[bulk_column], errors="coerce").to_numpy(dtype=float)
        bulk_film = (
            pd.to_numeric(df["ave_aff_bf"], errors="coerce").to_numpy(dtype=float)
            if "ave_aff_bf" in df.columns
            else None
        )
        valid_x = np.isfinite(x)
        if not np.any(valid_x):
            continue
        if normalize_time:
            final_time = float(x[np.flatnonzero(valid_x)[-1]])
            if not np.isfinite(final_time) or final_time == 0:
                continue
            x = x / final_time

        bulk_valid = np.isfinite(x) & np.isfinite(bulk)
        if not np.any(bulk_valid):
            continue
        line = ax.plot(
            x[bulk_valid], bulk[bulk_valid], linewidth=1.2, linestyle="-", label=label
        )[0]
        case_handles.append(line)
        all_y.append(bulk[bulk_valid])

        if bulk_film is not None:
            film_valid = np.isfinite(x) & np.isfinite(bulk_film)
            if np.any(film_valid):
                ax.plot(
                    x[film_valid],
                    bulk_film[film_valid],
                    color=line.get_color(),
                    linewidth=1.2,
                    linestyle="--",
                    label="_nolegend_",
                )
                all_y.append(bulk_film[film_valid])

    if not case_handles:
        plt.close(fig)
        return None

    ax.set_xlabel("Normalized time" if normalize_time else "Time")
    ax.set_ylabel("$A_{wn}$")
    ax.set_title("Fluid-fluid interfacial area over time")
    ax.tick_params(axis="both", direction="in")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 2))
    _set_axis_with_top_gap(ax, np.concatenate(all_y))

    case_legend = ax.legend(handles=case_handles, frameon=False, loc="best")
    ax.add_artist(case_legend)
    style_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.2, label="Bulk"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.2, label="Bulk + Film"),
    ]
    ax.legend(handles=style_handles, frameon=False, loc="upper right")
    fig.subplots_adjust(left=0.17, right=0.96, bottom=0.17, top=0.88)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / filename
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved plot: {fig_path}")
    return fig_path


def load_case_results(
    folders: list[Path],
    labels: list[str] | None = None,
) -> list[tuple[str, Path, pd.DataFrame]]:
    case_results = []
    labels = labels or []
    for index, folder in enumerate(folders):
        result_file = folder / "time_series.txt"
        df = read_dynamic_time_series(result_file)
        if df is None:
            print(f"Skipping {folder.name}: could not read {result_file.name}")
            continue
        label = labels[index] if index < len(labels) else _case_label(folder)
        case_results.append((label, folder, df))
    return case_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare dynamic PNM time-series results from single_run_dyn_pnm_* folders."
    )
    parser.add_argument(
        "folders",
        nargs="*",
        help=(
            "Dynamic result folders. If omitted, folders are discovered automatically, "
            "unless --suffixes or --case is used."
        ),
    )
    parser.add_argument(
        "--suffixes",
        nargs="+",
        help=(
            "Suffixes after single_run_dyn_pnm_. Example: --suffixes low mid high "
            "selects single_run_dyn_pnm_low, single_run_dyn_pnm_mid, single_run_dyn_pnm_high."
        ),
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help=(
            "Custom legend labels, matched in order to --suffixes or folder arguments. "
            "Use quotes for labels with spaces."
        ),
    )
    parser.add_argument(
        "--case",
        action="append",
        help=(
            "One suffix=label pair. Can be repeated. Example: "
            "--case low=\"Low Ca\" --case high=\"High Ca\"."
        ),
    )
    parser.add_argument(
        "--root",
        default=CUSTOM_ROOT,
        help="Root folder used for automatic discovery. Default: current folder.",
    )
    parser.add_argument(
        "--pattern",
        default=CUSTOM_PATTERN,
        help="Folder name pattern used for automatic discovery.",
    )
    parser.add_argument(
        "--output",
        default=CUSTOM_OUTPUT,
        help="Output folder for saved comparison plots.",
    )
    parser.add_argument(
        "--raw-time",
        action="store_true",
        help="Use raw Production_Time instead of normalizing each case by its final time.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = Path(args.output)

    case_specs = args.case or CUSTOM_CASES
    suffixes = args.suffixes or CUSTOM_SUFFIXES
    folder_args = args.folders or CUSTOM_FOLDERS

    labels = None
    if case_specs:
        folders, labels = resolve_case_specs(root, case_specs)
        if args.labels or CUSTOM_LABELS:
            print("Ignoring --labels because --case already provides labels.")
    elif suffixes:
        folders = resolve_suffix_args(root, suffixes)
        labels = args.labels if args.labels is not None else (CUSTOM_LABELS or None)
    elif folder_args:
        folders = resolve_folder_args(folder_args)
        labels = args.labels if args.labels is not None else (CUSTOM_LABELS or None)
    else:
        folders = discover_dynamic_folders(root, args.pattern)
        labels = args.labels if args.labels is not None else (CUSTOM_LABELS or None)

    if not folders:
        print(
            "No dynamic result folders found. Expected folders containing time_series.txt, "
            f"for example {args.pattern}."
        )
        return 1

    if labels is not None and len(labels) != len(folders):
        print(
            f"Number of labels ({len(labels)}) must match number of selected folders "
            f"({len(folders)})."
        )
        return 1

    print("Using dynamic folders:")
    for index, folder in enumerate(folders):
        label_text = labels[index] if labels is not None else _case_label(folder)
        print(f"  - {folder}  label={label_text}")

    case_results = load_case_results(folders, labels)
    if not case_results:
        print("No valid dynamic time-series results were found.")
        return 1

    normalize_time = not (args.raw_time or CUSTOM_RAW_TIME)
    plot_specs = [
        (
            "ave_pcap",
            "Average $P^{c}$ (Pa)",
            "Average capillary pressure over time",
            "Comparison Average capillary pressure-Time curve.png",
        ),
        (
            "Sg_all",
            "$S_{g}$ (-)",
            "Gas saturation over time",
            "Comparison Gas saturation-Time curve.png",
        ),
    ]

    saved = []
    for column, ylabel, title, filename in plot_specs:
        fig_path = save_overlay_plot(
            case_results,
            column,
            ylabel,
            title,
            filename,
            output_dir,
            normalize_time,
        )
        if fig_path is not None:
            saved.append(fig_path)

    area_path = save_area_overlay_plot(
        case_results,
        "Comparison Fluid-fluid interfacial area-Time curve.png",
        output_dir,
        normalize_time,
    )
    if area_path is not None:
        saved.append(area_path)

    if not saved:
        print("No plots were saved.")
        return 1

    print(f"Saved {len(saved)} plot(s) in: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

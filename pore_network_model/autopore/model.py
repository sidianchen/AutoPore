import argparse
import ctypes
import gc
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import openpnm as op
import pandas as pd
import porespy as ps
import pyvista as pv

from .config import (
    ALLOWED_VALUES,
    DIAMETER_PROPERTY_NAMES,
    MODEL_DEFAULTS,
    PACKAGE_DIR,
    THIS_DIR,
)
from .results import archive_completed_cycle

from .utils import (
    clear_folder,
    coarsen_image_majority,
    enforce_dual_percolation,
    extract_scientific_number,
    generate_input_file,
    make_gif_from_folder,
    plot_3d_binary_image,
    plot_network_with_cubes,
    safe_boundary_pores_with_area,
    save_plot,
    save_pore_size_distribution_plot,
    save_time_series_plot,
    save_uq_results_plots,
    read_dynamic_time_series,
    show_image,
    find_pvbatch,
    avi_to_gif,
    swap_xz_and_scale_domain,
    _handle_remove_readonly,
)

# ---------------------------------------------------------------------------
# Default (fixed) parameters – these stay the same for every run
# ---------------------------------------------------------------------------
DEFAULTS = MODEL_DEFAULTS

# ---------------------------------------------------------------------------
# Allowed values for the user-customisable parameters
# ---------------------------------------------------------------------------
def _diameter_property_name(element: str, diameter_type: str) -> str:
    try:
        return DIAMETER_PROPERTY_NAMES[element][diameter_type]
    except KeyError as exc:
        allowed = sorted(DIAMETER_PROPERTY_NAMES.get(element, {}))
        raise ValueError(
            f"Invalid {element} diameter type '{diameter_type}'. Allowed: {allowed}"
        ) from exc


def _network_diameter_values(network, element: str, diameter_type: str) -> np.ndarray:
    property_name = _diameter_property_name(element, diameter_type)
    key = f"{element}.{property_name}"
    fallback_keys = []
    if diameter_type == "equivalent":
        fallback_keys.append(f"{element}.equivelant_diameter")

    for candidate in [key] + fallback_keys:
        if candidate in network.keys():
            return np.asarray(network[candidate], dtype=float).reshape(-1, 1)

    available = [k for k in network.keys() if k.startswith(f"{element}.")]
    raise KeyError(
        f"Network does not contain '{key}'. Available {element} fields include: "
        f"{available[:30]}"
    )

WINDOWS_CMD_CANCELLED = 0xC000013A
WINDOWS_HEAP_CORRUPTION = 0xC0000374
WINDOWS_HEAP_CORRUPTION_SIGNED = -1073740940
DEFAULT_EXE_CORRUPTION_MAX_ATTEMPTS = 0
SW_SHOWNOACTIVATE = 4
HWND_BOTTOM = 1
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010


class ExeCorruptionError(RuntimeError):
    def __init__(self, pnm_label: str, returncode: int):
        self.pnm_label = pnm_label
        self.returncode = returncode
        super().__init__(
            f"{pnm_label} simulation failed with executable heap corruption "
            f"exit code {returncode}."
        )


def _is_exe_corruption_returncode(returncode: int | None) -> bool:
    if returncode is None:
        return False
    try:
        return int(returncode) & 0xFFFFFFFF == WINDOWS_HEAP_CORRUPTION
    except (TypeError, ValueError):
        return returncode in {WINDOWS_HEAP_CORRUPTION, WINDOWS_HEAP_CORRUPTION_SIGNED}


def _exe_corruption_retry_limit() -> int | None:
    raw = os.environ.get("PNM_EXE_CORRUPTION_MAX_ATTEMPTS", str(DEFAULT_EXE_CORRUPTION_MAX_ATTEMPTS))
    try:
        attempts = int(raw)
    except (TypeError, ValueError):
        attempts = DEFAULT_EXE_CORRUPTION_MAX_ATTEMPTS
    if attempts <= 0:
        return None
    return attempts


def _cmd_startupinfo_no_activate():
    if os.name != "nt":
        return None
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = SW_SHOWNOACTIVATE
    return startupinfo


def _send_process_windows_to_back(
    pid: int | None = None,
    title_substring: str | None = None,
    retries: int = 1,
    delay: float = 0.0,
) -> None:
    if os.name != "nt":
        return
    if pid is None and not title_substring:
        return
    try:
        user32 = ctypes.windll.user32
        title_substring_lower = title_substring.lower() if title_substring else None

        enum_proc_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        def get_window_title(hwnd) -> str:
            length = user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return ""
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            return buffer.value

        for attempt in range(max(1, int(retries))):
            windows = []

            def enum_proc(hwnd, lparam):
                if user32.IsWindowVisible(hwnd):
                    matched = False
                    if pid is not None:
                        window_pid = ctypes.c_ulong()
                        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(window_pid))
                        matched = window_pid.value == pid
                    if not matched and title_substring_lower:
                        matched = title_substring_lower in get_window_title(hwnd).lower()
                    if matched:
                        windows.append(hwnd)
                return True

            user32.EnumWindows(enum_proc_type(enum_proc), 0)
            for hwnd in windows:
                user32.ShowWindow(hwnd, SW_SHOWNOACTIVATE)
                user32.SetWindowPos(
                    hwnd,
                    HWND_BOTTOM,
                    0,
                    0,
                    0,
                    0,
                    SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
                )
            if windows:
                return
            if attempt < retries - 1 and delay > 0:
                time.sleep(delay)
    except Exception:
        pass


def _mode_output_folder(pnm_mode: str) -> Path:
    if pnm_mode == "dynamic":
        return THIS_DIR / "single_run_dyn_pnm"
    return THIS_DIR / "single_run_qs_pnm"


def _mode_realization_folder(pnm_mode: str) -> Path:
    if pnm_mode == "dynamic":
        return THIS_DIR / "uq_dyn_pnm"
    return THIS_DIR / "uq_qs_pnm"


def _reset_output_folder(output_folder: Path, retries=5, delay=0.5):
    output_folder = Path(output_folder)
    for attempt in range(retries):
        try:
            if output_folder.exists():
                shutil.rmtree(output_folder, onerror=_handle_remove_readonly)
            output_folder.mkdir(parents=True, exist_ok=True)
            return
        except PermissionError:
            gc.collect()
            if attempt == retries - 1:
                raise
            time.sleep(delay)


def _prepare_simulation_output_folder(simoutputname) -> Path:
    output_folder = Path(simoutputname)
    _reset_output_folder(output_folder)
    for subfolder in ("profile", "figs", "vtks"):
        (output_folder / subfolder).mkdir(parents=True, exist_ok=True)
    return output_folder


def _next_realization_index(results_root: Path) -> int:
    results_root.mkdir(parents=True, exist_ok=True)
    indexes = []
    for child in results_root.iterdir():
        if not child.is_dir() or not child.name.startswith("case_"):
            continue
        suffix = child.name.removeprefix("case_")
        if suffix.isdigit():
            indexes.append(int(suffix))
    return max(indexes, default=0) + 1


def _copy_realization_output(output_folder: Path, results_root: Path, case_index: int) -> Path:
    case_folder = results_root / f"case_{case_index:03d}"
    while case_folder.exists():
        case_index += 1
        case_folder = results_root / f"case_{case_index:03d}"
    shutil.copytree(output_folder, case_folder)
    return case_folder


def _copy_input_pnm_folder(destination_root: Path) -> str | None:
    source = Path(DEFAULTS["outputfolder"])
    if not source.exists():
        print(f"input_pnm folder not found: {source}")
        return None

    destination_root = Path(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / source.name
    if destination.exists():
        shutil.rmtree(destination, onerror=_handle_remove_readonly)
    shutil.copytree(source, destination)
    return str(destination)


def _case_folder_from_identifier(pnm_mode: str, case) -> Path:
    case_text = str(case).strip()
    if not case_text:
        raise ValueError("case must be an integer or case_###.")

    if case_text.lower().startswith("case_"):
        suffix = case_text[5:]
        if suffix.isdigit():
            case_name = f"case_{int(suffix):03d}"
        else:
            case_name = case_text
    elif case_text.isdigit():
        case_name = f"case_{int(case_text):03d}"
    else:
        raise ValueError("case must be an integer or case_###.")

    return _mode_realization_folder(pnm_mode) / case_name


def _replace_input_pnm_from_case(case_folder: Path) -> str:
    source = case_folder / Path(DEFAULTS["outputfolder"]).name
    if not source.exists():
        raise FileNotFoundError(f"input_pnm snapshot not found for {case_folder.name}: {source}")

    destination = Path(DEFAULTS["outputfolder"])
    if destination.exists():
        shutil.rmtree(destination, onerror=_handle_remove_readonly)
    shutil.copytree(source, destination)
    return str(destination)


def _replace_case_with_output(output_folder: Path, case_folder: Path) -> str:
    if not output_folder.exists():
        raise FileNotFoundError(f"simulation output folder not found: {output_folder}")
    if case_folder.exists():
        shutil.rmtree(case_folder, onerror=_handle_remove_readonly)
    shutil.copytree(output_folder, case_folder)
    return str(case_folder)


def _single_run_input_pnm_folder(args: dict) -> Path:
    output_folder = Path(args["simoutputname"])
    return output_folder / Path(DEFAULTS["outputfolder"]).name


def _active_input_pnm_folder(args: dict) -> Path:
    local_input = _single_run_input_pnm_folder(args)
    if args["action"] == "plot_result" and args["task_mode"] == "single_run" and local_input.exists():
        return local_input
    return Path(args["outputfolder"])


def _single_run_figs_folder(simoutputname) -> Path:
    figs_folder = Path(simoutputname) / "figs"
    figs_folder.mkdir(parents=True, exist_ok=True)
    return figs_folder


def _input_pnm_3d_preview_path(input_folder: Path) -> Path:
    input_folder = Path(input_folder)
    input_folder.mkdir(parents=True, exist_ok=True)
    return input_folder / "3D_image.png"


def _save_3d_preview_to_input_pnm(im0, input_folder: Path) -> Path:
    preview_path = _input_pnm_3d_preview_path(input_folder)
    plot_3d_binary_image(
        im0,
        color="lightgray",
        camera_azimuth=180,
        camera_elevation=0,
        camera_roll=0,
        filename=str(preview_path),
    )
    return preview_path


def _move_root_media_to_figs(simoutputname) -> list[str]:
    output_folder = Path(simoutputname)
    figs_folder = _single_run_figs_folder(output_folder)
    moved = []
    for path in output_folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".png", ".avi", ".gif"}:
            continue
        destination = figs_folder / path.name
        if destination.exists():
            path.unlink()
        else:
            shutil.move(str(path), str(destination))
        moved.append(str(destination))
    return moved


def _dynamic_time_series_has_data(path: Path) -> bool:
    """Return True when a dynamic result contains at least one numeric row."""
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            next(handle, None)
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith(";"):
                    continue
                try:
                    float(stripped.split()[0])
                    return True
                except (ValueError, IndexError):
                    continue
    except OSError:
        return False
    return False


def _ensure_dynamic_solver_inputs() -> None:
    """Upgrade legacy dynamic inputs to the format required by dyn_pnm.exe."""
    input_folder = THIS_DIR / "input_pnm"
    input_folder.mkdir(parents=True, exist_ok=True)
    dyn_param = input_folder / "dyn_param.txt"
    try:
        lines = dyn_param.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []

    numeric = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("//"):
            numeric.append(line)

    defaults = [
        "1.000000e-03", "1.000000e-01", "1.000000e+00", "1.000000e+02",
        "1.000000e-8", "2.000000e+01", "1e-8", "1e5", "1", "1",
    ]
    if len(numeric) < 10:
        numeric.extend(defaults[len(numeric):])
        comments = [
            "// gas viscosity", "// liquid viscosity", "// flow rate",
            "// capillary number", "// tolerance to scale pressure update",
            "// output time scaler with maxT", "// output time step size",
            "// maxPc", "// outlet condition (0 - zero gas flux, 1 - zero gradient)",
            "// shrink throat sizes if larger than connecting bodies",
        ]
        dyn_param.write_text(
            "\n".join(numeric[:10] + comments) + "\n", encoding="utf-8"
        )

    dyn_control = input_folder / "dyn_SimControl.txt"
    if not dyn_control.exists():
        dyn_control.write_text(
            "1e-6\n1e-5\n1e20\n1.5\n1e-6\n20\n1e-4\n"
            "// dt\n// DT_MAX\n// totTime\n// totTime_vs_BT_time\n"
            "// L_eps\n// Max_ITER\n// tol_oci\n",
            encoding="utf-8",
        )


def _run_pnm_executable(args: dict) -> dict | None:
    if args["pnm_mode"] == "quasi_static":
        bat = str(THIS_DIR / "run_qs_pnm.bat")
        pnm_label = "Quasi-static"
        start_message = "Start runing quasi static pore network model ..."
        finish_message = "Finish runing quasi static pore network model !!!"
    elif args["pnm_mode"] == "dynamic":
        _ensure_dynamic_solver_inputs()
        bat = str(THIS_DIR / "run_dyn_pnm.bat")
        pnm_label = "Dynamic"
        start_message = "Start runing dynamic pore network model ..."
        finish_message = "Finish runing dynamic pore network model !!!"
    else:
        print("model is not available!")
        return None

    print(start_message)
    old_cwd = os.getcwd()
    try:
        os.chdir(str(THIS_DIR))
        proc = subprocess.Popen(
            ["cmd", "/c", bat],
            cwd=str(THIS_DIR),
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            startupinfo=_cmd_startupinfo_no_activate(),
        )
        _send_process_windows_to_back(
            proc.pid,
            title_substring=Path(bat).name,
            retries=10,
            delay=0.15,
        )
        returncode = proc.wait()
    finally:
        os.chdir(old_cwd)

    if args["task_mode"] == "single_run":
        args["input_pnm_copy"] = _copy_input_pnm_folder(Path(args["simoutputname"]))

    if returncode == WINDOWS_CMD_CANCELLED:
        message = f"{pnm_label} simulation window was closed manually."
        print(message)
        return {
            "status": "manually_closed",
            "message": message,
            "returncode": returncode,
            "args": args,
        }

    if returncode != 0:
        if _is_exe_corruption_returncode(returncode):
            raise ExeCorruptionError(pnm_label, returncode)
        raise RuntimeError(
            f"{pnm_label} simulation failed with exit code {returncode}."
        )

    if args["pnm_mode"] == "dynamic":
        time_series_file = Path(args["simoutputname"]) / "time_series.txt"
        if not _dynamic_time_series_has_data(time_series_file):
            raise RuntimeError(
                "Dynamic simulation produced no converged time steps. "
                "The solver returned exit code 0, but time_series.txt contains "
                "no data rows. Check log.txt and adjust the dynamic model setup "
                "before rerunning."
            )

    print(finish_message)
    return None


def _needs_simulation_response(args: dict, missing_paths: list[Path], plot_kind: str) -> dict:
    pnm_label = "dynamic" if args["pnm_mode"] == "dynamic" else "quasi-static"
    if args["task_mode"] == "uq_run":
        recommended_action = "run_realizations"
        message = (
            f"{plot_kind} files are not ready. Please run the {pnm_label} "
            "uncertainty quantification simulations first."
        )
    else:
        recommended_action = "run_simulation"
        message = (
            f"{plot_kind} files are not ready. Please run the {pnm_label} "
            "simulation first."
        )

    print(message)
    if missing_paths:
        print("Missing required plotting files:")
        for path in missing_paths:
            print(f"  {path}")

    response = dict(args)
    response.update({
        "status": "needs_simulation",
        "needs_simulation": True,
        "recommended_action": recommended_action,
        "message": message,
        "missing_files": [str(path) for path in missing_paths],
    })
    return response


def _missing_single_run_plot_files(args: dict) -> list[Path]:
    output_folder = Path(args["simoutputname"])
    if not output_folder.exists():
        return [output_folder]

    profile_folder = output_folder / "profile"
    missing = []
    if not profile_folder.exists():
        missing.append(profile_folder)
    elif not any(profile_folder.glob("*.txt")):
        missing.append(profile_folder / "*.txt")

    for required_file in [output_folder / "profile_range.txt"]:
        if not required_file.exists():
            missing.append(required_file)

    if args["pnm_mode"] == "dynamic":
        mode_file = output_folder / "time_series.txt"
    else:
        mode_file = output_folder / "qs_results.txt"
    if not mode_file.exists() or (
        args["pnm_mode"] == "dynamic"
        and not _dynamic_time_series_has_data(mode_file)
    ):
        missing.append(mode_file)

    if args["img_src"] == "porespy":
        local_vti = _single_run_input_pnm_folder(args) / "generated_image.vti"
        if not local_vti.exists():
            missing.append(local_vti)

    return missing


def _missing_uq_plot_files(pnm_mode: str) -> list[Path]:
    results_root = _mode_realization_folder(pnm_mode)
    if not results_root.exists():
        return [results_root]

    case_folders = [
        child for child in sorted(results_root.iterdir())
        if child.is_dir() and child.name.startswith("case_")
    ]
    if not case_folders:
        return [results_root / "case_###"]

    result_name = "time_series.txt" if pnm_mode == "dynamic" else "qs_results.txt"
    return [
        case_folder / result_name
        for case_folder in case_folders
        if not (case_folder / result_name).exists()
        or (
            pnm_mode == "dynamic"
            and not _dynamic_time_series_has_data(case_folder / result_name)
        )
    ]


def _save_dynamic_time_series_plots(simoutputname: str, display: bool = True):
    output_folder = Path(simoutputname)
    figs_folder = _single_run_figs_folder(output_folder)
    time_series_file = output_folder / "time_series.txt"
    if not time_series_file.exists():
        print(f"Dynamic time-series file not found: {time_series_file}")
        return []

    df = read_dynamic_time_series(time_series_file)
    if df is None:
        return []

    plots = [
        (
            "Sg_all",
            "S_{g} (-)",
            "Gas saturation-Time curve.png",
            "Gas saturation over time",
        ),
        (
            "ave_pcap",
            "Average P^{c} (Pa)",
            "Average capillary pressure-Time curve.png",
            "Average capillary pressure over time",
        ),
    ]

    saved = []
    x = pd.to_numeric(df["Production_Time"], errors="coerce")
    for column, ylabel, filename, title in plots:
        if column not in df.columns:
            print(f"Dynamic time-series column not found for plot: {column}")
            continue
        y = pd.to_numeric(df[column], errors="coerce")
        fig_path = save_time_series_plot(
            figs_folder,
            x,
            y,
            case_name=filename,
            xlab="Normalized time",
            ylab=ylabel,
            title=title,
            normalize_x=True,
        )
        if fig_path is None:
            continue
        if display:
            show_image(str(fig_path))
        saved.append(str(fig_path))

    bulk_column = "ave_aff_b" if "ave_aff_b" in df.columns else "ave_aff"
    area_columns = [
        column for column in (bulk_column, "ave_aff_bf") if column in df.columns
    ]
    if area_columns:
        area_labels = [
            "Bulk" if column == bulk_column else "Bulk + Film"
            for column in area_columns
        ]
        area_x = pd.to_numeric(df["Production_Time"], errors="coerce").to_numpy(dtype=float)
        finite_x = area_x[np.isfinite(area_x)]
        if finite_x.size and finite_x[-1] != 0:
            area_x = area_x / finite_x[-1]
        area_y = df[area_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        fig_path = save_plot(
            figs_folder,
            area_x,
            area_y,
            case_name="Fluid-fluid interfacial area-Time curve.png",
            xlab="Normalized time",
            ylab="A_{wn}",
            title="Fluid-fluid interfacial area over time",
            labels=area_labels,
        )
        if display:
            show_image(str(fig_path))
        saved.append(str(fig_path))

    return saved


def _save_quasi_static_curve_plots(simoutputname: str, display: bool = True) -> list[str]:
    output_folder = Path(simoutputname)
    figs_folder = _single_run_figs_folder(output_folder)
    qs_results_path = output_folder / "qs_results.txt"
    if not qs_results_path.exists():
        print(f"Quasi-static results file not found: {qs_results_path}")
        return []

    qs_df = pd.read_csv(qs_results_path, sep=r"\s+", engine="python")
    required_columns = {"Pc", "Sw", "Krw", "Krg", "Aaw"}
    missing_columns = sorted(required_columns - set(qs_df.columns))
    if missing_columns:
        print(
            "Quasi-static results file is missing columns for curve plots: "
            + ", ".join(missing_columns)
        )
        return []

    for column in ("Pc", "Sw", "Krw", "Krg", "Aaw", "Aaw_b"):
        if column in qs_df.columns:
            qs_df[column] = pd.to_numeric(qs_df[column], errors="coerce")

    area_columns = []
    if "Aaw_b" in qs_df.columns:
        area_columns.append("Aaw_b")
    area_columns.append("Aaw")
    area_labels = [
        "Bulk" if column == "Aaw_b" else "Bulk + Film"
        for column in area_columns
    ]

    saved = []
    plot_specs = [
        (
            qs_df[["Pc"]].to_numpy(dtype=float) / 1e3,
            "Pc-Sw curve.png",
            "P^{c} (kPa)",
            "Capillary pressure curve",
            None,
        ),
        (
            qs_df[["Krw", "Krg"]].to_numpy(dtype=float),
            "Kr-Sw curve.png",
            "k_{r} (-)",
            "Relative permeability curve",
            ["Wetting", "Nonwetting"],
        ),
        (
            qs_df[area_columns].to_numpy(dtype=float) / 1e2,
            "A_w_nw-Sw curve.png",
            "A_{wn} (cm^{-1})",
            "Fluid-fluid interfacial area curve",
            area_labels,
        ),
    ]

    x = qs_df["Sw"].to_numpy(dtype=float)
    for y, filename, ylabel, title, labels in plot_specs:
        save_plot(
            figs_folder,
            x,
            y,
            case_name=filename,
            xlab="S_{w} (-)",
            ylab=ylabel,
            title=title,
            labels=labels,
        )
        fig_path = figs_folder / filename
        if fig_path.exists():
            saved.append(str(fig_path))
            if display:
                show_image(str(fig_path))

    return saved


def _save_uq_case_curve_plots(case_folder: Path, pnm_mode: str) -> list[str]:
    case_folder = Path(case_folder)
    if pnm_mode == "dynamic":
        return _save_dynamic_time_series_plots(str(case_folder), display=False)
    return _save_quasi_static_curve_plots(str(case_folder), display=False)


def _selected_local_variables(quantity: str) -> list[str]:
    if quantity == "all":
        return [item for item in ALLOWED_VALUES["quantity"] if item != "all"]
    return [quantity]


def _paraview_local_variable_command(
    args: dict,
    pvpython: Path,
    input_path: Path,
    output_path: Path,
    quantity: str,
    output_mode: str,
    vmin: float,
    vmax: float,
    start_index: int | str,
) -> list[str]:
    return [
        str(pvpython),
        str(PACKAGE_DIR / "paraview_script.py"),
        str(Path(input_path).resolve()),
        str(Path(output_path).resolve()),
        quantity,
        output_mode,
        str(vmin),
        str(vmax),
        args["pnm_mode"],
        str(args["time_digits"]),
        str(start_index),
        _diameter_property_name("pore", args["pore_diameter_type"]),
        _diameter_property_name("throat", args["throat_diameter_type"]),
    ]


def _render_local_variable_png(
    args: dict,
    pn,
    input_vtp: Path,
    output_png: Path,
    quantity: str,
    pore_values: np.ndarray,
    time_value: str,
    time_fmt: str,
    clim: tuple[float, float],
) -> Path:
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    pvpython = find_pvbatch(THIS_DIR)
    if pvpython:
        command = _paraview_local_variable_command(
            args,
            pvpython,
            Path(input_vtp),
            output_png,
            quantity,
            "png",
            clim[0],
            clim[1],
            "0",
        )
        subprocess.run(command, check=True)
    else:
        print("pvpython.exe not found, pyvista is used for plot resutls! Plot is slower!")
        plot_network_with_cubes(
            pn,
            pore_size=f"pore.{_diameter_property_name('pore', args['pore_diameter_type'])}",
            throat_size=f"throat.{_diameter_property_name('throat', args['throat_diameter_type'])}",
            pore_values_name=quantity,
            filename=str(output_png),
            pore_values=pore_values,
            time_value=time_value,
            time_fmt=time_fmt,
            pore_size_factor=0.8,
            throat_size_factor=0.5,
            throat_color_mode="max",
            throat_style="square",
            cmap="viridis",
            clim=clim,
            camera_azimuth=180,
            camera_elevation=0,
            camera_roll=0,
        )
    return output_png


def _save_local_variable_animation(
    args: dict,
    vtk_folder: Path,
    figs_folder: Path,
    quantity: str,
    clim: tuple[float, float],
) -> Path | None:
    pvpython = find_pvbatch(THIS_DIR)
    if not pvpython:
        return None

    output_avi = Path(figs_folder) / f"ps_{quantity}.avi"
    if args["pnm_mode"] == "quasi_static":
        qs_results = np.loadtxt(Path(args["simoutputname"]) / "qs_results.txt", skiprows=1)
        qs_results = np.atleast_2d(qs_results)
        less_than_one = np.where(qs_results[:, 1] < 1)[0]
        start_index = int(less_than_one[0]) if len(less_than_one) else 0
    else:
        start_index = 0

    command = _paraview_local_variable_command(
        args,
        pvpython,
        vtk_folder,
        output_avi,
        quantity,
        "animation",
        clim[0],
        clim[1],
        start_index,
    )
    subprocess.run(command, check=True)
    output_gif = output_avi.with_suffix(".gif")
    avi_to_gif(output_avi, output_gif, fps=5)
    return output_gif


def _plot_local_variable_profiles(
    args: dict,
    pn,
    profile_folder: Path,
    files_sorted: list[tuple[float, str]],
    first_profile_df: pd.DataFrame,
    profile_range: np.ndarray,
    figs_folder: Path,
    quantity: str,
    isolate_outputs: bool,
) -> list[str]:
    if quantity not in first_profile_df.columns:
        print(f"{quantity} not found!")
        return []

    col_id = first_profile_df.columns.get_loc(quantity)
    print(f"Found {quantity} at column {col_id}!")
    vtk_folder = Path(args["simoutputname"]) / "vtks"
    if isolate_outputs:
        vtk_folder = vtk_folder / quantity
    clear_folder(vtk_folder)

    clim = (
        float(np.floor(profile_range[0, col_id])),
        float(np.ceil(profile_range[1, col_id])),
    )
    saved_figures: list[str] = []
    rendered_steps: list[tuple[Path, str, np.ndarray, str, int]] = []
    old_value = -1.0
    checkresults = 0

    for i, (value, file_name) in enumerate(files_sorted):
        if i <= 0:
            continue

        profile = np.loadtxt(profile_folder / file_name, skiprows=2)
        profile = np.atleast_2d(profile)
        value0 = value
        value /= args["dSdP"]
        rounded = round(value0, args["time_digits"])
        if np.round(value, args["time_digits"]) <= old_value:
            old_value = np.round(value, args["time_digits"])
            continue

        if pn["pore.coords"].shape[0] != len(profile[:, col_id]):
            print("The simulation results were for other pore networks, please regenerate the pore network!")
            checkresults = 1
            break

        label = f"{rounded:0.{args['time_digits']}f}"
        if args["pnm_mode"] == "quasi_static":
            time_fmt = f"Pres = {label} Pa"
            print(f"Save result: Pressure difference is {value0} Pa, {label} Pa")
        else:
            time_fmt = f"Time = {label} s"
            print(f"Save result: Time step is {value0} s, {label} s")

        pore_values = np.asarray(profile[:, col_id], dtype=float).ravel()
        pn["pore." + quantity] = pore_values
        safe_label = str(label).replace(".", "p")
        vtp_stem = vtk_folder / f"ps_{safe_label}"
        op.io.project_to_vtk(project=pn.project, filename=vtp_stem)

        old_vtp = vtp_stem.with_suffix(".vtp")
        new_vtp = vtk_folder / f"ps_{label}.vtp"
        if old_vtp.exists() and old_vtp != new_vtp:
            if new_vtp.exists():
                new_vtp.unlink()
            old_vtp.rename(new_vtp)

        rendered_steps.append((new_vtp, label, pore_values, time_fmt, i))

        if args["vtk"] == "png" and i % 50 == 1:
            png_name = f"{quantity}_ps_{label}.png" if isolate_outputs else f"ps_{label}.png"
            png_path = figs_folder / png_name
            _render_local_variable_png(
                args,
                pn,
                new_vtp,
                png_path,
                quantity,
                pore_values,
                label,
                time_fmt,
                clim,
            )
            show_image(str(png_path))
            saved_figures.append(str(png_path))

        old_value = np.round(value, args["time_digits"])

    if checkresults == 1 or not rendered_steps or args["vtk"] != "png":
        return saved_figures

    last_vtp, last_label, last_pore_values, last_time_fmt, _last_i = rendered_steps[-1]
    preview_path = figs_folder / f"ps_{quantity}_last.png"
    _render_local_variable_png(
        args,
        pn,
        last_vtp,
        preview_path,
        quantity,
        last_pore_values,
        last_label,
        last_time_fmt,
        clim,
    )
    saved_figures.append(str(preview_path))

    gif_path = _save_local_variable_animation(args, vtk_folder, figs_folder, quantity, clim)
    if gif_path is not None:
        saved_figures.append(str(gif_path))
    show_image(str(preview_path))
    return saved_figures


def run_pore_network_model(
    action: str = "generate_input",
    img_src: str = "porespy",
    pnm_mode: str = "quasi_static",
    task_mode: str = "single_run",
    quantity: str = "sat_nw",
    vtk: str = "png",
    domain_size: list = None,
    voxel_size: float = None,
    image_file: str = None,
    voxels: list = None,
    coarse_factor: int = None,
    porosity: float = None,
    blobiness: float = None,
    pore_diameter_type: str = "equivalent",
    throat_diameter_type: str = "equivalent",
):
    """Execute the pore-network model with the given customisable parameters."""

    # --- validate -----------------------------------------------------------
    for name, value in [
        ("action", action),
        ("img_src", img_src),
        ("pnm_mode", pnm_mode),
        ("task_mode", task_mode),
        ("quantity", quantity),
        ("vtk", vtk),
        ("pore_diameter_type", pore_diameter_type),
        ("throat_diameter_type", throat_diameter_type),
    ]:
        if value not in ALLOWED_VALUES[name]:
            raise ValueError(
                f"Invalid {name}='{value}'. Must be one of {ALLOWED_VALUES[name]}"
            )

    # --- build full args dict -----------------------------------------------
    args = {**DEFAULTS}
    args["img_src"] = img_src
    args["action"] = action
    args["pnm_mode"] = pnm_mode
    args["task_mode"] = task_mode
    args["quantity"] = quantity
    args["vtk"] = vtk
    args["pore_diameter_type"] = pore_diameter_type
    args["throat_diameter_type"] = throat_diameter_type

    # Override image/domain properties if provided
    if domain_size is not None:
        args["shape"] = domain_size
    if voxel_size is not None:
        args["resolution"] = voxel_size
    if image_file is not None:
        candidate = Path(image_file)
        if not candidate.is_absolute():
            candidate = THIS_DIR / "image" / candidate
        args["file"] = str(candidate)
    if voxels is not None:
        args["voxels"] = voxels
    if coarse_factor is not None:
        args["coarse_factor"] = coarse_factor
    if porosity is not None:
        args["poro"] = porosity
    if blobiness is not None:
        args["blobiness"] = blobiness

    args["blobiness"] = float(args["blobiness"])
    if not np.isfinite(args["blobiness"]) or args["blobiness"] <= 0:
        raise ValueError("blobiness must be a positive finite value.")

    # Scale resolution by coarse factor: after merging, each voxel spans
    # coarse_factor × the original voxel size.
    args["resolution"] = args["resolution"] * args["coarse_factor"]

    shape = args["shape"]
    porosity = args["poro"]
    blobiness = args["blobiness"]

    if args["pnm_mode"] == "dynamic":
        args["simoutputname"] = str(THIS_DIR / "single_run_dyn_pnm") + "/"

    # -----------------------------------------------------------------------
    generate = 1 if args["img_src"] == "porespy" else 0
    active_input_folder = _active_input_pnm_folder(args)
    args["active_input_pnm"] = str(active_input_folder)
    image_name = str(active_input_folder / "generated_image")
    image_folder = THIS_DIR / "image"

    # --- Check for user-supplied image when using x_ray mode ---------------
    if args["img_src"] == "x_ray" and not os.path.exists(args["file"]):
        print("\n" + "=" * 60)
        print("IMAGE NOT FOUND")
        print("=" * 60)
        print(f"Expected image file: {args['file']}")
        print(f"Please place your .raw image in the folder:")
        print(f"  {image_folder}")
        print()
        print("Supported format: unsigned 8-bit binary (.raw)")
        print(f"Expected dimensions (voxels): {args['voxels']}")
        print("=" * 60 + "\n")
        return
    
    print(args["action"] )

    if args["action"] == "generate_input":
        clear_folder(Path(args["simoutputname"] + "/profile"))
        args["outputname"] = args["file"].replace(".raw", "")
        print(args)
        get_image = 0

        if generate == 0:
            if os.path.exists(args["file"]):
                im0 = np.fromfile(args["file"], dtype="uint8", sep="").reshape(args["voxels"])
                get_image = 1
            else:
                print(f"Error: File {args['file']} not found")
        elif generate == 1:
            print("Start generating image ...")
            im0 = ps.generators.blobs(shape=shape, porosity=porosity, blobiness=blobiness)
            ps.io.to_vtk(im0, filename=image_name)
            get_image = 1
        else:
            print("please provide the image or ask for image generation")

        if get_image == 0:
            print("Error: Image not found")
        else:
            figs_folder = _single_run_figs_folder(args["simoutputname"])
            clear_folder(figs_folder)
            preview_path = _save_3d_preview_to_input_pnm(im0, active_input_folder)
            show_image(str(preview_path))

            im1, phi = coarsen_image_majority(im0, factor=args["coarse_factor"])
            im = enforce_dual_percolation(im1)
            im = ~np.array(im, dtype=bool)
            im = np.swapaxes(im, 0, 2)

            net0 = ps.networks.snow2(phases=im, boundary_width=0)
            pn0 = op.io.network_from_porespy(net0.network)

            proj_copy = pn0.project.copy()
            pn = proj_copy.network

            h = op.utils.check_network_health(pn)
            pores_to_trim = np.unique(
                np.hstack([h["isolated_pores"], h["disconnected_pores"]])
            ).astype(int)

            pn["pore.old_id"] = np.arange(pn.Np)
            pn["throat.old_id"] = np.arange(pn.Nt)

            Pkeep = np.ones(pn.Np, dtype=bool)
            Pkeep[pores_to_trim] = False
            old_to_new_pore = -np.ones(pn.Np, dtype=int)
            old_to_new_pore[Pkeep] = np.arange(Pkeep.sum())
            op.topotools.trim(network=pn, pores=pores_to_trim)

            pore_diameter_property = _diameter_property_name("pore", args["pore_diameter_type"])
            throat_diameter_property = _diameter_property_name("throat", args["throat_diameter_type"])

            diameters = _network_diameter_values(pn, "pore", args["pore_diameter_type"])
            Nb = diameters.shape[0]
            indices = np.arange(Nb).reshape(-1, 1)
            shapes = np.ones(Nb).reshape(-1, 1)
            #lengths = diameters
            materials = 2 - pn["pore.phase"].reshape(-1, 1)
            eletrodes = -np.ones(Nb).reshape(-1, 1)
            pore_volumes = np.asarray(pn["pore.region_volume"]).reshape(-1, 1)
            lengths = np.divide(pore_volumes, diameters ** 2, out=diameters.copy(), where=diameters > 0) 
            pore_bodies = np.hstack(
                (
                    indices,
                    shapes,
                    diameters * args["resolution"] / 2.0,
                    lengths * args["resolution"],
                    pore_volumes * args["resolution"] ** 3,
                )
            )

            links = pn["throat.conns"]
            Nt = links.shape[0]
            indices = np.arange(Nt).reshape(-1, 1)
            shapes = np.ones(Nt).reshape(-1, 1)
            diameters = _network_diameter_values(pn, "throat", args["throat_diameter_type"])
            lengths = pn["throat.total_length"].reshape(-1, 1)
            throat_areas = np.asarray(pn["throat.cross_sectional_area"]).reshape(-1, 1)
            pore_throats = np.hstack(
                (
                    indices,
                    links,
                    shapes,
                    diameters * args["resolution"] / 2.0,
                    lengths * args["resolution"],
                    throat_areas * args["resolution"] ** 2,
                )
            )

            regions = net0["regions"].astype(np.int32)
            inlet_pore_ids0, inlet_pore_area0, outlet_pore_ids0, outlet_pore_area0 = safe_boundary_pores_with_area(
                pn0, regions, args, axis=0
            )

            inlet_pore_ids_mapped = old_to_new_pore[inlet_pore_ids0]
            outlet_pore_ids_mapped = old_to_new_pore[outlet_pore_ids0]

            inlet_mask = inlet_pore_ids_mapped >= 0
            outlet_mask = outlet_pore_ids_mapped >= 0
            inlet_pore_ids = inlet_pore_ids_mapped[inlet_mask]
            inlet_pore_area = inlet_pore_area0[inlet_mask]
            outlet_pore_ids = outlet_pore_ids_mapped[outlet_mask]
            outlet_pore_area = outlet_pore_area0[outlet_mask]

            bound_id = np.hstack((inlet_pore_ids, outlet_pore_ids)).reshape(-1, 1)
            bound_type = np.vstack(
                (eletrodes[inlet_pore_ids] * 0 + 1, 0 * eletrodes[outlet_pore_ids])
            ).reshape(-1, 1)
            bound_value = np.vstack(
                (eletrodes[inlet_pore_ids] * 0 + 1, eletrodes[outlet_pore_ids] * 0)
            ).reshape(-1, 1)

            indices = np.arange(Nb).reshape(-1, 1)
            coordinates = np.hstack((indices, pn["pore.geometric_centroid"] * args["resolution"]))
            boundaries = np.hstack((bound_id, bound_type, bound_value))

            thickness_value = args["thickness"] * (np.max(coordinates[:, 1]) - np.min(coordinates[:, 1]))
            xin = coordinates[inlet_pore_ids, 1]
            xin_id = np.where(xin <= np.min(coordinates[:, 1]) + thickness_value)[0]
            xout = coordinates[outlet_pore_ids, 1]
            xout_id = np.where(xout >= np.max(coordinates[:, 1]) - thickness_value)[0]
            selected = np.hstack((inlet_pore_ids[xin_id], outlet_pore_ids[xout_id]))

            if len(xin_id) == 0 or len(xout_id) == 0:
                print("Thickness is too small, Increase the value!!!")

            if len(selected) > 0:
                common_vals = np.intersect1d(bound_id, selected)
                idx_in_b = [np.where(bound_id == selected_val)[0] for selected_val in common_vals]
                idx = np.array(idx_in_b).ravel().reshape(-1, 1)
                boundaries = np.hstack(
                    (
                        bound_id[idx, :].reshape(-1, 1),
                        bound_type[idx, :].reshape(-1, 1),
                        bound_value[idx, :].reshape(-1, 1),
                    )
                )

            x_domain_size = float(im.shape[0]) * float(args["resolution"])
            if not np.isfinite(x_domain_size) or x_domain_size <= 0:
                x_domain_size = float(np.max(coordinates[:, 1]) - np.min(coordinates[:, 1]))
            ghost_throat_length = x_domain_size / 100.0

            generate_input_file(
                pore_bodies,
                pore_throats,
                boundaries,
                coordinates,
                output_filename=args["outputfolder"] + args["pnm_name"] + ".txt",
                ghost_throat_length=ghost_throat_length,
            )
            
            op.io.project_to_vtk(
                project=pn.project,
                filename=Path(args["outputfolder"]) / args["pnm_name"],
            )

            print("Finish reading/generating image and input files !!!")
            print("Visualizing the pore network ...")
            pore_values = _network_diameter_values(pn, "pore", args["pore_diameter_type"])
            throat_values = _network_diameter_values(pn, "throat", args["throat_diameter_type"])
            size_distribution_path = save_pore_size_distribution_plot(
                Path(args["outputfolder"]),
                pore_values.ravel() * args["resolution"],
                throat_values.ravel() * args["resolution"],
            )
            pvpython = find_pvbatch(THIS_DIR)
            if pvpython:
                script = str(PACKAGE_DIR / "paraview_script.py")
                arg1 = Path(args["outputfolder"] + args["pnm_name"] + ".vtp").resolve()
                arg2 = arg1.parent / arg1.with_suffix(".png").name
                arg3 = pore_diameter_property
                arg4 = "pnm"
                arg5 = str(np.min(pore_values))
                arg6 = str(np.max(pore_values))
                subprocess.run(
                    [
                        pvpython,
                        script,
                        arg1,
                        arg2,
                        arg3,
                        arg4,
                        arg5,
                        arg6,
                        "none",
                        "5",
                        "1",
                        pore_diameter_property,
                        throat_diameter_property,
                    ],
                    check=True,
                )
                #show_image(args["outputfolder"] + args["pnm_name"] + ".png")
            else:
                print("pvpython.exe not found, pyvista is used for plot resutls! Plot is slower!")
                plot_network_with_cubes(
                    pn,
                    pore_size=f"pore.{pore_diameter_property}",
                    throat_size=f"throat.{throat_diameter_property}",
                    pore_values=pore_values,
                    pore_values_name= "Pore size (um)",
                    pore_size_factor=0.8,
                    throat_size_factor=0.25,
                    throat_color_mode="max",
                    throat_style="square",
                    cmap="viridis",
                    clim=(np.min(pore_values), np.max(pore_values)),
                    filename=args["outputfolder"] + args["pnm_name"] + ".png",
                )
                
            #print("Finish visualize the pore network !!!")
            show_image(args["outputfolder"] + args["pnm_name"] + ".png")
            if size_distribution_path is not None:
                show_image(str(size_distribution_path))
            if args["task_mode"] == "single_run":
                args["input_pnm_copy"] = _copy_input_pnm_folder(Path(args["simoutputname"]))
            print("Finish visualize the pore network !!!")

    elif args["action"] == "run_simulation":
        max_attempts = _exe_corruption_retry_limit()
        attempt = 1

        while True:
            _prepare_simulation_output_folder(args["simoutputname"])

            try:
                simulation_result = _run_pnm_executable(args)
            except ExeCorruptionError as exc:
                if max_attempts is not None and attempt >= max_attempts:
                    raise RuntimeError(
                        f"{exc.pnm_label} simulation failed after {attempt} attempts "
                        f"because the executable kept returning heap-corruption "
                        f"exit code {exc.returncode}."
                    ) from exc

                next_attempt = attempt + 1
                attempt_text = (
                    f"{next_attempt}/{max_attempts}"
                    if max_attempts is not None
                    else f"{next_attempt}"
                )
                print(
                    f"{exc.pnm_label} executable returned heap-corruption exit code "
                    f"{exc.returncode}. Regenerating input and retrying "
                    f"({attempt_text}) ..."
                )
                run_pore_network_model(
                    action="generate_input",
                    img_src=img_src,
                    pnm_mode=pnm_mode,
                    task_mode=task_mode,
                    quantity=quantity,
                    vtk=vtk,
                    domain_size=domain_size,
                    voxel_size=voxel_size,
                    image_file=image_file,
                    voxels=voxels,
                    coarse_factor=coarse_factor,
                    porosity=porosity,
                    blobiness=blobiness,
                    pore_diameter_type=pore_diameter_type,
                    throat_diameter_type=throat_diameter_type,
                )
                attempt = next_attempt
                continue

            if isinstance(simulation_result, dict):
                return simulation_result
            break

    elif args["action"] == "plot_result":
        if args["task_mode"] == "uq_run":
            missing_files = _missing_uq_plot_files(args["pnm_mode"])
            if missing_files:
                return _needs_simulation_response(args, missing_files, "UQ plotting")

            print("Start plotting uncertainty quantification results ...")
            saved_uq_figures = save_uq_results_plots(args["pnm_mode"], THIS_DIR)
            args["uq_figures"] = saved_uq_figures
            args["model_results_archive"] = archive_completed_cycle(
                THIS_DIR,
                args["task_mode"],
                args["pnm_mode"],
            )
            args["status"] = "success"
            print("Finish plotting uncertainty quantification results !!!")
            return args

        missing_files = _missing_single_run_plot_files(args)
        if missing_files:
            return _needs_simulation_response(args, missing_files, "Single-run plotting")

        print("Start ploting results ...")
        get_image = 0

        if generate == 0:
            if os.path.exists(args["file"]):
                im0 = np.fromfile(args["file"], dtype="uint8", sep="").reshape(args["voxels"])
                get_image = 1
            else:
                print(f"Error: File {args['file']} not found")

        elif generate == 1:
            vtk_file = image_name + ".vti"
            if os.path.exists(vtk_file):
                mesh = pv.read(vtk_file)
                im0 = mesh.cell_data.active_scalars.reshape(np.array(mesh.dimensions) - 1, order="F")
                get_image = 1
            else:
                print(f"Error: File {vtk_file} not found")

        if get_image == 0:
            print("Error: Image not found")
        else:
            figs_folder = _single_run_figs_folder(args["simoutputname"])
            clear_folder(figs_folder)
            preview_path = _input_pnm_3d_preview_path(active_input_folder)
            if not preview_path.exists():
                preview_path = _save_3d_preview_to_input_pnm(im0, active_input_folder)
            show_image(str(preview_path))
            if args["pnm_mode"] == "dynamic":
                _save_dynamic_time_series_plots(args["simoutputname"])

            im1, phi = coarsen_image_majority(im0, factor=args["coarse_factor"])
            im = enforce_dual_percolation(im1)
            im = ~np.array(im, dtype=bool)
            im = np.swapaxes(im, 0, 2)

            net0 = ps.networks.snow2(phases=im, boundary_width=0)
            pn0 = op.io.network_from_porespy(net0.network)

            proj_copy = pn0.project.copy()
            pn = proj_copy.network

            h = op.utils.check_network_health(pn)
            pores_to_trim = np.unique(
                np.hstack([h["isolated_pores"], h["disconnected_pores"]])
            ).astype(int)

            pn["pore.old_id"] = np.arange(pn.Np)
            pn["throat.old_id"] = np.arange(pn.Nt)

            Pkeep = np.ones(pn.Np, dtype=bool)
            Pkeep[pores_to_trim] = False
            old_to_new_pore = -np.ones(pn.Np, dtype=int)
            old_to_new_pore[Pkeep] = np.arange(Pkeep.sum())
            op.topotools.trim(network=pn, pores=pores_to_trim)

            folder = Path(args["simoutputname"] + "profile")
            files = [p.name for p in folder.glob("*.txt")]
            files_sorted = sorted(((extract_scientific_number(f), f) for f in files), key=lambda x: x[0])

            print("")
            if len(files) == 0:
                print("No files in the profile folder...")
            else:
                clear_folder(Path(args["simoutputname"] + "vtks"))

                df = pd.read_csv(
                    folder / files_sorted[0][1],
                    sep=r"\s+",
                    header=1,
                    comment=";",
                    engine="python",
                )

                profile_range = np.loadtxt(args["simoutputname"] + "profile_range.txt", skiprows=2)
                profile_range = np.atleast_2d(profile_range)
                if args["pnm_mode"] == "quasi_static":
                    qs_results_path = Path(args["simoutputname"]) / "qs_results.txt"
                    qs_df = pd.read_csv(qs_results_path, sep=r"\s+", engine="python")
                    for column in ("Pc", "Sw", "Krw", "Krg", "Aaw", "Aaw_b"):
                        if column in qs_df.columns:
                            qs_df[column] = pd.to_numeric(qs_df[column], errors="coerce")
                    area_columns = []
                    if "Aaw_b" in qs_df.columns:
                        area_columns.append("Aaw_b")
                    area_columns.append("Aaw")
                    area_labels = [
                        "Bulk" if column == "Aaw_b" else "Bulk + Film"
                        for column in area_columns
                    ]
                    save_plot(
                        figs_folder,
                        qs_df["Sw"].to_numpy(dtype=float),
                        qs_df[["Pc"]].to_numpy(dtype=float) / 1e3,
                        case_name="Pc-Sw curve.png",
                        xlab="S_{w} (-)",
                        ylab="P^{c} (kPa)",
                        title="Capillary pressure curve",
                    )
                    show_image(str(figs_folder / "Pc-Sw curve.png"))

                    save_plot(
                        figs_folder,
                        qs_df["Sw"].to_numpy(dtype=float),
                        qs_df[["Krw", "Krg"]].to_numpy(dtype=float),
                        case_name="Kr-Sw curve.png",
                        xlab="S_{w} (-)",
                        ylab="k_{r} (-)",
                        title="Relative permeability curve",
                        labels=["Wetting", "Nonwetting"],
                    )
                    show_image(str(figs_folder / "Kr-Sw curve.png"))

                    save_plot(
                        figs_folder,
                        qs_df["Sw"].to_numpy(dtype=float),
                        qs_df[area_columns].to_numpy(dtype=float) / 1e2,
                        case_name="A_w_nw-Sw curve.png",
                        xlab="S_{w} (-)",
                        ylab="A_{wn} (cm^{-1})",
                        title="Fluid-fluid interfacial area curve",
                        labels=area_labels,
                    )
                    show_image(str(figs_folder / "A_w_nw-Sw curve.png"))

                rounded = round(files_sorted[-1][0], args["time_digits"])
                label = f"{rounded:0.{args['time_digits']}f}"
                if args["pnm_mode"] == "quasi_static":
                    print(f"Producing and showing image at the largest pressure difference: {rounded} Pa, {label} Pa")
                elif args["pnm_mode"] == "dynamic":
                    print(f"Producing and showing image at the last time step: {rounded} s, {label} s")

                quantities = _selected_local_variables(args["quantity"])
                saved_local_figures = []
                for quantity in quantities:
                    saved_local_figures.extend(
                        _plot_local_variable_profiles(
                            args,
                            pn,
                            folder,
                            files_sorted,
                            df,
                            profile_range,
                            figs_folder,
                            quantity,
                            isolate_outputs=args["quantity"] == "all",
                        )
                    )
                if saved_local_figures:
                    args["local_variable_figures"] = saved_local_figures
                args["media_files"] = _move_root_media_to_figs(args["simoutputname"])
                print("Finish ploting results !!!")
                    
    else:
        print(f"Task '{args['action']}' undefined!!!")

    if args["action"] == "plot_result":
        args["model_results_archive"] = archive_completed_cycle(
            THIS_DIR,
            args["task_mode"],
            args["pnm_mode"],
        )

    return args


def run_pore_network_realizations(
    realizations: int,
    img_src: str = "porespy",
    pnm_mode: str = "quasi_static",
    task_mode: str = "uq_run",
    quantity: str = "sat_nw",
    vtk: str = "vtk",
    domain_size: list = None,
    voxel_size: float = None,
    image_file: str = None,
    voxels: list = None,
    coarse_factor: int = None,
    porosity: float = None,
    blobiness: float = None,
    pore_diameter_type: str = "equivalent",
    throat_diameter_type: str = "equivalent",
    progress_callback=None,
):
    """
    Run multiple generate-input + simulation realizations in the backend.

    For each case, this function:
      1. Generates a fresh PNM input from the configured image source.
      2. Runs one quasi-static or dynamic simulation.
      3. Copies the current input_pnm folder into the single-run output folder.
      4. Copies single_run_qs_pnm or single_run_dyn_pnm into uq_qs_pnm/case_###
         or uq_dyn_pnm/case_###.
      5. Saves curve plots into that UQ case's figs folder.
    """
    try:
        realizations = int(realizations)
    except (TypeError, ValueError):
        raise ValueError("realizations must be an integer.")
    if realizations < 1:
        raise ValueError("realizations must be at least 1.")
    if pnm_mode not in ALLOWED_VALUES["pnm_mode"]:
        raise ValueError(f"Invalid pnm_mode='{pnm_mode}'. Must be one of {ALLOWED_VALUES['pnm_mode']}")
    if task_mode not in ALLOWED_VALUES["task_mode"]:
        raise ValueError(f"Invalid task_mode='{task_mode}'. Must be one of {ALLOWED_VALUES['task_mode']}")
    if img_src not in ALLOWED_VALUES["img_src"]:
        raise ValueError(f"Invalid img_src='{img_src}'. Must be one of {ALLOWED_VALUES['img_src']}")
    if img_src != "porespy":
        raise ValueError("UQ realizations only support img_src='porespy'.")
    if pore_diameter_type not in ALLOWED_VALUES["pore_diameter_type"]:
        raise ValueError(
            f"Invalid pore_diameter_type='{pore_diameter_type}'. "
            f"Must be one of {ALLOWED_VALUES['pore_diameter_type']}"
        )
    if throat_diameter_type not in ALLOWED_VALUES["throat_diameter_type"]:
        raise ValueError(
            f"Invalid throat_diameter_type='{throat_diameter_type}'. "
            f"Must be one of {ALLOWED_VALUES['throat_diameter_type']}"
        )

    output_folder = _mode_output_folder(pnm_mode)
    results_root = _mode_realization_folder(pnm_mode)
    _reset_output_folder(results_root)
    next_case_index = _next_realization_index(results_root)
    cases = []

    common_kwargs = {
        "img_src": img_src,
        "pnm_mode": pnm_mode,
        "task_mode": task_mode,
        "quantity": quantity,
        "vtk": vtk,
        "domain_size": domain_size,
        "voxel_size": voxel_size,
        "image_file": image_file,
        "voxels": voxels,
        "coarse_factor": coarse_factor,
        "porosity": porosity,
        "blobiness": blobiness,
        "pore_diameter_type": pore_diameter_type,
        "throat_diameter_type": throat_diameter_type,
    }

    for offset in range(realizations):
        case_index = next_case_index + offset
        case_name = f"case_{case_index:03d}"

        print("\n" + "=" * 60)
        print(f"UQ realization {offset + 1}/{realizations}: {case_name}")
        print("=" * 60)

        if progress_callback is not None:
            progress_callback({
                "event": "realization_started",
                "index": offset + 1,
                "total": realizations,
                "case": case_name,
            })

        run_pore_network_model(action="generate_input", **common_kwargs)
        if progress_callback is not None:
            progress_callback({
                "event": "realization_generated",
                "index": offset + 1,
                "total": realizations,
                "case": case_name,
            })

        simulation_result = run_pore_network_model(action="run_simulation", **common_kwargs)
        if progress_callback is not None:
            progress_callback({
                "event": "realization_simulated",
                "index": offset + 1,
                "total": realizations,
                "case": case_name,
            })

        input_pnm_in_output = _copy_input_pnm_folder(output_folder)
        copied_to = _copy_realization_output(output_folder, results_root, case_index)
        case_input_pnm = copied_to / Path(DEFAULTS["outputfolder"]).name
        input_pnm_copy = (
            str(case_input_pnm)
            if input_pnm_in_output and case_input_pnm.exists()
            else None
        )
        case_curve_figures = _save_uq_case_curve_plots(copied_to, pnm_mode)
        case_info = {
            "case": copied_to.name,
            "status": "success",
            "copied_to": str(copied_to),
            "input_pnm_copy": input_pnm_copy,
            "curve_figures": case_curve_figures,
        }

        if isinstance(simulation_result, dict) and simulation_result.get("status") == "manually_closed":
            case_info["status"] = "manually_closed"
            cases.append(case_info)
            return {
                "status": "manually_closed",
                "message": (
                    f"Stopped after {copied_to.name} because the simulation window "
                    "was manually closed. Output was copied before stopping."
                ),
                "results_root": str(results_root),
                "cases": cases,
            }

        cases.append(case_info)
        if progress_callback is not None:
            progress_callback({
                "event": "realization_finished",
                "index": offset + 1,
                "total": realizations,
                "case": copied_to.name,
                "status": case_info["status"],
            })

    return {
        "status": "success",
        "message": (
            f"Finished {realizations} {pnm_mode} realizations. "
            f"Copied outputs to {results_root}."
        ),
        "results_root": str(results_root),
        "cases": cases,
    }


def rerun_pore_network_case(
    case,
    pnm_mode: str = "quasi_static",
    task_mode: str = "uq_run",
    quantity: str = "sat_nw",
    vtk: str = "vtk",
):
    """
    Rerun an existing UQ case using that case's saved input_pnm snapshot.

    The saved snapshot is restored into the active input_pnm folder first, then
    the selected simulation is rerun, the active input_pnm folder is copied
    into the single-run output folder, the corresponding case folder is
    replaced by that full output folder, and curve plots are saved into that
    UQ case's figs folder.
    """
    if pnm_mode not in ALLOWED_VALUES["pnm_mode"]:
        raise ValueError(f"Invalid pnm_mode='{pnm_mode}'. Must be one of {ALLOWED_VALUES['pnm_mode']}")
    if task_mode not in ALLOWED_VALUES["task_mode"]:
        raise ValueError(f"Invalid task_mode='{task_mode}'. Must be one of {ALLOWED_VALUES['task_mode']}")

    case_folder = _case_folder_from_identifier(pnm_mode, case)
    if not case_folder.exists():
        raise FileNotFoundError(f"UQ case folder not found: {case_folder}")

    restored_input = _replace_input_pnm_from_case(case_folder)
    output_folder = _mode_output_folder(pnm_mode)
    simulation_result = run_pore_network_model(
        action="run_simulation",
        img_src="porespy",
        pnm_mode=pnm_mode,
        task_mode=task_mode,
        quantity=quantity,
        vtk=vtk,
    )

    input_pnm_in_output = _copy_input_pnm_folder(output_folder)
    copied_to = _replace_case_with_output(output_folder, case_folder)
    case_input_pnm = case_folder / Path(DEFAULTS["outputfolder"]).name
    input_pnm_copy = (
        str(case_input_pnm)
        if input_pnm_in_output and case_input_pnm.exists()
        else None
    )
    case_curve_figures = _save_uq_case_curve_plots(case_folder, pnm_mode)

    status = "success"
    message = f"Reran {case_folder.name} for {pnm_mode} and replaced its output."
    if isinstance(simulation_result, dict) and simulation_result.get("status") == "manually_closed":
        status = "manually_closed"
        message = (
            f"Stopped after rerunning {case_folder.name}; output was copied "
            "before stopping."
        )

    return {
        "status": status,
        "message": message,
        "case": case_folder.name,
        "pnm_mode": pnm_mode,
        "restored_input_pnm": restored_input,
        "copied_to": copied_to,
        "input_pnm_copy": input_pnm_copy,
        "curve_figures": case_curve_figures,
    }



# ---------------------------------------------------------------------------
# CLI entry-point (kept for backward compatibility)
# ---------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        description="extract network from image using PoreSpy"
    )
    parser.add_argument("file", type=str, help="The image file")
    parser.add_argument("-sp", "--img_src", type=str, default="porespy")
    parser.add_argument("-b", "--action", type=str, default="generate_input")
    parser.add_argument("-e", "--pnm_mode", type=str, default="quasi_static")
    parser.add_argument("-m", "--task_mode", type=str, default="single_run")
    parser.add_argument("-r", "--resolution", type=float, required=True)
    parser.add_argument("-v", "--voxels", nargs="+", type=int, default=[1])
    parser.add_argument("-sh", "--shape", nargs="+", type=int, default=[])
    parser.add_argument("-po", "--poro", type=float, default=0.5)
    parser.add_argument("-bl", "--blobiness", type=float, default=2)
    parser.add_argument("-c", "--coarse_factor", type=int, default=1)
    parser.add_argument("-f", "--outputfolder", type=str, default="./input_pnm/")
    parser.add_argument("-n", "--pnm_name", type=str, default="")
    parser.add_argument("-t", "--thickness", type=float, default=1)
    parser.add_argument("-s", "--simoutputname", type=str, default="./single_run_qs_pnm/")
    parser.add_argument("-g", "--dSdP", type=int, default=5)
    parser.add_argument("-a", "--min_boundary_face_area", type=float)
    parser.add_argument("-d", "--time_digits", type=int, default=2)
    parser.add_argument("-p", "--quantity", type=str, default="sat_nw")
    parser.add_argument("-k", "--vtk", type=str, default="png")
    parser.add_argument(
        "--pore_diameter_type",
        "--pore-diameter-type",
        type=str,
        default=DEFAULTS["pore_diameter_type"],
        choices=ALLOWED_VALUES["pore_diameter_type"],
    )
    parser.add_argument(
        "--throat_diameter_type",
        "--throat-diameter-type",
        type=str,
        default=DEFAULTS["throat_diameter_type"],
        choices=ALLOWED_VALUES["throat_diameter_type"],
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    cli_args = vars(parser.parse_args())
    run_pore_network_model(
        img_src=cli_args["img_src"],
        action=cli_args["action"],
        pnm_mode=cli_args["pnm_mode"],
        task_mode=cli_args["task_mode"],
        quantity=cli_args["quantity"],
        vtk=cli_args["vtk"],
        blobiness=cli_args["blobiness"],
        pore_diameter_type=cli_args["pore_diameter_type"],
        throat_diameter_type=cli_args["throat_diameter_type"],
    )
    

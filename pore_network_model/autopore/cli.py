"""
Run default PNM workflows without the LLM/chat GUI.

The simulation and plotting steps are intentionally separate. Run the
simulation command first, wait for it to finish, then run the matching
plot command.

Examples:
    python -m pore_network_model.autopore.cli qs-single-run
    python -m pore_network_model.autopore.cli qs-single-plot
    python -m pore_network_model.autopore.cli dyn-single-run
    python -m pore_network_model.autopore.cli dyn-single-plot
    python -m pore_network_model.autopore.cli qs-uq-run
    python -m pore_network_model.autopore.cli qs-uq-plot
    python -m pore_network_model.autopore.cli dyn-uq-run
    python -m pore_network_model.autopore.cli dyn-uq-plot
"""

from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

from .model import (
    ALLOWED_VALUES,
    DEFAULTS as MODEL_DEFAULTS,
    run_pore_network_model,
    run_pore_network_realizations,
)


DEFAULT_REALIZATIONS = 5
XRAY_VOXELS = [1920, 1920, 1190]
XRAY_VOXEL_SIZE = 3.2
XRAY_DEFAULT_IMAGE_FILE = Path(MODEL_DEFAULTS["file"]).name
DEFAULT_PORE_DIAMETER_TYPE = MODEL_DEFAULTS["pore_diameter_type"]
DEFAULT_THROAT_DIAMETER_TYPE = MODEL_DEFAULTS["throat_diameter_type"]


def _print_result(result):
    if isinstance(result, dict):
        status = result.get("status")
        message = result.get("message")
        if status:
            print(f"Status: {status}")
        if message:
            print(message)
        if result.get("missing_files"):
            print("Missing files:")
            for path in result["missing_files"]:
                print(f"  {path}")
        if not status and not message:
            pprint(result)
    else:
        print(result)


def _diameter_kwargs(args):
    return {
        "pore_diameter_type": args.pore_diameter_type,
        "throat_diameter_type": args.throat_diameter_type,
    }


def run_single_quasi_static(args):
    """Generate the default image/input and run one quasi-static simulation."""
    diameter_kwargs = _diameter_kwargs(args)
    run_pore_network_model(
        action="generate_input",
        pnm_mode="quasi_static",
        task_mode="single_run",
        blobiness=args.blobiness,
        **diameter_kwargs,
    )
    return run_pore_network_model(
        action="run_simulation",
        pnm_mode="quasi_static",
        task_mode="single_run",
        **diameter_kwargs,
    )


def plot_single_quasi_static():
    """Plot the default single-run quasi-static results."""
    return run_pore_network_model(
        action="plot_result",
        pnm_mode="quasi_static",
        task_mode="single_run",
    )


def run_single_dynamic(args):
    """Generate the default image/input and run one dynamic simulation."""
    diameter_kwargs = _diameter_kwargs(args)
    run_pore_network_model(
        action="generate_input",
        pnm_mode="dynamic",
        task_mode="single_run",
        blobiness=args.blobiness,
        **diameter_kwargs,
    )
    return run_pore_network_model(
        action="run_simulation",
        pnm_mode="dynamic",
        task_mode="single_run",
        **diameter_kwargs,
    )


def plot_single_dynamic():
    """Plot the default single-run dynamic results."""
    return run_pore_network_model(
        action="plot_result",
        pnm_mode="dynamic",
        task_mode="single_run",
    )


def _xray_kwargs(args, image_file: str | None = None):
    return {
        "img_src": "x_ray",
        "image_file": image_file or XRAY_DEFAULT_IMAGE_FILE,
        "voxels": XRAY_VOXELS,
        "voxel_size": XRAY_VOXEL_SIZE,
        **_diameter_kwargs(args),
    }


def run_xray_single_quasi_static(args):
    """Generate x-ray input and run one quasi-static simulation."""
    kwargs = _xray_kwargs(args, args.image_file)
    run_pore_network_model(
        action="generate_input",
        pnm_mode="quasi_static",
        task_mode="single_run",
        **kwargs,
    )
    return run_pore_network_model(
        action="run_simulation",
        pnm_mode="quasi_static",
        task_mode="single_run",
        **kwargs,
    )


def plot_xray_single_quasi_static(args):
    """Plot x-ray single-run quasi-static results."""
    return run_pore_network_model(
        action="plot_result",
        pnm_mode="quasi_static",
        task_mode="single_run",
        **_xray_kwargs(args, args.image_file),
    )


def run_xray_single_dynamic(args):
    """Generate x-ray input and run one dynamic simulation."""
    kwargs = _xray_kwargs(args, args.image_file)
    run_pore_network_model(
        action="generate_input",
        pnm_mode="dynamic",
        task_mode="single_run",
        **kwargs,
    )
    return run_pore_network_model(
        action="run_simulation",
        pnm_mode="dynamic",
        task_mode="single_run",
        **kwargs,
    )


def plot_xray_single_dynamic(args):
    """Plot x-ray single-run dynamic results."""
    return run_pore_network_model(
        action="plot_result",
        pnm_mode="dynamic",
        task_mode="single_run",
        **_xray_kwargs(args, args.image_file),
    )


def run_uq_quasi_static(args):
    """Run default quasi-static uncertainty quantification simulations."""
    return run_pore_network_realizations(
        realizations=args.realizations,
        pnm_mode="quasi_static",
        task_mode="uq_run",
        blobiness=args.blobiness,
        **_diameter_kwargs(args),
    )


def plot_uq_quasi_static():
    """Plot default quasi-static uncertainty quantification results."""
    return run_pore_network_model(
        action="plot_result",
        pnm_mode="quasi_static",
        task_mode="uq_run",
    )


def run_uq_dynamic(args):
    """Run default dynamic uncertainty quantification simulations."""
    return run_pore_network_realizations(
        realizations=args.realizations,
        pnm_mode="dynamic",
        task_mode="uq_run",
        blobiness=args.blobiness,
        **_diameter_kwargs(args),
    )


def plot_uq_dynamic():
    """Plot default dynamic uncertainty quantification results."""
    return run_pore_network_model(
        action="plot_result",
        pnm_mode="dynamic",
        task_mode="uq_run",
    )


TASKS = {
    "qs-single-run": lambda args: run_single_quasi_static(args),
    "qs-single-plot": lambda args: plot_single_quasi_static(),
    "dyn-single-run": lambda args: run_single_dynamic(args),
    "dyn-single-plot": lambda args: plot_single_dynamic(),
    "xray-qs-single-run": lambda args: run_xray_single_quasi_static(args),
    "xray-qs-single-plot": lambda args: plot_xray_single_quasi_static(args),
    "xray-dyn-single-run": lambda args: run_xray_single_dynamic(args),
    "xray-dyn-single-plot": lambda args: plot_xray_single_dynamic(args),
    "qs-uq-run": lambda args: run_uq_quasi_static(args),
    "qs-uq-plot": lambda args: plot_uq_quasi_static(),
    "dyn-uq-run": lambda args: run_uq_dynamic(args),
    "dyn-uq-plot": lambda args: plot_uq_dynamic(),
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run default PNM tasks without the LLM/chat GUI."
    )
    parser.add_argument(
        "task",
        choices=sorted(TASKS),
        help="Run step or plotting step to execute.",
    )
    parser.add_argument(
        "--realizations",
        type=int,
        default=DEFAULT_REALIZATIONS,
        help="Number of UQ realizations for qs-uq-run or dyn-uq-run.",
    )
    parser.add_argument(
        "--blobiness",
        type=float,
        default=None,
        help=f"PoreSpy blobiness for generated synthetic images. Default: {MODEL_DEFAULTS['blobiness']}.",
    )
    parser.add_argument(
        "--image-file",
        type=str,
        default=XRAY_DEFAULT_IMAGE_FILE,
        help=(
            "Raw x-ray image filename for xray-* tasks. Relative paths are "
            "resolved from pore_network_model/image/. "
            f"Default: {XRAY_DEFAULT_IMAGE_FILE}. "
            f"xray-* tasks use voxels={XRAY_VOXELS} and voxel_size={XRAY_VOXEL_SIZE} um."
        ),
    )
    parser.add_argument(
        "--pore-diameter-type",
        choices=ALLOWED_VALUES["pore_diameter_type"],
        default=DEFAULT_PORE_DIAMETER_TYPE,
        help=f"Pore diameter field for input generation and PNM visualization. Default: {DEFAULT_PORE_DIAMETER_TYPE}.",
    )
    parser.add_argument(
        "--throat-diameter-type",
        choices=ALLOWED_VALUES["throat_diameter_type"],
        default=DEFAULT_THROAT_DIAMETER_TYPE,
        help=f"Throat diameter field for input generation and PNM visualization. Default: {DEFAULT_THROAT_DIAMETER_TYPE}.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    result = TASKS[args.task](args)
    _print_result(result)


if __name__ == "__main__":
    main()

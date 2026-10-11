"""Archive completed AutoPore workflow results."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path


OUTPUT_FOLDERS = {
    ("single_run", "quasi_static"): "single_run_qs_pnm",
    ("single_run", "dynamic"): "single_run_dyn_pnm",
    ("uq_run", "quasi_static"): "uq_qs_pnm",
    ("uq_run", "dynamic"): "uq_dyn_pnm",
}


def _unique_archive_folder(model_results: Path, prefix: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = model_results / f"{prefix}_{stamp}"
    suffix = 2
    while destination.exists():
        destination = model_results / f"{prefix}_{stamp}_{suffix:02d}"
        suffix += 1
    return destination


def archive_completed_cycle(
    project_root: str | Path,
    task_mode: str,
    pnm_mode: str,
) -> dict[str, str]:
    """Copy the active input and output folders into the workspace model_results."""
    project_root = Path(project_root).resolve()
    try:
        output_name = OUTPUT_FOLDERS[(task_mode, pnm_mode)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported result archive mode: {task_mode}/{pnm_mode}"
        ) from exc

    input_source = project_root / "input_pnm"
    output_source = project_root / output_name
    missing = [str(path) for path in (input_source, output_source) if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Cannot archive the completed cycle; missing folder(s): "
            + ", ".join(missing)
        )

    model_results = project_root.parent / "model_results"
    model_results.mkdir(parents=True, exist_ok=True)
    archive_dir = _unique_archive_folder(model_results, output_name)
    archive_dir.mkdir(parents=False)

    input_destination = archive_dir / input_source.name
    output_destination = archive_dir / output_source.name
    try:
        shutil.copytree(input_source, input_destination, copy_function=shutil.copy2)
        shutil.copytree(output_source, output_destination, copy_function=shutil.copy2)
        manifest = {
            "status": "success",
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "task_mode": task_mode,
            "pnm_mode": pnm_mode,
            "archive_dir": str(archive_dir),
            "archived_folders": {
                "input_pnm": str(input_destination),
                "output": str(output_destination),
            },
            "source_folders": {
                "input_pnm": str(input_source),
                "output": str(output_source),
            },
        }
        (archive_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(archive_dir, ignore_errors=True)
        raise

    return {
        "archive_dir": str(archive_dir),
        "input_pnm": str(input_destination),
        "output": str(output_destination),
        "manifest": str(archive_dir / "run_manifest.json"),
    }

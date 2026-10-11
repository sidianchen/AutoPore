"""
AutoPore GUI - Chat interface for the Pore Network Model agent.
===================================================================
A tkinter-based GUI for the Pore Network Model agent, providing a
modern chat-like interface.

Usage:
    python AutoPore.py
"""

import glob
import json
import os
import re
import shutil
import textwrap
import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, scrolledtext, ttk
from pathlib import Path
from .config import ALLOWED_VALUES, MODEL_DEFAULTS, THIS_DIR
from .llm import (
    LLMClient,
    LLMSettings,
    VALID_PROVIDERS,
    load_settings,
    provider_defaults,
    save_settings,
)
from .workflow import WORKFLOW_STEPS, WorkflowTracker

# ---------------------------------------------------------------------------
# Intercept show_image calls to collect figure paths for the GUI viewer
# ---------------------------------------------------------------------------
_shown_figures: list[str] = []
_model_backend = None
_utils_backend = None
_backend_load_lock = threading.Lock()
_original_show_image = None
_figure_callback = None
_workflow_progress_callback = None
Image = None
ImageTk = None


def _load_pillow():
    global Image, ImageTk
    if Image is None or ImageTk is None:
        from PIL import Image as PILImage
        from PIL import ImageTk as PILImageTk

        Image = PILImage
        ImageTk = PILImageTk
    return Image, ImageTk

def _normalize_figure_path(filepath):
    try:
        return str(Path(filepath).resolve())
    except Exception:
        return str(filepath)

def _capture_show_image(filepath):
    """Replace show_image: collect the path instead of opening externally."""
    filepath = _normalize_figure_path(filepath)
    if os.path.isfile(filepath) and filepath not in _shown_figures:
        _shown_figures.append(filepath)
    if (
        _workflow_progress_callback is not None
        and Path(filepath).name == "3D_image.png"
    ):
        try:
            _workflow_progress_callback({
                "event": "input_image_visualized",
                "filepath": filepath,
            })
        except Exception:
            pass
    if _figure_callback is not None:
        _figure_callback(filepath)


def _load_model_backend():
    """Load the scientific stack only when a workflow action needs it."""
    global _model_backend, _utils_backend, _original_show_image

    if _model_backend is not None:
        return _model_backend

    with _backend_load_lock:
        if _model_backend is None:
            from . import model as model_backend
            from . import utils as utils_backend

            _original_show_image = utils_backend.show_image
            utils_backend.show_image = _capture_show_image
            model_backend.show_image = _capture_show_image
            _utils_backend = utils_backend
            _model_backend = model_backend

    return _model_backend


def run_pore_network_model(**kwargs):
    return _load_model_backend().run_pore_network_model(**kwargs)


def run_pore_network_realizations(**kwargs):
    return _load_model_backend().run_pore_network_realizations(**kwargs)


def rerun_pore_network_case(**kwargs):
    return _load_model_backend().rerun_pore_network_case(**kwargs)

# ---------------------------------------------------------------------------
# Default parameter values
# ---------------------------------------------------------------------------
DEFAULTS = {
    "action": "generate_input",
    "img_src": "porespy",
    "pnm_mode": "quasi_static",
    "task_mode": "single_run",
    "quantity": "sat_nw",
    "vtk": "vtk",
}

IMAGE_DEFAULTS = {
    "shape": None,
    "voxel_size": None,
    "image_file": None,
    "voxels": None,
    "coarse_factor": None,
    "porosity": None,
    "blobiness": None,
    "pore_diameter_type": None,
    "throat_diameter_type": None,
}

config = dict(DEFAULTS)
image_config = dict(IMAGE_DEFAULTS)
DEFAULT_REALIZATIONS = 5
uq_config = {"realizations": DEFAULT_REALIZATIONS}
STATE_DIR = THIS_DIR / ".autopore"
WORKFLOW_LOG_FILE = STATE_DIR / "workflow.log.jsonl"
LLM_SETTINGS_FILE = STATE_DIR / "llm_settings.json"
WORKFLOW_TRACKER = WorkflowTracker(
    WORKFLOW_LOG_FILE,
    task_mode=config["task_mode"],
    pnm_mode=config["pnm_mode"],
)

WORKFLOW_TOOL_STEP_VALUES = ["generate_input", "run_simulation", "plot_result", "process_data"]
WORKFLOW_STEP_ALIASES = {
    "generate_image": "generate_input",
    "generate_input": "generate_input",
    "run": "run_simulation",
    "run_simulation": "run_simulation",
    "plot": "plot_result",
    "plot_results": "plot_result",
    "plot_result": "plot_result",
    "process_data": "plot_result",
    "process_result": "plot_result",
    "process_results": "plot_result",
}

# ---------------------------------------------------------------------------
# Restore default simulation parameter files
# ---------------------------------------------------------------------------
QS_PARAM_DEFAULTS = (
    "0\n5e4\n1e2\n1\n"
    "// minimum pressure\n// maximum pressure\n"
    "// pressure step\n// compute permeability\n"
)
DYN_PARAM_DEFAULTS = (
    "1.000000e-03\n1.000000e-01\n1.000000e+00\n1.000000e+02\n"
    "1.000000e-8\n2.000000e+01\n1e-8\n1e5\n1\n1\n"
    "// gas viscosity\n// liquid viscosity\n// flow rate\n"
    "// capillary number\n// scale update for large pressure\n"
    "// output time scaler with maxT\n// output time step size\n// maxPc\n"
    "// outlet condition (0 - zero gas flux, 1 - zero gradient)\n"
    "// shrink throat sizes if larger than connecting bodies\n"
)

_qs_file = THIS_DIR / "input_pnm" / "qs_param.txt"
_dyn_file = THIS_DIR / "input_pnm" / "dyn_param.txt"
_dyn_control_file = THIS_DIR / "input_pnm" / "dyn_SimControl.txt"
DYN_SIM_CONTROL_DEFAULTS = (
    "1e-6\n1e-5\n1e20\n1.5\n1e-6\n20\n1e-4\n"
    "// dt\n// DT_MAX\n// totTime\n// totTime_vs_BT_time\n"
    "// L_eps\n// Max_ITER\n// tol_oci\n"
)


def _ensure_param_file(path: Path, defaults: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(defaults)


def _reset_simulation_param_files() -> None:
    _qs_file.write_text(QS_PARAM_DEFAULTS)
    _dyn_file.write_text(DYN_PARAM_DEFAULTS)
    _dyn_control_file.write_text(DYN_SIM_CONTROL_DEFAULTS)


_ensure_param_file(_qs_file, QS_PARAM_DEFAULTS)
_ensure_param_file(_dyn_file, DYN_PARAM_DEFAULTS)
_ensure_param_file(_dyn_control_file, DYN_SIM_CONTROL_DEFAULTS)

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------
PARAM_PROPS = {
    "action": {
        "type": "string",
        "enum": ALLOWED_VALUES["action"],
        "description": "generate_input | run_simulation | plot_result",
    },
    "img_src": {
        "type": "string",
        "enum": ALLOWED_VALUES["img_src"],
        "description": "porespy synthetic image or x_ray raw image",
    },
    "pnm_mode": {
        "type": "string",
        "enum": ALLOWED_VALUES["pnm_mode"],
        "description": "quasi_static or dynamic",
    },
    "task_mode": {
        "type": "string",
        "enum": ALLOWED_VALUES["task_mode"],
        "description": "single_run or uq_run for uncertainty quantification",
    },
    "quantity": {
        "type": "string",
        "enum": ALLOWED_VALUES["quantity"],
        "description": "quantity for plotting: sat_nw, sat_w, pc, pres_nw, pres_w, or all for every local variable",
    },
    "vtk": {
        "type": "string",
        "enum": ALLOWED_VALUES["vtk"],
        "description": "flow-pattern files: png images/video plus vtk files, or vtk files only",
    },
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "set_parameters",
            "description": (
                "Update one or more customisable parameters. "
                "Only include the parameters the user wants to change; "
                "omitted parameters keep their current values. "
                "This does NOT execute the model — it only stores settings."
            ),
            "parameters": {
                "type": "object",
                "properties": PARAM_PROPS,
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_config",
            "description": (
                "Return the current parameter configuration so the user "
                "can review it before executing."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_workflow",
            "description": (
                "Return the current workflow progress, log file path, completed "
                "steps, and next allowed top-to-bottom actions."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_workflow_over",
            "description": (
                "Start a fresh top-to-bottom workflow cycle while keeping the "
                "current model parameters."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_model",
            "description": (
                "Execute the pore-network model using the current parameter "
                "configuration. Optional main parameters passed here are applied "
                "before execution. Call this only when the user explicitly asks "
                "to run / execute / go / start / plot / process the model."
            ),
            "parameters": {
                "type": "object",
                "properties": PARAM_PROPS,
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_workflow",
            "description": (
                "Execute one, two, or three consecutive workflow steps in the requested order. "
                "Use this when the user asks to generate input, run a simulation, and/or process "
                "results in one prompt. If run_simulation is manually closed and plot_result or "
                "process_data was requested afterward, continue to process the available outputs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": WORKFLOW_TOOL_STEP_VALUES,
                        },
                        "minItems": 1,
                        "maxItems": 3,
                        "description": (
                            "Consecutive workflow steps to execute, e.g. "
                            "['generate_input', 'run_simulation'], "
                            "['run_simulation', 'plot_result'], or all three. "
                            "Use plot_result/process_data for processing simulation data."
                        ),
                    },
                    "pnm_mode": {
                        "type": "string",
                        "enum": ALLOWED_VALUES["pnm_mode"],
                        "description": (
                            "Simulation/results mode for run_simulation and plot_result. "
                            "Omit to keep the current mode."
                        ),
                    },
                },
                "required": ["steps"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_realizations",
            "description": (
                "Run multiple realizations. For each realization, generate a new input, "
                "run one simulation mode, then copy the matching single_run_qs_pnm or "
                "single_run_dyn_pnm folder into uq_qs_pnm/case_### or uq_dyn_pnm/case_###."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "realizations": {
                        "type": "integer",
                        "minimum": 1,
                        "description": (
                            "Number of generate-input plus simulation loops to run. "
                            "Omit it to use the total configured in the workflow tracker."
                        ),
                    },
                    "pnm_mode": {
                        "type": "string",
                        "enum": ALLOWED_VALUES["pnm_mode"],
                        "description": (
                            "Simulation mode to run for every realization. "
                            "Omit to keep the current mode."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rerun_case",
            "description": (
                "Rerun an existing UQ case. This restores the case's saved "
                "input_pnm snapshot first, reruns the selected simulation mode, "
                "then replaces that case folder with the new output."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "case": {
                        "type": "string",
                        "description": "Case number or folder name, e.g. '3' or 'case_003'.",
                    },
                    "pnm_mode": {
                        "type": "string",
                        "enum": ALLOWED_VALUES["pnm_mode"],
                        "description": (
                            "Simulation mode for the UQ case. Omit to keep "
                            "the current mode."
                        ),
                    },
                },
                "required": ["case"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reset_parameters",
            "description": (
                "Reset all customisable parameters back to their default values."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_image_properties",
            "description": (
                "Set image and domain properties. Use when the user specifies "
                "domain size, voxel size, image file name, or image voxel "
                "dimensions. Only include the properties the user provides."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dimensions": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": (
                            "User-facing image dimensions as [nx, ny, nz]. "
                            "This is mode-aware: it updates shape for porespy "
                            "mode and voxels for x_ray mode."
                        ),
                    },
                    "shape": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": (
                            "Shape as [nx, ny, nz] in voxels for "
                            "PoreSpy image generation (default: [200, 200, 350])."
                        ),
                    },
                    "voxel_size": {
                        "type": "number",
                        "description": (
                            "Voxel size / resolution in micrometres "
                            "(default: 2.5 µm)."
                        ),
                    },
                    "image_file": {
                        "type": "string",
                        "description": (
                            "Name of the user's .raw image file for x_ray mode "
                            "(e.g. 'my_rock.raw'). File should be in "
                            "pore_network_model/image/ folder."
                        ),
                    },
                    "voxels": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": (
                            "Image dimensions as [nx, ny, nz] for x_ray mode "
                            "(must match the .raw file, e.g. [252, 252, 252])."
                        ),
                    },
                    "coarse_factor": {
                        "type": "integer",
                        "description": (
                            "Image coarsening: number of voxels to merge "
                            "in each direction. E.g. 2 means every 2x2x2 "
                            "block becomes 1 voxel (default: 1)."
                        ),
                    },
                    "porosity": {
                        "type": "number",
                        "description": (
                            "Porosity of the generated image, between 0 and 1 "
                            "(default: 0.5). Only used for PoreSpy image generation."
                        ),
                    },
                    "blobiness": {
                        "type": "number",
                        "description": (
                            "PoreSpy blobiness for generated synthetic images "
                            "(default: 2). Larger values make smoother/larger blobs."
                        ),
                    },
                    "pore_diameter_type": {
                        "type": "string",
                        "enum": ALLOWED_VALUES["pore_diameter_type"],
                        "description": (
                            "Pore diameter field used for PNM input generation "
                            "and PNM visualization sizing."
                        ),
                    },
                    "throat_diameter_type": {
                        "type": "string",
                        "enum": ALLOWED_VALUES["throat_diameter_type"],
                        "description": (
                            "Throat diameter field used for PNM input generation "
                            "and PNM visualization sizing."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_qs_params",
            "description": (
                "Set the quasi-static simulation pressure parameters. "
                "These are written to qs_param.txt: minimum pressure (Pa), "
                "maximum pressure (Pa), and pressure step (Pa). "
                "Only include the values the user provides."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "min_pressure": {
                        "type": "number",
                        "description": (
                            "Minimum pressure in Pa (default: 0)."
                        ),
                    },
                    "max_pressure": {
                        "type": "number",
                        "description": (
                            "Maximum pressure in Pa (default: 5e4)."
                        ),
                    },
                    "pressure_step": {
                        "type": "number",
                        "description": (
                            "Pressure step in Pa (default: 1e2)."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_dyn_params",
            "description": (
                "Set dynamic simulation parameters. "
                "These are written to dyn_param.txt: gas viscosity, "
                "liquid viscosity, capillary number, and max capillary "
                "pressure (maxPc). Only include the values the user provides."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gas_viscosity": {
                        "type": "number",
                        "description": (
                            "Gas viscosity in Pa·s (default: 1e-3)."
                        ),
                    },
                    "liquid_viscosity": {
                        "type": "number",
                        "description": (
                            "Liquid viscosity in Pa·s (default: 1e-1)."
                        ),
                    },
                    "capillary_number": {
                        "type": "number",
                        "description": (
                            "Capillary number (default: 100)."
                        ),
                    },
                    "max_pc": {
                        "type": "number",
                        "description": (
                            "Maximum capillary pressure allowed in the "
                            "driest pores, in Pa (default: 1e5)."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
]

def _compact_tool_schema(tools):
    descriptions = {
        "set_parameters": "Update main settings.",
        "show_config": "Show current settings.",
        "show_workflow": "Show workflow progress.",
        "start_workflow_over": "Start a fresh workflow cycle.",
        "execute_model": "Run current action once.",
        "execute_workflow": "Run consecutive workflow steps.",
        "run_realizations": "Run N generate+simulate realizations.",
        "rerun_case": "Rerun an existing UQ case.",
        "reset_parameters": "Reset settings.",
        "set_image_properties": "Set image/domain settings.",
        "set_qs_params": "Set quasi-static pressure settings.",
        "set_dyn_params": "Set dynamic simulation settings.",
    }

    def strip_descriptions(obj):
        if isinstance(obj, dict):
            obj.pop("description", None)
            for value in obj.values():
                strip_descriptions(value)
        elif isinstance(obj, list):
            for value in obj:
                strip_descriptions(value)

    for tool in tools:
        fn = tool.get("function", {})
        name = fn.get("name", "")
        strip_descriptions(fn.get("parameters", {}))
        fn["description"] = descriptions.get(name, name)


_compact_tool_schema(TOOLS)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert assistant for a Pore Network Model (PNM) simulation tool.

The user will configure the model step-by-step through natural language and
then ask you to execute it.  You have nine tools:

1. **set_parameters** – update one or more of the 5 customisable parameters.
   Call this whenever the user mentions changing a setting.  Only include
   the parameters the user wants to change.
2. **show_config** – display the current configuration.  Call when the user
   asks to see / review / show the current settings.
3. **execute_model** – run the model with the current configuration.  Call
   ONLY when the user explicitly says "run", "execute", "go", "start",
   "do it", or similar.
4. **execute_workflow** – run one or more consecutive workflow steps:
   generate_input, run_simulation, and plot_result/process_data. Use this
   when the user asks for two or more steps in the same prompt, or asks for
   a full workflow. If the simulation is manually closed and a processing
   step was requested afterward, continue processing the available data.
5. **run_realizations** – run multiple realizations. For each realization,
   generate a new input, run one quasi-static or dynamic simulation, then copy
   single_run_qs_pnm or single_run_dyn_pnm into uq_qs_pnm/case_### or uq_dyn_pnm/case_###.
6. **reset_parameters** – reset everything to defaults.
7. **set_image_properties** – set domain/image properties:
   - dimensions: [nx, ny, nz] voxels; mode-aware, updates shape for porespy and voxels for x_ray
   - shape: [nx, ny, nz] voxels for generated image (default: [200, 200, 350])
   - voxel_size: resolution in µm (default: 2.5)
   - coarse_factor: number of voxels to merge in each direction (default: 1)
   - porosity: porosity of generated image, 0–1 (default: 0.5)
   - blobiness: PoreSpy blobiness for generated image (default: 2)
   - image_file: .raw file name for x_ray mode
   - voxels: [nx, ny, nz] dimensions of the .raw file for x_ray mode (default: [252, 252, 252])
   - pore_diameter_type: inscribed | equivalent | extended
   - throat_diameter_type: inscribed | equivalent
8. **set_qs_params** – set quasi-static simulation pressure parameters:
   - min_pressure: minimum pressure in Pa (default: 0)
   - max_pressure: maximum pressure in Pa (default: 5e4)
   - pressure_step: pressure step in Pa (default: 1e2)
   These values are written to the qs_param.txt file before running
   a quasi-static simulation.
9. **set_dyn_params** – set dynamic simulation parameters:
   - gas_viscosity: gas viscosity in Pa·s (default: 1e-3)
   - liquid_viscosity: liquid viscosity in Pa·s (default: 1e-1)
   - capillary_number: capillary number (default: 100)
   - max_pc: max capillary pressure in driest pores, Pa (default: 1e5)
   These values are written to the dyn_param.txt file before running
   a dynamic simulation.

Customisable parameters and allowed values:
  action   : generate_input | run_simulation | plot_result  (default: generate_input)
  img_src  : porespy | x_ray                  (default: porespy)
  pnm_mode : quasi_static | dynamic           (default: quasi_static)
  task_mode: single_run | uq_run              (default: single_run)
  quantity : sat_nw | sat_w | pc | pres_nw | pres_w | all  (default: sat_nw)
  vtk      : png | vtk                        (default: vtk)

IMPORTANT — when the user wants to run a simulation or plot results:
  ALWAYS ask the user to confirm:
    "Would you like to proceed directly using the existing PNM files,
     or would you like to create/provide an image first?"

  If the user chooses to proceed directly (use existing files):
    IMMEDIATELY call set_parameters to set action = run_simulation (or
    plot_result) and pnm_mode = quasi_static (or dynamic) as soon as
    the user confirms.  Do NOT wait — update the config right away.
    If pnm_mode = quasi_static and action = run_simulation, ask:
      "Do you want to set the pressure range for the simulation?
       Current defaults: min Pressure = 0 Pa, max Pressure = 5e4 Pa, Pressure step = 1e2 Pa."
      If the user provides values, call set_qs_params, then call execute_model.
      If the user is happy with defaults, call execute_model immediately.
    If pnm_mode = dynamic and action = run_simulation, ask:
      "Do you want to set the dynamic simulation parameters?
       Current defaults: gas viscosity = 1e-3 Pa·s, liquid viscosity = 1e-1 Pa·s,
       capillary number = 100, maxPc = 1e5 Pa."
      If the user provides values, call set_dyn_params, then call execute_model.
      If the user is happy with defaults, call execute_model immediately.

  If the user chooses to create/provide an image first, ask:
    "Would you like to generate a synthetic image or use your own image?"

    Option A — Generate a synthetic image (porespy):
      IMMEDIATELY call set_parameters to set action = generate_input,
      img_src = porespy.
      Ask if they want custom shape [nx, ny, nz], voxel size (µm),
      porosity (0–1, default: 0.5), blobiness (default: 2),
      and coarse factor (number of voxels to merge, default: 1),
      or keep the defaults ([200, 200, 350], 2.5 µm, porosity=0.5, blobiness=2, coarse_factor=1).
      If custom, use set_image_properties to store dimensions, voxel_size,
      porosity, blobiness, and/or coarse_factor.

    Option B — Use their own image (x_ray):
      IMMEDIATELY call set_parameters to set action = generate_input,
      img_src = x_ray.
      Tell the user to place their .raw image in the pore_network_model/image/ folder.
      Ask for:
         - image file name (e.g. 'my_rock.raw')
         - image dimensions [nx, ny, nz] in voxels
         - voxel size in µm
         - coarse factor (number of voxels to merge)
      Use set_image_properties to store image_file, dimensions, voxel_size,
      and coarse_factor.

IMPORTANT — when the user asks to plot results:
  Directly ask: "Would you like to plot quasi-static or dynamic results?"
  Once the user confirms, IMMEDIATELY call set_parameters to set
  action = plot_result and pnm_mode accordingly, then call execute_model.
  No need to ask about image source or existing files for plotting.

  Do NOT execute until the user has confirmed their choice and provided
  the required information (or accepted defaults).

Rules:
- Whenever the user confirms a choice or provides configuration values,
  IMMEDIATELY call the appropriate tool (set_parameters, set_image_properties,
  set_qs_params, or set_dyn_params) to update the config. Never just
  acknowledge — always call the tool to make the change.
- When the user sets parameters, call set_parameters with ONLY the changed
  values.  Then confirm what was updated and show the full current config.
- When the user says "run" / "execute" / "go" (with or without parameter
  changes in the same sentence), first call set_parameters if any parameters
  are being changed, then call execute_model.
- Never execute without the user's explicit instruction.
- If the request is ambiguous, ask the user to clarify rather than guessing.
- If the user asks something unrelated, answer helpfully without calling tools.
"""

SYSTEM_PROMPT += """

Additional clarification for image/run flow:
- If the user has just configured or chosen to generate/provide an image, do NOT immediately ask:
  "Would you like to proceed directly using the existing PNM files, or would you like to create/provide an image first?"
- After image configuration, continue naturally with the image-generation flow instead of asking that question.
- Only ask a follow-up about existing PNM files when an image or PNM input files already exist from an earlier step, or when the user is clearly referring to existing files.
- In that case, ask exactly:
  "Would you like to run simulations using the existing PNM files, or would you like to recreate/provide an image first?"
- The user may give multiple instructions in one prompt. You should handle all requested steps in a logical order instead of addressing only one part.
- If the user explicitly asks for a combined workflow such as "generate input, then run simulation, then plot results", first extract and store all requested settings with tools, then call execute_workflow with the requested consecutive steps.
- Do not ask for a second confirmation when the user's prompt already clearly says to run, execute, do, generate, process, or plot the requested steps.
- If the user is only discussing a possible workflow and has not asked you to execute it, summarize the sequence and ask before calling execute_workflow.
- Use execute_workflow for valid consecutive sequences: generate_input; run_simulation; plot_result/process_data; generate_input -> run_simulation; run_simulation -> plot_result; generate_input -> run_simulation -> plot_result.
- Treat "process data", "process results", "analyze results", and "plot results" as the plot_result step.
- If execute_workflow runs a simulation that is manually closed or terminated and a plot_result/process_data step was requested afterward, it will continue and process whatever output files are available.
- After the user confirms a previously proposed sequence, carry it out with execute_workflow without asking them to restate the same instructions.
- If the user asks to run N realizations, N loops, N cases, or multiple simulations,
  set any requested parameters first, then call run_realizations with realizations = N.
- If the user asks for UQ/realizations but does not specify a count, omit the count so the workflow tracker setting is used.
- run_realizations already performs generate_input followed by run_simulation for each case.
  Do not call execute_workflow for the same loops.
- If the user does not specify quasi-static or dynamic for realizations, use the current pnm_mode.
- Use only short-lived workflow memory for the current closed loop. The only state you should rely on across turns is whether an input has been generated in this session.
- If input has not been generated in this session and the user asks to run a simulation or plot results without clearly choosing to use existing files, remind them that they need to generate or provide an image first.
- If input has already been generated in this session, the user may request a single task or a combined sequence of tasks.
- If a step generates files or images, briefly remind the user what was generated and where it is located. Summarize similar files together instead of listing long repetitive sequences.
- If the user has just finished configuring image defaults or image properties, and they have not yet explicitly asked to execute generation, clearly tell them that the image has only been configured, not generated yet, and ask:
  "Do you want me to generate the input now?"
"""

SYSTEM_PROMPT = """\
You control a Pore Network Model tool via function calls.

Use tools, not plain text, when changing settings or executing.
- set_parameters: action/img_src/pnm_mode/task_mode/quantity/vtk.
- show_workflow: inspect what has been done, the log file path, and the next allowed workflow actions.
- start_workflow_over: start a fresh workflow cycle while keeping current model parameters.
- set_image_properties: dimensions, shape, voxel_size, image_file, voxels, coarse_factor, porosity, blobiness, pore_diameter_type, throat_diameter_type.
- When the user asks to change image dimensions, use dimensions. It updates shape in porespy mode and voxels in x_ray mode.
- Diameter type settings control both PNM input-file diameters and PNM visualization sizes.
- Diameter aliases: Eq/equiv/equivalent -> equivalent, In/ins/inscribed -> inscribed, Ext/extended -> extended. Use pore_diameter_type for pore bodies and throat_diameter_type for throats/pore throats.
- set_qs_params or set_dyn_params for simulation parameter files.
- execute_model for one requested action.
- execute_workflow for consecutive requested steps, e.g. generate_input -> run_simulation -> plot_result.
- run_realizations when the user asks for N realizations/cases/loops; it already generates input and runs one simulation per case. This is displayed as UQ action "generate_input & run_simulation".
- rerun_case when the user asks to rerun an existing UQ case; it restores that case's input_pnm snapshot before rerunning.
- Treat task_mode and pnm_mode as standalone model settings. Set them before choosing task-specific parameters.
- Single-run main parameters: img_src, action, quantity, vtk (shown as Flow pattern files).
- UQ uses img_src=porespy only. Do not use x_ray for UQ.
- UQ action is either "generate_input & run_simulation" or "plot_result". Use run_realizations for "generate_input & run_simulation"; do not pass that display label as action.
- UQ-run inputs: realizations for run_realizations; omit it to use the workflow tracker total. For plotting UQ results use action=plot_result and task_mode=uq_run.
- quantity and vtk are not required for UQ plots because all UQ summary figures are generated together.
- Treat process/analyze/plot results as plot_result.
- Treat uncertainty quantification, uncertain quantification, and UQ as task_mode=uq_run.
- When the user asks to plot UQ results, set action=plot_result and task_mode=uq_run, set pnm_mode if specified, then execute_model.
- When the user asks to run realizations and plot UQ results in one request, call run_realizations first, then plot_result with task_mode=uq_run.
- When the user asks for ordinary/current single-run results, use task_mode=single_run.
- If execute_model or execute_workflow returns status=needs_simulation, tell the user the requested plot files are not ready and ask them to run the recommended simulation step first.
- If execute_model, execute_workflow, run_realizations, or rerun_case returns status=workflow_blocked, do not call another execution tool. Tell the user nothing was run and repeat the returned message explaining which step sequence is needed before the requested step.
- Workflow order inside one cycle is generate_input -> run_simulation -> plot_result. After plot_result closes a cycle, the next request may start over from generate_input or start from the middle using existing files.
- When the user says start over, first call start_workflow_over. If they also ask to run/generate/plot after starting over, continue with the requested next tool call.
- Use show_workflow when the user asks what has been done, what can be done next, or where the workflow log is.

Do not execute unless the user clearly asks to run/generate/process/plot/do it.
If a mode is omitted, use current pnm_mode. If needed information is missing, ask briefly.
"""


DIAMETER_TYPE_ALIASES = {
    "eq": "equivalent",
    "equi": "equivalent",
    "equiv": "equivalent",
    "equivalent": "equivalent",
    "equivelant": "equivalent",
    "equilivalent": "equivalent",
    "in": "inscribed",
    "ins": "inscribed",
    "inscribed": "inscribed",
    "ext": "extended",
    "extended": "extended",
}


_NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _normalize_diameter_type(value):
    if not isinstance(value, str):
        return value
    key = re.sub(r"[^a-z0-9]+", "", value.strip().lower())
    return DIAMETER_TYPE_ALIASES.get(key, value)


def _number_after_any_pattern(text: str, patterns: tuple[str, ...]) -> float | None:
    for pattern in patterns:
        match = re.search(
            rf"{pattern}\s*(?:viscosity|mu)?\s*(?:to|as|=|:)?\s*({_NUMBER_RE})",
            text,
        )
        if match:
            return float(match.group(1))
    return None


def _parse_direct_dyn_param_setting(text: str) -> dict | None:
    lowered = text.strip().lower()
    if not re.search(r"\b(change|set|switch|use|make|update)\b", lowered):
        return None

    updates = {}
    gas_patterns = (
        r"\bgas(?:\s+phase)?",
        r"\bnon[-\s]?wetting(?:[-\s]?phase)?",
        r"\bnw\b",
        r"\bviscosity\s+(?:of|for)\s+(?:gas|non[-\s]?wetting(?:[-\s]?phase)?|nw)",
    )
    liquid_patterns = (
        r"\bliquid(?:\s+phase)?",
        r"\bwater",
        r"\bwetting(?:[-\s]?phase)?",
        r"\bviscosity\s+(?:of|for)\s+(?:liquid|water|wetting(?:[-\s]?phase)?)",
    )

    if "viscos" in lowered:
        gas_value = _number_after_any_pattern(lowered, gas_patterns)
        liquid_value = _number_after_any_pattern(lowered, liquid_patterns)
        if gas_value is not None:
            updates["gas_viscosity"] = gas_value
        if liquid_value is not None:
            updates["liquid_viscosity"] = liquid_value

        if not updates:
            numbers = [float(value) for value in re.findall(_NUMBER_RE, lowered)]
            set_both = re.search(r"\b(both|all|viscosities)\b", lowered) is not None
            mentions_gas = re.search(r"\b(gas|non[-\s]?wetting|nw)\b", lowered) is not None
            mentions_liquid = re.search(r"\b(liquid|water|wetting)\b", lowered) is not None
            if set_both and len(numbers) == 1:
                updates["gas_viscosity"] = numbers[0]
                updates["liquid_viscosity"] = numbers[0]
            elif mentions_gas and mentions_liquid and len(numbers) >= 2:
                updates["gas_viscosity"] = numbers[0]
                updates["liquid_viscosity"] = numbers[1]

    capillary_match = re.search(
        rf"\bcapillary\s+number\b\s*(?:to|as|=|:)?\s*({_NUMBER_RE})",
        lowered,
    )
    if capillary_match:
        updates["capillary_number"] = float(capillary_match.group(1))

    max_pc_match = re.search(
        rf"\b(?:max\s*pc|maxpc|maximum\s+capillary\s+pressure)\b\s*(?:to|as|=|:)?\s*({_NUMBER_RE})",
        lowered,
    )
    if max_pc_match:
        updates["max_pc"] = float(max_pc_match.group(1))

    return updates or None


def _parse_direct_diameter_setting(text: str) -> dict | None:
    lowered = text.strip().lower()
    if "diameter" not in lowered:
        return None
    if not re.search(r"\b(change|set|switch|use|make|update)\b", lowered):
        return None

    value_match = re.search(
        r"(?:\bto\b|\bas\b|=|:)\s*"
        r"(equivalent|equiv|equi|eq|equivelant|equilivalent|"
        r"inscribed|ins|in|extended|ext)\b",
        lowered,
    )
    if not value_match:
        return None

    diameter_type = _normalize_diameter_type(value_match.group(1))
    pore_throat_phrase = re.search(r"\bpore\s+throat\b", lowered) is not None
    both_requested = re.search(
        r"\b(pore\s*(and|&|/)\s*throat|throat\s*(and|&|/)\s*pore)\b",
        lowered,
    ) is not None
    mentions_pore = re.search(r"\bpore\b", lowered) is not None and not pore_throat_phrase
    mentions_throat = re.search(r"\bthroat\b", lowered) is not None

    updates = {}
    if both_requested or (not mentions_pore and not mentions_throat):
        updates["pore_diameter_type"] = diameter_type
        updates["throat_diameter_type"] = diameter_type
    else:
        if mentions_pore:
            updates["pore_diameter_type"] = diameter_type
        if mentions_throat:
            updates["throat_diameter_type"] = diameter_type
    return updates or None


def _build_run_kwargs(action: str | None = None, pnm_mode: str | None = None) -> dict:
    run_kwargs = dict(config)
    if action is not None:
        run_kwargs["action"] = action
    if pnm_mode is not None:
        run_kwargs["pnm_mode"] = pnm_mode
    for key, value in image_config.items():
        if value is not None:
            run_kwargs["domain_size" if key == "shape" else key] = value
    return run_kwargs


def _sync_workflow_tracker() -> None:
    WORKFLOW_TRACKER.set_modes(config["task_mode"], config["pnm_mode"])


def _workflow_payload() -> dict:
    return {
        "workflow": WORKFLOW_TRACKER.snapshot(),
        "workflow_status": WORKFLOW_TRACKER.format_status(),
    }


def _notify_workflow_progress(event: str = "workflow_state_changed", **payload) -> None:
    if _workflow_progress_callback is None:
        return
    try:
        _workflow_progress_callback({"event": event, **payload})
    except Exception:
        pass


def _normalize_workflow_steps(steps) -> list[str]:
    if isinstance(steps, str):
        steps = [steps]
    if not isinstance(steps, list) or not steps:
        raise ValueError("execute_workflow requires at least one workflow step.")

    normalized = []
    for raw_step in steps:
        step = WORKFLOW_STEP_ALIASES.get(str(raw_step).strip(), str(raw_step).strip())
        if step not in WORKFLOW_STEPS:
            raise ValueError(
                f"Invalid workflow step '{raw_step}'. Allowed: {WORKFLOW_TOOL_STEP_VALUES}"
            )
        normalized.append(step)

    indexes = [WORKFLOW_STEPS.index(step) for step in normalized]
    expected = list(range(indexes[0], indexes[0] + len(indexes)))
    if indexes != expected or len(set(indexes)) != len(indexes):
        allowed = (
            "generate_input; run_simulation; plot_result; "
            "generate_input -> run_simulation; run_simulation -> plot_result; "
            "generate_input -> run_simulation -> plot_result"
        )
        raise ValueError(f"Workflow steps must be consecutive. Allowed sequences: {allowed}.")

    return normalized


def _workflow_input_source_node() -> str:
    return (
        "image_generation"
        if config.get("task_mode") == "uq_run" or config.get("img_src") == "porespy"
        else "image_import"
    )


def _workflow_nodes_for_step(step: str) -> list[str]:
    if step == "generate_input":
        source_node = _workflow_input_source_node()
        return [source_node, "pore_extraction"]
    if step == "run_simulation":
        return ["dynamic" if config.get("pnm_mode") == "dynamic" else "quasi_static"]
    if step == "plot_result":
        return ["plot_result"]
    return [step]


def _workflow_active_nodes_for_step(step: str) -> list[str]:
    if step == "generate_input":
        return [_workflow_input_source_node()]
    return _workflow_nodes_for_step(step)


def _workflow_upstream_nodes_for_step(step: str) -> list[str]:
    if step not in WORKFLOW_STEPS:
        return []
    nodes = []
    for upstream_step in WORKFLOW_STEPS[:WORKFLOW_STEPS.index(step)]:
        nodes.extend(_workflow_nodes_for_step(upstream_step))
    return nodes


def _execute_model_action(action: str, pnm_mode: str | None = None) -> dict:
    if action not in ALLOWED_VALUES["action"]:
        raise ValueError(f"Invalid action='{action}'. Allowed: {ALLOWED_VALUES['action']}")
    if pnm_mode is not None and pnm_mode not in ALLOWED_VALUES["pnm_mode"]:
        raise ValueError(f"Invalid pnm_mode='{pnm_mode}'. Allowed: {ALLOWED_VALUES['pnm_mode']}")

    config["action"] = action
    if pnm_mode is not None:
        config["pnm_mode"] = pnm_mode
    _sync_workflow_tracker()

    action_nodes = _workflow_nodes_for_step(action)
    active_nodes = _workflow_active_nodes_for_step(action)
    allowed, message = WORKFLOW_TRACKER.prepare_step(
        action,
        source="execute_model",
        active_nodes=active_nodes,
        upstream_nodes=_workflow_upstream_nodes_for_step(action),
    )
    if not allowed:
        return {
            "step": action,
            "status": "workflow_blocked",
            "message": message,
            "executed_with": _build_run_kwargs(action=action, pnm_mode=pnm_mode),
            **_workflow_payload(),
        }
    _notify_workflow_progress("step_started", step=action)

    run_kwargs = _build_run_kwargs(action=action, pnm_mode=pnm_mode)
    result = run_pore_network_model(**run_kwargs)

    if isinstance(result, dict) and result.get("status") == "manually_closed":
        message = result.get("message", "Simulation window was closed manually.")
        WORKFLOW_TRACKER.complete_step(
            action,
            "manually_closed",
            message=message,
            source="execute_model",
            completed_nodes=action_nodes,
        )
        _notify_workflow_progress("step_finished", step=action, status="manually_closed")
        return {
            "step": action,
            "status": "manually_closed",
            "message": message,
            "executed_with": run_kwargs,
            **_workflow_payload(),
        }

    if isinstance(result, dict) and result.get("status") == "needs_simulation":
        message = result.get(
            "message",
            "The required simulation output files are missing. Please run the simulation first.",
        )
        WORKFLOW_TRACKER.complete_step(
            action,
            "needs_simulation",
            message=message,
            source="execute_model",
        )
        _notify_workflow_progress("step_finished", step=action, status="needs_simulation")
        return {
            "step": action,
            "status": "needs_simulation",
            "message": message,
            "needs_simulation": True,
            "recommended_action": result.get("recommended_action", "run_simulation"),
            "missing_files": result.get("missing_files", []),
            "executed_with": run_kwargs,
            **_workflow_payload(),
        }

    WORKFLOW_TRACKER.complete_step(
        action,
        "success",
        source="execute_model",
        completed_nodes=action_nodes,
    )
    _notify_workflow_progress("step_finished", step=action, status="success")
    response = {
        "step": action,
        "status": "success",
        "executed_with": run_kwargs,
        **_workflow_payload(),
    }
    if isinstance(result, dict) and result.get("model_results_archive"):
        response["model_results_archive"] = result["model_results_archive"]
    return response


def _execute_workflow(steps, pnm_mode: str | None = None) -> dict:
    try:
        normalized_steps = _normalize_workflow_steps(steps)
    except ValueError as exc:
        message = f"I did not run anything. {exc}"
        WORKFLOW_TRACKER.note("workflow_blocked", reason=message)
        return {
            "status": "workflow_blocked",
            "message": message,
            "steps": [],
            **_workflow_payload(),
        }

    results = []
    saw_manual_close = False

    for index, step in enumerate(normalized_steps):
        try:
            step_result = _execute_model_action(step, pnm_mode=pnm_mode)
        except Exception as exc:
            return {
                "status": "error",
                "message": f"{step} failed: {exc}",
                "steps": results,
            }

        results.append(step_result)

        if step_result["status"] == "workflow_blocked":
            return {
                "status": "workflow_blocked",
                "message": step_result["message"],
                "steps": results,
                **_workflow_payload(),
            }

        if step_result["status"] == "needs_simulation":
            return {
                "status": "needs_simulation",
                "message": step_result["message"],
                "needs_simulation": True,
                "recommended_action": step_result.get("recommended_action", "run_simulation"),
                "missing_files": step_result.get("missing_files", []),
                "steps": results,
                **_workflow_payload(),
            }

        if step_result["status"] == "manually_closed":
            saw_manual_close = True
            remaining_steps = normalized_steps[index + 1:]
            if "plot_result" not in remaining_steps:
                return {
                    "status": "manually_closed",
                    "message": step_result["message"],
                    "steps": results,
                    **_workflow_payload(),
                }

    message = "Workflow executed successfully."
    if saw_manual_close:
        message = (
            "Workflow continued after the simulation window was manually closed; "
            "processing used the available output files."
        )

    response = {
        "status": "success",
        "message": message,
        "steps": results,
        **_workflow_payload(),
    }
    for step_result in reversed(results):
        if step_result.get("model_results_archive"):
            response["model_results_archive"] = step_result["model_results_archive"]
            break
    return response


def _run_realizations(realizations, pnm_mode: str | None = None) -> dict:
    if realizations is None:
        realizations = uq_config["realizations"]
    try:
        realizations = int(realizations)
    except (TypeError, ValueError) as exc:
        raise ValueError("Realizations must be an integer greater than zero.") from exc
    if realizations < 1:
        raise ValueError("Realizations must be an integer greater than zero.")
    uq_config["realizations"] = realizations
    if pnm_mode is None:
        pnm_mode = config["pnm_mode"]

    config["pnm_mode"] = pnm_mode
    config["task_mode"] = "uq_run"
    config["img_src"] = "porespy"
    config["action"] = "generate_input"
    _sync_workflow_tracker()
    WORKFLOW_TRACKER.set_uq_progress(0, realizations)
    generate_nodes = _workflow_nodes_for_step("generate_input")
    generate_active_nodes = _workflow_active_nodes_for_step("generate_input")
    simulation_nodes = _workflow_nodes_for_step("run_simulation")
    allowed, message = WORKFLOW_TRACKER.prepare_step(
        "generate_input",
        source="run_realizations",
        active_nodes=generate_active_nodes,
        upstream_nodes=[],
    )
    if not allowed:
        return {
            "status": "workflow_blocked",
            "message": message,
            **_workflow_payload(),
        }
    _notify_workflow_progress("step_started", step="generate_input")
    run_kwargs = _build_run_kwargs(action=None, pnm_mode=pnm_mode)
    run_kwargs.pop("action", None)
    run_kwargs["img_src"] = "porespy"
    if _workflow_progress_callback is not None:
        run_kwargs["progress_callback"] = _workflow_progress_callback
    result = run_pore_network_realizations(realizations=realizations, **run_kwargs)
    if isinstance(result, dict):
        completed_realizations = len(result.get("cases", []))
        if completed_realizations:
            WORKFLOW_TRACKER.set_uq_progress(completed_realizations, realizations)
    if isinstance(result, dict) and result.get("cases"):
        status = result.get("status", "success")
        WORKFLOW_TRACKER.complete_step(
            "generate_input",
            status,
            message=result.get("message"),
            source="run_realizations",
            completed_nodes=generate_nodes,
        )
        WORKFLOW_TRACKER.complete_step(
            "run_simulation",
            status,
            message=result.get("message"),
            source="run_realizations",
            completed_nodes=simulation_nodes,
        )
        result.update(_workflow_payload())
    elif isinstance(result, dict):
        WORKFLOW_TRACKER.complete_step(
            "generate_input",
            result.get("status", "error"),
            message=result.get("message"),
            source="run_realizations",
            completed_nodes=generate_nodes,
        )
        result.update(_workflow_payload())
    return result


def _rerun_case(case, pnm_mode: str | None = None) -> dict:
    if pnm_mode is None:
        pnm_mode = config["pnm_mode"]

    config["pnm_mode"] = pnm_mode
    config["task_mode"] = "uq_run"
    _sync_workflow_tracker()
    simulation_nodes = _workflow_nodes_for_step("run_simulation")
    allowed, message = WORKFLOW_TRACKER.prepare_step(
        "run_simulation",
        source="rerun_case",
        active_nodes=simulation_nodes,
        upstream_nodes=_workflow_upstream_nodes_for_step("run_simulation"),
    )
    if not allowed:
        return {
            "status": "workflow_blocked",
            "message": message,
            **_workflow_payload(),
        }
    _notify_workflow_progress("step_started", step="run_simulation")
    result = rerun_pore_network_case(
        case=case,
        pnm_mode=pnm_mode,
        task_mode="uq_run",
        quantity=config["quantity"],
        vtk=config["vtk"],
    )
    if isinstance(result, dict):
        status = result.get("status", "success")
        WORKFLOW_TRACKER.complete_step(
            "run_simulation",
            status,
            message=result.get("message"),
            source="rerun_case",
            completed_nodes=simulation_nodes,
        )
        result.update(_workflow_payload())
    return result


def _apply_main_parameter_arguments(arguments: dict) -> tuple[dict, str | None]:
    updated = {}
    for key, value in arguments.items():
        if key not in ALLOWED_VALUES:
            continue
        if value not in ALLOWED_VALUES[key]:
            return updated, f"Invalid {key}='{value}'. Allowed: {ALLOWED_VALUES[key]}"
        config[key] = value
        updated[key] = value
    if (
        config.get("quantity") == "all"
        and config.get("vtk") != "png"
        and "vtk" not in arguments
    ):
        config["vtk"] = "png"
        updated["vtk"] = "png"
    if config.get("task_mode") == "uq_run" and config.get("img_src") != "porespy":
        config["img_src"] = "porespy"
        updated["img_src"] = "porespy"
    return updated, None


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------
def dispatch_tool_call(name: str, arguments: dict) -> str:
    if name == "set_parameters":
        updated, error_message = _apply_main_parameter_arguments(arguments)
        if error_message:
            return json.dumps({
                "status": "error",
                "message": error_message,
            })
        result = {
            "status": "ok",
            "updated": updated,
            "current_config": dict(config),
        }
        _sync_workflow_tracker()
        if updated:
            WORKFLOW_TRACKER.note("parameters_updated", updated=updated, config=dict(config))
        result.update(_workflow_payload())
        if updated.get("img_src") == "x_ray":
            image_folder = str(THIS_DIR / "image")
            result["note"] = (
                f"Using x_ray mode requires a .raw image file. "
                f"Please place your image in: {image_folder}  "
                f"(expected file: {MODEL_DEFAULTS['file']}, "
                f"dimensions: {MODEL_DEFAULTS['voxels']})"
            )
        return json.dumps(result)

    if name == "show_config":
        return json.dumps({"current_config": dict(config), **_workflow_payload()})

    if name == "show_workflow":
        _sync_workflow_tracker()
        return json.dumps({"status": "ok", **_workflow_payload()})

    if name == "start_workflow_over":
        _sync_workflow_tracker()
        WORKFLOW_TRACKER.reset("Started a fresh workflow cycle while keeping model parameters.")
        return json.dumps({
            "status": "ok",
            "message": "Workflow cycle restarted. Model parameters were kept.",
            **_workflow_payload(),
        })

    if name == "execute_model":
        try:
            updated, error_message = _apply_main_parameter_arguments(arguments)
            if error_message:
                return json.dumps({
                    "status": "error",
                    "message": error_message,
                })
            result = _execute_model_action(config["action"], pnm_mode=config["pnm_mode"])
            if result.get("status") == "manually_closed":
                return json.dumps({
                    "status": "manually_closed",
                    "message": result.get("message", "Simulation window was closed manually."),
                    "updated": updated,
                    "executed_with": result.get("executed_with", {}),
                    **_workflow_payload(),
                })
            if result.get("status") == "needs_simulation":
                return json.dumps({
                    "status": "needs_simulation",
                    "message": result.get(
                        "message",
                        "The required simulation output files are missing. Please run the simulation first.",
                    ),
                    "needs_simulation": True,
                    "recommended_action": result.get("recommended_action", "run_simulation"),
                    "missing_files": result.get("missing_files", []),
                    "updated": updated,
                    "executed_with": result.get("executed_with", {}),
                    **_workflow_payload(),
                })
            if result.get("status") == "workflow_blocked":
                return json.dumps({
                    "status": "workflow_blocked",
                    "message": result.get("message", "Workflow order blocked this action."),
                    "updated": updated,
                    "executed_with": result.get("executed_with", {}),
                    **_workflow_payload(),
                })
            response = {
                "status": "success",
                "updated": updated,
                "executed_with": result["executed_with"],
                **_workflow_payload(),
            }
            if result.get("model_results_archive"):
                response["model_results_archive"] = result["model_results_archive"]
            return json.dumps(response)
        except Exception as exc:
            return json.dumps({"status": "error", "message": str(exc)})

    if name == "execute_workflow":
        try:
            result = _execute_workflow(
                arguments.get("steps", []),
                pnm_mode=arguments.get("pnm_mode"),
            )
            return json.dumps(result)
        except Exception as exc:
            return json.dumps({"status": "error", "message": str(exc)})

    if name == "run_realizations":
        try:
            result = _run_realizations(
                arguments.get("realizations"),
                pnm_mode=arguments.get("pnm_mode"),
            )
            return json.dumps(result)
        except Exception as exc:
            return json.dumps({"status": "error", "message": str(exc)})

    if name == "rerun_case":
        try:
            result = _rerun_case(
                arguments.get("case"),
                pnm_mode=arguments.get("pnm_mode"),
            )
            return json.dumps(result)
        except Exception as exc:
            return json.dumps({"status": "error", "message": str(exc)})

    if name == "reset_parameters":
        config.update(DEFAULTS)
        image_config.update(IMAGE_DEFAULTS)
        uq_config["realizations"] = DEFAULT_REALIZATIONS
        _reset_simulation_param_files()
        WORKFLOW_TRACKER.task_mode = config["task_mode"]
        WORKFLOW_TRACKER.pnm_mode = config["pnm_mode"]
        WORKFLOW_TRACKER.reset("Reset parameters and started a fresh workflow cycle.")
        return json.dumps({
            "status": "ok",
            "message": "All parameters reset to defaults.",
            "current_config": dict(config),
            **_workflow_payload(),
        })

    if name == "set_image_properties":
        updated = {}
        dimension_key = "voxels" if config.get("img_src") == "x_ray" else "shape"
        for source_key in ("dimensions", "shape", "voxels"):
            if source_key in arguments:
                image_config[dimension_key] = arguments[source_key]
                updated[dimension_key] = arguments[source_key]
        for key in (
            "voxel_size",
            "image_file",
            "coarse_factor",
            "porosity",
            "blobiness",
            "pore_diameter_type",
            "throat_diameter_type",
        ):
            if key in arguments:
                if key in {"pore_diameter_type", "throat_diameter_type"}:
                    arguments[key] = _normalize_diameter_type(arguments[key])
                if key in ALLOWED_VALUES and arguments[key] not in ALLOWED_VALUES[key]:
                    return json.dumps({
                        "status": "error",
                        "message": f"Invalid {key}='{arguments[key]}'. Allowed: {ALLOWED_VALUES[key]}",
                    })
                image_config[key] = arguments[key]
                updated[key] = arguments[key]
        if updated:
            WORKFLOW_TRACKER.note("image_properties_updated", updated=updated)
        return json.dumps({
            "status": "ok",
            "updated": updated,
            "image_properties": {k: v for k, v in image_config.items() if v is not None},
            **_workflow_payload(),
        })

    if name == "set_qs_params":
        qs_file = THIS_DIR / "input_pnm" / "qs_param.txt"
        try:
            lines = qs_file.read_text().splitlines()
        except FileNotFoundError:
            lines = ["0", "5e4", "1e2", "1",
                     "// minimum pressure", "// maximum pressure",
                     "// pressure step", "// compute permeability"]
        if "min_pressure" in arguments:
            lines[0] = str(arguments["min_pressure"])
        if "max_pressure" in arguments:
            lines[1] = str(arguments["max_pressure"])
        if "pressure_step" in arguments:
            lines[2] = str(arguments["pressure_step"])
        qs_file.write_text("\n".join(lines) + "\n")
        WORKFLOW_TRACKER.note(
            "qs_parameters_updated",
            min_pressure=lines[0],
            max_pressure=lines[1],
            pressure_step=lines[2],
        )
        return json.dumps({
            "status": "ok",
            "qs_params": {
                "min_pressure": lines[0],
                "max_pressure": lines[1],
                "pressure_step": lines[2],
            },
            "file": str(qs_file),
            **_workflow_payload(),
        })

    if name == "set_dyn_params":
        dyn_file = THIS_DIR / "input_pnm" / "dyn_param.txt"
        try:
            lines = dyn_file.read_text().splitlines()
        except FileNotFoundError:
            lines = DYN_PARAM_DEFAULTS.splitlines()
        if "gas_viscosity" in arguments:
            lines[0] = f"{arguments['gas_viscosity']:.6e}"
        if "liquid_viscosity" in arguments:
            lines[1] = f"{arguments['liquid_viscosity']:.6e}"
        if "capillary_number" in arguments:
            lines[3] = f"{arguments['capillary_number']:.6e}"
        if "max_pc" in arguments:
            lines[7] = str(arguments["max_pc"])
        dyn_file.write_text("\n".join(lines) + "\n")
        WORKFLOW_TRACKER.note(
            "dynamic_parameters_updated",
            gas_viscosity=lines[0],
            liquid_viscosity=lines[1],
            capillary_number=lines[3],
            max_pc=lines[7],
        )
        return json.dumps({
            "status": "ok",
            "dyn_params": {
                "gas_viscosity": lines[0],
                "liquid_viscosity": lines[1],
                "capillary_number": lines[3],
                "max_pc": lines[7],
            },
            "file": str(dyn_file),
            **_workflow_payload(),
        })

    return json.dumps({"status": "error", "message": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# GUI Application
# ---------------------------------------------------------------------------
class AutoPoreGUI:
    # Color scheme
    BG_DARK = "#1e1e2e"
    BG_CHAT = "#181825"
    BG_INPUT = "#313244"
    BG_USER = "#45475a"
    BG_AGENT = "#1e1e2e"
    BG_TOOL = "#11111b"
    BG_SIDEBAR = BG_DARK
    FG_MAIN = "#cdd6f4"
    FG_DIM = "#a6adc8"
    FG_OPTION = "#6c7086"
    FG_ACCENT = "#89b4fa"
    FG_GREEN = "#a6e3a1"
    FG_YELLOW = "#f9e2af"
    FG_RED = "#f38ba8"
    FG_TOOL = "#94e2d5"
    BORDER_COLOR = "#313244"

    
    # Color scheme
    # BG_DARK       = "#0E0E0E"   # app background
    # BG_CHAT       = "#141414"   # main panels
    # BG_INPUT      = "#1C1C1C"   # input box
    # BG_USER       = "#1A1A1A"   # user bubble / user area
    # BG_AGENT      = "#141414"   # assistant area
    # BG_TOOL       = "#121212"   # tool/status blocks
    # BG_SIDEBAR    = "#101010"   # sidebar / config background
    # FG_MAIN       = "#FFFFFF"   # main text
    # FG_DIM        = "#B3B3B3"   # secondary text
    # FG_ACCENT     = "#FFFFFF"   # keep accent white for this style
    # FG_GREEN      = "#FFFFFF"   # avoid colored accents
    # FG_YELLOW     = "#FFFFFF"
    # FG_RED        = "#FFFFFF"
    # FG_TOOL       = "#E5E5E5"   # readable tool text
    # BORDER_COLOR  = "#2A2A2A"
    
    def __init__(self, root: tk.Tk):
        global _figure_callback, _workflow_progress_callback

        self.root = root
        self.llm_settings = load_settings(LLM_SETTINGS_FILE)
        self.provider_var = tk.StringVar(value=self.llm_settings.provider)
        self.model_var = tk.StringVar(value=self.llm_settings.model)
        self.base_url_var = tk.StringVar(value=self.llm_settings.base_url)
        self.task_mode_var = tk.StringVar(value=config.get("task_mode", "single_run"))
        self.model_combo = None
        self._conversation_llm_key = self._llm_key()
        self._update_window_title()
        self.root.geometry("1540x820+40+120")
        self.root.minsize(1540, 820)
        self.root.configure(bg=self.BG_DARK)

        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": self._build_workflow_state_message()},
        ]
        self._busy = False
        self._simulation_running = False
        self.workflow_display = None
        self.workflow_log_label = None
        self.flowchart_canvas = None
        self.generated_files_display = None
        self.chat_panel = None
        self.task_single_btn = None
        self.task_uq_btn = None
        self.run_next_btn = None
        self.run_full_btn = None
        self.start_over_btn = None
        self._active_panel_action = None
        self.step_indicator_labels = {}
        self._flowchart_photo = None
        self._flowchart_startup_checked = False
        self._workflow_running_dot_count = 0
        self._workflow_animation_after_id = None
        self._workflow_control_widgets = []
        self._workflow_controls_refreshing = False
        self._button_phase = "idle"
        self._button_dot_count = 0
        self._button_animation_after_id = None
        self._output_log_tag_counter = 0

        # Figure state
        self._figure_paths: list[str] = []
        self._figure_index = 0
        self._photo_image = None
        self._figure_slide_after_id = None
        self._figure_transition_after_id = None
        self._transition_photo_image = None
        self._figure_slide_pause_ms = 1500
        self._figure_slide_last_pause_ms = 3000
        self._figure_transition_ms = 320
        self._figure_transition_frames = 12

        # Figure popup window state
        self.fig_window = None
        self.fig_canvas = None
        self.fig_name_label = None
        self.fig_counter_label = None
        self.fig_prev_btn = None
        self.fig_next_btn = None
        self.fig_open_btn = None
        self.clear_history_btn = None

        # GIF animation state
        self._gif_frames = []
        self._gif_durations = []
        self._gif_frame_index = 0
        self._gif_after_id = None

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        _figure_callback = self._queue_figure_popup
        _workflow_progress_callback = self._queue_workflow_progress

        self._build_ui()
        self._refresh_workflow_panel()
        self.root.after(250, self._show_flowchart_on_startup)
        if self.provider_var.get() == "local":
            self.root.after(350, self._detect_local_model)
        self._show_welcome()
        self._show_initial_parameters()

    def _llm_key(self) -> tuple[str, str, str]:
        return (
            self.provider_var.get().strip().lower(),
            self.model_var.get().strip(),
            self.base_url_var.get().strip(),
        )

    def _current_llm_settings(self) -> LLMSettings:
        settings = LLMSettings(
            provider=self.provider_var.get(),
            model=self.model_var.get(),
            base_url=self.base_url_var.get(),
        )
        settings.validate()
        return settings

    def _update_window_title(self) -> None:
        provider = self.provider_var.get().strip().lower() or "local"
        provider_label = {
            "local": "Ollama",
            "openai": "OpenAI",
            "openrouter": "OpenRouter",
        }.get(provider, provider)
        model = self.model_var.get().strip() or "auto"
        self.root.title(f"AutoPore ({model}, {provider_label})")

    def _save_llm_settings(self, *, reset_conversation: bool = True) -> None:
        try:
            settings = self._current_llm_settings()
            save_settings(LLM_SETTINGS_FILE, settings)
        except Exception as exc:
            self._append_chat("", f"Could not save LLM settings: {exc}", "error")
            return
        self.llm_settings = settings
        self._update_window_title()
        key = self._llm_key()
        if reset_conversation and key != self._conversation_llm_key:
            self.messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "system", "content": self._build_workflow_state_message()},
            ]
            self._conversation_llm_key = key

    def _provider_changed(self, _event=None) -> None:
        provider = self.provider_var.get().strip().lower()
        model, base_url = provider_defaults(provider)
        self.model_var.set(model)
        self.base_url_var.set(base_url)
        if self.model_combo is not None:
            self.model_combo.configure(values=())
        self._save_llm_settings()
        if provider == "local":
            self._detect_local_model()

    def _llm_fields_changed(self, _event=None) -> None:
        self._save_llm_settings()

    def _apply_resolved_model(
        self,
        expected_provider: str,
        expected_base_url: str,
        selected: str,
        available: list[str],
        announce: bool = False,
    ) -> None:
        current_provider, _, current_url = self._llm_key()
        if (
            current_provider != expected_provider
            or current_url != expected_base_url
        ):
            return
        if self.model_combo is not None:
            self.model_combo.configure(values=available)
        self.model_var.set(selected)
        self._save_llm_settings(reset_conversation=announce)
        if not announce:
            self._conversation_llm_key = self._llm_key()
        if announce:
            self._append_generated_files_message(
                f"Local model selected: {selected}"
            )

    def _detect_local_model(self) -> None:
        if self.provider_var.get().strip().lower() != "local":
            return
        settings = self._current_llm_settings()
        requested_key = self._llm_key()

        def work():
            try:
                selected, available = LLMClient(settings).resolve_model()
            except Exception as exc:
                self.root.after(
                    0,
                    self._append_chat,
                    "",
                    f"Local model detection failed: {exc}",
                    "error",
                )
                return

            self.root.after(
                0,
                self._apply_resolved_model,
                requested_key[0],
                requested_key[2],
                selected,
                available,
                True,
            )

        threading.Thread(target=work, daemon=True).start()

    def _build_workflow_state_message(self) -> str:
        snapshot = WORKFLOW_TRACKER.snapshot()
        return (
            "Workflow state: "
            f"branch={snapshot['branch']}; "
            f"cycle_id={snapshot['cycle_id']}; "
            f"closed={snapshot['closed']}; "
            f"completed_steps={snapshot['completed_steps']}; "
            f"active_step={snapshot['active_step']}; "
            f"highlighted_steps={snapshot['highlighted_steps']}; "
            f"completed_nodes={snapshot.get('completed_nodes', [])}; "
            f"active_nodes={snapshot.get('active_nodes', [])}; "
            f"highlighted_nodes={snapshot.get('highlighted_nodes', [])}; "
            f"configured_realizations={uq_config['realizations']}; "
            f"next_allowed_steps={snapshot['next_allowed_steps']}; "
            f"log_file={snapshot['log_file']}."
        )

    def _refresh_workflow_state_message(self):
        self.messages[1] = {
            "role": "system",
            "content": self._build_workflow_state_message(),
        }

    def _update_workflow_state_from_tool(self, func_name: str, tool_result: str):
        try:
            json.loads(tool_result)
        except json.JSONDecodeError:
            return

        self._refresh_workflow_state_message()
        self.root.after(0, self._refresh_workflow_panel)

    def _queue_workflow_progress(self, event: dict):
        event = dict(event or {})
        event_name = event.get("event")
        index = event.get("index")
        total = event.get("total")
        case = event.get("case")

        if event_name == "realization_started":
            WORKFLOW_TRACKER.start_uq_realization(index, total, case=case)
            self.root.after(0, self._reset_live_figures)
        elif event_name == "input_image_visualized":
            if WORKFLOW_TRACKER.active_step != "generate_input":
                return
            source_node = _workflow_input_source_node()
            source_label = "Image generated" if source_node == "image_generation" else "Image imported"
            WORKFLOW_TRACKER.switch_active_nodes(
                "generate_input",
                completed_nodes=[source_node],
                active_nodes=["pore_extraction"],
                message=f"{source_label}; extracting pore network.",
                source="show_image",
            )
        elif event_name == "realization_generated":
            generate_nodes = _workflow_nodes_for_step("generate_input")
            simulation_nodes = _workflow_nodes_for_step("run_simulation")
            WORKFLOW_TRACKER.complete_step(
                "generate_input",
                "success",
                message=f"UQ realization {index}/{total}: input generated.",
                source="run_realizations",
                completed_nodes=generate_nodes,
            )
            WORKFLOW_TRACKER.prepare_step(
                "run_simulation",
                source="run_realizations",
                active_nodes=simulation_nodes,
                upstream_nodes=generate_nodes,
            )
        elif event_name == "realization_simulated":
            simulation_nodes = _workflow_nodes_for_step("run_simulation")
            WORKFLOW_TRACKER.complete_step(
                "run_simulation",
                "success",
                message=f"UQ realization {index}/{total}: simulation finished.",
                source="run_realizations",
                completed_nodes=simulation_nodes,
            )
        elif event_name == "realization_finished":
            WORKFLOW_TRACKER.note(
                "uq_realization_finished",
                realization_index=index,
                total=total,
                case=case,
                status=event.get("status"),
            )
        elif event_name in {"step_started", "step_finished", "workflow_state_changed"}:
            if (
                event_name == "step_started"
                and event.get("step") == "run_simulation"
                and config.get("pnm_mode") == "dynamic"
            ):
                self.root.after(0, self._ensure_single_3d_preview)
        else:
            return

        self.root.after(0, self._refresh_workflow_state_message)
        self.root.after(0, self._refresh_workflow_panel)
        if event_name in {"input_image_visualized", "realization_generated", "realization_simulated", "realization_finished"}:
            self.root.after(0, self._sync_figures_to_workflow_state)

    def _snapshot_artifact_files(self) -> dict[str, float]:
        roots = [
            THIS_DIR / "image",
            THIS_DIR / "input_pnm",
            THIS_DIR / "single_run_qs_pnm",
            THIS_DIR / "single_run_dyn_pnm",
            THIS_DIR / "uq_qs_pnm",
            THIS_DIR / "uq_dyn_pnm",
        ]
        files: dict[str, float] = {}
        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_file():
                    resolved = str(path.resolve())
                    try:
                        files[resolved] = path.stat().st_mtime
                    except OSError:
                        continue
        return files

    def _summarize_generated_artifacts(self, before: dict[str, float], after: dict[str, float]) -> str | None:
        created_files = sorted(path for path in after if path not in before)
        updated_files = sorted(
            path for path, mtime in after.items()
            if path in before and mtime > before[path]
        )
        if not created_files and not updated_files:
            return None

        image_exts = {".png", ".gif", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        groups: dict[tuple[str, str], list[str]] = {}

        def add_grouped(file_paths: list[str], change_label: str):
            for file_path in file_paths:
                path = Path(file_path)
                kind = "Images" if path.suffix.lower() in image_exts else "Files"
                key = (str(path.parent), f"{change_label} {kind}")
                groups.setdefault(key, []).append(path.name)

        add_grouped(created_files, "Created")
        add_grouped(updated_files, "Updated")

        lines = ["Generated artifacts summary:"]
        for (folder, kind), names in sorted(groups.items()):
            names = sorted(names)
            if len(names) <= 3:
                item_text = ", ".join(names)
            else:
                item_text = f"{len(names)} similar files"
                first = ", ".join(names[:3])
                item_text += f" ({first}, ...)"
            lines.append(f"  {kind}: {item_text}")
            lines.append(f"  Location: {folder}")

        return "\n".join(lines)

    def _show_generated_artifacts(self, before: dict[str, float], after: dict[str, float]):
        summary = self._summarize_generated_artifacts(before, after)
        if summary:
            self._append_generated_files_message(summary)

    def _clear_generated_files_display(self):
        if self.generated_files_display is None:
            return
        self.generated_files_display.configure(state=tk.NORMAL)
        self.generated_files_display.delete("1.0", tk.END)
        self.generated_files_display.configure(state=tk.DISABLED)

    def _insert_output_log_text(self, text: str):
        if self.generated_files_display is None:
            return
        log_font = tkfont.Font(font=self.generated_files_display.cget("font"))
        widget_width = self.generated_files_display.winfo_width() or 700
        char_width = max(1, log_font.measure("0"))
        wrap_width = max(32, int((widget_width - 32) / char_width))

        def wrapped_log_lines(line: str) -> list[tuple[str, int | None]]:
            colon_index = line.find(":")
            if colon_index < 0 or colon_index >= len(line) - 1:
                return [(part, None) for part in textwrap.wrap(
                    line,
                    width=wrap_width,
                    break_long_words=False,
                    break_on_hyphens=False,
                    replace_whitespace=False,
                    drop_whitespace=False,
                )] or [(line, None)]
            prefix_end = colon_index + 1
            if len(line) > prefix_end and line[prefix_end] == " ":
                prefix_end += 1

            prefix = line[:prefix_end]
            value = line[prefix_end:]
            if "\\" in value or "/" in value:
                indent = " " * prefix_end
                pieces = []
                current_piece = ""
                for char in value:
                    current_piece += char
                    if char in "\\/":
                        pieces.append(current_piece)
                        current_piece = ""
                if current_piece:
                    pieces.append(current_piece)

                physical_lines = []
                current = prefix
                for piece in pieces:
                    if (
                        current != prefix
                        and len(current) + len(piece) > wrap_width
                    ):
                        physical_lines.append(current.rstrip())
                        current = indent + piece
                    else:
                        current += piece
                if current.strip():
                    physical_lines.append(current.rstrip())
                return [(part, prefix_end) for part in physical_lines] or [(line, prefix_end)]

            wrapped = textwrap.wrap(
                line,
                width=wrap_width,
                subsequent_indent=" " * prefix_end,
                break_long_words=False,
                break_on_hyphens=False,
                replace_whitespace=False,
                drop_whitespace=False,
            ) or [line]
            return [(part, prefix_end) for part in wrapped]

        first_line = True
        for line in text.splitlines():
            for physical_line, prefix_end in wrapped_log_lines(line):
                if not first_line:
                    self.generated_files_display.insert(tk.END, "\n")
                first_line = False
                if prefix_end is None:
                    self.generated_files_display.insert(tk.END, physical_line)
                    continue
                prefix_text = line[:prefix_end]
                self._output_log_tag_counter += 1
                tag_name = f"output_log_line_{self._output_log_tag_counter}"
                self.generated_files_display.tag_configure(
                    tag_name,
                    lmargin2=log_font.measure(prefix_text),
                )
                self.generated_files_display.insert(tk.END, physical_line, tag_name)

    def _append_generated_files_message(self, text: str):
        if not text or self.generated_files_display is None:
            return
        self.generated_files_display.configure(state=tk.NORMAL)
        if self.generated_files_display.get("1.0", tk.END).strip():
            self.generated_files_display.insert(tk.END, "\n\n")
        self._insert_output_log_text(text.strip())
        self.generated_files_display.configure(state=tk.DISABLED)
        self.generated_files_display.see(tk.END)

    def _remove_path_for_history_clear(self, path: Path) -> None:
        if path.is_symlink() or path.is_file():
            path.unlink()
            return
        if path.is_dir():
            shutil.rmtree(path)

    def _clear_history(self):
        if self._busy or self._simulation_running:
            messagebox.showwarning(
                "Clear history",
                "A workflow is currently running. Please wait for it to finish before clearing history.",
                parent=self.root,
            )
            return

        project_root = THIS_DIR.resolve()
        workspace_root = project_root.parent.resolve()
        results_root = (workspace_root / "model_results").resolve()
        generated_roots = [
            (project_root / "image").resolve(),
            (project_root / "input_pnm").resolve(),
        ]
        protected_roots = [(results_root, workspace_root)] + [
            (path, project_root) for path in generated_roots
        ]
        for path, allowed_root in protected_roots:
            if allowed_root not in path.parents and path != allowed_root:
                messagebox.showerror(
                    "Clear history",
                    f"Cleanup was cancelled because a target folder is unexpected:\n\n{path}",
                    parent=self.root,
                )
                return

        confirmed = messagebox.askyesno(
            "Clear history",
            "This permanently removes archived model results in:\n"
            f"{results_root}\n\n"
            "It also removes AutoPore-generated files named generated_* in the image and input folders.\n"
            "Original imported images will be preserved. Continue?",
            icon="warning",
            parent=self.root,
        )
        if not confirmed:
            return

        removed_results = 0
        removed_generated = 0
        try:
            results_root.mkdir(parents=True, exist_ok=True)
            for path in list(results_root.iterdir()):
                self._remove_path_for_history_clear(path)
                removed_results += 1

            for root in generated_roots:
                if not root.exists():
                    continue
                for path in list(root.glob("generated_*")):
                    self._remove_path_for_history_clear(path)
                    removed_generated += 1
        except OSError as exc:
            messagebox.showerror(
                "Could not clear history",
                f"Cleanup stopped because a file could not be removed:\n\n{exc}",
                parent=self.root,
            )
            return

        global _shown_figures
        _shown_figures = []
        try:
            WORKFLOW_TRACKER._clear_log()
        except Exception:
            pass
        WORKFLOW_TRACKER.reset("History cleared.")
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": self._build_workflow_state_message()},
        ]
        self._conversation_llm_key = self._llm_key()
        self._clear_generated_files_display()
        self._figure_paths = []
        self._figure_index = 0
        if self.fig_window is not None and self.fig_window.winfo_exists():
            self._on_close_figure_window()
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.configure(state=tk.DISABLED)
        self._refresh_workflow_panel()
        self._show_welcome()
        self._append_generated_files_message(
            "History cleared.\n"
            f"  Removed result item(s): {removed_results}\n"
            f"  Removed generated file(s): {removed_generated}"
        )
        messagebox.showinfo(
            "History cleared",
            f"Removed {removed_results} result item(s) and {removed_generated} generated file(s).",
            parent=self.root,
        )

    def _execution_resets_generated_messages(self, func_name: str, func_args: dict) -> bool:
        if func_name == "run_realizations":
            return True
        if func_name == "execute_model":
            return func_args.get("action", config.get("action")) == "generate_input"
        if func_name == "execute_workflow":
            steps = func_args.get("steps", [])
            if isinstance(steps, str):
                steps = [steps]
            return "generate_input" in steps
        return False

    def _execution_resets_figures(self, func_name: str, func_args: dict) -> bool:
        if func_name == "run_realizations":
            return True
        if func_name == "execute_model":
            return func_args.get("action", config.get("action")) == "generate_input"
        if func_name == "execute_workflow":
            steps = func_args.get("steps", [])
            if isinstance(steps, str):
                steps = [steps]
            return "generate_input" in steps
        return False

    def _append_existing_figure(self, paths: list[str], candidate) -> None:
        if not candidate:
            return
        path = _normalize_figure_path(candidate)
        if os.path.isfile(path) and path not in paths:
            paths.append(path)

    def _active_output_folder(self) -> Path:
        return THIS_DIR / (
            "single_run_dyn_pnm"
            if config.get("pnm_mode") == "dynamic"
            else "single_run_qs_pnm"
        )

    def _preferred_input_visual_folder(self) -> Path:
        output_input = self._active_output_folder() / "input_pnm"
        if (output_input / "3D_image.png").is_file():
            return output_input
        return THIS_DIR / "input_pnm"

    def _resolve_figure_path(self, path: str) -> str | None:
        """Resolve transiently removed copied-input figures to their master copy."""
        normalized = _normalize_figure_path(path)
        if os.path.isfile(normalized):
            return normalized

        source = Path(normalized)
        if source.parent.name == "input_pnm":
            candidates = (
                THIS_DIR / "input_pnm" / source.name,
                self._active_output_folder() / "input_pnm" / source.name,
            )
            for candidate in candidates:
                if candidate.is_file():
                    return _normalize_figure_path(candidate)
        return None

    def _ensure_single_3d_preview(self) -> None:
        """Keep exactly one valid 3D preview visible during dynamic simulation."""
        candidates = (
            THIS_DIR / "input_pnm" / "3D_image.png",
            self._active_output_folder() / "input_pnm" / "3D_image.png",
        )
        preview = next((path for path in candidates if path.is_file()), None)
        if preview is None:
            return

        normalized_preview = _normalize_figure_path(preview)
        # Keep one canonical preview during simulation without stopping an
        # already-running slideshow timer. Post-processing will add the result
        # figures and continue normal slideshow advancement automatically.
        self._figure_paths = [normalized_preview]
        self._figure_index = 0

        self._ensure_figure_window()
        self.fig_window.deiconify()
        self._show_current_figure()

    def _input_visual_paths(self, *, include_extraction: bool, root: Path | None = None) -> list[str]:
        folder = Path(root) if root is not None else self._preferred_input_visual_folder()
        paths: list[str] = []
        self._append_existing_figure(paths, folder / "3D_image.png")
        if include_extraction:
            self._append_existing_figure(paths, folder / f"{MODEL_DEFAULTS['pnm_name']}.png")
            self._append_existing_figure(paths, folder / "pore_size_distribution.png")
        return paths

    @staticmethod
    def _dynamic_postprocess_sort_key(path) -> tuple[int, str]:
        name = Path(path).name.lower()
        if "gas saturation" in name:
            priority = 0
        elif "capillary pressure" in name:
            priority = 1
        elif "interfacial area" in name:
            priority = 2
        else:
            priority = 3
        return priority, name

    def _single_run_postprocess_paths(self) -> list[str]:
        input_paths = self._input_visual_paths(include_extraction=True)
        result_paths: list[str] = []
        figs_folder = self._active_output_folder() / "figs"
        if not figs_folder.is_dir():
            return input_paths

        figure_files = list(figs_folder.glob("*.png"))
        if config.get("pnm_mode") == "dynamic":
            figure_files.sort(key=self._dynamic_postprocess_sort_key)
        else:
            figure_files.sort()
        for path in figure_files:
            self._append_existing_figure(result_paths, path)
        for path in sorted(figs_folder.glob("*.gif")):
            preview = path.with_name(f"{path.stem}_last.png")
            if preview.is_file() and str(preview.resolve()) in result_paths:
                continue
            self._append_existing_figure(result_paths, path)
        if config.get("pnm_mode") == "dynamic":
            return result_paths + input_paths
        return input_paths + result_paths

    def _uq_summary_paths(self) -> list[str]:
        uq_root = THIS_DIR / (
            "uq_dyn_pnm"
            if config.get("pnm_mode") == "dynamic"
            else "uq_qs_pnm"
        )
        paths: list[str] = []
        if uq_root.is_dir():
            figure_files = list(uq_root.glob("UQ*.png"))
            if config.get("pnm_mode") == "dynamic":
                figure_files.sort(key=self._dynamic_postprocess_sort_key)
            else:
                figure_files.sort()
            for path in figure_files:
                self._append_existing_figure(paths, path)
        return paths

    def _workflow_completed_steps_from_result(self, result: dict | None) -> set[str]:
        if isinstance(result, dict):
            workflow = result.get("workflow")
            if isinstance(workflow, dict):
                return set(workflow.get("completed_steps", []))
        return set(WORKFLOW_TRACKER.snapshot().get("completed_steps", []))

    def _figures_for_workflow_stage(self, result: dict | None = None) -> list[str]:
        if isinstance(result, dict) and result.get("status") in {"error", "cancelled"}:
            return []

        completed = self._workflow_completed_steps_from_result(result)
        if config.get("task_mode") == "uq_run":
            if "plot_result" in completed or (isinstance(result, dict) and result.get("step") == "plot_result"):
                return self._uq_summary_paths()
            if completed & {"generate_input", "run_simulation"}:
                return self._input_visual_paths(include_extraction=True)
            return self._input_visual_paths(include_extraction=False)

        if "plot_result" in completed or (isinstance(result, dict) and result.get("step") == "plot_result"):
            return self._single_run_postprocess_paths()
        if completed & {"generate_input", "run_simulation"}:
            return self._input_visual_paths(include_extraction=True)
        return self._input_visual_paths(include_extraction=False)

    def _sync_figures_to_workflow_state(self, result: dict | None = None):
        figures = self._figures_for_workflow_stage(result)
        if figures:
            self._load_figures(figures)
        elif self.fig_window is not None and self.fig_window.winfo_exists():
            preview = THIS_DIR / "input_pnm" / "3D_image.png"
            if preview.is_file():
                self._load_figures([str(preview)])
            else:
                self._reset_live_figures()

    def _collect_displayable_figure_paths(self) -> list[str]:
        figure_paths: list[str] = []

        for path in _shown_figures:
            normalized_path = _normalize_figure_path(path)
            if os.path.isfile(normalized_path) and normalized_path not in figure_paths:
                figure_paths.append(normalized_path)

        if config.get("action") == "generate_input":
            expected_paths = [
                _normalize_figure_path(
                    THIS_DIR / "input_pnm" / "3D_image.png"
                ),
                _normalize_figure_path(
                    THIS_DIR / "input_pnm" / f"{MODEL_DEFAULTS['pnm_name']}.png"
                ),
            ]
            for path in expected_paths:
                if os.path.isfile(path) and path not in figure_paths:
                    figure_paths.append(path)

        if config.get("action") == "plot_result" and config.get("task_mode") == "uq_run":
            uq_root = THIS_DIR / ("uq_dyn_pnm" if config.get("pnm_mode") == "dynamic" else "uq_qs_pnm")
            if uq_root.exists():
                for path in sorted(uq_root.glob("UQ*.png")):
                    normalized_path = _normalize_figure_path(path)
                    if os.path.isfile(normalized_path) and normalized_path not in figure_paths:
                        figure_paths.append(normalized_path)

        return figure_paths

    def _refresh_workflow_panel(self):
        self._refresh_task_selector()
        self._refresh_flowchart_preview()
        self._refresh_step_indicators()

    def _refresh_task_selector(self):
        selected_mode = config.get("task_mode", "single_run")
        self.task_mode_var.set(selected_mode)
        label_specs = (
            (self.task_single_btn, "single_run"),
            (self.task_uq_btn, "uq_run"),
        )
        for label, mode in label_specs:
            if label is None:
                continue
            selected = selected_mode == mode
            label.configure(
                bg=self.BG_CHAT,
                fg=self.FG_ACCENT if selected else self.FG_DIM,
                activebackground=self.BG_CHAT,
                activeforeground=self.FG_ACCENT if selected else self.FG_DIM,
                state=tk.DISABLED if self._busy else tk.NORMAL,
            )

    def _task_mode_selected(self):
        selected_mode = self.task_mode_var.get()
        if selected_mode == config.get("task_mode"):
            return
        if self._apply_workflow_control_update(
            "set_parameters", {"task_mode": selected_mode}
        ):
            self._clear_generated_files_display()
        else:
            self.task_mode_var.set(config.get("task_mode", "single_run"))

    def _refresh_step_indicators(self):
        if not self.step_indicator_labels:
            return

        snapshot = WORKFLOW_TRACKER.snapshot()
        highlighted = set(snapshot.get("highlighted_steps", []))
        active_step = snapshot.get("active_step")
        started_from = snapshot.get("started_from")
        start_index = (
            ("generate_input", "run_simulation", "plot_result").index(started_from)
            if started_from in {"generate_input", "run_simulation", "plot_result"}
            else 0
        )

        for step, label in self.step_indicator_labels.items():
            step_index = ("generate_input", "run_simulation", "plot_result").index(step)
            if step == active_step:
                bg = self.FG_ACCENT
                fg = self.BG_DARK
                text_prefix = "RUNNING"
                relief = tk.SUNKEN
            elif step in highlighted:
                bg = self.FG_GREEN
                fg = self.BG_DARK
                text_prefix = "DONE"
                relief = tk.SUNKEN
            elif started_from and step_index < start_index:
                bg = self.FG_GREEN
                fg = self.BG_DARK
                text_prefix = "READY"
                relief = tk.SUNKEN
            else:
                bg = self.BG_INPUT
                fg = self.FG_MAIN
                text_prefix = "NEXT" if step in snapshot.get("next_allowed_steps", []) else "WAIT"
                relief = tk.FLAT

            label.configure(
                bg=bg,
                fg=fg,
                relief=relief,
                text=f"{text_prefix}\n{label.step_text}",
            )

    def _select_workflow_task(self, task_mode: str):
        if task_mode not in ALLOWED_VALUES["task_mode"]:
            return
        if config.get("task_mode") == task_mode:
            self._refresh_task_selector()
            return

        config["task_mode"] = task_mode
        if task_mode == "uq_run":
            config["img_src"] = "porespy"
        self._clear_generated_files_display()
        _sync_workflow_tracker()
        WORKFLOW_TRACKER.note("task_selected_from_panel", task_mode=task_mode)
        self._refresh_workflow_state_message()
        self._refresh_workflow_panel()

    def _workflow_overlay_text(self) -> str:
        snapshot = WORKFLOW_TRACKER.snapshot()
        if snapshot["closed"]:
            return "Closed loop complete"
        next_steps = snapshot.get("next_allowed_steps", [])
        if next_steps:
            label_map = {
                "generate_input": "Generate input",
                "run_simulation": "Run simulation",
                "plot_result": "Plot results",
            }
            if snapshot.get("task_mode") == "uq_run" and next_steps[0] == "generate_input":
                return "Next: Generate input & run simulation"
            return f"Next: {label_map.get(next_steps[0], next_steps[0])}"
        return "Workflow waiting"

    def _flowchart_node_style(self, node: str, active_nodes: set[str], highlighted_nodes: set[str], active: bool = True):
        if not active:
            return "#25283a", self.FG_DIM, self.BORDER_COLOR, 1
        if node in active_nodes:
            return self.FG_ACCENT, self.BG_DARK, "#ffffff", 2
        if node in highlighted_nodes:
            return self.FG_GREEN, self.BG_DARK, "#ffffff", 2
        return self.BG_INPUT, self.FG_MAIN, self.BORDER_COLOR, 1

    def _draw_flowchart_node(self, x0, y0, x1, y1, text, node, active_nodes, highlighted_nodes, active=True):
        fill, text_fill, outline, width = self._flowchart_node_style(
            node,
            active_nodes,
            highlighted_nodes,
            active=active,
        )
        show_running_dots = active and node in active_nodes and self._simulation_running
        dots = "." * max(1, self._workflow_running_dot_count) if show_running_dots else ""
        self.flowchart_canvas.create_rectangle(
            x0,
            y0,
            x1,
            y1,
            fill=fill,
            outline=outline,
            width=width,
        )
        node_font = ("Segoe UI", 10, "bold")
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        self.flowchart_canvas.create_text(
            center_x,
            center_y,
            text=text,
            fill=text_fill,
            font=node_font,
            justify=tk.CENTER,
            width=max(60, x1 - x0 - 8),
        )
        if show_running_dots:
            measured_font = tkfont.Font(font=node_font)
            text_width = max(
                measured_font.measure(line)
                for line in str(text).splitlines()
            )
            dot_width = measured_font.measure("...")
            dot_x = min(
                x1 - dot_width - 8,
                center_x + text_width / 2 + 5,
            )
            self.flowchart_canvas.create_text(
                dot_x,
                center_y,
                text=dots,
                fill=text_fill,
                font=node_font,
                anchor=tk.W,
                tags=("workflow_running_dots",),
            )

    def _refresh_workflow_running_dots(self):
        if self.flowchart_canvas is None:
            return
        dots = "." * max(1, self._workflow_running_dot_count)
        for item in self.flowchart_canvas.find_withtag("workflow_running_dots"):
            try:
                self.flowchart_canvas.itemconfigure(item, text=dots)
            except tk.TclError:
                pass

    def _clear_workflow_running_dots(self):
        if self.flowchart_canvas is None:
            return
        for item in self.flowchart_canvas.find_withtag("workflow_running_dots"):
            try:
                self.flowchart_canvas.itemconfigure(item, text="")
            except tk.TclError:
                pass

    def _draw_arrow(
        self,
        x1,
        y1,
        x2,
        y2,
        from_node=None,
        to_node=None,
        highlighted_nodes=None,
        active_nodes=None,
    ):
        highlighted_nodes = set(highlighted_nodes or [])
        active_nodes = set(active_nodes or [])
        executed_link = (
            from_node is not None
            and to_node is not None
            and from_node in highlighted_nodes
            and (to_node in highlighted_nodes or to_node in active_nodes)
        )
        self.flowchart_canvas.create_line(
            x1,
            y1,
            x2,
            y2,
            fill=self.FG_GREEN if executed_link else self.FG_DIM,
            arrow=tk.LAST,
            width=3 if executed_link else 2,
        )

    def _draw_parameter_block(self, x, y, title, lines, width):
        title_font = ("Segoe UI", 10, "bold")
        param_font = ("Consolas", 9)
        param_measure = tkfont.Font(family="Consolas", size=9)
        self.flowchart_canvas.create_text(
            x,
            y,
            anchor=tk.NW,
            text=title,
            fill=self.FG_DIM,
            font=title_font,
            width=width,
        )
        line_y = y + 17
        for line in lines:
            text = line.rstrip()
            option_start = text.find(" (opts:")
            if option_start < 0:
                option_start = text.find("opts:")
            if option_start >= 0:
                before = text[:option_start]
                option_text = text[option_start:]
                if before:
                    self.flowchart_canvas.create_text(
                        x,
                        line_y,
                        anchor=tk.NW,
                        text=before,
                        fill=self.FG_DIM,
                        font=param_font,
                    )
                option_x = x + param_measure.measure(before)
                self.flowchart_canvas.create_text(
                    option_x,
                    line_y,
                    anchor=tk.NW,
                    text=option_text,
                    fill=self.FG_OPTION,
                    font=param_font,
                    width=max(40, width - param_measure.measure(before)),
                )
            else:
                self.flowchart_canvas.create_text(
                    x,
                    line_y,
                    anchor=tk.NW,
                    text=text,
                    fill=self.FG_DIM,
                    font=param_font,
                    width=width,
                )
            line_y += 14

    def _clear_workflow_controls(self):
        self._workflow_controls_refreshing = True
        try:
            for widget in self._workflow_control_widgets:
                try:
                    widget.destroy()
                except tk.TclError:
                    pass
            self._workflow_control_widgets.clear()
        finally:
            self._workflow_controls_refreshing = False

    def _add_workflow_widget(self, widget, x, y, width, height=18):
        self._workflow_control_widgets.append(widget)
        self.flowchart_canvas.create_window(
            x,
            y,
            anchor=tk.NW,
            window=widget,
            width=max(1, int(width)),
            height=height,
        )
        return widget

    def _draw_workflow_parameter_title(self, x, y, title, width):
        self.flowchart_canvas.create_text(
            x,
            y,
            anchor=tk.NW,
            text=title,
            fill=self.FG_DIM,
            font=("Segoe UI", 10, "bold"),
            width=width,
        )

    def _show_workflow_control_error(self, message):
        messagebox.showerror("Invalid parameter", str(message), parent=self.root)

    def _apply_workflow_control_update(self, tool_name, arguments):
        if self._workflow_controls_refreshing or self._busy:
            return False
        try:
            result = json.loads(dispatch_tool_call(tool_name, arguments))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._show_workflow_control_error(exc)
            return False
        if result.get("status") == "error":
            self._show_workflow_control_error(
                result.get("message", "The parameter could not be updated.")
            )
            return False
        self.root.after_idle(self._refresh_workflow_panel)
        return True

    @staticmethod
    def _parse_workflow_dimensions(value):
        parts = [part for part in re.split(r"[\s,;xX\[\]()]+", value.strip()) if part]
        if len(parts) != 3:
            raise ValueError("Dimensions must contain exactly three voxel counts.")
        dimensions = [int(part) for part in parts]
        if any(item <= 0 for item in dimensions):
            raise ValueError("Each dimension must be greater than zero.")
        return dimensions

    @staticmethod
    def _parse_positive_float(value):
        parsed = float(value)
        if parsed <= 0:
            raise ValueError("The value must be greater than zero.")
        return parsed

    @staticmethod
    def _parse_nonnegative_float(value):
        parsed = float(value)
        if parsed < 0:
            raise ValueError("The value cannot be negative.")
        return parsed

    @staticmethod
    def _parse_porosity(value):
        parsed = float(value)
        if not 0 < parsed < 1:
            raise ValueError("Porosity must be greater than zero and less than one.")
        return parsed

    @staticmethod
    def _parse_positive_int(value):
        parsed = int(value)
        if parsed <= 0:
            raise ValueError("The value must be an integer greater than zero.")
        return parsed

    def _workflow_control_geometry(self, x, width, label_width=170):
        control_x = x + label_width
        control_width = max(92, min(132, width - label_width - 58))
        return control_x, control_width

    def _draw_workflow_entry_row(
        self,
        x,
        y,
        width,
        label,
        value,
        tool_name,
        key,
        parser=str,
        unit="",
        label_width=170,
        enabled=True,
    ):
        label_font = ("Segoe UI", 10)
        control_font = ("Segoe UI", 9)
        self.flowchart_canvas.create_text(
            x + 2,
            y + 1,
            anchor=tk.NW,
            text=f"{label}:",
            fill=self.FG_DIM if enabled else self.FG_OPTION,
            font=label_font,
        )
        control_x, control_width = self._workflow_control_geometry(
            x,
            width,
            label_width=label_width,
        )
        text_value = str(value)
        variable = tk.StringVar(value=text_value)
        entry = tk.Entry(
            self.flowchart_canvas,
            textvariable=variable,
            bg=self.BG_INPUT,
            fg=self.FG_MAIN,
            insertbackground=self.FG_MAIN,
            disabledbackground=self.BG_INPUT,
            disabledforeground=self.FG_OPTION,
            relief=tk.FLAT,
            bd=0,
            highlightthickness=1,
            highlightbackground=self.BORDER_COLOR,
            highlightcolor=self.FG_ACCENT,
            font=control_font,
        )
        if self._busy or not enabled:
            entry.configure(state=tk.DISABLED)
        self._add_workflow_widget(entry, control_x, y, control_width)
        if unit:
            self.flowchart_canvas.create_text(
                control_x + control_width + 7,
                y + 1,
                anchor=tk.NW,
                text=unit,
                fill=self.FG_DIM if enabled else self.FG_OPTION,
                font=label_font,
            )

        committed = {"text": text_value}

        def commit(_event=None):
            if self._workflow_controls_refreshing or self._busy or not enabled:
                return
            raw_value = variable.get().strip()
            if raw_value == committed["text"]:
                return
            try:
                parsed_value = parser(raw_value)
            except (TypeError, ValueError) as exc:
                variable.set(committed["text"])
                self._show_workflow_control_error(exc)
                return
            if self._apply_workflow_control_update(tool_name, {key: parsed_value}):
                committed["text"] = raw_value
            else:
                variable.set(committed["text"])

        entry.bind("<Return>", commit)
        entry.bind("<FocusOut>", commit)
        return entry

    def _draw_workflow_combo_row(
        self,
        x,
        y,
        width,
        label,
        current_value,
        options,
        tool_name,
        key,
        label_width=170,
    ):
        label_font = ("Segoe UI", 10)
        self.flowchart_canvas.create_text(
            x + 2,
            y + 1,
            anchor=tk.NW,
            text=f"{label}:",
            fill=self.FG_DIM,
            font=label_font,
        )
        display_to_value = {
            str(self._format_parameter_value(key, value)): value for value in options
        }
        current_display = str(self._format_parameter_value(key, current_value))
        variable = tk.StringVar(value=current_display)
        combo = ttk.Combobox(
            self.flowchart_canvas,
            textvariable=variable,
            values=list(display_to_value),
            state="readonly",
            width=12,
            style=(
                "Pore.WorkflowDim.TCombobox"
                if self._busy
                else "Pore.Workflow.TCombobox"
            ),
        )
        control_x, control_width = self._workflow_control_geometry(
            x,
            width,
            label_width=label_width,
        )
        self._add_workflow_widget(combo, control_x, y, control_width)

        def selected(_event=None):
            if self._busy:
                variable.set(current_display)
                return
            selected_value = display_to_value.get(variable.get())
            if selected_value is None or selected_value == current_value:
                return
            self._apply_workflow_control_update(tool_name, {key: selected_value})

        combo.bind("<<ComboboxSelected>>", selected)
        if self._busy:
            combo.bind("<Button-1>", lambda _event: "break")
            combo.bind("<Key>", lambda _event: "break")
        return combo

    def _draw_uq_realization_total(self, current, total, right_x, y):
        entry_width = 38
        entry_x = right_x - entry_width - 8
        self.flowchart_canvas.create_text(
            entry_x - 4,
            y + 1,
            anchor=tk.NE,
            text=f"{current}/",
            fill=self.FG_ACCENT,
            font=("Segoe UI", 9, "bold"),
        )

        variable = tk.StringVar(value=str(total))
        entry = tk.Entry(
            self.flowchart_canvas,
            textvariable=variable,
            bg=self.BG_INPUT,
            fg=self.FG_MAIN,
            insertbackground=self.FG_MAIN,
            disabledbackground=self.BG_INPUT,
            disabledforeground=self.FG_OPTION,
            relief=tk.FLAT,
            bd=0,
            highlightthickness=1,
            highlightbackground=self.BORDER_COLOR,
            highlightcolor=self.FG_ACCENT,
            justify=tk.CENTER,
            font=("Segoe UI", 9, "bold"),
        )
        if self._busy:
            entry.configure(state=tk.DISABLED)
        self._add_workflow_widget(entry, entry_x, y, entry_width)
        committed = {"text": str(total)}

        def commit(_event=None):
            if self._workflow_controls_refreshing or self._busy:
                return
            raw_value = variable.get().strip()
            if raw_value == committed["text"]:
                return
            try:
                value = self._parse_positive_int(raw_value)
            except (TypeError, ValueError) as exc:
                variable.set(committed["text"])
                self._show_workflow_control_error(exc)
                return
            uq_config["realizations"] = value
            WORKFLOW_TRACKER.set_uq_progress(current, value)
            WORKFLOW_TRACKER.note(
                "uq_realizations_updated",
                realizations=value,
            )
            committed["text"] = str(value)
            self._refresh_workflow_state_message()
            self.root.after_idle(self._refresh_workflow_panel)

        entry.bind("<Return>", commit)
        entry.bind("<FocusOut>", commit)
        return entry

    def _draw_editable_image_block(self, x, y, width):
        self._draw_workflow_parameter_title(x, y, "Image properties", width)
        image_defaults = self._get_image_display_defaults()
        is_uq = config.get("task_mode") == "uq_run"
        source_value = "Generate" if config.get("img_src") == "porespy" else "Import"
        source_var = tk.StringVar(value=source_value)
        row_y = y + 17
        font = ("Segoe UI", 10)
        self.flowchart_canvas.create_text(
            x + 2,
            row_y + 1,
            anchor=tk.NW,
            text="Source:",
            fill=self.FG_DIM,
            font=font,
        )

        def select_source():
            source = "porespy" if source_var.get() == "Generate" else "x_ray"
            self._apply_workflow_control_update("set_parameters", {"img_src": source})

        radio_state = tk.DISABLED if self._busy else tk.NORMAL
        generate_radio = tk.Radiobutton(
            self.flowchart_canvas,
            text="Generate",
            value="Generate",
            variable=source_var,
            command=select_source,
            state=radio_state,
            bg=self.BG_TOOL,
            fg=self.FG_DIM,
            activebackground=self.BG_TOOL,
            activeforeground=self.FG_MAIN,
            disabledforeground=self.FG_OPTION,
            selectcolor=self.BG_INPUT,
            bd=0,
            highlightthickness=0,
            font=("Segoe UI", 9),
        )
        import_radio = tk.Radiobutton(
            self.flowchart_canvas,
            text="Import",
            value="Import",
            variable=source_var,
            command=select_source,
            state=tk.DISABLED if is_uq or self._busy else tk.NORMAL,
            bg=self.BG_TOOL,
            fg=self.FG_DIM,
            activebackground=self.BG_TOOL,
            activeforeground=self.FG_MAIN,
            disabledforeground=self.FG_OPTION,
            selectcolor=self.BG_INPUT,
            bd=0,
            highlightthickness=0,
            font=("Segoe UI", 9),
        )
        self._add_workflow_widget(generate_radio, x + 48, row_y, 74, height=20)
        self._add_workflow_widget(import_radio, x + 122, row_y, 62, height=20)

        file_value = image_config.get("image_file") or image_defaults["image_file"]
        file_var = tk.StringVar(value=str(file_value))
        file_x = x + 190
        lower_control_x, lower_control_width = self._workflow_control_geometry(x, width)
        lower_control_right = lower_control_x + lower_control_width
        unit_column_x = lower_control_right + 7
        file_width = max(70, lower_control_right - file_x)
        file_entry = tk.Entry(
            self.flowchart_canvas,
            textvariable=file_var,
            bg=self.BG_INPUT,
            fg=self.FG_MAIN,
            insertbackground=self.FG_MAIN,
            disabledbackground=self.BG_INPUT,
            disabledforeground=self.FG_OPTION,
            relief=tk.FLAT,
            bd=0,
            highlightthickness=1,
            highlightbackground=self.BORDER_COLOR,
            highlightcolor=self.FG_ACCENT,
            font=("Segoe UI", 9),
        )
        import_enabled = config.get("img_src") == "x_ray" and not is_uq and not self._busy
        if not import_enabled:
            file_entry.configure(state=tk.DISABLED)
        self._add_workflow_widget(file_entry, file_x, row_y, file_width, height=20)

        committed_file = {"text": str(file_value)}

        def commit_file(_event=None):
            if not import_enabled or self._workflow_controls_refreshing:
                return
            selected_file = file_var.get().strip()
            if not selected_file or selected_file == committed_file["text"]:
                return
            if self._apply_workflow_control_update(
                "set_image_properties", {"image_file": selected_file}
            ):
                committed_file["text"] = selected_file

        file_entry.bind("<Return>", commit_file)
        file_entry.bind("<FocusOut>", commit_file)

        def browse_image():
            selected_file = filedialog.askopenfilename(
                parent=self.root,
                title="Import image",
                initialdir=str(THIS_DIR / "image"),
                filetypes=(("RAW image", "*.raw"), ("All files", "*.*")),
            )
            if not selected_file:
                return
            file_var.set(selected_file)
            committed_file["text"] = selected_file
            self._apply_workflow_control_update(
                "set_image_properties", {"image_file": selected_file}
            )

        browse_button = tk.Button(
            self.flowchart_canvas,
            text="Browse",
            command=browse_image,
            state=tk.NORMAL if import_enabled else tk.DISABLED,
            bg=self.BG_INPUT,
            fg=self.FG_MAIN,
            activebackground=self.BG_USER,
            activeforeground=self.FG_MAIN,
            disabledforeground=self.FG_OPTION,
            relief=tk.FLAT,
            bd=0,
            anchor=tk.W,
            padx=4,
            font=("Segoe UI", 9),
        )
        browse_x = unit_column_x - 4
        browse_width = 52
        self._add_workflow_widget(
            browse_button,
            browse_x,
            row_y,
            browse_width,
            height=20,
        )

        if config.get("img_src") == "x_ray":
            dimensions = image_config.get("voxels") or image_defaults["voxels"]
        else:
            dimensions = image_config.get("shape") or image_defaults["shape"]
        if isinstance(dimensions, (list, tuple)):
            dimensions = ", ".join(str(value) for value in dimensions)
        rows = (
            (
                "dimensions",
                dimensions,
                self._parse_workflow_dimensions,
                " voxels",
            ),
            (
                "voxel_size",
                image_config.get("voxel_size") or image_defaults["voxel_size"],
                self._parse_positive_float,
                " µm",
            ),
            (
                "porosity",
                image_config.get("porosity")
                if image_config.get("porosity") is not None
                else image_defaults["porosity"],
                self._parse_porosity,
                "",
            ),
            (
                "blobiness",
                image_config.get("blobiness")
                if image_config.get("blobiness") is not None
                else image_defaults["blobiness"],
                self._parse_positive_float,
                "",
            ),
        )
        for index, (key, value, parser, unit) in enumerate(rows, start=1):
            self._draw_workflow_entry_row(
                x,
                row_y + index * 22,
                width,
                self._format_parameter_label(key),
                value,
                "set_image_properties",
                "dimensions" if key == "dimensions" else key,
                parser=parser,
                unit=unit,
                enabled=(
                    config.get("img_src") != "x_ray"
                    or key not in {"porosity", "blobiness"}
                ),
            )

    def _draw_editable_extraction_block(self, x, y, width):
        self._draw_workflow_parameter_title(x, y, "Extraction parameters", width)
        image_defaults = self._get_image_display_defaults()
        row_y = y + 17
        coarse_factor = image_config.get("coarse_factor")
        if coarse_factor is None:
            coarse_factor = image_defaults["coarse_factor"]
        self._draw_workflow_entry_row(
            x,
            row_y,
            width,
            self._format_parameter_label("coarse_factor"),
            coarse_factor,
            "set_image_properties",
            "coarse_factor",
            parser=self._parse_positive_int,
            unit=" voxels",
        )
        for index, key in enumerate(
            ("pore_diameter_type", "throat_diameter_type"), start=1
        ):
            value = image_config.get(key) or image_defaults[key]
            self._draw_workflow_combo_row(
                x,
                row_y + index * 22,
                width,
                self._format_parameter_label(key),
                value,
                ALLOWED_VALUES[key],
                "set_image_properties",
                key,
            )

    def _draw_editable_simulation_block(self, x, y, width):
        self._draw_workflow_parameter_title(x, y, "Model setup", width)
        row_y = y + 17
        self._draw_workflow_combo_row(
            x,
            row_y,
            width,
            self._format_parameter_label("mode"),
            config.get("pnm_mode"),
            ALLOWED_VALUES["pnm_mode"],
            "set_parameters",
            "pnm_mode",
        )
        if config.get("pnm_mode") == "dynamic":
            params = self._read_dyn_params()
            keys = (
                "gas_viscosity",
                "liquid_viscosity",
                "capillary_number",
                "max_pc",
            )
            for index, key in enumerate(keys):
                self._draw_workflow_entry_row(
                    x,
                    row_y + (index + 1) * 22,
                    width,
                    self._format_parameter_label(key),
                    params[key],
                    "set_dyn_params",
                    key,
                    parser=self._parse_positive_float,
                )
            return

        params = self._read_qs_params()
        parsers = {
            "min_pressure": self._parse_nonnegative_float,
            "max_pressure": self._parse_positive_float,
            "pressure_step": self._parse_positive_float,
        }
        for index, key in enumerate(("min_pressure", "max_pressure", "pressure_step")):
            self._draw_workflow_entry_row(
                x,
                row_y + (index + 1) * 22,
                width,
                self._format_parameter_label(key),
                params[key],
                "set_qs_params",
                key,
                parser=parsers[key],
                unit=" Pa",
            )

    def _draw_editable_plot_block(self, x, y, width):
        self._draw_workflow_parameter_title(x, y, "Plot setting", width)
        row_y = y + 17
        if config.get("task_mode") != "single_run":
            return
        self._draw_workflow_combo_row(
            x,
            row_y,
            width,
            self._format_parameter_label("quantity"),
            config.get("quantity"),
            ALLOWED_VALUES["quantity"],
            "set_parameters",
            "quantity",
        )
        self._draw_workflow_combo_row(
            x,
            row_y + 22,
            width,
            self._format_parameter_label("vtk"),
            config.get("vtk"),
            ALLOWED_VALUES["vtk"],
            "set_parameters",
            "vtk",
        )

    def _workflow_simulation_parameter_block(self) -> tuple[str, list[str]]:
        if config.get("pnm_mode") == "dynamic":
            return (
                "Model setup",
                [self._format_parameter_line("mode", config.get("pnm_mode"))] + [
                    f"  {self._format_parameter_label(key)}: {value}"
                    for key, value in self._read_dyn_params().items()
                ],
            )
        return (
            "Model setup",
            [self._format_parameter_line("mode", config.get("pnm_mode"))] + [
                self._format_qs_param_line(key, value)
                for key, value in self._read_qs_params().items()
            ],
        )

    def _workflow_plot_parameter_block(self) -> tuple[str, list[str]]:
        lines = []
        if config.get("task_mode") == "single_run":
            lines.extend([
                self._format_parameter_line("quantity", config.get("quantity")),
                self._format_parameter_line("vtk", config.get("vtk")),
            ])
        return (
            "Plot setting",
            lines,
        )

    def _draw_flowchart_map(self):
        if self.flowchart_canvas is None:
            return

        actual_cw = self.flowchart_canvas.winfo_width()
        cw = actual_cw if actual_cw > 10 else 870
        ch = max(self.flowchart_canvas.winfo_height(), 455)
        snapshot = WORKFLOW_TRACKER.snapshot()
        highlighted_nodes = set(snapshot.get("highlighted_nodes", []))
        active_nodes = set(snapshot.get("active_nodes", []))
        is_uq = config.get("task_mode") == "uq_run"

        self._clear_workflow_controls()
        self.flowchart_canvas.delete("all")

        box_h = 63
        margin = 14
        box_x = margin
        small_w = 160
        small_gap = 12
        pair_w = small_w * 2 + small_gap
        center_w = pair_w
        center_x = box_x
        param_x = box_x + pair_w + 12
        param_w = max(260, cw - param_x - margin)
        top_y = 52
        # Keep post processing close to the action strip immediately below the canvas.
        plot_y = max(top_y + 326, ch - 8 - box_h)
        row_gap = (plot_y - top_y) / 3
        extract_y = int(round(top_y + row_gap))
        sim_y = int(round(top_y + row_gap * 2))

        if is_uq:
            uq_top = top_y - 30
            self.flowchart_canvas.create_text(
                margin,
                uq_top + 5,
                anchor=tk.NW,
                text="Repeat for each realization",
                fill=self.FG_DIM,
                font=("Segoe UI", 10, "italic"),
            )
            gen_x0 = box_x
            gen_x1 = box_x + center_w
            current = snapshot.get("uq_realization_index") or 0
            total = snapshot.get("uq_total") or uq_config["realizations"]
            self._draw_flowchart_node(
                gen_x0,
                top_y,
                gen_x1,
                top_y + box_h,
                "Image generation",
                "image_generation",
                active_nodes,
                highlighted_nodes,
            )
            self._draw_uq_realization_total(
                current,
                total,
                gen_x1,
                uq_top + 1,
            )
            self._draw_arrow(
                gen_x0 + center_w / 2,
                top_y + box_h,
                center_x + center_w / 2,
                extract_y,
                from_node="image_generation",
                to_node="pore_extraction",
                highlighted_nodes=highlighted_nodes,
                active_nodes=active_nodes,
            )
        else:
            left_col_x = box_x + small_w / 2
            right_col_x = box_x + small_w + small_gap + small_w / 2
            self._draw_flowchart_node(
                box_x,
                top_y,
                box_x + small_w,
                top_y + box_h,
                "Image generation",
                "image_generation",
                active_nodes,
                highlighted_nodes,
            )
            self._draw_flowchart_node(
                box_x + small_w + small_gap,
                top_y,
                box_x + pair_w,
                top_y + box_h,
                "Image import",
                "image_import",
                active_nodes,
                highlighted_nodes,
            )
            self._draw_arrow(
                left_col_x,
                top_y + box_h,
                left_col_x,
                extract_y,
                from_node="image_generation",
                to_node="pore_extraction",
                highlighted_nodes=highlighted_nodes,
                active_nodes=active_nodes,
            )
            self._draw_arrow(
                right_col_x,
                top_y + box_h,
                right_col_x,
                extract_y,
                from_node="image_import",
                to_node="pore_extraction",
                highlighted_nodes=highlighted_nodes,
                active_nodes=active_nodes,
            )

        self._draw_editable_image_block(param_x, top_y - 3, param_w)

        self._draw_flowchart_node(
            center_x,
            extract_y,
            center_x + center_w,
            extract_y + box_h,
            "Pore network extraction",
            "pore_extraction",
            active_nodes,
            highlighted_nodes,
        )
        self._draw_editable_extraction_block(param_x, extract_y - 2, param_w)
        left_col_x = box_x + small_w / 2
        right_col_x = box_x + small_w + small_gap + small_w / 2
        self._draw_arrow(
            left_col_x,
            extract_y + box_h,
            left_col_x,
            sim_y,
            from_node="pore_extraction",
            to_node="quasi_static",
            highlighted_nodes=highlighted_nodes,
            active_nodes=active_nodes,
        )
        self._draw_arrow(
            right_col_x,
            extract_y + box_h,
            right_col_x,
            sim_y,
            from_node="pore_extraction",
            to_node="dynamic",
            highlighted_nodes=highlighted_nodes,
            active_nodes=active_nodes,
        )

        self._draw_flowchart_node(
            box_x,
            sim_y,
            box_x + small_w,
            sim_y + box_h,
            "Quasi-static PNM",
            "quasi_static",
            active_nodes,
            highlighted_nodes,
        )
        self._draw_flowchart_node(
            box_x + small_w + small_gap,
            sim_y,
            box_x + pair_w,
            sim_y + box_h,
            "Dynamic PNM",
            "dynamic",
            active_nodes,
            highlighted_nodes,
        )
        self._draw_editable_simulation_block(param_x, sim_y - 2, param_w)

        plot_w = 140
        plot_x0 = box_x + (center_w - plot_w) / 2
        plot_x1 = plot_x0 + plot_w
        self._draw_arrow(
            box_x + small_w / 2,
            sim_y + box_h,
            plot_x0 + plot_w / 2,
            plot_y,
            from_node="quasi_static",
            to_node="plot_result",
            highlighted_nodes=highlighted_nodes,
            active_nodes=active_nodes,
        )
        self._draw_arrow(
            box_x + small_w + small_gap + small_w / 2,
            sim_y + box_h,
            plot_x0 + plot_w / 2,
            plot_y,
            from_node="dynamic",
            to_node="plot_result",
            highlighted_nodes=highlighted_nodes,
            active_nodes=active_nodes,
        )
        self._draw_flowchart_node(
            plot_x0,
            plot_y,
            plot_x1,
            plot_y + box_h,
            "Post processing",
            "plot_result",
            active_nodes,
            highlighted_nodes,
        )
        self._draw_editable_plot_block(param_x, plot_y - 2, param_w)

    def _refresh_flowchart_preview(self):
        if self.flowchart_canvas is None:
            return

        self._flowchart_photo = None
        self._draw_flowchart_map()

    def _show_flowchart_on_startup(self):
        if self._flowchart_startup_checked:
            return
        self._flowchart_startup_checked = True
        self.root.update_idletasks()
        self._refresh_flowchart_preview()
        self._refresh_step_indicators()

    # ----- UI construction -------------------------------------------------
    def _build_ui(self):
        # Main container: chat plus workflow tracker
        main_frame = tk.Frame(self.root, bg=self.BG_DARK)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = tk.Frame(main_frame, bg=self.BG_DARK)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.chat_panel = left_frame

        workflow_frame = tk.Frame(
            main_frame,
            bg=self.BG_CHAT,
            width=900,
            highlightthickness=1,
            highlightbackground=self.BORDER_COLOR,
        )
        workflow_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=8)
        workflow_frame.pack_propagate(False)

        # Title bar
        title_bar = tk.Frame(left_frame, bg=self.BG_DARK, pady=8)
        title_bar.pack(fill=tk.X)

        tk.Label(
            title_bar,
            text="  AutoPore",
            font=("Segoe UI", 18, "bold"),
            bg=self.BG_DARK,
            fg=self.FG_ACCENT,
            anchor="w",
        ).pack(side=tk.LEFT, padx=8)

        tk.Label(
            title_bar,
            text="Natural Language-driven Pore-Scale Modeling Workflow",
            font=("Segoe UI", 12),
            bg=self.BG_DARK,
            fg=self.FG_DIM,
            anchor="w",
        ).pack(side=tk.LEFT, padx=4)

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "Pore.TCombobox",
            fieldbackground=self.BG_INPUT,
            background=self.BG_INPUT,
            foreground=self.FG_MAIN,
            selectforeground=self.FG_MAIN,
            selectbackground=self.BG_INPUT,
            arrowcolor=self.FG_MAIN,
        )
        style.map(
            "Pore.TCombobox",
            foreground=[
                ("readonly", self.FG_MAIN),
                ("active", self.FG_MAIN),
                ("disabled", self.FG_DIM),
            ],
            fieldbackground=[
                ("readonly", self.BG_INPUT),
                ("active", self.BG_INPUT),
                ("disabled", self.BG_INPUT),
            ],
            selectforeground=[
                ("readonly", self.FG_MAIN),
                ("active", self.FG_MAIN),
            ],
            selectbackground=[
                ("readonly", self.BG_INPUT),
                ("active", self.BG_INPUT),
            ],
        )
        style.configure(
            "Pore.Workflow.TCombobox",
            fieldbackground=self.BG_INPUT,
            background=self.BG_INPUT,
            foreground=self.FG_MAIN,
            selectforeground=self.FG_MAIN,
            selectbackground=self.BG_INPUT,
            arrowcolor=self.FG_MAIN,
        )
        style.map(
            "Pore.Workflow.TCombobox",
            foreground=[
                ("disabled", self.FG_OPTION),
                ("readonly", self.FG_MAIN),
                ("active", self.FG_MAIN),
            ],
            fieldbackground=[
                ("disabled", self.BG_INPUT),
                ("readonly", self.BG_INPUT),
                ("active", self.BG_INPUT),
            ],
            selectforeground=[
                ("disabled", self.FG_OPTION),
                ("readonly", self.FG_MAIN),
                ("active", self.FG_MAIN),
            ],
            selectbackground=[
                ("disabled", self.BG_INPUT),
                ("readonly", self.BG_INPUT),
                ("active", self.BG_INPUT),
            ],
            arrowcolor=[
                ("disabled", self.FG_OPTION),
                ("readonly", self.FG_MAIN),
                ("active", self.FG_MAIN),
            ],
        )
        style.configure(
            "Pore.WorkflowDim.TCombobox",
            fieldbackground=self.BG_INPUT,
            background=self.BG_INPUT,
            foreground=self.FG_OPTION,
            selectforeground=self.FG_OPTION,
            selectbackground=self.BG_INPUT,
            arrowcolor=self.FG_OPTION,
        )
        style.map(
            "Pore.WorkflowDim.TCombobox",
            foreground=[
                ("readonly", self.FG_OPTION),
                ("active", self.FG_OPTION),
                ("disabled", self.FG_OPTION),
            ],
            fieldbackground=[
                ("readonly", self.BG_INPUT),
                ("active", self.BG_INPUT),
                ("disabled", self.BG_INPUT),
            ],
            selectforeground=[
                ("readonly", self.FG_OPTION),
                ("active", self.FG_OPTION),
                ("disabled", self.FG_OPTION),
            ],
            selectbackground=[
                ("readonly", self.BG_INPUT),
                ("active", self.BG_INPUT),
                ("disabled", self.BG_INPUT),
            ],
            arrowcolor=[
                ("readonly", self.FG_OPTION),
                ("active", self.FG_OPTION),
                ("disabled", self.FG_OPTION),
            ],
        )

        llm_bar = tk.Frame(left_frame, bg=self.BG_CHAT, padx=9, pady=7)
        llm_bar.pack(fill=tk.X, padx=8, pady=(0, 6))
        tk.Label(
            llm_bar,
            text="LLM",
            font=("Segoe UI", 10, "bold"),
            bg=self.BG_CHAT,
            fg=self.FG_MAIN,
        ).grid(row=0, column=0, padx=(0, 5))
        provider_combo = ttk.Combobox(
            llm_bar,
            textvariable=self.provider_var,
            values=VALID_PROVIDERS,
            state="readonly",
            width=10,
            style="Pore.TCombobox",
        )
        provider_combo.grid(row=0, column=1, padx=(0, 8))
        provider_combo.bind("<<ComboboxSelected>>", self._provider_changed)
        tk.Label(
            llm_bar,
            text="Model",
            bg=self.BG_CHAT,
            fg=self.FG_DIM,
        ).grid(row=0, column=2, padx=(0, 4))
        self.model_combo = ttk.Combobox(
            llm_bar,
            textvariable=self.model_var,
            values=(),
            state="normal",
            width=18,
            style="Pore.TCombobox",
        )
        self.model_combo.grid(row=0, column=3, sticky="ew", padx=(0, 5))
        self.model_combo.bind("<<ComboboxSelected>>", self._llm_fields_changed)
        self.model_combo.bind("<Return>", self._llm_fields_changed)
        self.model_combo.bind("<FocusOut>", self._llm_fields_changed)
        tk.Button(
            llm_bar,
            text="Detect",
            command=self._detect_local_model,
            bg=self.BG_INPUT,
            fg=self.FG_MAIN,
            activebackground=self.BG_USER,
            activeforeground=self.FG_MAIN,
            relief=tk.FLAT,
            padx=8,
            pady=3,
        ).grid(row=0, column=4, padx=(0, 8))
        tk.Label(
            llm_bar,
            text="Base URL",
            bg=self.BG_CHAT,
            fg=self.FG_DIM,
        ).grid(row=0, column=5, padx=(0, 4))
        base_url_entry = tk.Entry(
            llm_bar,
            textvariable=self.base_url_var,
            width=22,
            bg=self.BG_INPUT,
            fg=self.FG_MAIN,
            insertbackground=self.FG_MAIN,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=self.BORDER_COLOR,
            highlightcolor=self.FG_ACCENT,
        )
        base_url_entry.grid(row=0, column=6, sticky="ew")
        base_url_entry.bind("<Return>", self._llm_fields_changed)
        base_url_entry.bind("<FocusOut>", self._llm_fields_changed)
        llm_bar.grid_columnconfigure(3, weight=1)
        llm_bar.grid_columnconfigure(6, weight=1)

        # Chat display
        self.chat_display = scrolledtext.ScrolledText(
            left_frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
            bg=self.BG_CHAT,
            fg=self.FG_MAIN,
            font=("Consolas", 13),
            relief=tk.FLAT,
            insertbackground=self.FG_MAIN,
            selectbackground="#585b70",
            padx=12,
            pady=8,
            borderwidth=0,
            highlightthickness=0,
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))

        # Tags
        self.chat_display.tag_configure(
            "user_label", foreground=self.FG_ACCENT, font=("Segoe UI", 13, "bold")
        )
        self.chat_display.tag_configure(
            "agent_label", foreground=self.FG_GREEN, font=("Segoe UI", 13, "bold")
        )
        self.chat_display.tag_configure(
            "user_msg", foreground=self.FG_MAIN, font=("Consolas", 13)
        )
        self.chat_display.tag_configure(
            "agent_msg", foreground=self.FG_MAIN, font=("Consolas", 13)
        )
        self.chat_display.tag_configure(
            "tool_msg", foreground=self.FG_TOOL, font=("Consolas", 12, "italic")
        )
        self.chat_display.tag_configure(
            "status_msg", foreground=self.FG_YELLOW, font=("Consolas", 12, "italic")
        )
        self.chat_display.tag_configure(
            "error_msg", foreground=self.FG_RED, font=("Consolas", 13, "bold")
        )
        self.chat_display.tag_configure("separator", foreground="#585b70")

        tk.Label(
            workflow_frame,
            text="Workflow Tracker",
            font=("Segoe UI", 18, "bold"),
            bg=self.BG_CHAT,
            fg=self.FG_ACCENT,
            anchor=tk.CENTER,
        ).pack(fill=tk.X, padx=10, pady=(10, 4))

        task_selector = tk.Frame(workflow_frame, bg=self.BG_CHAT)
        task_selector.pack(fill=tk.X, padx=10, pady=(0, 8))

        self.task_single_btn = tk.Radiobutton(
            task_selector,
            text="Deterministic Simulation",
            variable=self.task_mode_var,
            value="single_run",
            command=self._task_mode_selected,
            font=("Segoe UI", 11, "bold"),
            bg=self.BG_CHAT,
            fg=self.FG_ACCENT,
            activebackground=self.BG_CHAT,
            activeforeground=self.FG_ACCENT,
            selectcolor=self.BG_INPUT,
            bd=0,
            highlightthickness=0,
            padx=0,
            pady=2,
        )
        self.task_single_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))

        self.task_uq_btn = tk.Radiobutton(
            task_selector,
            text="Uncertainty Quantification",
            variable=self.task_mode_var,
            value="uq_run",
            command=self._task_mode_selected,
            font=("Segoe UI", 11, "bold"),
            bg=self.BG_CHAT,
            fg=self.FG_DIM,
            activebackground=self.BG_CHAT,
            activeforeground=self.FG_DIM,
            selectcolor=self.BG_INPUT,
            bd=0,
            highlightthickness=0,
            padx=0,
            pady=2,
        )
        self.task_uq_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

        self.flowchart_canvas = tk.Canvas(
            workflow_frame,
            height=455,
            bg=self.BG_TOOL,
            highlightthickness=0,
            relief=tk.FLAT,
        )

        file_logs_frame = tk.Frame(workflow_frame, bg=self.BG_CHAT)
        file_logs_frame.pack(side=tk.BOTTOM, fill=tk.X)

        file_logs_header = tk.Frame(file_logs_frame, bg=self.BG_CHAT)
        file_logs_header.pack(fill=tk.X, padx=10, pady=(0, 4))
        tk.Label(
            file_logs_header,
            text="",
            bg=self.BG_CHAT,
            width=13,
        ).pack(side=tk.LEFT)
        tk.Label(
            file_logs_header,
            text="File Logs",
            font=("Segoe UI", 18, "bold"),
            bg=self.BG_CHAT,
            fg=self.FG_ACCENT,
            anchor=tk.CENTER,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.clear_history_btn = tk.Button(
            file_logs_header,
            text="Clear history",
            font=("Segoe UI", 10, "bold"),
            bg=self.BG_INPUT,
            fg=self.FG_MAIN,
            activebackground="#585b70",
            activeforeground=self.FG_MAIN,
            relief=tk.FLAT,
            width=13,
            cursor="hand2",
            command=self._clear_history,
        )
        self.clear_history_btn.pack(side=tk.RIGHT)

        self.generated_files_display = scrolledtext.ScrolledText(
            file_logs_frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
            bg=self.BG_TOOL,
            fg=self.FG_DIM,
            font=("Consolas", 10),
            relief=tk.FLAT,
            insertbackground=self.FG_MAIN,
            selectbackground="#585b70",
            padx=10,
            pady=8,
            borderwidth=0,
            highlightthickness=0,
            height=5,
        )
        self.generated_files_display.pack(fill=tk.X, padx=10, pady=(0, 8))

        # Keep the controls in one framed strip matching the 332 px flowchart column.
        workflow_actions_frame = tk.Frame(
            workflow_frame,
            bg=self.BG_TOOL,
            width=332,
            height=43,
            highlightbackground=self.BG_USER,
            highlightthickness=1,
        )
        workflow_actions_frame.pack(
            side=tk.BOTTOM,
            anchor=tk.W,
            padx=(24, 0),
            pady=(0, 8),
        )
        workflow_actions_frame.pack_propagate(False)

        workflow_actions = tk.Frame(workflow_actions_frame, bg=self.BG_TOOL)
        workflow_actions.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        workflow_actions.grid_propagate(False)
        workflow_actions.grid_columnconfigure((0, 1, 2), weight=1, uniform="workflow_action")
        workflow_actions.grid_rowconfigure(0, weight=1)

        self.run_next_btn = tk.Button(
            workflow_actions,
            text="Run next",
            command=self._run_next_from_panel,
            font=("Segoe UI", 9, "bold"),
            bg=self.FG_ACCENT,
            fg=self.BG_DARK,
            activebackground="#74a8f5",
            activeforeground=self.BG_DARK,
            relief=tk.FLAT,
            cursor="hand2",
            padx=6,
            pady=3,
        )
        self.run_next_btn.grid(row=0, column=0, sticky="nsew", padx=(0, 3))

        self.run_full_btn = tk.Button(
            workflow_actions,
            text="Run full / UQ",
            command=self._run_full_from_panel,
            font=("Segoe UI", 9, "bold"),
            bg=self.BG_INPUT,
            fg=self.FG_MAIN,
            activebackground=self.BG_USER,
            activeforeground=self.FG_MAIN,
            relief=tk.FLAT,
            cursor="hand2",
            padx=6,
            pady=3,
        )
        self.run_full_btn.grid(row=0, column=1, sticky="nsew", padx=3)

        self.start_over_btn = tk.Button(
            workflow_actions,
            text="Start over",
            command=self._start_workflow_over_from_panel,
            font=("Segoe UI", 9, "bold"),
            bg=self.BG_INPUT,
            fg=self.FG_MAIN,
            activebackground=self.BG_USER,
            activeforeground=self.FG_MAIN,
            relief=tk.FLAT,
            cursor="hand2",
            padx=6,
            pady=3,
        )
        self.start_over_btn.grid(row=0, column=2, sticky="nsew", padx=(3, 0))

        self.flowchart_canvas.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=(0, 2),
        )
        self.flowchart_canvas.bind(
            "<Configure>",
            lambda event: self._refresh_flowchart_preview(),
        )

        # Input area
        input_frame = tk.Frame(left_frame, bg=self.BG_DARK, pady=6)
        input_frame.pack(fill=tk.X, padx=8)

        self.user_input = tk.Text(
            input_frame,
            font=("Consolas", 14),
            bg=self.BG_INPUT,
            fg=self.FG_MAIN,
            insertbackground=self.FG_MAIN,
            relief=tk.FLAT,
            highlightthickness=1,
            height=3,
            width=52,
            undo=True,
            highlightbackground=self.BORDER_COLOR,
            highlightcolor=self.FG_ACCENT,
        )
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.user_input.bind("<Return>", self._on_send_key)
        self.user_input.bind("<Shift-Return>", lambda e: None)

        self.send_btn = tk.Button(
            input_frame,
            text="Send",
            font=("Segoe UI", 13, "bold"),
            bg=self.FG_ACCENT,
            fg=self.BG_DARK,
            activebackground="#b4d0fb",
            relief=tk.FLAT,
            width=13,
            padx=18,
            pady=4,
            cursor="hand2",
            command=self._on_send,
        )
        self.send_btn.pack(side=tk.RIGHT)

    # ----- Figure popup window ---------------------------------------------
    def _ensure_figure_window(self):
        self.root.update_idletasks()
        main_x = self.root.winfo_x()
        main_y = self.root.winfo_y()
        main_h = self.root.winfo_height()
        chat_w = 0
        if self.chat_panel is not None:
            chat_w = self.chat_panel.winfo_width()
        if chat_w <= 1 and self.chat_display is not None:
            chat_w = self.chat_display.winfo_width()
        fig_w = max(480, chat_w)
        fig_h = main_h
        fig_x = max(0, main_x)
        fig_y = max(0, main_y)

        if self.fig_window is not None and self.fig_window.winfo_exists():
            self.fig_window.geometry(f"{fig_w}x{fig_h}+{fig_x}+{fig_y}")
            return

        self.fig_window = tk.Toplevel(self.root)
        self.fig_window.title("AutoPore Figures")
        self.fig_window.geometry(f"{fig_w}x{fig_h}+{fig_x}+{fig_y}")
        self.fig_window.minsize(480, 700)
        self.fig_window.configure(bg=self.BG_DARK)
        self.fig_window.protocol("WM_DELETE_WINDOW", self._on_close_figure_window)

        fig_frame = tk.Frame(self.fig_window, bg=self.BG_DARK)
        fig_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        fig_header = tk.Frame(fig_frame, bg=self.BG_DARK)
        fig_header.pack(fill=tk.X, padx=8, pady=(8, 2))

        tk.Label(
            fig_header,
            text="  Figures",
            font=("Segoe UI", 15, "bold"),
            bg=self.BG_DARK,
            fg=self.FG_ACCENT,
            anchor="w",
        ).pack(side=tk.LEFT)

        self.fig_counter_label = tk.Label(
            fig_header,
            text="",
            font=("Segoe UI", 12),
            bg=self.BG_DARK,
            fg=self.FG_DIM,
        )
        self.fig_counter_label.pack(side=tk.RIGHT, padx=4)

        self.fig_next_btn = tk.Button(
            fig_header,
            text="▶",
            font=("Segoe UI", 13),
            bg=self.BG_INPUT,
            fg=self.FG_MAIN,
            activebackground="#585b70",
            relief=tk.FLAT,
            padx=6,
            cursor="hand2",
            command=self._fig_next,
        )
        self.fig_next_btn.pack(side=tk.RIGHT, padx=2)

        self.fig_prev_btn = tk.Button(
            fig_header,
            text="◀",
            font=("Segoe UI", 13),
            bg=self.BG_INPUT,
            fg=self.FG_MAIN,
            activebackground="#585b70",
            relief=tk.FLAT,
            padx=6,
            cursor="hand2",
            command=self._fig_prev,
        )
        self.fig_prev_btn.pack(side=tk.RIGHT, padx=2)

        sep_fig = tk.Frame(fig_frame, bg=self.BORDER_COLOR, height=1)
        sep_fig.pack(fill=tk.X, padx=12, pady=4)

        self.fig_canvas = tk.Canvas(
            fig_frame,
            bg=self.BG_CHAT,
            highlightthickness=0,
            relief=tk.FLAT,
        )
        self.fig_canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))

        self.fig_open_btn = tk.Button(
            fig_frame,
            text="",
            font=("Segoe UI", 12, "bold"),
            bg=self.BG_INPUT,
            fg=self.FG_MAIN,
            activebackground="#585b70",
            relief=tk.FLAT,
            cursor="hand2",
            command=self._open_current_figure_externally,
        )

        self.fig_name_label = tk.Label(
            fig_frame,
            text="No figures yet",
            font=("Consolas", 11),
            bg=self.BG_DARK,
            fg=self.FG_DIM,
            anchor="center",
        )
        self.fig_name_label.pack(fill=tk.X, padx=8, pady=(0, 6))

        self.fig_canvas.bind("<Configure>", self._on_fig_canvas_resize)
        self.fig_canvas.bind("<Button-1>", self._on_figure_manual_interaction)

    def _on_close_figure_window(self):
        self._stop_figure_slideshow()
        self._stop_gif_animation()

        if self.fig_window is not None:
            try:
                self.fig_window.destroy()
            except Exception:
                pass

        self.fig_window = None
        self.fig_canvas = None
        self.fig_name_label = None
        self.fig_counter_label = None
        self.fig_prev_btn = None
        self.fig_next_btn = None
        self.fig_open_btn = None
        self._photo_image = None

    # ----- Figure viewer helpers -------------------------------------------
    def _queue_figure_popup(self, path: str):
        self.root.after(0, self._handle_new_figure, _normalize_figure_path(path), 0)

    def _handle_new_figure(self, path: str, retry_count: int = 0):
        path = _normalize_figure_path(path)
        if not os.path.isfile(path):
            if retry_count < 20:
                self.root.after(200, self._handle_new_figure, path, retry_count + 1)
            return

        if path not in _shown_figures:
            _shown_figures.append(path)

        existing_3d_index = next(
            (
                index
                for index, existing in enumerate(self._figure_paths)
                if Path(existing).name.lower() == "3d_image.png"
            ),
            None,
        )
        if Path(path).name.lower() == "3d_image.png" and existing_3d_index is not None:
            self._figure_paths[existing_3d_index] = path
            self._figure_index = existing_3d_index
        elif path in self._figure_paths:
            self._figure_index = self._figure_paths.index(path)
        else:
            self._figure_paths.append(path)
            self._figure_index = len(self._figure_paths) - 1

        self._ensure_figure_window()
        self.fig_window.deiconify()
        self.fig_window.lift()
        self.fig_window.focus_force()
        self._show_current_figure()

    def _reset_live_figures(self):
        self._stop_figure_slideshow()
        self._figure_paths = []
        self._figure_index = 0

        if self.fig_window is not None and self.fig_window.winfo_exists():
            self._show_current_figure()

    def _load_figures(self, paths: list[str]):
        normalized_paths = [_normalize_figure_path(p) for p in paths]
        self._figure_paths = []
        has_3d_preview = False
        for path in normalized_paths:
            if not os.path.isfile(path) or path in self._figure_paths:
                continue
            is_3d_preview = Path(path).name.lower() == "3d_image.png"
            if is_3d_preview and has_3d_preview:
                continue
            self._figure_paths.append(path)
            has_3d_preview = has_3d_preview or is_3d_preview
        self._figure_index = 0

        if not self._figure_paths:
            return

        # Always repopup when figures are loaded
        self._ensure_figure_window()
        self.fig_window.deiconify()
        self.fig_window.lift()
        self.fig_window.focus_force()

        self._show_current_figure()
        self._start_figure_slideshow()

    def _fig_prev(self):
        self._stop_figure_slideshow()
        if self._figure_paths and self._figure_index > 0:
            self._figure_index -= 1
            self._show_current_figure()

    def _fig_next(self):
        self._stop_figure_slideshow()
        if self._figure_paths and self._figure_index < len(self._figure_paths) - 1:
            self._figure_index += 1
            self._show_current_figure()

    def _on_figure_manual_interaction(self, event=None):
        self._stop_figure_slideshow()
        if not self._figure_paths:
            return
        path = self._figure_paths[self._figure_index]
        target = self._figure_external_target(path)
        if target != path:
            self._open_path_externally(target)

    def _start_figure_slideshow(self):
        self._stop_figure_slideshow()
        if len(self._figure_paths) <= 1:
            return
        self._figure_slide_after_id = self.root.after(
            self._current_figure_slide_pause_ms(),
            self._advance_figure_slideshow,
        )

    def _current_figure_slide_pause_ms(self) -> int:
        if self._figure_paths and self._figure_index == len(self._figure_paths) - 1:
            return self._figure_slide_last_pause_ms
        return self._figure_slide_pause_ms

    def _advance_figure_slideshow(self):
        self._figure_slide_after_id = None
        if self.fig_window is None or not self.fig_window.winfo_exists():
            return
        if not self._figure_paths:
            return
        self._figure_index = (self._figure_index + 1) % len(self._figure_paths)
        self._show_current_figure(animate=True)
        delay = self._figure_transition_ms + self._current_figure_slide_pause_ms()
        self._figure_slide_after_id = self.root.after(delay, self._advance_figure_slideshow)

    def _stop_figure_slideshow(self, settle_transition: bool = True):
        if self._figure_slide_after_id is not None:
            try:
                self.root.after_cancel(self._figure_slide_after_id)
            except Exception:
                pass
            self._figure_slide_after_id = None
        if self._figure_transition_after_id is not None:
            try:
                self.root.after_cancel(self._figure_transition_after_id)
            except Exception:
                pass
            self._figure_transition_after_id = None
            if settle_transition and self._figure_paths:
                self._show_current_figure(animate=False)

    def _show_current_figure(self, animate: bool = False):
        if self.fig_window is None or not self.fig_window.winfo_exists():
            return
        if self.fig_canvas is None:
            return

        self._stop_gif_animation()

        if not self._figure_paths:
            self.fig_canvas.delete("all")
            self._photo_image = None
            self._transition_photo_image = None
            self._hide_external_open_button()
            if self.fig_counter_label is not None:
                self.fig_counter_label.configure(text="")
            if self.fig_name_label is not None:
                self.fig_name_label.configure(text="No figures yet")
            return

        while self._figure_paths:
            self._figure_index = min(self._figure_index, len(self._figure_paths) - 1)
            original_path = self._figure_paths[self._figure_index]
            resolved_path = self._resolve_figure_path(original_path)
            if resolved_path is not None:
                self._figure_paths[self._figure_index] = resolved_path
                break
            self._figure_paths.pop(self._figure_index)

        if not self._figure_paths:
            self.fig_canvas.delete("all")
            self._photo_image = None
            self._transition_photo_image = None
            self._hide_external_open_button()
            if self.fig_counter_label is not None:
                self.fig_counter_label.configure(text="")
            if self.fig_name_label is not None:
                self.fig_name_label.configure(text="No figures yet")
            return

        idx = self._figure_index
        total = len(self._figure_paths)
        path = self._figure_paths[idx]

        if self.fig_counter_label is not None:
            self.fig_counter_label.configure(text=f"{idx + 1} / {total}")
        if self.fig_name_label is not None:
            self.fig_name_label.configure(text=Path(path).name)

        self._render_image(path, animate=animate)

    def _render_image(self, path: str, animate: bool = False):
        if self.fig_canvas is None:
            return

        self._stop_gif_animation()
        if self._figure_transition_after_id is not None:
            try:
                self.root.after_cancel(self._figure_transition_after_id)
            except Exception:
                pass
            self._figure_transition_after_id = None

        cw = self.fig_canvas.winfo_width()
        ch = self.fig_canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        try:
            if not animate or not self.fig_canvas.find_withtag("figure_display"):
                self.fig_canvas.delete("all")
                item_id, photo = self._create_figure_canvas_item(
                    path,
                    cw // 2,
                    ch // 2,
                    "figure_display",
                )
                self._photo_image = photo
                self._transition_photo_image = None
            else:
                old_items = self.fig_canvas.find_withtag("figure_display")
                item_id, photo = self._create_figure_canvas_item(
                    path,
                    cw + cw // 2,
                    ch // 2,
                    "figure_display_new",
                )
                self._transition_photo_image = photo
                self._animate_figure_swap(old_items, item_id, cw, ch, 0)

        except Exception as exc:
            self.fig_canvas.delete("all")
            self._photo_image = None
            self._transition_photo_image = None
            self.fig_canvas.create_text(
                cw // 2,
                ch // 2,
                text=f"Cannot display image\n{exc}",
                fill=self.FG_DIM,
                font=("Consolas", 12),
                tags=("figure_display",),
            )

    def _create_figure_canvas_item(self, path: str, x: int, y: int, tag: str):
        Image, ImageTk = _load_pillow()
        ext = Path(path).suffix.lower()
        if ext == ".gif":
            self._show_external_open_button(path)
            item_id = self.fig_canvas.create_text(
                x,
                y,
                text="Click link below to open GIF file",
                fill=self.FG_DIM,
                font=("Consolas", 10),
                justify=tk.CENTER,
                tags=(tag,),
            )
            return item_id, None

        target = self._figure_external_target(path)
        if target != path:
            self._show_external_open_button(target)
        else:
            self._hide_external_open_button()
        img = Image.open(path)
        cw = max(self.fig_canvas.winfo_width(), 10)
        ch = max(self.fig_canvas.winfo_height(), 10)
        img.thumbnail((cw, ch), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        item_id = self.fig_canvas.create_image(
            x,
            y,
            anchor=tk.CENTER,
            image=photo,
            tags=(tag,),
        )
        return item_id, photo

    def _animate_figure_swap(self, old_items, new_item, cw: int, ch: int, frame: int):
        if self.fig_canvas is None:
            return

        frames = max(int(self._figure_transition_frames), 1)
        progress = min(frame / frames, 1.0)
        eased = progress * progress * (3 - 2 * progress)
        old_x = cw // 2 - int(cw * eased)
        new_x = cw + cw // 2 - int(cw * eased)
        y = ch // 2

        for item in old_items:
            try:
                self.fig_canvas.coords(item, old_x, y)
            except Exception:
                pass
        try:
            self.fig_canvas.coords(new_item, new_x, y)
        except Exception:
            return

        if frame < frames:
            frame_delay = max(1, int(self._figure_transition_ms / frames))
            self._figure_transition_after_id = self.root.after(
                frame_delay,
                self._animate_figure_swap,
                old_items,
                new_item,
                cw,
                ch,
                frame + 1,
            )
            return

        self._figure_transition_after_id = None
        for item in old_items:
            try:
                self.fig_canvas.delete(item)
            except Exception:
                pass
        self.fig_canvas.itemconfigure(new_item, tags=("figure_display",))
        self._photo_image = self._transition_photo_image
        self._transition_photo_image = None

    def _show_external_open_button(self, path: str):
        if self.fig_open_btn is None:
            return
        self.fig_open_btn.configure(text=f"Open GIF file: {Path(path).name}")
        self.fig_open_btn.pack(fill=tk.X, padx=8, pady=(0, 4))

    def _hide_external_open_button(self):
        if self.fig_open_btn is None:
            return
        self.fig_open_btn.pack_forget()

    def _open_current_figure_externally(self):
        self._stop_figure_slideshow()
        if not self._figure_paths:
            return
        path = self._figure_external_target(self._figure_paths[self._figure_index])
        self._open_path_externally(path)

    def _figure_external_target(self, path: str) -> str:
        path_obj = Path(path)
        if path_obj.suffix.lower() == ".gif":
            return str(path_obj)
        if path_obj.name.startswith("ps_") and path_obj.name.endswith("_last.png"):
            gif_path = path_obj.with_name(path_obj.name.replace("_last.png", ".gif"))
            if gif_path.is_file():
                return str(gif_path)
        return str(path_obj)

    def _open_path_externally(self, path: str):
        if os.path.isfile(path):
            try:
                os.startfile(path)  # type: ignore[attr-defined]
            except (AttributeError, OSError) as exc:
                messagebox.showerror("Could not open figure", str(exc), parent=self.root)

    def _render_gif(self, path: str, cw: int, ch: int):
        Image, ImageTk = _load_pillow()
        img = Image.open(path)

        self._gif_frames = []
        self._gif_durations = []

        for frame in ImageSequence.Iterator(img):
            fr = frame.copy().convert("RGBA")
            fr.thumbnail((cw, ch), Image.LANCZOS)
            self._gif_frames.append(ImageTk.PhotoImage(fr))
            self._gif_durations.append(frame.info.get("duration", 100))

        if not self._gif_frames:
            raise ValueError("No GIF frames found")

        self._gif_frame_index = 0
        self.fig_canvas.delete("all")
        self.fig_canvas.create_image(
            cw // 2,
            ch // 2,
            anchor=tk.CENTER,
            image=self._gif_frames[0],
            tags="gif_frame",
        )
        self._photo_image = self._gif_frames[0]
        self._animate_gif()

    def _animate_gif(self):
        if not self._gif_frames:
            return
        if self.fig_canvas is None:
            return
        if self.fig_window is None or not self.fig_window.winfo_exists():
            return

        self._gif_frame_index = (self._gif_frame_index + 1) % len(self._gif_frames)
        frame = self._gif_frames[self._gif_frame_index]

        self.fig_canvas.delete("gif_frame")
        self.fig_canvas.create_image(
            self.fig_canvas.winfo_width() // 2,
            self.fig_canvas.winfo_height() // 2,
            anchor=tk.CENTER,
            image=frame,
            tags="gif_frame",
        )
        self._photo_image = frame

        delay = self._gif_durations[self._gif_frame_index]
        if delay <= 0:
            delay = 100

        self._gif_after_id = self.root.after(delay, self._animate_gif)

    def _stop_gif_animation(self):
        if self._gif_after_id is not None:
            try:
                self.root.after_cancel(self._gif_after_id)
            except Exception:
                pass
            self._gif_after_id = None

        self._gif_frames = []
        self._gif_durations = []
        self._gif_frame_index = 0

    def _on_fig_canvas_resize(self, event=None):
        if self.fig_canvas is None:
            return
        if self._figure_paths:
            self._show_current_figure()

    # ----- Welcome message -------------------------------------------------
    def _show_welcome(self):
        self._append_chat(
            "Agent",
            (
                "Welcome to AutoPore!\n"
                "I can help you configure and run image-based pore-scale simulations.\n"
                "The Workflow panel tracks completed steps, next allowed steps, and the log file.\n\n"
                "Try saying things like:\n"
                '  • "Generate an image using default setting"\n'
                '  • "Import an x-ray image from my folder"\n'
                '  • "Run a quasi-static simulation"\n'
                '  • "Run a dynamic simulation"\n'
                '  • "Plot the results"\n'
                '  • "Run an uncertainty quantification of 5 quasi-static simulations"\n'
                '  • "Reset everything"'
            ),
            "agent",
        )

    # ----- Chat display helpers --------------------------------------------
    def _append_chat(self, sender: str, text: str, kind: str = "agent"):
        self.chat_display.configure(state=tk.NORMAL)

        if kind == "user":
            self.chat_display.insert(tk.END, "You:  ", "user_label")
            self.chat_display.insert(tk.END, f"{text}\n", "user_msg")

        elif kind == "agent":
            self.chat_display.insert(tk.END, "Agent:  ", "agent_label")
            self.chat_display.insert(tk.END, f"{text}\n", "agent_msg")

        elif kind == "tool":
            self.chat_display.insert(tk.END, f"  ⚙ {text}\n", "tool_msg")

        elif kind == "status":
            self.chat_display.insert(tk.END, f"  {text}\n", "status_msg")

        elif kind == "error":
            self.chat_display.insert(tk.END, f"  ✖ {text}\n", "error_msg")

        self.chat_display.insert(tk.END, "\n", "separator")
        self.chat_display.configure(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def _set_busy(self, busy: bool):
        was_busy = self._busy
        self._busy = busy
        if not busy and not self._simulation_running:
            self._button_phase = "idle"
            self._stop_button_status_animation()
        self._refresh_send_button()
        if was_busy and not busy:
            self._refresh_workflow_panel()

    def _set_button_phase(self, phase: str):
        self._button_phase = phase
        if phase == "idle":
            self._stop_button_status_animation()
        else:
            self._start_button_status_animation()
        self._refresh_send_button()

    def _refresh_send_button(self):
        state = tk.DISABLED if (self._busy or self._simulation_running) else tk.NORMAL
        self.send_btn.configure(state=state)
        self.user_input.configure(state=state)
        if self.clear_history_btn is not None:
            self.clear_history_btn.configure(state=state)
        for button in (self.run_next_btn, self.run_full_btn, self.start_over_btn):
            if button is not None:
                button.configure(state=state)
        self._refresh_workflow_action_buttons()

        if self._button_phase == "idle":
            button_text = "Send"
            button_anchor = tk.CENTER
        else:
            label = {
                "sending": "Sending",
                "running": "Running",
                "returning": "Returning",
            }.get(self._button_phase, "Send")
            dots = "." * max(1, self._button_dot_count)
            button_text = f"{label}{dots:<3}"
            button_anchor = tk.CENTER

        self.send_btn.configure(text=button_text, anchor=button_anchor)

    def _refresh_workflow_action_buttons(self):
        selected = self._active_panel_action or "next"
        buttons = {
            "next": self.run_next_btn,
            "full": self.run_full_btn,
            "reset": self.start_over_btn,
        }
        for action, button in buttons.items():
            if button is None:
                continue
            is_selected = action == selected
            button.configure(
                bg=self.FG_ACCENT if is_selected else self.BG_INPUT,
                fg=self.BG_DARK if is_selected else self.FG_MAIN,
                activebackground="#74a8f5" if is_selected else self.BG_USER,
                activeforeground=self.BG_DARK if is_selected else self.FG_MAIN,
            )

    def _start_button_status_animation(self):
        if self._button_animation_after_id is not None:
            return
        self._button_dot_count = 0
        self._animate_button_status()

    def _stop_button_status_animation(self):
        if self._button_animation_after_id is not None:
            try:
                self.root.after_cancel(self._button_animation_after_id)
            except Exception:
                pass
            self._button_animation_after_id = None
        self._button_dot_count = 0

    def _animate_button_status(self):
        if self._button_phase == "idle":
            self._button_animation_after_id = None
            return
        self._button_dot_count = (self._button_dot_count % 3) + 1
        self._refresh_send_button()
        self._button_animation_after_id = self.root.after(450, self._animate_button_status)

    def _start_workflow_running_animation(self):
        if self._workflow_animation_after_id is not None:
            return
        self._workflow_running_dot_count = 0
        self._animate_workflow_running_step()

    def _stop_workflow_running_animation(self):
        if self._workflow_animation_after_id is not None:
            try:
                self.root.after_cancel(self._workflow_animation_after_id)
            except Exception:
                pass
            self._workflow_animation_after_id = None
        self._workflow_running_dot_count = 0
        self._clear_workflow_running_dots()

    def _animate_workflow_running_step(self):
        if not self._simulation_running:
            self._workflow_animation_after_id = None
            return
        self._workflow_running_dot_count = (self._workflow_running_dot_count % 3) + 1
        self._refresh_workflow_running_dots()
        self._set_busy(self._busy)
        self._workflow_animation_after_id = self.root.after(450, self._animate_workflow_running_step)

    def _set_simulation_running(self, running: bool):
        self._simulation_running = running
        if running:
            self._set_button_phase("running")
            self._start_workflow_running_animation()
        else:
            self._stop_workflow_running_animation()
            self._set_button_phase("returning" if self._busy else "idle")
        self._set_busy(self._busy)

    # ----- Parameter formatting ----------------------------------------
    def _format_config_snapshot(self, upcoming_func: str | None = None) -> str:
        lines = []
    
        lines.append("Current Parameters:")
        for k, v in config.items():
            is_default = (k in DEFAULTS and v == DEFAULTS[k])
            suffix = " (default)" if is_default else "  ★ changed"
            lines.append(f"  {self._format_parameter_label(k)}: {v}{suffix}")
    
        active_img = {k: v for k, v in image_config.items() if v is not None}
        if active_img:
            lines.append("")
            lines.append("Current Image Properties:")
            for k, v in active_img.items():
                lines.append(f"  {self._format_parameter_label(k)}: {v}")
    
        lines.append("")
        lines.append("Optional / Allowed Parameters:")
        for k, vals in ALLOWED_VALUES.items():
            if isinstance(vals, (list, tuple)):
                val_text = " | ".join(str(x) for x in vals)
            else:
                val_text = str(vals)
            lines.append(f"  {self._format_parameter_label(k)}: {val_text}")
    
        return "\n".join(lines)
    
    
    def _read_qs_params(self) -> dict[str, str]:
        qs_file = THIS_DIR / "input_pnm" / "qs_param.txt"
        try:
            lines = qs_file.read_text().splitlines()
        except FileNotFoundError:
            lines = QS_PARAM_DEFAULTS.splitlines()

        return {
            "min_pressure": lines[0],
            "max_pressure": lines[1],
            "pressure_step": lines[2],
        }

    def _read_dyn_params(self) -> dict[str, str]:
        dyn_file = THIS_DIR / "input_pnm" / "dyn_param.txt"
        try:
            lines = dyn_file.read_text().splitlines()
        except FileNotFoundError:
            lines = DYN_PARAM_DEFAULTS.splitlines()

        return {
            "gas_viscosity": lines[0],
            "liquid_viscosity": lines[1],
            "capillary_number": lines[3],
            "max_pc": lines[7],
        }

    def _format_parameter_label(self, key: str) -> str:
        label_overrides = {
            "action": "Action",
            "img_src": "Image",
            "task_mode": "Task mode",
            "pnm_mode": "PNM mode",
            "quantity": "Local variable",
            "vtk": "Local file format",
            "dimensions": "Dimensions",
            "shape": "Dimensions",
            "voxels": "Image voxels",
            "voxel_size": "Voxel size",
            "image_file": "Image file",
            "coarse_factor": "Image coarsen",
            "porosity": "Porosity",
            "blobiness": "Blobiness",
            "pore_diameter_type": "Pore diameter",
            "throat_diameter_type": "Throat diameter",
            "min_pressure": "Min capillary pressure",
            "max_pressure": "Max capillary pressure",
            "pressure_step": "Pressure step",
            "gas_viscosity": "Nonwet viscosity",
            "liquid_viscosity": "Wet viscosity",
            "capillary_number": "Capillary number",
            "max_pc": "Max capillary pressure",
            "realizations": "Realizations",
            "mode": "Mode",
        }
        if key in label_overrides:
            return label_overrides[key]
        label = str(key).replace("_", " ")
        return label[:1].upper() + label[1:]

    def _format_parameter_value(self, key: str, value: object) -> object:
        if key == "quantity" and isinstance(value, str):
            return {
                "sat_nw": "Snw",
                "sat_w": "Sw",
                "pc": "Pc",
                "pres_nw": "Pnw",
                "pres_w": "Pw",
                "all": "All variables",
            }.get(value, value)
        if key in {"pore_diameter_type", "throat_diameter_type"} and isinstance(value, str):
            return {
                "equivalent": "Equivalent sphere",
                "inscribed": "Inscribed sphere",
                "extended": "Extended sphere",
            }.get(value, value)
        if key in {"mode", "pnm_mode"} and isinstance(value, str):
            return {
                "quasi_static": "Quasi static",
                "dynamic": "Dynamic",
            }.get(value, value)
        if key == "vtk" and isinstance(value, str):
            return value.upper()
        return value

    def _format_parameter_options(self, key: str, current_value: object | None = None) -> str:
        option_key = "pnm_mode" if key == "mode" else key
        values = ALLOWED_VALUES.get(option_key)
        if not isinstance(values, (list, tuple)):
            return ""
        current_display = (
            str(self._format_parameter_value(key, current_value))
            if current_value is not None
            else None
        )
        formatted_options = []
        for value in values:
            display = str(self._format_parameter_value(key, value))
            if current_value is not None and (value == current_value or display == current_display):
                continue
            formatted_options.append(display)
        return " | ".join(formatted_options)

    def _format_parameter_line(self, key: str, value: object, unit: str = "") -> str:
        if key in {"pore_diameter_type", "throat_diameter_type"}:
            line = f"  {self._format_parameter_label(key)}: {self._format_parameter_value(key, value)}{unit}"
        else:
            line = f"  {self._format_parameter_label(key)}: {self._format_parameter_value(key, value)}{unit}"
        options = self._format_parameter_options(key, value)
        if options:
            line += f" (opts: {options})"
        return line

    def _format_qs_param_line(self, key: str, value: str) -> str:
        return f"  {self._format_parameter_label(key)}: {value} Pa"

    def _get_image_display_defaults(self) -> dict[str, object]:
        return {
            "shape": MODEL_DEFAULTS["shape"],
            "voxel_size": MODEL_DEFAULTS["resolution"],
            "image_file": Path(MODEL_DEFAULTS["file"]).name,
            "voxels": MODEL_DEFAULTS["voxels"],
            "coarse_factor": MODEL_DEFAULTS["coarse_factor"],
            "porosity": MODEL_DEFAULTS["poro"],
            "blobiness": MODEL_DEFAULTS["blobiness"],
            "pore_diameter_type": MODEL_DEFAULTS["pore_diameter_type"],
            "throat_diameter_type": MODEL_DEFAULTS["throat_diameter_type"],
        }

    def _format_image_properties_lines(self) -> list[str]:
        image_defaults = self._get_image_display_defaults()
        lines = []

        if config.get("img_src") == "x_ray":
            dimensions_value = image_config.get("voxels")
            if dimensions_value is None:
                dimensions_value = image_defaults["voxels"]
        else:
            dimensions_value = image_config.get("shape")
            if dimensions_value is None:
                dimensions_value = image_defaults["shape"]
        if isinstance(dimensions_value, (list, tuple)):
            dimensions_value = ", ".join(str(value) for value in dimensions_value)
        lines.append(f"  {self._format_parameter_label('dimensions')}: {dimensions_value} voxels")

        for key in (
            "voxel_size",
            "porosity",
            "blobiness",
        ):
            value = image_config.get(key)
            if value is None:
                value = image_defaults[key]
            if key == "voxel_size":
                lines.append(f"  {self._format_parameter_label(key)}: {value} µm")
            else:
                lines.append(f"  {self._format_parameter_label(key)}: {value}")

        if config.get("img_src") == "x_ray":
            image_file = image_config.get("image_file")
            if image_file is None:
                image_file = image_defaults["image_file"]
            lines.append(f"  {self._format_parameter_label('image_file')}: {image_file}")

        return lines

    def _format_pore_network_extraction_lines(self) -> list[str]:
        image_defaults = self._get_image_display_defaults()
        lines = []
        coarse_factor = image_config.get("coarse_factor")
        if coarse_factor is None:
            coarse_factor = image_defaults["coarse_factor"]
        lines.append(self._format_parameter_line("coarse_factor", coarse_factor, " voxels"))
        for key in ("pore_diameter_type", "throat_diameter_type"):
            value = image_config.get(key)
            if value is None:
                value = image_defaults[key]
            lines.append(self._format_parameter_line(key, value))
        return lines

    def _show_initial_parameters(self):
        self._refresh_workflow_panel()

    def _announce_tool_call(self, func_name: str, func_args: dict):
        return

    def _refresh_after_tool_call(self, func_name: str, tool_result: str):
        try:
            result = json.loads(tool_result)
        except json.JSONDecodeError:
            result = {}
        if func_name == "set_parameters" and "task_mode" in result.get("updated", {}):
            self._clear_generated_files_display()
        self._refresh_workflow_panel()

    # ----- Send / receive --------------------------------------------------
    def _quick_send(self, text: str):
        if self._busy:
            return
        self.user_input.delete("1.0", tk.END)
        self.user_input.insert("1.0", text)
        self._on_send()

    def _run_next_from_panel(self):
        if self._busy or self._simulation_running:
            return
        snapshot = WORKFLOW_TRACKER.snapshot()
        next_steps = snapshot.get("next_allowed_steps", [])
        if not next_steps:
            self._append_generated_files_message(
                "The workflow is complete. Click Start over to begin a new run."
            )
            return
        self._start_panel_execution("next", next_steps[0])

    def _run_full_from_panel(self):
        if self._busy or self._simulation_running:
            return
        snapshot = WORKFLOW_TRACKER.snapshot()
        next_steps = snapshot.get("next_allowed_steps", [])
        if not next_steps:
            self._append_generated_files_message(
                "The workflow is complete. Click Start over to begin a new run."
            )
            return
        self._start_panel_execution("full", next_steps[0])

    def _start_panel_execution(self, mode: str, first_step: str):
        global _shown_figures

        resets_outputs = first_step == "generate_input"
        if resets_outputs:
            _shown_figures.clear()
            self._reset_live_figures()
            self._clear_generated_files_display()

        before = self._snapshot_artifact_files()
        self._active_panel_action = mode
        self._set_busy(True)
        self._set_simulation_running(True)
        worker = threading.Thread(
            target=self._panel_execution_worker,
            args=(mode, first_step, before),
            daemon=True,
        )
        worker.start()

    def _panel_execution_worker(self, mode: str, first_step: str, before: dict[str, float]):
        try:
            task_mode = config.get("task_mode", "single_run")
            pnm_mode = config.get("pnm_mode", "quasi_static")

            if task_mode == "uq_run":
                if first_step == "generate_input":
                    run_result = _run_realizations(uq_config.get("realizations"), pnm_mode)
                    if mode == "full" and run_result.get("status") == "success":
                        plot_result = _execute_model_action("plot_result", pnm_mode=pnm_mode)
                        result = {
                            "status": plot_result.get("status", "success"),
                            "message": "UQ workflow executed successfully.",
                            "steps": [run_result, plot_result],
                            **_workflow_payload(),
                        }
                        if plot_result.get("model_results_archive"):
                            result["model_results_archive"] = plot_result["model_results_archive"]
                    else:
                        result = run_result
                else:
                    result = _execute_model_action(first_step, pnm_mode=pnm_mode)
            elif mode == "full":
                start_index = WORKFLOW_STEPS.index(first_step)
                result = _execute_workflow(
                    list(WORKFLOW_STEPS[start_index:]),
                    pnm_mode=pnm_mode,
                )
            else:
                result = _execute_model_action(first_step, pnm_mode=pnm_mode)
        except Exception as exc:
            result = {"status": "error", "message": str(exc)}

        self.root.after(0, self._finish_panel_execution, result, before)

    def _finish_panel_execution(self, result: dict, before: dict[str, float]):
        self._set_simulation_running(False)
        after = self._snapshot_artifact_files()
        self._show_generated_artifacts(before, after)

        archive_info = result.get("model_results_archive")
        if isinstance(archive_info, dict) and archive_info.get("archive_dir"):
            self._append_generated_files_message(
                "Archived completed cycle:\n"
                f"  Location: {archive_info['archive_dir']}"
            )

        status = result.get("status", "error")
        if status not in {"success", "manually_closed"}:
            self._append_generated_files_message(
                result.get("message", "The workflow action did not complete.")
            )

        self._refresh_workflow_state_message()
        self._refresh_workflow_panel()
        self._sync_figures_to_workflow_state(result)
        self._active_panel_action = None
        self._set_busy(False)

    def _start_workflow_over_from_panel(self):
        if self._busy:
            return
        WORKFLOW_TRACKER.reset("Start Over button clicked in the workflow panel.")
        self._clear_generated_files_display()
        self._reset_live_figures()
        self._refresh_workflow_state_message()
        self._refresh_workflow_panel()

    def _try_handle_direct_diameter_setting(self, text: str) -> bool:
        updates = _parse_direct_diameter_setting(text)
        if not updates:
            return False

        tool_result = dispatch_tool_call("set_image_properties", updates)
        self._refresh_after_tool_call("set_image_properties", tool_result)
        try:
            result = json.loads(tool_result)
        except json.JSONDecodeError:
            result = {"status": "error", "message": tool_result}

        if result.get("status") == "ok":
            updated = result.get("updated", {})
            summary = "; ".join(
                f"{self._format_parameter_label(key)}: "
                f"{self._format_parameter_value(key, value)}"
                for key, value in updated.items()
            )
            if summary:
                self._append_chat("Agent", f"Updated {summary}.", "agent")
            else:
                self._append_chat("Agent", "No diameter settings were changed.", "agent")
        else:
            self._append_chat("", result.get("message", "Could not update diameter settings."), "error")
        return True

    def _try_handle_direct_dyn_param_setting(self, text: str) -> bool:
        updates = _parse_direct_dyn_param_setting(text)
        if not updates:
            return False

        tool_result = dispatch_tool_call("set_dyn_params", updates)
        self._refresh_after_tool_call("set_dyn_params", tool_result)
        try:
            result = json.loads(tool_result)
        except json.JSONDecodeError:
            result = {"status": "error", "message": tool_result}

        if result.get("status") == "ok":
            dyn_params = result.get("dyn_params", {})
            summary = "; ".join(
                f"{self._format_parameter_label(key)}: {dyn_params.get(key)}"
                for key in updates
                if key in dyn_params
            )
            if summary:
                self._append_chat("Agent", f"Updated {summary}.", "agent")
            else:
                self._append_chat("Agent", "No dynamic parameters were changed.", "agent")
        else:
            self._append_chat("", result.get("message", "Could not update dynamic parameters."), "error")
        return True

    def _on_send_key(self, event=None):
        # Shift+Enter -> newline
        if event and (event.state & 0x1):
            return
        self._on_send()
        return "break"

    def _on_send(self, event=None):
        if self._busy:
            return

        text = self.user_input.get("1.0", tk.END).strip()
        if not text:
            return

        self.user_input.delete("1.0", tk.END)

        if text.lower() in ("quit", "exit", "q"):
            self.root.destroy()
            return

        self._append_chat("You", text, "user")
        if self._try_handle_direct_dyn_param_setting(text):
            return
        if self._try_handle_direct_diameter_setting(text):
            return

        self.messages.append({"role": "user", "content": text})
        self._set_button_phase("sending")
        self._set_busy(True)

        thread = threading.Thread(target=self._agent_loop, daemon=True)
        thread.start()

    def _agent_loop(self):
        """Run the LLM tool-calling loop in a background thread."""
        try:
            settings = self._current_llm_settings()
            llm = LLMClient(settings)
            client = llm.client()
            model, available = llm.resolve_model(client)
            self.root.after(
                0,
                self._apply_resolved_model,
                settings.provider,
                settings.base_url,
                model,
                available,
                False,
            )
            while True:
                response = client.chat.completions.create(
                    model=model,
                    messages=self.messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    max_tokens=1024,
                )
                reply = response.choices[0].message
                self.messages.append(reply)
    
                if reply.tool_calls:
                    for tc in reply.tool_calls:
                        func_name = tc.function.name
                        func_args = json.loads(tc.function.arguments)
    
                        is_execution_call = func_name in {
                            "execute_model",
                            "execute_workflow",
                            "run_realizations",
                            "rerun_case",
                        }

                        # Clear captured figures before executing model/workflow
                        artifact_snapshot_before = None
                        if is_execution_call:
                            _shown_figures.clear()
                            if self._execution_resets_figures(func_name, func_args):
                                self.root.after(0, self._reset_live_figures)
                            if self._execution_resets_generated_messages(func_name, func_args):
                                self.root.after(0, self._clear_generated_files_display)
                            self.root.after(0, self._set_simulation_running, True)
                            artifact_snapshot_before = self._snapshot_artifact_files()
    
                        tool_result = dispatch_tool_call(func_name, func_args)
                        self._update_workflow_state_from_tool(func_name, tool_result)
                        self.root.after(0, self._refresh_after_tool_call, func_name, tool_result)
    
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_result,
                        })
    
                        if is_execution_call:
                            result = json.loads(tool_result)
                            self.root.after(0, self._set_simulation_running, False)
                            archive_info = result.get("model_results_archive")
                            if isinstance(archive_info, dict) and archive_info.get("archive_dir"):
                                self.root.after(
                                    0,
                                    self._append_generated_files_message,
                                    "Archived completed cycle:\n"
                                    f"  Location: {archive_info['archive_dir']}",
                                )
                            if result.get("status") in {"success", "manually_closed", "needs_simulation", "workflow_blocked"}:
                                artifact_snapshot_after = self._snapshot_artifact_files()
                                if result.get("status") in {"manually_closed", "needs_simulation", "workflow_blocked"}:
                                    self.root.after(
                                        0,
                                        self._append_generated_files_message,
                                        result.get("message", ""),
                                    )
                                if artifact_snapshot_before is not None:
                                    self.root.after(
                                        0,
                                        self._show_generated_artifacts,
                                        artifact_snapshot_before,
                                        artifact_snapshot_after,
                                    )
                                self.root.after(0, self._sync_figures_to_workflow_state, result)
                            elif result.get("status") == "cancelled":
                                self.root.after(
                                    0,
                                    self._append_generated_files_message,
                                    result.get("message", "Simulation was cancelled."),
                                )
                            else:
                                self.root.after(
                                    0,
                                    self._append_generated_files_message,
                                    f"Execution error: {result.get('message', 'unknown')}",
                                )
                    continue
    
                if reply.content:
                    self.root.after(0, self._append_chat, "Agent", reply.content, "agent")
                break
    
        except Exception as exc:
            provider = self.provider_var.get().strip().lower()
            error_text = str(exc)
            if provider == "local" and (
                "unable to allocate" in error_text.lower()
                or "cuda_host" in error_text.lower()
                or "llama runner process has terminated" in error_text.lower()
            ):
                error_text += (
                    "\nThe selected Ollama model does not fit the available memory. "
                    "Choose a smaller installed model and click Detect."
                )
            self.root.after(
                0,
                self._append_chat,
                "",
                f"Error communicating with LLM: {error_text}",
                "error",
            )
        finally:
            self.root.after(0, self._set_busy, False)

    # ----- Close -----------------------------------------------------------
    def _on_close(self):
        global _figure_callback, _workflow_progress_callback

        _figure_callback = None
        _workflow_progress_callback = None
        self._stop_figure_slideshow()
        self._stop_workflow_running_animation()
        self._stop_button_status_animation()
        self._stop_gif_animation()

        if self.fig_window is not None and self.fig_window.winfo_exists():
            try:
                self.fig_window.destroy()
            except Exception:
                pass

        self.root.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    root = tk.Tk()
    app = AutoPoreGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

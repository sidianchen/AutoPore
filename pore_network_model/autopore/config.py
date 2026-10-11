"""Lightweight paths and parameter definitions shared by the GUI and model."""

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
THIS_DIR = PACKAGE_DIR.parent

MODEL_DEFAULTS = {
    "file": str(THIS_DIR / "image" / "my_rock.raw"),
    "resolution": 2.5,
    "voxels": [252, 252, 252],
    "shape": [200, 200, 350],
    "poro": 0.5,
    "blobiness": 2,
    "coarse_factor": 1,
    "outputfolder": str(THIS_DIR / "input_pnm") + "/",
    "pnm_name": "pore_network",
    "thickness": 1.0,
    "simoutputname": str(THIS_DIR / "single_run_qs_pnm") + "/",
    "dSdP": 5,
    "min_boundary_face_area": 15.0,
    "time_digits": 5,
    "pore_diameter_type": "equivalent",
    "throat_diameter_type": "equivalent",
}

ALLOWED_VALUES = {
    "action": ["generate_input", "run_simulation", "plot_result"],
    "img_src": ["porespy", "x_ray"],
    "pnm_mode": ["quasi_static", "dynamic"],
    "task_mode": ["single_run", "uq_run"],
    "quantity": ["sat_nw", "sat_w", "pc", "pres_nw", "pres_w", "all"],
    "vtk": ["png", "vtk"],
    "pore_diameter_type": ["inscribed", "equivalent", "extended"],
    "throat_diameter_type": ["inscribed", "equivalent"],
}

DIAMETER_PROPERTY_NAMES = {
    "pore": {
        "inscribed": "inscribed_diameter",
        "equivalent": "equivalent_diameter",
        "extended": "extended_diameter",
    },
    "throat": {
        "inscribed": "inscribed_diameter",
        "equivalent": "equivalent_diameter",
    },
}

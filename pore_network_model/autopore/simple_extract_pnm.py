"""Simple raw-image to pore-network input-file extractor.

Run this file from Spyder or from a terminal. It only processes the image,
extracts the pore network, and writes the input files. It does not run
simulations and does not create plots.

Required packages:
    numpy, porespy, openpnm, scipy, scikit-image

Optional package:
    vtk, only if WRITE_VTP = True
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openpnm as op
import porespy as ps


PACKAGE_DIR = Path(__file__).resolve().parent
MODEL_DIR = PACKAGE_DIR.parent
SCRIPT_DIR = MODEL_DIR / "image"

from .image_utils import (
    coarsen_image_majority,
    enforce_dual_percolation,
    generate_input_file,
    safe_boundary_pores_with_area,
)


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------

# Put the raw image in this same folder. Use a filename or an absolute path.
RAW_FILE = "my_rock.raw"

# Raw files do not store dimensions. Set this if auto-detection cannot infer it.
# Example for your large x-ray image: VOXELS = [1920, 1920, 1190]
VOXELS = None

# Known uint8 raw-image sizes. The script uses these only when VOXELS is None.
KNOWN_VOXEL_SHAPES = [
    [1920, 1920, 1190],
    [252, 252, 252],
]

VOXEL_SIZE = 3.2
COARSE_FACTOR = 4
PORE_DIAMETER_TYPE = "equivalent"    # inscribed, equivalent, or extended
THROAT_DIAMETER_TYPE = "equivalent"  # inscribed or equivalent

OUTPUT_DIR = SCRIPT_DIR
PNM_NAME = "pore_network"
THICKNESS = 1.0

# Set True only if you also want a ParaView/OpenPNM .vtp file.
# The simulator input file is the .txt file, so this can stay False.
WRITE_VTP = False

# Save the imported raw image as a .vti file in this same folder.
WRITE_VTI = True
VTI_NAME = "image"

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


def diameter_property_name(element: str, diameter_type: str) -> str:
    try:
        return DIAMETER_PROPERTY_NAMES[element][diameter_type]
    except KeyError as exc:
        allowed = sorted(DIAMETER_PROPERTY_NAMES.get(element, {}))
        raise ValueError(
            f"Invalid {element} diameter type {diameter_type!r}. Allowed: {allowed}"
        ) from exc


def network_diameter_values(network, element: str, diameter_type: str) -> np.ndarray:
    property_name = diameter_property_name(element, diameter_type)
    key = f"{element}.{property_name}"
    fallback_keys = []
    if diameter_type == "equivalent":
        fallback_keys.append(f"{element}.equivelant_diameter")

    for candidate in [key] + fallback_keys:
        if candidate in network.keys():
            return np.asarray(network[candidate], dtype=float).reshape(-1, 1)

    available = [k for k in network.keys() if k.startswith(f"{element}.")]
    raise KeyError(
        f"Network does not contain {key!r}. Available {element} fields include: "
        f"{available[:30]}"
    )


def resolve_raw_path(raw_file: str) -> Path:
    raw_path = Path(raw_file)
    if not raw_path.is_absolute():
        raw_path = SCRIPT_DIR / raw_path
    return raw_path


def resolve_voxels(raw_path: Path) -> list[int]:
    if VOXELS is not None:
        return list(VOXELS)

    raw_size = raw_path.stat().st_size
    for shape in KNOWN_VOXEL_SHAPES:
        if int(np.prod(shape)) == raw_size:
            return list(shape)

    raise ValueError(
        "Could not infer raw-image dimensions. Set VOXELS = [nx, ny, nz] "
        f"near the top of this file. Raw byte count is {raw_size}."
    )


def read_raw_image(raw_path: Path, voxels: list[int]) -> np.ndarray:
    expected_size = int(np.prod(voxels))
    actual_size = raw_path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"Raw file has {actual_size} bytes, but voxels {voxels} require "
            f"{expected_size} bytes for uint8 data."
        )
    return np.fromfile(raw_path, dtype=np.uint8).reshape(voxels)


def extract_network_from_image(im0: np.ndarray):
    im1, _ = coarsen_image_majority(im0, factor=COARSE_FACTOR)
    im = enforce_dual_percolation(im1)
    im = ~np.asarray(im, dtype=bool)
    im = np.swapaxes(im, 0, 2)

    net0 = ps.networks.snow2(phases=im, boundary_width=0)
    pn0 = op.io.network_from_porespy(net0.network)

    project = pn0.project.copy()
    pn = project.network

    health = op.utils.check_network_health(pn)
    pores_to_trim = np.unique(
        np.hstack([health["isolated_pores"], health["disconnected_pores"]])
    ).astype(int)

    pn["pore.old_id"] = np.arange(pn.Np)
    pn["throat.old_id"] = np.arange(pn.Nt)

    keep_pores = np.ones(pn.Np, dtype=bool)
    keep_pores[pores_to_trim] = False
    old_to_new_pore = -np.ones(pn.Np, dtype=int)
    old_to_new_pore[keep_pores] = np.arange(keep_pores.sum())
    op.topotools.trim(network=pn, pores=pores_to_trim)

    return im, net0, pn0, pn, old_to_new_pore


def build_input_arrays(im, net0, pn0, pn, old_to_new_pore):
    resolution = float(VOXEL_SIZE)

    pore_diameters = network_diameter_values(pn, "pore", PORE_DIAMETER_TYPE)
    pore_count = pore_diameters.shape[0]
    pore_indices = np.arange(pore_count).reshape(-1, 1)
    pore_shapes = np.ones(pore_count).reshape(-1, 1)
    pore_volumes = np.asarray(pn["pore.region_volume"]).reshape(-1, 1)
    pore_lengths = np.divide(
        pore_volumes,
        pore_diameters ** 2,
        out=pore_diameters.copy(),
        where=pore_diameters > 0,
    )
    pore_bodies = np.hstack(
        (
            pore_indices,
            pore_shapes,
            pore_diameters * resolution / 2.0,
            pore_lengths * resolution,
            pore_volumes * resolution ** 3,
        )
    )

    throat_links = pn["throat.conns"]
    throat_count = throat_links.shape[0]
    throat_indices = np.arange(throat_count).reshape(-1, 1)
    throat_shapes = np.ones(throat_count).reshape(-1, 1)
    throat_diameters = network_diameter_values(pn, "throat", THROAT_DIAMETER_TYPE)
    throat_lengths = pn["throat.total_length"].reshape(-1, 1)
    throat_areas = np.asarray(pn["throat.cross_sectional_area"]).reshape(-1, 1)
    pore_throats = np.hstack(
        (
            throat_indices,
            throat_links,
            throat_shapes,
            throat_diameters * resolution / 2.0,
            throat_lengths * resolution,
            throat_areas * resolution ** 2,
        )
    )

    regions = net0["regions"].astype(np.int32)
    inlet0, inlet_area0, outlet0, outlet_area0 = safe_boundary_pores_with_area(
        pn0,
        regions,
        {"min_boundary_face_area": 1},
        axis=0,
    )

    inlet_mapped = old_to_new_pore[inlet0]
    outlet_mapped = old_to_new_pore[outlet0]

    inlet_mask = inlet_mapped >= 0
    outlet_mask = outlet_mapped >= 0
    inlet = inlet_mapped[inlet_mask]
    outlet = outlet_mapped[outlet_mask]

    electrodes = -np.ones(pore_count).reshape(-1, 1)
    bound_id = np.hstack((inlet, outlet)).reshape(-1, 1)
    bound_type = np.vstack((electrodes[inlet] * 0 + 1, electrodes[outlet] * 0))
    bound_value = np.vstack((electrodes[inlet] * 0 + 1, electrodes[outlet] * 0))

    coordinates = np.hstack(
        (pore_indices, pn["pore.geometric_centroid"] * resolution)
    )
    boundaries = np.hstack((bound_id, bound_type.reshape(-1, 1), bound_value.reshape(-1, 1)))

    thickness_value = THICKNESS * (np.max(coordinates[:, 1]) - np.min(coordinates[:, 1]))
    xin = coordinates[inlet, 1]
    xout = coordinates[outlet, 1]
    xin_id = np.where(xin <= np.min(coordinates[:, 1]) + thickness_value)[0]
    xout_id = np.where(xout >= np.max(coordinates[:, 1]) - thickness_value)[0]
    selected = np.hstack((inlet[xin_id], outlet[xout_id]))

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

    x_domain_size = float(im.shape[0]) * resolution
    ghost_throat_length = x_domain_size / 100.0

    return pore_bodies, pore_throats, boundaries, coordinates, ghost_throat_length


def main():
    raw_path = resolve_raw_path(RAW_FILE)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw image not found: {raw_path}")

    voxels = resolve_voxels(raw_path)
    print(f"Reading raw image: {raw_path}")
    print(f"Voxels: {voxels}")
    print(f"Voxel size: {VOXEL_SIZE}")

    im0 = read_raw_image(raw_path, voxels)
    if WRITE_VTI:
        vti_base = OUTPUT_DIR / VTI_NAME
        ps.io.to_vtk(im0, filename=str(vti_base))
        print(f"Wrote: {vti_base.with_suffix('.vti')}")

    im, net0, pn0, pn, old_to_new_pore = extract_network_from_image(im0)
    arrays = build_input_arrays(im, net0, pn0, pn, old_to_new_pore)
    pore_bodies, pore_throats, boundaries, coordinates, ghost_throat_length = arrays

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    txt_file = OUTPUT_DIR / f"{PNM_NAME}.txt"

    generate_input_file(
        pore_bodies,
        pore_throats,
        boundaries,
        coordinates,
        output_filename=str(txt_file),
        ghost_throat_length=ghost_throat_length,
    )

    print("Finished extracting pore network.")
    print(f"Wrote: {txt_file}")

    if WRITE_VTP:
        vtk_file = OUTPUT_DIR / PNM_NAME
        op.io.project_to_vtk(project=pn.project, filename=vtk_file)
        print(f"Wrote: {vtk_file.with_suffix('.vtp')}")


if __name__ == "__main__":
    main()

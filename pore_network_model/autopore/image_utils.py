"""Essential helper functions for simple_extract_pnm.py.

This local utility file intentionally avoids plotting, video, PyVista, and VTK
imports. It contains only the image-processing and input-file-writing helpers
needed by the simple raw-image pore-network extractor.
"""

from __future__ import annotations

import io

import numpy as np
from scipy.ndimage import label
from skimage.graph import route_through_array


def coarsen_image_majority(im, factor, threshold=0.5):
    """Coarsen a 3D binary image using block averaging and majority rule."""
    factor = int(factor)
    if factor <= 1:
        return np.asarray(im, dtype=bool), np.asarray(im, dtype=float)

    im = np.asarray(im, dtype=float)
    sx, sy, sz = im.shape

    sx2 = (sx // factor) * factor
    sy2 = (sy // factor) * factor
    sz2 = (sz // factor) * factor
    im_trim = im[:sx2, :sy2, :sz2]

    phi = im_trim.reshape(
        sx2 // factor,
        factor,
        sy2 // factor,
        factor,
        sz2 // factor,
        factor,
    ).mean(axis=(1, 3, 5))

    return phi > threshold, phi


def enforce_dual_percolation(
    im,
    axis=0,
    connectivity=3,
    penalty=1000.0,
    keep_spanning_only=True,
    ensure_face_contact=True,
    max_iters=5,
):
    """Ensure both pore and solid phases span from inlet to outlet."""
    im2 = np.asarray(im, dtype=bool).copy()

    if im2.ndim == 2:
        structure = np.ones((3, 3), bool) if connectivity > 1 else None
    else:
        structure = np.ones((3, 3, 3), bool) if connectivity > 1 else None

    def spanning_labels(mask):
        labeled, _ = label(mask, structure=structure)
        inlet = np.take(labeled, 0, axis=axis)
        outlet = np.take(labeled, -1, axis=axis)
        labs = np.intersect1d(
            np.unique(inlet[inlet > 0]),
            np.unique(outlet[outlet > 0]),
        )
        return labeled, labs

    def spanning_exists(mask):
        _, labs = spanning_labels(mask)
        return labs.size > 0

    def endpoints_from_faces(target_mask):
        sl_in = [slice(None)] * target_mask.ndim
        sl_out = [slice(None)] * target_mask.ndim
        sl_in[axis] = 0
        sl_out[axis] = -1

        inlet_points = np.argwhere(target_mask[tuple(sl_in)])
        outlet_points = np.argwhere(target_mask[tuple(sl_out)])
        if inlet_points.size == 0 or outlet_points.size == 0:
            return None, None

        a0 = inlet_points[
            np.argmin(np.sum((inlet_points - inlet_points.mean(axis=0)) ** 2, axis=1))
        ]
        b0 = outlet_points[np.argmin(np.sum((outlet_points - a0) ** 2, axis=1))]

        start = a0.tolist()
        end = b0.tolist()
        start.insert(axis, 0)
        end.insert(axis, target_mask.shape[axis] - 1)
        return tuple(start), tuple(end)

    def add_minimal_bridge(im_bool, target_is_pore=True):
        target = im_bool if target_is_pore else ~im_bool
        start, end = endpoints_from_faces(target)
        if start is None:
            center = [s // 2 for s in target.shape]
            start_list = center.copy()
            end_list = center.copy()
            start_list[axis] = 0
            end_list[axis] = target.shape[axis] - 1
            start, end = tuple(start_list), tuple(end_list)

        cost = np.where(target, 1.0, penalty).astype(float)
        path, _ = route_through_array(
            cost,
            start,
            end,
            fully_connected=(connectivity > 1),
            geometric=True,
        )
        path = np.asarray(path, dtype=int)

        if target_is_pore:
            im_bool[tuple(path.T)] = True
        else:
            im_bool[tuple(path.T)] = False
        return im_bool

    def keep_only_spanning(mask):
        labeled, labs = spanning_labels(mask)
        if labs.size == 0:
            return mask
        return np.isin(labeled, labs)

    def force_face_contact(im_bool):
        shape = im_bool.shape
        center = [s // 2 for s in shape]

        a = tuple(0 if d == axis else center[d] for d in range(im_bool.ndim))
        b = tuple(shape[axis] - 1 if d == axis else center[d] for d in range(im_bool.ndim))

        im_bool[a] = True
        im_bool[b] = True

        a2 = list(a)
        b2 = list(b)
        a2[axis] = min(1, shape[axis] - 1)
        b2[axis] = max(shape[axis] - 2, 0)
        im_bool[tuple(a2)] = False
        im_bool[tuple(b2)] = False
        return im_bool

    for _ in range(max_iters):
        if ensure_face_contact:
            im2 = force_face_contact(im2)

        if not spanning_exists(im2):
            im2 = add_minimal_bridge(im2, target_is_pore=True)

        if not spanning_exists(~im2):
            im2 = add_minimal_bridge(im2, target_is_pore=False)

        if keep_spanning_only:
            im2 = keep_only_spanning(im2)
            im2 = ~keep_only_spanning(~im2)

        if spanning_exists(im2) and spanning_exists(~im2):
            break

    return im2


def safe_boundary_pores_with_area(pn, regions, args=None, axis=0):
    """Find inlet/outlet boundary pore IDs and their boundary-face voxel areas."""
    if args is None:
        args = {}
    min_face_area = int(float(args.get("min_boundary_face_area", 1)))

    coords = pn["pore.coords"].astype(float)
    reg_labels = pn["pore.region_label"].astype(np.int64)
    axis_coords = coords[:, axis]

    def one_face(face_index, pick_min=True):
        slicer = [slice(None)] * 3
        slicer[axis] = face_index
        face_map = regions[tuple(slicer)]

        face_regs, counts = np.unique(face_map, return_counts=True)
        valid = (face_regs > 0) & (counts >= min_face_area)
        face_regs = face_regs[valid]
        counts = counts[valid]

        chosen_pores = []
        chosen_areas = []
        for reg, area_vox in zip(face_regs, counts):
            pores_in_reg = np.where(reg_labels == reg)[0]
            if pores_in_reg.size == 0:
                continue
            if pick_min:
                pore = pores_in_reg[np.argmin(axis_coords[pores_in_reg])]
            else:
                pore = pores_in_reg[np.argmax(axis_coords[pores_in_reg])]
            chosen_pores.append(pore)
            chosen_areas.append(int(area_vox))

        if not chosen_pores:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        chosen_pores = np.asarray(chosen_pores, dtype=np.int64)
        chosen_areas = np.asarray(chosen_areas, dtype=np.int64)
        unique_pores, inverse = np.unique(chosen_pores, return_inverse=True)
        area_sum = np.zeros_like(unique_pores, dtype=np.int64)
        np.add.at(area_sum, inverse, chosen_areas)
        return unique_pores, area_sum

    left_pores, left_area = one_face(face_index=0, pick_min=True)
    right_pores, right_area = one_face(face_index=-1, pick_min=False)
    return left_pores, left_area, right_pores, right_area


def format_general_params(params):
    """Format simple key-value parameters at the start of the input file."""
    return "\n".join(f"[{key.upper()}]\n{value}" for key, value in params.items())


def format_table(section_title, comments, data, int_col=4, col_separator="\t"):
    """Format one tabular input-file section."""
    with io.StringIO() as buffer:
        buffer.write(f"{section_title}\n")
        buffer.write(f"{comments.strip()}\n")
        for row in data:
            str_row = [
                str(int(item)) if i < int_col else str(item)
                for i, item in enumerate(row)
            ]
            buffer.write(f"{col_separator.join(str_row)}\n")
        buffer.write(";\n")
        return buffer.getvalue()


def add_ghost_layer(
    pore_body_data,
    pore_throat_data,
    boundary_data,
    coordination_data,
    boundary_type_for_ghost=0,
    pore_radius_col=2,
    pore_length_col=3,
    ghost_throat_length=None,
):
    """Add one ghost pore and ghost throat for each outlet boundary pore."""

    def as_int(x):
        return int(float(x))

    def as_float(x):
        return float(x)

    new_pores = [list(row) for row in pore_body_data]
    new_throats = [list(row) for row in pore_throat_data]
    new_boundaries = [list(row) for row in boundary_data]
    new_coords = [list(row) for row in coordination_data]

    pore_by_id = {as_int(row[0]): row for row in new_pores}
    coord_by_id = {as_int(row[0]): row for row in new_coords}

    target_rows = [
        row for row in new_boundaries
        if as_int(row[1]) == boundary_type_for_ghost
    ]
    if not target_rows:
        return new_pores, new_throats, new_boundaries, new_coords

    target_ids = [as_int(row[0]) for row in target_rows]
    missing_pores = [pid for pid in target_ids if pid not in pore_by_id]
    missing_coords = [pid for pid in target_ids if pid not in coord_by_id]
    if missing_pores:
        raise ValueError(f"Boundary pore IDs missing from pore_body_data: {missing_pores[:20]}")
    if missing_coords:
        raise ValueError(f"Boundary pore IDs missing from coordination_data: {missing_coords[:20]}")

    if ghost_throat_length is None:
        ghost_length = max(as_float(pore_by_id[pid][pore_radius_col]) for pid in target_ids)
    else:
        ghost_length = float(ghost_throat_length)
        if not np.isfinite(ghost_length) or ghost_length <= 0:
            raise ValueError("ghost_throat_length must be a positive finite value.")

    next_pore_id = max(as_int(row[0]) for row in new_pores) + 1
    next_throat_id = max(as_int(row[0]) for row in new_throats) + 1
    ghost_id_map = {}

    for original_id in target_ids:
        ghost_id = next_pore_id
        next_pore_id += 1
        ghost_id_map[original_id] = ghost_id

        original_pore = list(pore_by_id[original_id])
        original_coord = list(coord_by_id[original_id])

        ghost_pore = list(original_pore)
        ghost_pore[0] = ghost_id
        new_pores.append(ghost_pore)

        ghost_coord = list(original_coord)
        ghost_coord[0] = ghost_id
        new_coords.append(ghost_coord)

        pore_shape = original_pore[1]
        pore_radius = original_pore[pore_radius_col]
        extra_geometry_cols = original_pore[pore_length_col + 1:]
        ghost_throat = [
            next_throat_id,
            original_id,
            ghost_id,
            pore_shape,
            pore_radius,
            ghost_length,
        ] + extra_geometry_cols
        next_throat_id += 1
        new_throats.append(ghost_throat)

    for row in new_boundaries:
        if as_int(row[1]) == boundary_type_for_ghost:
            row[0] = ghost_id_map[as_int(row[0])]

    print(f"Added {len(ghost_id_map)} ghost pore bodies.")
    print(f"Added {len(ghost_id_map)} ghost pore throats.")
    print(f"Ghost throat length = {ghost_length}")

    return new_pores, new_throats, new_boundaries, new_coords


def generate_input_file(
    pore_body_data,
    pore_throat_data,
    boundary_data,
    coordination_data,
    output_filename="ExtractedPNM",
    add_ghost=True,
    ghost_throat_length=None,
):
    """Write the pore-network model text input file."""
    if add_ghost:
        pore_body_data, pore_throat_data, boundary_data, coordination_data = add_ghost_layer(
            pore_body_data=pore_body_data,
            pore_throat_data=pore_throat_data,
            boundary_data=boundary_data,
            coordination_data=coordination_data,
            boundary_type_for_ghost=0,
            ghost_throat_length=ghost_throat_length,
        )

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

    file_content = [
        format_general_params(general_params),
        format_table("[PORE_BODY]", pore_body_comments, pore_body_data, int_col=2).rstrip(),
        format_table("[PORE_THROAT]", pore_throat_comments, pore_throat_data, int_col=4).rstrip(),
        format_table("[BOUNDARY]", boundary_comments, boundary_data, int_col=3).rstrip(),
        format_table("[COORDINATION]", coordination_comments, coordination_data, int_col=1).rstrip(),
        notes.strip(),
    ]

    with open(output_filename, "w") as f:
        f.write("\n".join(file_content))

    print(f"\n--- File successfully saved as '{output_filename}' ---")
    print("Finish generating PNM file!!!")

import os
import sys
import numpy as np
import meshio

# =========================
# Point projector
# =========================


def radial_project_no_clamp(p, center, a, b, c, n):
    """
    Radially project p into the super-ellipsoid.
    """
    d = p - center
    if np.allclose(d, 0.0):
        return p.copy()
    val = (abs(d[0]/a)**n + abs(d[1]/b)**n + abs(d[2]/c)**n)
    if val <= 0.0:
        return p.copy()
    s = (1.0 / val)**(1.0 / n)
    return center + d * s


def clamp_z_to_cut(q, z_cut):
    """Clamp z to z_cut."""
    if q[2] > z_cut:
        qq = q.copy()
        qq[2] = z_cut
        return qq
    return q

# =========================
# Bounding box detection
# =========================


def build_move_map_for_boundaries(
    inp_path,
    *,
    plane_tol=1e-5,
    xc=0.0, yc=0.0, zc=0.0, a=1.0, b=1.0, c=1.0, n=2.0,
    z_cut=None,
    threshold=None, tol=1e-12
):
    """
    1. Reads mesh.
    2. Auto-detects Min/Max X, Min/Max Y, Min/Max Z.
    3. Computes projections and returns {tag: new_xyz}.
    """
    mesh = meshio.read(inp_path)
    P = mesh.points

    node_tags = None
    if "gmsh:node_tags" in mesh.point_data:
        node_tags = mesh.point_data[
            "gmsh:node_tags"].astype(np.int64).reshape(-1)
    else:
        node_tags = np.arange(1, P.shape[0] + 1, dtype=np.int64)

    # Auto-detect Bounds
    min_x, min_y, min_z = np.min(P, axis=0)
    max_x, max_y, max_z = np.max(P, axis=0)

    print(f"[Info] Mesh Bounds Detected:")
    print(f"       X: [{min_x:.4f}, {max_x:.4f}]")
    print(f"       Y: [{min_y:.4f}, {max_y:.4f}]")
    print(f"       Z: [{min_z:.4f}, {max_z:.4f}]")

    # Select Nodes on the 5 Faces
    # use np.isclose to find nodes on the planes
    mask_min_x = np.isclose(P[:, 0], min_x, atol=plane_tol)
    mask_max_x = np.isclose(P[:, 0], max_x, atol=plane_tol)
    mask_min_y = np.isclose(P[:, 1], min_y, atol=plane_tol)
    mask_max_y = np.isclose(P[:, 1], max_y, atol=plane_tol)
    mask_min_z = np.isclose(P[:, 2], min_z, atol=plane_tol)

    # Combine masks
    final_mask = mask_min_x | mask_max_x | mask_min_y | mask_max_y | mask_min_z

    ids = np.where(final_mask)[0]

    center = np.array([xc, yc, zc], dtype=float)
    move_map = {}
    considered = 0
    snapped = 0

    for i in ids:
        p = P[i]
        q_raw = radial_project_no_clamp(p, center, a, b, c, n)
        dist = np.linalg.norm(q_raw - p)

        # threshold check
        if threshold is not None and not (dist <= float(threshold) + tol):
            continue

        considered += 1
        q = clamp_z_to_cut(q_raw, z_cut) if (z_cut is not None) else q_raw
        move_map[int(node_tags[i])] = q
        snapped += 1

    print(f"[move_map] Boundary nodes selected: {len(ids)}, "
          f"considered: {considered}, to move: {snapped}")
    return move_map

# =========================
# ASCII clone editor (Gmsh v2.2 and v4.x)
# =========================


def _format_triplet(xyz):
    return f"{xyz[0]:.16g} {xyz[1]:.16g} {xyz[2]:.16g}"


def clone_with_moved_nodes_v22(lines, move_map):
    out = list(lines)
    i = 0
    n = len(lines)
    while i < n:
        if lines[i].strip() == "$Nodes":
            j = i + 1
            if j >= n:
                break
            try:
                N = int(lines[j].strip())
            except ValueError:
                i += 1
                continue

            k0 = j + 1
            k1 = k0 + N
            for k in range(k0, min(k1, n)):
                parts = lines[k].strip().split()
                if not parts:
                    continue
                try:
                    tag = int(float(parts[0]))
                except Exception:
                    continue

                if tag in move_map:
                    new_xyz = _format_triplet(move_map[tag])
                    out[k] = f"{tag} {new_xyz}\n"
            i = k1
        i += 1
    return out


def clone_with_moved_nodes_v4(lines, move_map):
    out = list(lines)
    i = 0
    n = len(lines)
    while i < n:
        if lines[i].strip() == "$Nodes":
            j = i + 1
            if j >= n:
                break
            hdr = lines[j].strip().split()
            if len(hdr) != 4:
                break
            num_blocks = int(hdr[0])
            ptr = j + 1
            for _ in range(num_blocks):
                if ptr >= n:
                    break
                bh = lines[ptr].strip().split()
                ptr += 1
                if len(bh) != 4:
                    break
                entityDim, entityTag, parametric, \
                    numNodesInBlock = map(int, bh)

                tags = []
                for _t in range(numNodesInBlock):
                    if ptr >= n:
                        break
                    tags.append(int(float(lines[ptr].strip().split()[0])))
                    ptr += 1

                for idx_in_block in range(numNodesInBlock):
                    if ptr >= n:
                        break
                    tag = tags[idx_in_block]
                    if tag in move_map:
                        parts = lines[ptr].rstrip("\n").split()
                        tail = ""
                        if len(parts) > 3:
                            tail = " " + " ".join(parts[3:])
                        out[ptr] = f"{_format_triplet(move_map[tag])}{tail}\n"
                    ptr += 1
            break
        i += 1
    return out


def detect_gmsh_ascii_version(lines):
    n = len(lines)
    for i in range(n):
        if lines[i].strip() == "$Nodes":
            if i + 1 >= n:
                break
            parts = lines[i+1].strip().split()
            if len(parts) == 1:
                return "2.2"
            if len(parts) == 4:
                return "4"
            break
    return None


def clone_ascii_and_move_nodes(input_msh_path, output_msh_path, move_map):
    with open(input_msh_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    version = detect_gmsh_ascii_version(lines)
    if version is None:
        raise RuntimeError("Could not detect $Nodes layout (v2.2 or v4.x).")

    if version == "2.2":
        new_lines = clone_with_moved_nodes_v22(lines, move_map)
    else:
        new_lines = clone_with_moved_nodes_v4(lines, move_map)

    with open(output_msh_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"Wrote ASCII file with moved nodes: "
          f"{output_msh_path} (layout v{version})")

# =========================
# Public API
# =========================


def snap_boundaries_by_text_cloning(
    input_msh_path,
    output_msh_path,
    *,
    plane_tol=1e-5,
    xc=0.0, yc=0.0, zc=0.0, a=1.0, b=1.0, c=1.0, n=2.0,
    z_cut=None,
    threshold=None, tol=1e-12
):
    """
    Auto-detects mesh boundaries (Min/Max X, Min/Max Y, Min Z)
    and snaps them to the super-ellipsoid.
    """
    move_map = build_move_map_for_boundaries(
        input_msh_path,
        plane_tol=plane_tol,
        xc=xc, yc=yc, zc=zc, a=a, b=b, c=c, n=n,
        z_cut=z_cut, threshold=threshold, tol=tol
    )
    if not move_map:
        print("No nodes selected to move")
        with open(input_msh_path, "r", encoding="utf-8") as \
                fin, open(output_msh_path, "w", encoding="utf-8") as fout:
            fout.write(fin.read())
        return
    clone_ascii_and_move_nodes(input_msh_path, output_msh_path, move_map)


# =========================
# Usage
# =========================
if __name__ == "__main__":
    # Example parameters
    scale = 1.0
    domainX = 1.0 * scale
    domainY = 1.0 * scale
    domainZ = 1.0 * scale

    ellipseLx = 0.5 * scale
    ellipseLy = 0.5 * scale
    ellipseLz = 0.5 * scale
    ellipse_n = 3.0
    z_cut = 0.0

    # Super-ellipsoid center
    xc, yc, zc = domainX/2.0, domainY/2.0, -domainZ/2.0
    a = domainX/2.0 + ellipseLx
    b = domainY/2.0 + ellipseLy
    c = domainZ/2.0 + ellipseLz

    snap_boundaries_by_text_cloning(
        "n3.0sou.msh",
        "n3.0souSNAP.msh",
        plane_tol=1e-3,  # Tolerance to find nodes on the boundary
        xc=xc, yc=yc, zc=zc,
        a=a, b=b, c=c, n=ellipse_n,
        z_cut=z_cut,
        tol=1e-12
    )

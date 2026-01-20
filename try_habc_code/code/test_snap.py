import numpy as np
import meshio
import firedrake as fire


def radial_project_no_clamp(p, center, a, b, c, n):
    d = p - center
    if np.allclose(d, 0.0):
        return p.copy()
    val = (abs(d[0]/a) ** n + abs(d[1]/b) ** n + abs(d[2]/c) ** n)
    if val <= 0.0:
        return p.copy()
    s = (1.0 / val)**(1.0 / n)
    q_raw = center + d * s
    return q_raw


def clamp_z(q, z_cut):
    if q[2] > z_cut:
        qc = q.copy()
        qc[2] = z_cut
        return qc
    return q


def superellipsoid_F(X, Y, Z, xc, yc, zc, a, b, c, n):
    return (np.abs((X - xc) / a) ** n
            + np.abs((Y - yc) / b) ** n
            + np.abs((Z - zc) / c) ** n)


def snap_meshfile_to_superellipsoid(
    input_msh_path,
    output_msh_path,
    *,
    xc, yc, zc,
    a, b, c, n,
    z_cut=0.0,
    threshold=0.02,    # snap if radial ||q_raw - p|| <= threshold
    tol=1e-12,         # numerical tolerance
    exclude_z_plane=True,
    z_plane_tol=1e-12  # exclude nodes with |z - z_cut| <= z_plane_tol
):

    mesh = meshio.read(input_msh_path)
    P = mesh.points.copy()
    center = np.array([xc, yc, zc], dtype=float)
    Fvals = superellipsoid_F(P[:, 0], P[:, 1], P[:, 2], xc, yc, zc, a, b, c, n)
    inside_mask = (Fvals <= 1.0 + tol) & (P[:, 2] <= z_cut + tol)
    if exclude_z_plane:
        not_plane = np.abs(P[:, 2] - z_cut) > z_plane_tol
        inside_mask &= not_plane

    snapped = 0
    considered = 0

    for vid in np.where(inside_mask)[0]:
        p = P[vid]
        q_raw = radial_project_no_clamp(p, center, a, b, c, n)
        dist_rad = np.linalg.norm(q_raw - p)
        if not (dist_rad <= threshold + tol):
            continue

        considered += 1
        q = clamp_z(q_raw, z_cut)
        P[vid] = q
        snapped += 1

    out_mesh = meshio.Mesh(
        points=P,
        cells=mesh.cells,
        point_data=mesh.point_data,
        cell_data=mesh.cell_data,
        field_data=mesh.field_data,
        cell_sets=getattr(mesh, "cell_sets", None),
    )
    meshio.write(output_msh_path, out_mesh, file_format="gmsh22", binary=True)

    q = {"overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)}
    final_mesh = fire.Mesh(output_msh_path, distribution_parameters=q)
    fire.VTKFile("box_snap.pvd").write(final_mesh)

    print("Snapping complete.")
    print(f"  Considered (inside & within threshold): {considered}")
    print(f"  Snapped:                                {snapped}")
    print(f"  Wrote: {output_msh_path}")


if __name__ == "__main__":
    scale = 1.0
    domainX = 1.0 * scale
    domainY = 1.0 * scale
    domainZ = 1.0 * scale

    ellipseLx = 0.5 * scale
    ellipseLy = 0.5 * scale
    ellipseLz = 0.5 * scale
    ellipse_n = 2.8
    z_cut = 0.0

    xc, yc, zc = domainX/2.0, domainY/2.0, -domainZ/2.0
    a = domainX/2.0 + ellipseLx
    b = domainY/2.0 + ellipseLy
    c = domainZ/2.0 + ellipseLz

    snap_meshfile_to_superellipsoid(
        "ellipsoid_mesh.msh",
        "snapped_output.msh",
        xc=xc, yc=yc, zc=zc,
        a=a, b=b, c=c, n=ellipse_n,
        z_cut=z_cut,
        threshold=0.1,
        tol=1e-12,
        exclude_z_plane=True,
        z_plane_tol=1e-12
    )

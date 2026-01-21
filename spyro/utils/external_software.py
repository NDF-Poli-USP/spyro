def initialize_gmsh():
    '''
    Initialize Gmsh API if not already initialized.

    Returns
    -------
    None
    '''

    if gmsh.isInitialized():
        import gmsh
        print("Gmsh is already initialized.")
    else:
        gmsh.initialize()
        print("Gmsh was not initialized, now it is.")

    # -  0: disables all output messages
    # -  1: minimal output
    # -  2: default verbosity
    # - 99: maximum verbosity
    gmsh.option.setNumber("General.Verbosity", 1)


def detect_gmsh_version(info_file=None):
    '''
    Detect Gmsh version from ASCII file or installed
    in the system. If the version is detected from
    file, only Gmsh v2.2 and v4.x formats are supported.

    Parameters
    ----------
    info_file : list of str, optional
        Lines of an ASCII Gmsh file.
    Returns
    -------
    str_ver : str
        Gmsh version as a string. It returns "2.2" or "4" if
        detected from file, otherwise the installed version.
    '''

    if info_file:
        print("Getting Gmsh version from ASCII file")
        n = len(info_file)
        for i in range(n):
            if info_file[i].strip() == "$Nodes":
                if i + 1 >= n:
                    break
                parts = info_file[i+1].strip().split()
                if len(parts) == 1:
                    return "2.2"
                if len(parts) == 4:
                    return "4"
                break
        return None

    print("Getting Gmsh version installed in the system")
    try:
        initialize_gmsh()
        version = gmsh.option.getString("General.Version")
        print("Gmsh version:", version)
        gmsh.finalize()

    except ImportError:
        print("Gmsh is not installed.")

    str_ver = ".".join(version.split(".")[:2])

    return str_ver


def read_gmsh_file(input_msh_path):
    '''
    Read an ASCII Gmsh file and detect its version.

    Parameters
    ----------
    input_msh_path : str
        Path to the input ASCII Gmsh file.

    Returns
    -------
    lines : list of str
        Lines of the input ASCII Gmsh file.
    '''

    with open(input_msh_path, "r", encoding="utf-8") as f:
        info_file = f.readlines()

    version = detect_gmsh_version(info_file=info_file)
    if version is None:
        raise RuntimeError(
            f"Could not detect $Nodes layout in {input_msh_path}. "
            "Only Gmsh v2.2 and v4.x formats are supported.")

    return info_file


def report_mesh_quality(dim=3, quality_type=2):
    '''
    Report mesh quality statistics for elements in a Gmsh mesh

    Parameters
    ----------
    dim : `int`, optional
        Dimension of elements to evaluate (2 for surface, 3 for volume).
        Default is 3 (volume elements).
    quality_type : `int`, optional
        Quality metric type to use (0=gamma, 1=eta, 2=rho).
        gamma: vol/sum_face/max_edge, eta : vol^(2/3)/sum_edge^2,
        rho: min_edge/max_edge. Default is 2 (rho).

    Returns
    -------
    None
    '''

    # Initialize Gmsh API
    initialize_gmsh()
    gmsh.option.setNumber("Mesh.QualityType", quality_type)

    # Grab all elements of this dimension (returns per-type lists)
    ele_types, ele_tags, node_tags = gmsh.model.mesh.getElements(dim)

    # Flatten to a single list of element tags
    all_tags = []
    for tags in ele_tags:
        all_tags.extend(tags.tolist() if hasattr(
            tags, "tolist") else list(tags))

    if not all_tags:
        print(f"[Quality] No elements found for dim={dim}", flush=True)
        return

    # Compute qualities for elements
    q = gmsh.model.mesh.getElementQualities(all_tags)
    q = np.asarray(q, dtype=float)
    print(f"[Quality] Count={q.size} Min={q.min():.6g} "
          f"p1={np.percentile(q, 1):.6g} - p5={np.percentile(q, 5):.6g}\n"
          f"Median={np.median(q):.6g} p95={np.percentile(q, 95):.6g} - "
          f"Max={q.max():.6g} - Mean={q.mean():.6g}", flush=True)
    gmsh.finalize()

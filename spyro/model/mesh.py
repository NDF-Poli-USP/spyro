from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import firedrake as fire
from .. import meshing


class MeshType(Enum):
    FIREDRAKE_MESH = "firedrake_mesh"
    USER_MESH = "user_mesh"
    SEISMIC_MESH = "SeismicMesh"
    FILE = "file"
    SPYRO_MESH = "spyro_mesh"
    GMSH_MESH = "gmsh_mesh"


class FiniteElementMethod(Enum):
    MASS_LUMPED_TRIANGLE = "mass_lumped_triangle"
    DG_TRIANGLE = "DG_triangle"
    SPECTRAL_QUADRILATERAL = "spectral_quadrilateral"
    DG_QUATRILATERAL = "DG_quadrilateral"
    CG_TRIANGLE = "CG_triangle"


class CellType(Enum):
    QUADRILATERAL = "quadrilateral"
    TRIANGLE = "triangle"


class Mesh(ABC):
    @abstractmethod
    def get_mesh(self, mesh_definition):
        pass


@dataclass
class MeshDefinition:
    mesh: Mesh
    domain_dimensions: list[float]
    negative_z: bool
    mesh_type: MeshType
    finite_element_method: FiniteElementMethod
    abc_pad_length: float
    cell_type: CellType
    degree: int

    dimension: int = field(init=False)

    def __post_init__(self):
        self.dimension = len(self.domain_dimensions)
        if not all(x > 0 for x in self.domain_dimensions):
            raise ValueError("All dimension lengths should be positive")

        if self.negative_z:
            self.domain_dimensions[0] = -self.domain_dimensions[0]

    def assert_point_in_domain(self, point_coordinates):
        """
        Checks if a point is within the mesh domain.

        Parameters
        ----------
        point_coordinates : list
            Coordinates of the point to check.

        Raises
        ------
        ValueError
            If the point is outside the mesh domain.
        """
        for i, (coord, length) in enumerate(
            zip(point_coordinates, self.domain_dimensions)
        ):
            error_message = (
                f"Coordinate {coord} in dimension {i} is outside the domain "
            )
            self._assert_in_range(coord, length, error_message)

    def _assert_in_range(self, coord, length, error_message):
        low = min(0, length)
        high = max(0, length)
        if not low <= coord <= high:
            raise ValueError(error_message + f"[{length}, 0].")

    def get_mesh(self):
        return self.mesh.get_mesh(self)


@dataclass
class ExistingMesh(Mesh):
    mesh: object

    def get_mesh(self, _):
        return self.mesh


@dataclass
class MeshFile(Mesh):
    file_name: str

    def get_mesh(self, mesh_definition: MeshDefinition):
        method = mesh_definition.method

        return fire.Mesh(
            mesh_definition.mesh_file,
            comm=mesh_definition.comm,
            distribution_parameters={
                "overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)
            }
            if method
            in [
                FiniteElementMethod.CG_TRIANGLE,
                FiniteElementMethod.MASS_LUMPED_TRIANGLE,
            ]
            else None,
        )


class AutomatedMesh(Mesh):
    edges_length: list[float]
    cells_per_wavelength: object  # check type, mutually exclusive with edge length?, need frequency if set

    def get_mesh(self, mesh_definition):
        autoMeshing = meshing.AutomaticMesh(mesh_parameters=mesh_definition)
        return autoMeshing.create_mesh()

import numpy as np
import torch
from typing import NamedTuple


class BlockMeshPointCloud(NamedTuple):
    alpha: torch.Tensor
    points: torch.Tensor
    colors: np.array
    normals: np.array
    vertices: torch.Tensor
    faces: torch.Tensor
    transform_vertices_function: object
    triangles: torch.Tensor
    S:torch.Tensor
    R_4d:torch.Tensor
    T:torch.Tensor
    occupancy:torch.Tensor
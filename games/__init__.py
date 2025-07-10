from arguments import (
    OptimizationParamsBlock,
    OptimizationParamsPart
)
from games.block_mesh_splatting.scene.block_gaussian_model import BlockGaussianModel
from games.block_mesh_splatting.scene.two_gaussian_model import TwoGaussianModel

optimizationParamTypeCallbacks = {
    "gs_block": OptimizationParamsBlock,
    "gs_point": OptimizationParamsPart
}

gaussianModel = {
    'gs_block': BlockGaussianModel,
    "gs_point": TwoGaussianModel
}

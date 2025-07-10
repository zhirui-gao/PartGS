

from scene.dataset_readers import (
    readColmapSceneInfo,
    readNerfSyntheticInfo,
    readNeuSDTUInfo
)

from games.block_mesh_splatting.scene.dataset_readers import(
    readNeRFSyntheticBlockMeshInfo,
    readNeuSDTUBlockInfo,
    readColmapBlockMeshInfo,
    readReplicaBlockMeshInfo
)


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Colmap_Block": readColmapBlockMeshInfo,
    "Blender": readNerfSyntheticInfo,
    "Blender_Block": readNeRFSyntheticBlockMeshInfo,
    "sphere": readNeuSDTUInfo,
    "sphere_Block": readNeuSDTUBlockInfo,
    "Replica_Block": readReplicaBlockMeshInfo,
}

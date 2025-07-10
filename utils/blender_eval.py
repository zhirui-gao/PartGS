
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
from pytorch3d.structures import Meshes as P3DMeshes
from tqdm import tqdm

MAX_DIST = 1.0
DOWNSAMPLE_DENSITY = 0.005

def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1 + 1, :n2 + 1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1, 2, 0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:, :1] + v2 * k[:, 1:] + tri_vert
    return q
def evaluate_mesh(inp, stl, eval_dir, suffix='',step='-1', num_mesh=1):
    os.makedirs(eval_dir,exist_ok=True)
    if isinstance(inp, (str, Path)):
        data_mesh = o3d.io.read_triangle_mesh(str(inp))
    elif isinstance(inp, P3DMeshes):
        verts, faces = list(map(lambda t: t.detach().cpu().numpy(), inp.get_mesh_verts_faces(0)))
        verts, faces = o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces)
        data_mesh = o3d.geometry.TriangleMesh(verts, faces)

    data_mesh.remove_unreferenced_vertices()
    mp.freeze_support()

    pbar = tqdm(total=6)
    pbar.set_description('read data mesh')

    vertices = np.asarray(data_mesh.vertices)
    triangles = np.asarray(data_mesh.triangles)
    tri_vert = vertices[triangles]

    pbar.update(1)
    pbar.set_description('sample pcd from mesh')
    v1 = tri_vert[:, 1] - tri_vert[:, 0]
    v2 = tri_vert[:, 2] - tri_vert[:, 0]
    l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
    non_zero_area = (area2 > 0)[:, 0]
    l1, l2, area2, v1, v2, tri_vert = [
        arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]
    ]
    thr = DOWNSAMPLE_DENSITY * np.sqrt(l1 * l2 / area2)
    n1 = np.floor(l1 / thr)
    n2 = np.floor(l2 / thr)

    with mp.Pool() as mp_pool:
        new_pts = mp_pool.map(sample_single_tri,
                              ((n1[i, 0], n2[i, 0], v1[i:i + 1], v2[i:i + 1], tri_vert[i:i + 1, 0]) for i in
                               range(len(n1))), chunksize=1024)

    new_pts = np.concatenate(new_pts, axis=0)
    data_pcd = np.concatenate([vertices, new_pts], axis=0)

    pbar.update(1)
    pbar.set_description('random shuffle pcd index')
    shuffle_rng = np.random.default_rng()
    shuffle_rng.shuffle(data_pcd, axis=0)

    pbar.update(1)
    pbar.set_description('downsample pcd')
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=DOWNSAMPLE_DENSITY, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(data_pcd)
    rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=DOWNSAMPLE_DENSITY, return_distance=False)
    mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1
    data_down = data_pcd[mask]

    pbar.update(1)
    pbar.set_description('read STL pcd')
    pbar.update(1)
    pbar.set_description('compute data2stl')
    nn_engine.fit(stl)
    dist_d2s, idx_d2s = nn_engine.kneighbors(data_pcd, n_neighbors=1, return_distance=True)

    mean_d2s = dist_d2s[dist_d2s < MAX_DIST].mean()

    pbar.update(1)
    pbar.set_description('compute stl2data')

    nn_engine.fit(data_down)
    dist_s2d, idx_s2d = nn_engine.kneighbors(stl, n_neighbors=1, return_distance=True)
    mean_s2d = dist_s2d[dist_s2d < MAX_DIST].mean()

    pbar.close()
    avg = (mean_d2s + mean_s2d) / 2
    print("chamfer distance:", mean_d2s, mean_s2d, avg)

    with open(f'{eval_dir}/blender_scores{suffix}_{step}.tsv', 'w') as f:
        f.write(f'acc\tcomp\tavg\n')
        f.write(f'{mean_d2s}\t{mean_s2d}\t{avg}\n')
        f.write(f'num_mesh\n')
        f.write(f'{num_mesh}')
        print('dtu_scores{}: acc={:.5f}, comp={:.5f}, avg={:.5f}, num_mesh={:d}'.format(suffix, mean_d2s, mean_s2d,
                                                                                        avg, num_mesh))
    return avg


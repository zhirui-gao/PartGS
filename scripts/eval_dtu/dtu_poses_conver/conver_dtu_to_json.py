

import os
import numpy as np
import json
import sys
from pathlib import Path
from argparse import ArgumentParser
import trimesh

dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[2]
sys.path.append(dir_path.__str__())

from submodules.colmap.scripts.python.database import COLMAPDatabase  # NOQA
from submodules.colmap.scripts.python.read_write_model import read_model, rotmat2qvec  # NOQA


def create_init_files(pinhole_dict_file, db_file, out_dir):


    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # create template
    with open(pinhole_dict_file) as fp:
        pinhole_dict = json.load(fp)

    template = {}
    cameras_line_template = '{camera_id} PINHOLE {width} {height} {f} {cx} {cy} {k1} {k2}\n'
    images_line_template = '{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {image_name}\n\n'

    for img_name in pinhole_dict:
        # w, h, fx, fy, cx, cy, qvec, t
        params = pinhole_dict[img_name]
        w = params[0]
        h = params[1]
        fx = params[2]
        # fy = params[3]
        cx = params[4]
        cy = params[5]
        qvec = params[6:10]
        tvec = params[10:13]

        cam_line = cameras_line_template.format(
            camera_id="{camera_id}", width=w, height=h, f=fx, cx=cx, cy=cy, k1=0, k2=0)
        img_line = images_line_template.format(image_id="{image_id}", qw=qvec[0], qx=qvec[1], qy=qvec[2], qz=qvec[3],
                                               tx=tvec[0], ty=tvec[1], tz=tvec[2], camera_id="{camera_id}",
                                               image_name=img_name)
        template[img_name] = (cam_line, img_line)

    # read database
    db = COLMAPDatabase.connect(db_file)
    table_images = db.execute("SELECT * FROM images")
    img_name2id_dict = {}
    for row in table_images:
        img_name2id_dict[row[1]] = row[0]

    cameras_txt_lines = [template[img_name][0].format(camera_id=1)]
    images_txt_lines = []
    for img_name, img_id in img_name2id_dict.items():
        image_line = template[img_name][1].format(image_id=img_id, camera_id=1)
        images_txt_lines.append(image_line)

    with open(os.path.join(out_dir, 'cameras.txt'), 'w') as fp:
        fp.writelines(cameras_txt_lines)

    with open(os.path.join(out_dir, 'images.txt'), 'w') as fp:
        fp.writelines(images_txt_lines)
        fp.write('\n')

    # create an empty points3D.txt
    fp = open(os.path.join(out_dir, 'points3D.txt'), 'w')
    fp.close()

def load_K_Rt_from_P(filename, P=None):
    import cv2 as cv
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]]
                 for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return K, pose

def convert_cam_dict_to_pinhole_dict(camera_dict, pinhole_dict_file, img_names):


    print('Writing pinhole_dict to: ', pinhole_dict_file)
    h = 1200
    w = 1600
    pinhole_dict = {}
    for idx in range(0, len(camera_dict)//6):
        world_mat = camera_dict['world_mat_%d' % idx].astype(np.float32)
        scale_mat = camera_dict['scale_mat_%d' % idx].astype(np.float32)
        P = world_mat @ scale_mat
        P = P[:3, :4]
        K, pose_c2w = load_K_Rt_from_P(None, P)

        W2C  = np.linalg.inv(pose_c2w)

        # params
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        qvec = rotmat2qvec(W2C[:3, :3])  # Assuming this function returns a list of floats
        tvec = [float(x) for x in W2C[:3, 3]]

        params = [w, h, fx, fy, cx, cy,
                  qvec[0], qvec[1], qvec[2], qvec[3],
                  tvec[0], tvec[1], tvec[2]]
        pinhole_dict[img_names[idx]] = params


    with open(pinhole_dict_file, 'w') as fp:
        json.dump(pinhole_dict, fp, indent=2, sort_keys=True)


def load_COLMAP_poses(cam_file, img_dir, tf='w2c'):
    # load img_dir namges
    names = sorted(os.listdir(img_dir))

    with open(cam_file) as f:
        lines = f.readlines()

    # C2W
    poses = {}
    for idx, line in enumerate(lines):
        if idx % 5 == 0:  # header
            img_idx, valid, _ = line.split(' ')
            if valid != '-1':
                poses[int(img_idx)] = np.eye(4)
                poses[int(img_idx)]
        else:
            if int(img_idx) in poses:
                num = np.array([float(n) for n in line.split(' ')])
                poses[int(img_idx)][idx % 5-1, :] = num

    if tf == 'c2w':
        return poses
    else:
        # convert to W2C (follow nerf convention)
        poses_w2c = {}
        for k, v in poses.items():
            poses_w2c[names[k]] = np.linalg.inv(v)
        return poses_w2c


def load_transformation(trans_file):
    with open(trans_file) as f:
        lines = f.readlines()

    trans = np.eye(4)
    for idx, line in enumerate(lines):
        num = np.array([float(n) for n in line.split(' ')])
        trans[idx, :] = num

    return trans


def align_gt_with_cam(pts, trans):
    trans_inv = np.linalg.inv(trans)
    pts_aligned = pts @ trans_inv[:3, :3].transpose(-1, -2) + trans_inv[:3, -1]
    return pts_aligned


def compute_bound(pts):
    bounding_box = np.array([pts.min(axis=0), pts.max(axis=0)])
    center = bounding_box.mean(axis=0)
    radius = np.max(np.linalg.norm(pts - center, axis=-1)) * 1.01
    return center, radius, bounding_box.T.tolist()

def init_colmap(args):
    assert args.dtu_path, "Provide path to Tanks and Temples dataset"
    scene_list = os.listdir(args.dtu_path)

    for scene in scene_list:
        scene_path = os.path.join(args.dtu_path, scene)

        if not os.path.exists(f"{scene_path}/image"):
            raise Exception(f"'image` folder cannot be found in {scene_path}."
                            "Please check the expected folder structure in DATA_PREPROCESSING.md")

        # extract features
        os.system(f"colmap feature_extractor --database_path {scene_path}/database.db \
                --image_path {scene_path}/image \
                --ImageReader.camera_model=PINHOLE \
                --SiftExtraction.use_gpu=true \
                --SiftExtraction.num_threads=32 \
                --ImageReader.single_camera=true"
                  )

        # match features
        os.system(f"colmap sequential_matcher \
                --database_path {scene_path}/database.db \
                --SiftMatching.use_gpu=true"
                  )

        # read poses
        camera_dict = np.load(os.path.join(scene_path, 'cameras.npz'))
        # convert to colmap files
        pinhole_dict_file = os.path.join(scene_path, 'pinhole_dict.json')
        convert_cam_dict_to_pinhole_dict(camera_dict, pinhole_dict_file,img_names=os.listdir(os.path.join(scene_path, 'image')))

        db_file = os.path.join(scene_path, 'database.db')
        sfm_dir = os.path.join(scene_path, 'sparse')
        create_init_files(pinhole_dict_file, db_file, sfm_dir)

        # bundle adjustment
        os.system(f"colmap point_triangulator \
                --database_path {scene_path}/database.db \
                --image_path {scene_path}/image \
                --input_path {scene_path}/sparse \
                --output_path {scene_path}/sparse \
                --Mapper.tri_ignore_two_view_tracks=true"
                  )
        os.system(f"colmap bundle_adjuster \
                --input_path {scene_path}/sparse \
                --output_path {scene_path}/sparse \
                --BundleAdjustment.refine_extrinsics=false"
                  )

        # undistortion
        # os.system(f"colmap image_undistorter \
        #     --image_path {scene_path}/image \
        #     --input_path {scene_path}/sparse \
        #     --output_path {scene_path} \
        #     --output_type COLMAP \
        #     --max_image_size 1500"
        #           )




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dtu_path', type=str, default='./data/DTU_IDR', help='Path to dtu dataset')
    args = parser.parse_args()
    init_colmap(args)
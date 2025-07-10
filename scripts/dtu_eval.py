import os
from argparse import ArgumentParser
#
dtu_scenes = ['scan37','scan97', 'scan105',  'scan63', 'scan37', 'scan69', 'scan55','scan65',
             'scan24',  'scan106', 'scan110', 'scan114', 'scan118', 'scan122','scan40'] #

import pandas
import json
parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval/dtu")
parser.add_argument('--dtu', "-dtu", default='/media/gzr/gzr_2t/datasets/DTU_posed_colmap', type=str)
args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(dtu_scenes)

cd_error_eva = 0
part_num_eva = 0
cd_error_eva_stage1 = 0

if not args.skip_metrics:
    parser.add_argument('--DTU_Official', "-DTU", default='/media/gzr/gzr_2t/datasets/DTU_posed_colmap', type=str)
    # parser.add_argument('--eval_path', required=True, type=str)
    args = parser.parse_args()

for scene in dtu_scenes:
    if not args.skip_training:
        common_args = " --quiet  --data_type dtu --depth_ratio 1.0  -r 4 --lambda_dist 1000"
        source = args.dtu + "/" + scene
        print("python train.py  -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        os.system("python train.py  -s " + source + " -m " + args.output_path + "/" + scene + common_args)


    if not args.skip_rendering:
        all_sources = []
        common_args = " --quiet --skip_train --depth_ratio 1.0 --num_cluster 1 --voxel_size 0.004 --sdf_trunc 0.016 --depth_trunc 3.0"
        source = args.dtu + "/" + scene
        print("python render.py --iteration 30000 -s " + source + " -m" + args.output_path + "/" + scene + common_args)
        os.system("python render.py --iteration 30000 -s " + source + " -m" + args.output_path + "/" + scene + common_args)


    if not args.skip_metrics:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        scan_id = scene[4:]
        ply_file = f"{args.output_path}/{scene}/train/ours_30000/"
        iteration = 30000
        string = f"python {script_dir}/eval_dtu/evaluate_single_scene.py " + \
                 f"--input_mesh {args.output_path}/{scene}/train/ours_30000/fuse_post.ply " + \
                 f"--scan_id {scan_id} --output_dir {args.output_path}/tmp/scan{scan_id} " + \
                 f"--mask_dir {args.dtu} " + \
                 f"--DTU {args.DTU_Official}"

        os.system(string)
    txt_path = f"{args.output_path}/{scene}/dtu_scores_30000.tsv"
    data = pandas.read_csv(txt_path, sep="\t", header=None).to_numpy()
    part_num_eva += int(data[-1][0])
    cd_error_eva_stage1 += float(data[1][-1])
    print('ave part: ', int(data[-1][0]), 'error:', float(data[1][-1]))
    if not args.skip_metrics:
        data = json.load(open(f'{args.output_path}/tmp/scan{scan_id}/results.json'))
        cd_error_eva += float(data['overall'])

print("ave part: ", part_num_eva / len(all_scenes), "ave cd error stage1: ",
        cd_error_eva_stage1 / len(all_scenes), "ave cd error: ", cd_error_eva / len(all_scenes))

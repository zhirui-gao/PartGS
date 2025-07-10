

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(gs_type, model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        #try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test" #"images/test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / f"renders" # _{gs_type}
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + f"/results_{gs_type}.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + f"/per_view_{gs_type}.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)


    method_averages = {}
    for scene in full_dict:
        for method in full_dict[scene]:
            if method not in method_averages:
                method_averages[method] = {"SSIM": 0, "PSNR": 0, "LPIPS": 0, "count": 0}
            method_averages[method]["SSIM"] += full_dict[scene][method]["SSIM"]
            method_averages[method]["PSNR"] += full_dict[scene][method]["PSNR"]
            method_averages[method]["LPIPS"] += full_dict[scene][method]["LPIPS"]
            method_averages[method]["count"] += 1

    for method in method_averages:
        method_averages[method]["SSIM"] /= method_averages[method]["count"]
        method_averages[method]["PSNR"] /= method_averages[method]["count"]
        method_averages[method]["LPIPS"] /= method_averages[method]["count"]
        print(f"Scene: {scene_dir}, Method: {method}")
        print("  Average SSIM : {:>12.7f}".format(method_averages[method]["SSIM"]))
        print("  Average PSNR : {:>12.7f}".format(method_averages[method]["PSNR"]))
        print("  Average LPIPS: {:>12.7f}".format(method_averages[method]["LPIPS"]))
        print("")
        del method_averages[method]["count"]

    parent_dir = Path(model_paths[0]).parent
    with open(parent_dir / f"overall_results_{gs_type}.json", 'w') as fp:
        json.dump(method_averages, fp, indent=True)



if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Metrics script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--gs_type', type=str, default="gs_flat")
    args = parser.parse_args()
    model_paths = []
    for scene in os.listdir(args.model_paths[0]):
        if scene.startswith("scan"):
            model_paths.append(os.path.join(args.model_paths[0], scene))

    evaluate(args.gs_type, model_paths)

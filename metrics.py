#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

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

def yieldImages(renders_dir, gt_dir):
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        yield tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda(), tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda(), fname

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    print("")

    for scene_dir in model_paths:
        # try:
            print("Scene:", scene_dir)

            for test_dir in [Path(scene_dir) / "test", Path(scene_dir) / "train"]:
                if not test_dir.exists():
                    continue

                full_dict[test_dir] = {}
                per_view_dict[test_dir] = {}

                for method in [d for d in os.listdir(test_dir) if os.path.isdir(test_dir / d)]:
                    print("Method:", test_dir, method)

                    full_dict[test_dir][method] = {}
                    per_view_dict[test_dir][method] = {"SSIM": {}, "PSNR": {},"LPIPS": {}}

                    method_dir = test_dir / method
                    gt_dir = method_dir/ "gt"
                    renders_dir = method_dir / "renders"

                    for (render, gt, name) in yieldImages(renders_dir, gt_dir):
                        per_view_dict[test_dir][method]["SSIM"][name] = ssim(render, gt).item()
                        per_view_dict[test_dir][method]["PSNR"][name] = psnr(render, gt).item()
                        # per_view_dict[test_dir][method]["LPIPS"][name] = lpips(render, gt, net_type='vgg').item()

                    if not per_view_dict[test_dir][method]["PSNR"]:
                        continue

                    method_ssim = torch.tensor([v for v in per_view_dict[test_dir][method]["SSIM"].values()]).mean().item()
                    method_psnr = torch.tensor([v for v in per_view_dict[test_dir][method]["PSNR"].values()]).mean().item()
                    # method_lpips = torch.tensor([v for v in per_view_dict[test_dir][method]["LPIPS"].values()]).mean().item()
                    print("  SSIM : {:>12.7f}".format(method_ssim, ".5"))
                    print("  PSNR : {:>12.7f}".format(method_psnr, ".5"))
                    # print("  LPIPS: {:>12.7f}".format(method_lpips, ".5"))
                    print("")

                    full_dict[test_dir][method].update({"SSIM": method_ssim,
                                                        "PSNR": method_psnr,
                                                        # "LPIPS": method_lpips
                                                        })

                with open(test_dir / "results.json", 'w') as fp:
                    json.dump(full_dict[test_dir], fp, indent=True)
                with open(test_dir / "per_view.json", 'w') as fp:
                    json.dump(per_view_dict[test_dir], fp, indent=True)
        # except:
        #     print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)

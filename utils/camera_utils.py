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

from scene.cameras import Camera, TrainedCamera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import torch
from typing import NamedTuple

WARNED = False

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_name: str
    width: int
    height: int

def loadCam(args, id, cam_info, resolution_scale, tuned):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = resized_image_rgb[3:4, ...] if resized_image_rgb.shape[0] == 4 else torch.ones_like(resized_image_rgb[:1, ...])

    if tuned:
        return TrainedCamera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                            FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                            image=gt_image, gt_alpha_mask=loaded_mask,
                            image_name=cam_info.image_name, uid=id, data_device=args.data_device)
    else:
        return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                    FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                    image=gt_image, gt_alpha_mask=loaded_mask,
                    image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def camInfo(id, cam: Camera):
    image = torch.cat([cam._original_image, cam._gt_alpha_mask], dim=0)
    # cam.R and cam.T can be obsolete if tuned camera => reprocess them from get_world_view_transform
    w2c = cam.get_world_view_transform().t().detach().cpu().numpy()
    R = w2c[:3,:3].T # see getWorld2View2
    T = w2c[:3,3]
    fovx = cam.FoVx().detach().cpu().item()
    fovy = cam.FoVy().detach().cpu().item()
    cam_info = CameraInfo(uid=id, R=R, T=T, FovX=fovx, FovY=fovy, image=image,
                          image_name=cam.image_name, width=image.shape[2], height=image.shape[1])
    return cam_info

def cameraList_from_camInfos(cam_infos, resolution_scale, args, tuned = False):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, tuned))

    return camera_list

def cameraList_to_camInfos(cam_list):
    camera_infos = []

    for id, c in enumerate(cam_list):
        camera_infos.append(camInfo(id, c))

    return camera_infos

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : camera.height / 2 * fov2focal(camera.FovY),
        'fx' : camera.width / 2 * fov2focal(camera.FovX)
    }
    return camera_entry

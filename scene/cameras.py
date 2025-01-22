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

import torch
from torch import nn, sqrt
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from pathlib import Path

def quaternion_to_matrix(q):
    q = q / (q.norm() + 1.e-7)
    a,b,c,d = q[0], q[1], q[2], q[3]
    m = torch.stack([torch.stack([a**2+b**2-c**2-d**2, 2*b*c-2*a*d, 2*a*c+2*b*d]),
                     torch.stack([2*a*d+2*b*c, a**2-b**2+c**2-d**2, 2*c*d-2*a*b]),
                     torch.stack([2*b*d-2*a*c, 2*a*b+2*c*d, a**2-b**2-c**2+d**2])])
    return m

def matrix_to_quaternion(m):
    assert m.shape[0] == 3 and m.shape[1] == 3
    Qxx,Qyx,Qzx = m[0]
    Qxy,Qyy,Qzy = m[1]
    Qxz,Qyz,Qzz = m[2]
    K = 1 / 3 * torch.tensor([[Qxx - Qyy - Qzz, Qyx + Qxy, Qzx + Qxz, Qyz - Qzy],
                              [Qyx + Qxy, Qyy - Qxx - Qzz, Qzy + Qyz, Qzx - Qxz],
                              [Qzx + Qxz, Qzy + Qyz, Qzz - Qxx - Qyy, Qxy - Qyx],
                              [Qyz - Qzy, Qzx - Qxz, Qxy - Qyx, Qxx + Qyy + Qzz]], device=m.device)
    b,c,d,a = torch.linalg.eigh(K).eigenvectors[:,-1] # quaternion is the eigen vector of biggest eigen value
    return torch.stack([a,b,c,d])

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self._FoVx = torch.tensor(FoVx, device=data_device)
        self._FoVy = torch.tensor(FoVy, device=data_device)
        self.image_name = image_name
        self.is_test = False

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self._original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self._original_image.shape[2]
        self.image_height = self._original_image.shape[1]

        self._gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self._world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()

        self.is_trained = False

    def get_projection_matrix(self):
        return getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx(), fovY=self.FoVy()).transpose(0,1)

    def get_world_view_transform(self):
        return self._world_view_transform

    def get_camera_center(self):
        return self.get_world_view_transform().inverse()[3, :3]

    def get_full_proj_transform(self):
        return (self.get_world_view_transform().unsqueeze(0).bmm(self.get_projection_matrix().unsqueeze(0))).squeeze(0)

    def original_image(self, bg: torch.tensor = [0.,0.,0.]):
        return self._original_image * self._gt_alpha_mask + bg[:,None,None] * (1. - self._gt_alpha_mask)

    def gt_alpha_mask(self):
        return self._gt_alpha_mask

    def display_name(self):
        return Path(self.image_name).stem

    def FoVx(self):
        return self._FoVx

    def FoVy(self):
        return self._FoVy

class TrainedCamera(Camera):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(TrainedCamera, self).__init__(colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                                            image_name, uid, trans, scale, data_device)
        w2c = self._world_view_transform.transpose(0, 1)
        self.world_view_q = matrix_to_quaternion(w2c[:3,:3]).requires_grad_(True)
        self.world_view_xy = w2c[:2,3].requires_grad_(True)
        self._z = w2c[2,3].requires_grad_(True)
        self._FoVx = torch.tensor(FoVx, device=data_device).requires_grad_(True)
        self._FoVy = torch.tensor(FoVy, device=data_device).requires_grad_(True)
        self.is_trained = True
        self.improving = False
        self.score = 0.0
        self._a = torch.tensor(0., device=data_device).requires_grad_(True)
        self._b = torch.tensor(0., device=data_device).requires_grad_(True)
        self._c = torch.tensor(0., device=data_device).requires_grad_(True)
        self.abc_tuning = False
        self._FoVx0 = 0.
        self._FoVy0 = 0.
        self._z0 = 0.
        self.hessian_eigen_vectors = torch.zeros((3,3), device=data_device)

    def save_state(self):
        with torch.no_grad():
            return (self.world_view_q.clone(),
                    self.world_view_xy.clone(),
                    self._z.clone(),
                    self._z0,
                    self._FoVx.clone(),
                    self._FoVx0,
                    self._FoVy.clone(),
                    self._FoVy0,
                    self._a.clone(),
                    self._b.clone(),
                    self._c.clone(),
                    self.abc_tuning)

    def load_state(self, state):
        with torch.no_grad():
            self.world_view_q.set_(state[0])
            self.world_view_xy.set_(state[1])
            self._z.set_(state[2])
            self._z0 = state[3]
            self._FoVx.set_(state[4])
            self._FoVx0 = state[5]
            self._FoVy.set_(state[6])
            self._FoVy0 = state[7]
            self._a.set_(state[8])
            self._b.set_(state[9])
            self._c.set_(state[10])
            self.abc_tuning = state[11]

    def ndc2Pix(self, v, S):
        return ((v + 1.0) * S - 1.0) * 0.5

    def eval_hessianeigenvectors(self, means3D, screenspace_points_grad):
        assert (not self.abc_tuning)
        means4 = torch.cat([means3D, torch.ones_like(means3D[:,:1])], dim=1)
        p_hom = means4 @ self.get_full_proj_transform()
        p_w = 1.0 / (p_hom[:,3] + 1.e-7)
        p_proj = torch.stack([ p_hom[:,0] * p_w, p_hom[:,1] * p_w, p_hom[:,2] * p_w ], dim=-1) # (N,3)
        point_image = torch.stack([ self.ndc2Pix(p_proj[:,0], self.image_width), self.ndc2Pix(p_proj[:,1], self.image_height) ], dim=-1)
        uv = point_image
        params = [self._z, self._FoVx, self._FoVy]
        z_grad, fovx_grad, fovy_grad = torch.autograd.grad([uv], params, [screenspace_points_grad[:,:2]], create_graph=True)
        d2L_dz2, d2L_dzdfovx, d2L_dzdfovy = torch.autograd.grad(z_grad, params, retain_graph=True)
        d2L_dfovx_dz, d2L_dfovx2, d2L_dfovxdfovy = torch.autograd.grad(fovx_grad, params, retain_graph=True)
        d2L_dfovy_dz, d2L_dfovydfovx, d2L_dfovy2 = torch.autograd.grad(fovy_grad, params)
        hessian = torch.stack([torch.stack([d2L_dz2, d2L_dzdfovx, d2L_dzdfovy]),
                               torch.stack([d2L_dfovx_dz, d2L_dfovx2, d2L_dfovxdfovy]),
                               torch.stack([d2L_dfovy_dz, d2L_dfovydfovx, d2L_dfovy2])
                             ])
        eigval, eigvec = torch.linalg.eigh(hessian) # the eigen vectors are the columns of hessian_eigen_vectors
        return eigval, eigvec

    def to_abc(self, means3D, screenspace_points_grad):
        if self.abc_tuning: # TODO: consider staying in abc mode, processing hessian on abc itself
            self.to_zfovxfovy()
        eigval, eigvec = self.eval_hessianeigenvectors(means3D, screenspace_points_grad)
        self.hessian_eigen_vectors = eigvec
        with torch.no_grad():
            self._z0 = self._z.item()
            self._FoVx0 = self._FoVx.item()
            self._FoVy0 = self._FoVy.item()
            self._a.zero_()
            self._b.zero_()
            self._c.zero_()
        self.abc_tuning = True
        return eigval

    def abc(self):
        return torch.stack([self._a, self._b, self._c])

    def z(self):
        if self.abc_tuning:
            return self._z0 + torch.dot(self.hessian_eigen_vectors[0], self.abc())
        else:
            return self._z

    def FoVx(self):
        if self.abc_tuning:
            return self._FoVx0 + torch.dot(self.hessian_eigen_vectors[1], self.abc())
        else:
            return self._FoVx

    def FoVy(self):
        if self.abc_tuning:
            return self._FoVy0 + torch.dot(self.hessian_eigen_vectors[2], self.abc())
        else:
            return self._FoVy

    def to_zfovxfovy(self):
        with torch.no_grad():
            self._z.set_(self.z())
            self._FoVx.set_(self.FoVx())
            self._FoVy.set_(self.FoVy())
        self.abc_tuning = False

    def update_grads(self, means3D, screenspace_points_grad):
        means4 = torch.cat([means3D, torch.ones_like(means3D[:,:1])], dim=1)
        p_hom = means4 @ self.get_full_proj_transform()
        p_w = 1.0 / (p_hom[:,3] + 1.e-7)
        p_proj = torch.stack([ p_hom[:,0] * p_w, p_hom[:,1] * p_w, p_hom[:,2] * p_w ], dim=-1) # (N,3)
        point_image = torch.stack([ self.ndc2Pix(p_proj[:,0], self.image_width), self.ndc2Pix(p_proj[:,1], self.image_height) ], dim=-1)
        uv = point_image
        params = [self.world_view_q, self.world_view_xy, self._z, self._FoVx, self._FoVy, self._a, self._b, self._c]
        grads = torch.autograd.grad([uv], params, [screenspace_points_grad[:,:2]], allow_unused=True)
        with torch.no_grad():
            self.world_view_q._grad = grads[0]
            self.world_view_xy._grad = grads[1]
            self._z._grad = grads[2]
            self._FoVx._grad = grads[3]
            self._FoVy._grad = grads[4]
            self._a._grad = grads[5]
            self._b._grad = grads[6]
            self._c._grad = grads[7]

    def get_world_view_transform(self):
        matrix = torch.eye(4).to(self.world_view_q)
        matrix[:3,:3] = quaternion_to_matrix(self.world_view_q / self.world_view_q.norm())
        matrix[:2,3] = self.world_view_xy
        matrix[2,3] = self.z()
        return matrix.transpose(0,1)

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


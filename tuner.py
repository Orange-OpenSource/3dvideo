from gaussian_renderer import render
import torch
from tqdm import tqdm
import os, json
from torch.optim import Adam
from torchvision.transforms import ToPILImage

def tb_image(values, highlight = -1):
    if type(values) == list:
        values = torch.tensor(values)
    x = torch.arange(values.shape[0])
    try:
        from matplotlib import pyplot as plt
        import numpy as np
        fig = plt.figure()
        ax = fig.subplots()
        ax.bar(x[x != highlight], values[x != highlight], color="blue")
        ax.bar(x[x == highlight], values[x == highlight], color="red")
        fig.canvas.draw()
        a = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(fig.canvas.height(),-1,4)
        plt.close(fig)
        a = a.transpose(2,0,1) # channel first
        return a[1:] # remove alpha from argb to rgba
    except:
        img = torch.ones((3, 600, max(len(values) * 20 + 10, 800)))
        mn, mx = values.min(), values.max()
        if mn >= 0.:
            z = 0
            normalized = (values / mx * 550).long()
        else:
            z = (-mn / (mx - mn) * 550).long()
            normalized = ((values - mn) / (mx - mn) * 550).long()
        for i,v in enumerate(normalized):
            color = torch.tensor([1.,0.,0.] if (i == highlight) else [0.5,0.5,1.0] if (i % 10 == 0) else [0.,0.,1.])
            if v >= z:
                img[:,z:v,i*20+10:i*20+20] = color[:,None,None]
            else:
                img[:,v:z,i*20+10:i*20+20] = color[:,None,None]
        return img.flip(dims=[1])

class CamTuner():

    def __init__(self, scene, background, dataset, opt, pipe, tb_writer, first_iter, testing_iterations, saving_iterations):
        self.steps = 50
        self.scene = scene
        self.background = background
        self.opt = opt
        self.pipe = pipe
        self.tb_writer = tb_writer
        self.cams = sorted(self.scene.getTrainCameras() + self.scene.getTestCameras(), key = lambda c: c.image_name)
        self.percam_trainings = [0 for _ in range(len(self.cams))]
        self.percam_improvements = [0 for _ in range(len(self.cams))]
        # self.cams = self.cams[:5]
        self.tuned_cam = 0 # start with cam 0
        self.bg = torch.rand((3), device="cuda") if self.opt.random_background else self.background
        self.iter = first_iter
        self.cam_score_beta = max(1.0 / self.steps, 0.001)
        # roughly 5 chances to tune every cam
        self.period = int((max(opt.iterations, opt.tune_until_iter) - opt.tune_from_iter) / (5 * len(self.cams)))
        self.period = max(self.period // 100 * 100, 100)
        self.testing_iterations = testing_iterations
        self.saving_iterations = saving_iterations
        self.dataset = dataset
        self.last_init_iter = -1000

        self.tracked = [0, 1, 2]
        self.iter = 0
        self.thres = 0.01 / self.steps # get min 0.01dB in steps

        if self.opt.tune_cams:
            fov_params = [c._FoVx for c in self.cams] + [c._FoVy for c in self.cams]
            l = [{"name": "cam_q", "lr": opt.cam_q_lr, "params": [c.world_view_q for c in self.cams]},
                 {"name": "cam_xy", "lr": opt.cam_t_lr, "params": [c.world_view_xy for c in self.cams]},
                 {"name": "cam_z", "lr": opt.cam_t_lr, "params": [c._z for c in self.cams]},
                 {"name": "cam_a", "lr": opt.cam_abc_lr, "params": [c._a for c in self.cams]},
                 {"name": "cam_b", "lr": opt.cam_abc_lr, "params": [c._b for c in self.cams]},
                 {"name": "cam_c", "lr": opt.cam_abc_lr, "params": [c._c for c in self.cams]},
                 {"name": "cam_fov", "lr": opt.cam_fov_lr, "params": fov_params}]
            self.cam_optimizer = Adam(l, betas=(opt.cam_beta1, opt.cam_beta2), eps=opt.cam_eps)
        else:
            self.cam_optimizer = None

    def color_loss(self, image, gt_image):
        if self.opt.cam_L1:
            return torch.mean((image-gt_image).abs())
        else:
            return torch.mean((image-gt_image)**2)

    def init_list(self, iteration):
        print("initializing the scores of %d cams..." % len(self.cams))
        for c in tqdm(range(len(self.cams))):
            self.tuned_cam = c
            self.cams[c].score = 0.0
            self.tune_one(min_rounds=2, max_rounds=2)
        if not self.opt.cam_no_abc:
            print("to abc params...")
            for c in tqdm(range(len(self.cams))):
                self.tuned_cam = c
                # self.draw_loss()
                if not self.cams[c].abc_tuning:
                    self.switch_one_to_abc2()
        if self.tb_writer:
            self.tb_writer.add_image("tuning/init_scores", tb_image([c.score for c in self.cams]), self.iter)
        self.last_init_iter = iteration

    def tune_after_training(self):
        print("tuning cameras after training...")
        for iteration in tqdm(range(self.opt.iterations + 1, self.opt.tune_until_iter + 1)):
            self.tune(iteration)
            if iteration in self.saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                self.scene.save(iteration)
                self.save(iteration)

    def tune(self, iteration):
        if not self.opt.tune_cams:
            return False
        if iteration < self.opt.tune_from_iter:
            return False
        if iteration > self.opt.tune_until_iter:
            return False
        if iteration >= self.opt.densify_from_iter and iteration < self.opt.densify_until_iter and iteration % self.opt.opacity_reset_interval < 300:
            # after opacity reset, let the system stabilize
            return False
        if iteration < self.opt.iterations and iteration % self.period != 0:
            return False

        if all([c.score < self.thres for c in self.cams]):
            if iteration - self.last_init_iter < 1000:
                # last init gave low scores => let the system train before retrying
                return False
            self.init_list(iteration)
            next_cam = max(enumerate(self.cams), key=lambda c: c[1].score)[0]
        else:
            self.tune_one(min_rounds=0, max_rounds=20)
            next_cam = max(enumerate(self.cams), key=lambda c: c[1].score)[0]
        if self.tb_writer:
            self.tb_writer.add_image("tuning/trainings", tb_image(self.percam_trainings, highlight=self.tuned_cam), self.iter)
            self.tb_writer.add_image("tuning/improvements", tb_image(self.percam_improvements, highlight=self.tuned_cam), self.iter)
            self.tb_writer.add_image("tuning/scores", tb_image([c.score for c in self.cams], highlight=next_cam), self.iter)
        self.tuned_cam = next_cam

        return True

    def tune_one(self, min_rounds, max_rounds):
        cam = self.cams[self.tuned_cam]
        gt_image = cam.original_image(self.bg).cuda()
        psnr0 = None
        psnr = None

        self.tb_writer.add_scalar("tuning/delta_psnr", -0.1, self.iter-1)
        self.tb_writer.add_scalar("tuning/tuned", -1, self.iter-1)
        self.tb_writer.add_scalar("tuning/tuned", self.tuned_cam, self.iter)
        start_iter = self.iter

        saved_state = cam.save_state()

        for round in range(max_rounds): # number to repeat if score stays high enough
            if round >= min_rounds and cam.score < 0.1 * self.thres: # wait until far below threshold
                break
            for s in range(self.steps):
                render_pkg = render(cam, self.scene.gaussians, self.pipe, self.bg)
                image, viewspace_point_tensor, visibility_filter, _ = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                Ll2 = torch.mean((image-gt_image)**2)
                loss = self.color_loss(image, gt_image)
                prev_psnr = psnr
                psnr = (-10. * torch.log10(Ll2)).item()
                delta_psnr = 0.0 if prev_psnr is None else psnr - prev_psnr
                cam.score = cam.score * (1.0 - self.cam_score_beta) + delta_psnr * self.cam_score_beta
                if s == 0 and round == 0:
                    psnr0 = psnr
                loss.backward()
                cam.update_grads(self.scene.gaussians.get_xyz[visibility_filter], screenspace_points_grad=viewspace_point_tensor.grad[visibility_filter])
                if os.path.exists(os.environ["HOME"] + "/tmp/tracked.json"):
                    j = json.load(open(os.environ["HOME"] + "/tmp/tracked.json"))
                    if type(j) == list and len(j) > 0 and type(j[0]) == int:
                        self.tracked = j
                if self.tb_writer and self.tuned_cam in self.tracked:
                    for i,l in enumerate(["x", "y"]):
                        grad = 0. if cam.world_view_xy._grad is None else cam.world_view_xy._grad[i]
                        self.tb_writer.add_scalar("tuning%d/%s_grad" % (self.tuned_cam, l), grad, self.iter)
                        self.tb_writer.add_scalar("tuning%d/%s" % (self.tuned_cam, l), cam.world_view_xy[i].item(), self.iter)
                    for l,v in zip(["a", "b", "c"], [cam._a, cam._b, cam._c]):
                        if v._grad is not None:
                            self.tb_writer.add_scalar("tuning%d/%s" % (self.tuned_cam, l), v, self.iter)
                            self.tb_writer.add_scalar("tuning%d/%s_grad" % (self.tuned_cam, l), v._grad, self.iter)
                            if 'exp_avg' in self.cam_optimizer.state[v]:
                                self.tb_writer.add_scalar("tuning%d/%s_exp_avg" % (self.tuned_cam, l), self.cam_optimizer.state[v]['exp_avg'], self.iter)
                                self.tb_writer.add_scalar("tuning%d/%s_exp_avg_sq" % (self.tuned_cam, l), self.cam_optimizer.state[v]['exp_avg_sq'], self.iter)
                    if cam._z._grad is not None:
                        self.tb_writer.add_scalar("tuning%d/z_grad" % self.tuned_cam, cam._z._grad.item(), self.iter)
                    self.tb_writer.add_scalar("tuning%d/z" % self.tuned_cam, cam.z().item(), self.iter)
                    for v,l in zip([cam.FoVx(), cam.FoVy()], ["fovx", "fovy"]):
                        self.tb_writer.add_scalar("tuning%d/%s" % (self.tuned_cam, l), v, self.iter)
                    for v,l in zip([cam._FoVx, cam._FoVy], ["fovx", "fovy"]):
                        if v._grad is not None:
                            self.tb_writer.add_scalar("tuning%d/%s_grad" % (self.tuned_cam,l), v._grad, self.iter)
                    self.tb_writer.add_scalar("tuning%d/loss" % self.tuned_cam, loss, self.iter)
                olds = cam._a.item(), cam._b.item(), cam._c.item()
                self.cam_optimizer.step()
                if self.tb_writer and self.tuned_cam in self.tracked:
                    for l,v,old_v in zip(["a", "b", "c"], [cam._a, cam._b, cam._c], olds):
                        self.tb_writer.add_scalar("tuning%d/%s_inc" % (self.tuned_cam, l), v - old_v, self.iter)
                self.cam_optimizer.zero_grad()
                if self.tb_writer and (s % 10 == 0 or s == self.steps - 1):
                    self.tb_writer.add_scalar("tuning/delta_psnr", psnr - psnr0, self.iter)
                self.iter += 1
            if self.tb_writer and self.tuned_cam in self.tracked:
                self.tb_writer.add_scalar("tuning%d/score" % self.tuned_cam, cam.score, self.iter)
        if psnr - psnr0 < 0.:
            # should not happen: come back to previous state
            cam.score = 0.
            cam.load_state(saved_state)
            if self.tb_writer and self.tuned_cam in self.tracked:
                self.tb_writer.add_scalar("tuning%d/score" % self.tuned_cam, cam.score, self.iter)
        self.tb_writer.add_scalar("tuning/delta_psnr", -0.1, self.iter + 1)
        self.tb_writer.add_scalar("tuning/tuned", self.tuned_cam, self.iter)
        self.tb_writer.add_scalar("tuning/tuned", -1, self.iter + 1)
        self.percam_trainings[self.tuned_cam] += self.iter - start_iter
        self.percam_improvements[self.tuned_cam] += max(0., psnr - psnr0)

    def switch_one_to_abc(self):
        cam = self.cams[self.tuned_cam]
        gt_image = cam.original_image(self.bg).cuda()
        render_pkg = render(cam, self.scene.gaussians, self.pipe, self.bg)
        image, viewspace_point_tensor, visibility_filter, _ = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        loss = self.color_loss(image, gt_image)
        loss.backward()
        # cam.trace_hessianeigenvectors2d(self.scene.gaussians.get_xyz[visibility_filter], screenspace_points_grad=viewspace_point_tensor.grad[visibility_filter], idx=self.tuned_cam)
        eigval = cam.to_abc(self.scene.gaussians.get_xyz[visibility_filter], screenspace_points_grad=viewspace_point_tensor.grad[visibility_filter])
        if self.tuned_cam in self.tracked and self.tb_writer:
            for l,v in zip(["a", "b", "c"], eigval):
                self.tb_writer.add_scalar("tuning%d/eigval_%s" % (self.tuned_cam,l), v, self.iter)

    def draw_loss(self):
        delta_z = torch.linspace(-0.001, 0.001, 11)
        delta_fovx = delta_z
        cam = self.cams[self.tuned_cam]
        w2c = cam.get_world_view_transform().t().detach().cpu().numpy()
        X, Y = torch.meshgrid(delta_z, delta_fovx, indexing="xy")
        L = torch.zeros_like(X)
        ZG = torch.zeros_like(X)
        FG = torch.zeros_like(X)
        from scene.cameras import TrainedCamera
        import numpy as np
        gt_image = cam.original_image(self.bg).cuda()
        for i,dz in enumerate(delta_z):
            for j,df in enumerate(delta_fovx):
                T = w2c[:3,3] + np.array([0.,0.,X[i,j].item()])
                R = w2c[:3,:3].T
                idx = self.tuned_cam * len(delta_z) * len(delta_fovx) + i * len(delta_fovx) + j
                mcam = TrainedCamera(idx, R, T, (cam.FoVx() + Y[i,j]).item(), cam.FoVy().item(), cam._original_image, cam._gt_alpha_mask, cam.image_name, idx)
                render_pkg = render(mcam, self.scene.gaussians, self.pipe, self.bg)
                image, viewspace_point_tensor, visibility_filter, _ = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                loss = self.color_loss(image, gt_image)
                loss.backward()
                mcam.update_grads(self.scene.gaussians.get_xyz[visibility_filter], screenspace_points_grad=viewspace_point_tensor.grad[visibility_filter])
                L[i,j] = loss.item()
                ZG[i,j] = mcam._z._grad.item()
                FG[i,j] = mcam._FoVx._grad.item()
        torch.save({"z": (X + cam.z().detach().cpu()), "fovx": (Y + cam.FoVx().detach().cpu()), "loss": L.detach().cpu(), "z_grad": ZG.detach().cpu(), "fovx_grad": FG.detach().cpu()}, '/home/omge7332/tmp/heatmap%d.pth' % self.tuned_cam)

    def gradients(self, cam, dz, dfovx, dfovy):
        from scene.cameras import TrainedCamera
        import numpy as np
        w2c = cam.get_world_view_transform().t().detach().cpu().numpy()
        T = w2c[:3,3] + np.array([0., 0., dz])
        R = w2c[:3,:3].T
        idx = 0
        mcam = TrainedCamera(idx, R, T, (cam.FoVx() + dfovx).item(), (cam.FoVy() + dfovy).item(), cam._original_image, cam._gt_alpha_mask, cam.image_name, idx)
        render_pkg = render(mcam, self.scene.gaussians, self.pipe, self.bg)
        image, viewspace_point_tensor, visibility_filter, _ = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gt_image = cam.original_image(self.bg).cuda()
        loss = self.color_loss(image, gt_image)
        loss.backward()
        mcam.update_grads(self.scene.gaussians.get_xyz[visibility_filter], screenspace_points_grad=viewspace_point_tensor.grad[visibility_filter])
        return mcam._z._grad.item(), mcam._FoVx._grad.item(), mcam._FoVy._grad.item()

    def switch_one_to_abc2(self):
        dfovx = dfovy = dz = 1.e-4
        cam = self.cams[self.tuned_cam]
        dL_dz_pz, dL_dfovx_pz, dL_dfovy_pz = self.gradients(cam, dz, 0., 0., )
        dL_dz_mz, dL_dfovx_mz, dL_dfovy_mz = self.gradients(cam, dz, 0., 0., )
        dL_dz_pfovx, dL_dfovx_pfovx, dL_dfovy_pfovx = self.gradients(cam, 0., dfovx, 0.)
        dL_dz_mfovx, dL_dfovx_mfovx, dL_dfovy_mfovx = self.gradients(cam, 0., -dfovx, 0.)
        dL_dz_pfovy, dL_dfovx_pfovy, dL_dfovy_pfovy = self.gradients(cam, 0., 0., dfovy)
        dL_dz_mfovy, dL_dfovx_mfovy, dL_dfovy_mfovy = self.gradients(cam, 0., 0., -dfovy)
        d2L_dz2 = (dL_dz_pz - dL_dz_mz) / (2 * dz)
        d2L_dfovxdz = (dL_dfovx_pz - dL_dfovx_mz) / (2 * dz)
        d2L_dfovydz = (dL_dfovy_pz - dL_dfovy_mz) / (2 * dz)
        d2L_dzdfovx = (dL_dz_pfovx - dL_dz_mfovx) / (2 * dfovx)
        d2L_dfovx2 = (dL_dfovx_pfovx - dL_dfovx_mfovx) / (2 * dfovx)
        d2L_dfovydfovx = (dL_dfovy_pfovx - dL_dfovy_mfovx) / (2 * dfovx)
        d2L_dzdfovy = (dL_dz_pfovy - dL_dz_mfovy) / (2 * dfovy)
        d2L_dfovxdfovy = (dL_dfovx_pfovy - dL_dfovx_mfovy) / (2 * dfovy)
        d2L_dfovy2 = (dL_dfovy_pfovy - dL_dfovy_mfovy) / (2 * dfovy)

        # get the matrix symmetric:
        d2L_dzdfovx = 0.5 * (d2L_dzdfovx + d2L_dfovxdz)
        d2L_dzdfovy = 0.5 * (d2L_dzdfovy + d2L_dfovydz)
        d2L_dfovxdfovy = 0.5 * (d2L_dfovxdfovy + d2L_dfovydfovx)

        pseudo_hessian = torch.tensor([[d2L_dz2, d2L_dzdfovx, d2L_dzdfovy],
                                       [d2L_dzdfovx, d2L_dfovx2, d2L_dfovxdfovy],
                                       [d2L_dzdfovy, d2L_dfovxdfovy, d2L_dfovy2]])
        eigvec, eigval, _ = torch.linalg.svd(pseudo_hessian) # eigen vectors are columns of eigvec 
        # to positive eigen values:
        eigvec = eigvec * eigval.sign()[None]
        eigval = eigval * eigval.sign()
        # sort ascending:
        eigval, idx = eigval.sort()
        eigvec = eigvec[:,idx]
        cam.to_abc2(eigvec, eigval)
        if self.tuned_cam in self.tracked and self.tb_writer:
            for l,v in zip(["a", "b", "c"], eigval):
                self.tb_writer.add_scalar("tuning%d/eigval_%s" % (self.tuned_cam,l), v, self.iter)

    def save(self, iteration):
        if self.opt.tune_cams:
            with open(os.path.join(self.dataset.model_path, "cam_scores%06d.csv" % iteration), "w") as out:
                out.write("name;score\n")
                out.write("\n".join("%s;%f" % (c.image_name, c.score) for c in self.cams))


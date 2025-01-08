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
        self.score_ema = 0.
        # self.score = 0.
        self.steps = 25
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
        self.ema_moment = 1.0 / len(self.cams)
        # roughly 3 chances to tune every cam
        self.period = int((max(opt.iterations, opt.tune_until_iter) - opt.tune_from_iter) / (3 * len(self.cams))) // 100 * 100
        self.testing_iterations = testing_iterations
        self.saving_iterations = saving_iterations
        self.dataset = dataset

        if self.opt.tune_cams:
            fov_params = [c._FoVx for c in self.cams] + [c._FoVy for c in self.cams]
            l = [{"name": "cam_q", "lr": opt.cam_q_lr, "params": [c.world_view_q for c in self.cams]},
                 {"name": "cam_t", "lr": opt.cam_t_lr, "params": [c.world_view_t for c in self.cams]},
                 {"name": "cam_fov", "lr": opt.cam_fov_lr, "params": fov_params}]
            self.cam_optimizer = Adam(l, betas=(0.9,0.99))
        else:
            self.cam_optimizer = None

    def init_list(self, iteration):
        print("initializing the scores of %d cams..." % len(self.cams))
        for c in tqdm(range(len(self.cams))):
            self.tuned_cam = c
            self.tune(iteration, True)
        self.score_ema = sum([c.score for c in self.cams]) / len(self.cams)

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
        start_iter = self.iter
        for round in range(20): # repeat 20x if score stays high enough
            for s in range(self.steps):
                render_pkg = render(cam, self.scene.gaussians, self.pipe, self.bg)
                image, viewspace_point_tensor, visibility_filter, _ = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                Ll2 = torch.mean((image-gt_image)**2)
                psnr = (-10. * torch.log10(Ll2)).item()
                if s == 0:
                    round_psnr0 = psnr
                if psnr0 is None:
                    psnr0 = psnr
                Ll2.backward()
                cam.update_grads(self.scene.gaussians.get_xyz[visibility_filter], screenspace_points_grad=viewspace_point_tensor.grad[visibility_filter])
                self.cam_optimizer.step()
                self.cam_optimizer.zero_grad()
                if self.tb_writer and (s % 10 == 0 or s == self.steps - 1):
                    self.tb_writer.add_scalar("tuning/iteration", iteration, self.iter)
                    self.tb_writer.add_scalar("tuning/delta_psnr", psnr - psnr0, self.iter)
                self.iter += 1
            cam.score = cam.score * 0.66 + (psnr - round_psnr0) * 0.34
            if (init and round == 2) or (not init and round > 0 and cam.score < thres):
                cam.score = (psnr - psnr0) / 3
                break
        self.tb_writer.add_scalar("tuning/delta_psnr", -0.1, self.iter+1)
        self.percam_trainings[self.tuned_cam] += self.iter - start_iter
        self.percam_improvements[self.tuned_cam] += psnr - psnr0
        # optimized, update the score_ema and change the tuned cam:
        self.score_ema = self.score_ema * (1 - self.ema_moment) + cam.score * self.ema_moment
        next_cam = max(enumerate(self.cams), key=lambda c: c[1].score)[0]
        if self.tb_writer:
            if init:
                if self.tuned_cam == len(self.cams) - 1:
                    plt.bar(range(len(self.cams)), [c.score for c in self.cams])
                    self.tb_writer.add_figure("tuning/init_scores", plt.gcf(), 0)
            else:
                self.tb_writer.add_image("tuning/trainings", tb_image(self.percam_trainings, highlight=self.tuned_cam), self.iter)
                self.tb_writer.add_image("tuning/improvements", tb_image(self.percam_improvements, highlight=self.tuned_cam), self.iter)
                self.tb_writer.add_image("tuning/scores", tb_image([c.score for c in self.cams], highlight=next_cam), self.iter)
                self.tb_writer.add_scalar("tuning/score_ema", self.score_ema, self.iter+1)
        self.tuned_cam = next_cam

        return True

    def save(self, iteration):
        if self.opt.tune_cams:
            with open(os.path.join(self.dataset.model_path, "cam_scores%06d.csv" % iteration), "w") as out:
                out.write("name;score\n")
                out.write("\n".join("%s;%f" % (c.image_name, c.score) for c in self.cams))


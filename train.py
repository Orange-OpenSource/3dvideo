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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from datetime import datetime
import json

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def ndc2Pix(v, S):
	return ((v + 1.0) * S - 1.0) * 0.5

def world_view_grad(means3D, viewpoint_cam, screenspace_points_grad):
    means4 = torch.cat([means3D, torch.ones_like(means3D[:,:1])], dim=1)
    p_hom = means4 @ viewpoint_cam.get_full_proj_transform()
    p_w = 1.0 / (p_hom[:,3] + 1.e-7)
    p_proj = torch.stack([ p_hom[:,0] * p_w, p_hom[:,1] * p_w, p_hom[:,2] * p_w ], dim=-1) # (N,3)
    point_image = torch.stack([ ndc2Pix(p_proj[:,0], viewpoint_cam.image_width), ndc2Pix(p_proj[:,1], viewpoint_cam.image_height) ], dim=-1)
    uv = point_image
    params = [viewpoint_cam.world_view_q, viewpoint_cam.world_view_t, viewpoint_cam._FoVx, viewpoint_cam._FoVy]
    grads = torch.autograd.grad([uv], params, [screenspace_points_grad[:,:2]])
    return grads # (dl_dq, dl_dt, dl_dFoVx) or (dl_dq, dl_dt, dl_dFoVx, dl_dFoVy)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, opt)
    start_datetime = datetime.now().timestamp()
    gaussians = GaussianModel(dataset.sh_degree, dataset.z0, opt.densify_max, opt.densify_percent, dataset.max_opacity)
    scene = Scene(dataset, gaussians, opt=opt)

    if opt.start_ply:
        gaussians.load_ply(opt.start_ply)
        first_iter = opt.start_iter
        gaussians.max_radii2D = torch.zeros_like(gaussians._xyz[:,0])

    gaussians.training_setup(opt)
    if os.path.exists(scene.model_path + "/deadline-chkpnt.pth"):
        print("Deadline checkpoint detected, start from it", scene.model_path + "/deadline-chkpnt.pth")
        checkpoint = scene.model_path + "/deadline-chkpnt.pth"
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    report_bg = torch.tensor([255.,121.,0.], device="cuda")/255. if opt.random_background and not dataset.white_background else background # don't update at each iteration, to get comparable PSNR

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    viewpoint_cam = None
    trained_cam = None
    ema_loss_for_log = 0.0
    tunings = { c.image_name: 0 for c in scene.getTrainCameras() + scene.getTestCameras()}
    progress_bar = tqdm(range(first_iter, max(opt.iterations, opt.tune_until_iter)), desc="Training progress")
    first_iter += 1

    target_score = 1000.
    all_viewpoint_stack = scene.getTrainCameras().copy() + scene.getTestCameras().copy()
    for i,c in enumerate(all_viewpoint_stack):
        c.idx = i

    for iteration in range(first_iter, max(opt.iterations, opt.tune_until_iter) + 1):
        if iteration < opt.tune_from_iter and iteration > opt.iterations:
            continue
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if datetime.now().timestamp() - start_datetime > opt.deadline:
            # save as (iteration - 1) because iteration needs to be replayed
            print("\n[ITER {}] Saving Checkpoint for timeout".format(iteration - 1))
            torch.save((gaussians.capture(), iteration - 1), scene.model_path + "/deadline-chkpnt.pth")
            sys.exit(42)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack or iteration in [opt.iterations, opt.tune_from_iter, opt.tune_until_iter]:
            if iteration <= opt.iterations:
                viewpoint_stack = scene.getTrainCameras().copy()
            else:
                viewpoint_stack = []
            if opt.tune_cams and iteration == max(first_iter, opt.tune_from_iter):
                # initialize grad_scores
                for c in scene.getTrainCameras().copy() + scene.getTestCameras().copy():
                    bg = torch.rand((3), device="cuda") if opt.random_background else background
                    gt_image = c.original_image(bg).cuda()
                    render_pkg = render(c, gaussians, pipe, bg)
                    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                    Ll2 = torch.mean((image-gt_image)**2)
                    Ll2.backward()
                    c.update_grads(world_view_grad(gaussians.get_xyz[visibility_filter], c, screenspace_points_grad=viewspace_point_tensor.grad[visibility_filter]))
        if opt.tune_cams and iteration >= opt.tune_from_iter and iteration <= opt.tune_until_iter:
            if not all_viewpoint_stack:
                all_viewpoint_stack = scene.getTrainCameras().copy() + scene.getTestCameras().copy()
            if not trained_cam or trained_cam.grad_score() < target_score:
                # trained_cam score is low, we can chose a new one:
                target_score = 0.99 * target_score + 0.01 * trained_cam.grad_score() if trained_cam else target_score
                tb_writer.add_scalar('trained_cam/idx', trained_cam.idx if trained_cam else -1, iteration-1)
                tb_writer.add_scalar('trained_cam/tuned', (torch.tensor(list(tunings.values())) > 0).sum(), iteration-1)
                trained_cam = all_viewpoint_stack.pop(0)
                trained_cam.ema_loss = 0.
                tunings[trained_cam.image_name] += 1
                tb_writer.add_scalar('trained_cam/idx', trained_cam.idx if trained_cam else -1, iteration)
                tb_writer.add_scalar('trained_cam/tuned', (torch.tensor(list(tunings.values())) > 0).sum(), iteration)
            viewpoint_stack = [c for c in viewpoint_stack if c != trained_cam]
        else:
            trained_cam = None

        if viewpoint_stack and (not trained_cam or iteration % (1 + opt.cam_tuning_priority) == 0):
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        else:
            viewpoint_cam = trained_cam

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image(bg).cuda()

        Ll1 = l1_loss(image, gt_image)
        Ll2 = torch.mean((image-gt_image)**2)
        if viewpoint_cam == trained_cam:
            loss = Ll2
            trained_cam.ema_loss = opt.cam_ema_moment * trained_cam.ema_loss + (1. - opt.cam_ema_moment) * loss.item()
            trained_cam.ema_loss_short = opt.cam_ema_moment * 0.9 * trained_cam.ema_loss_short + (1. - opt.cam_ema_moment * 0.9) * loss.item()
        else:
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.lambda_psnr * Ll2
        loss.backward()

        iter_end.record()

        if viewpoint_cam == trained_cam:
            # MGE: should integrate clean backpropagation? This is a shortcut...
            grads = world_view_grad(gaussians.get_xyz[visibility_filter], viewpoint_cam, screenspace_points_grad=viewspace_point_tensor.grad[visibility_filter])
            viewpoint_cam.update_grads(grads)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                log_dict = {"Loss": f"{ema_loss_for_log:.{7}f}"}
                progress_bar.set_postfix(log_dict)
                progress_bar.update(10)
            if iteration == max(opt.iterations, opt.tune_until_iter):
                progress_bar.close()

            # Log and save
            if viewpoint_cam == trained_cam:
                tb_writer.add_scalar('trained_cam/psnr', -10.*torch.log10(Ll2), iteration)
                tb_writer.add_scalar('trained_cam/psnr_ema', -10.*torch.log10(torch.tensor(trained_cam.ema_loss)), iteration)
                tb_writer.add_scalar('trained_cam/grad_score', trained_cam.grad_score(), iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, tunings, render, (pipe, report_bg))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                if opt.tune_cams:
                    cams = scene.getTestCameras() + scene.getTrainCameras()
                    cams = sorted(cams, key = lambda c: c.grad_score(), reverse=True)
                    with open(os.path.join(dataset.model_path, "grad_debug%06d.csv" % iteration), "w") as out:
                        out.write("name,dL_dcamq,dL_dcamt,dL_dfovx,dL_dfovy,grad_score\n")
                        out.write("\n".join("%s,%f,%f,%f,%f,%f" % (c.image_name, c.dL_dcamq, c.dL_dcamt, c.dL_dfovx, c.dL_dfovy, c.grad_score()) for c in cams))

            # Densification
            if iteration < min(opt.densify_until_iter, opt.iterations):
                if viewpoint_cam != trained_cam:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if viewpoint_cam != trained_cam and iteration < opt.iterations:
                gaussians.optimizer.step()
            if viewpoint_cam == trained_cam:
                scene.cam_optimizer.step()
            if scene.cam_optimizer:
                scene.cam_optimizer.zero_grad()
            gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    # loop finished, get rid of deadline-chkpnt.pth (which we don't want to be erroneously used if job is launched again):
    if os.path.exists(scene.model_path + "/deadline-chkpnt.pth"):
        os.remove(scene.model_path + "/deadline-chkpnt.pth")

def prepare_output_and_logger(dataset_args, opt_args):
    if not dataset_args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        dataset_args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(dataset_args.model_path))
    os.makedirs(dataset_args.model_path, exist_ok = True)
    with open(os.path.join(dataset_args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(dataset_args))))
    with open(os.path.join(dataset_args.model_path, "opt_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(opt_args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(dataset_args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, tunings, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                masked_psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_dict = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_dict["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image(renderArgs[1]).to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.display_name()), image[None], global_step=iteration)
                        alpha_image = 1. - render_dict["debugBuffer"][1:2] # debugBuffer returns transparency
                        tb_writer.add_images(config['name'] + "_view_{}/alpha".format(viewpoint.display_name()), alpha_image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.display_name()), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    masked_psnr_test += psnr(image, gt_image, viewpoint.gt_alpha_mask()).mean().double()
                psnr_test /= len(config['cameras'])
                masked_psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} masked PSNR {}".format(iteration, config['name'], l1_test, psnr_test, masked_psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - masked_psnr', masked_psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000] + [3000 * i for i in range(100)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000] + [50000 * i for i in range(10)])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations += [args.iterations]
    if args.tune_cams:
        args.save_iterations += [args.tune_until_iter]
    else:
        args.tune_until_iter = 0

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    try:
        network_gui.init(args.ip, args.port)
    except:
        print("error with", args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")

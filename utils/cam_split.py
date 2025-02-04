import numpy as np
import json
from argparse import ArgumentParser
import os.path as path
import os
from graphics_utils import fov2focal
from PIL import Image
from scene.cameras import matrix_to_quaternion

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--folder")
    parser.add_argument("--test_images", nargs='+', type=str)
    parser.add_argument("--sparse_files", action="store_true")
    args = parser.parse_args()


    trainfile = path.join(args.folder, "transforms_train.json")
    jfile = json.load(open(trainfile))
    frames = jfile["frames"]
    testfile = path.join(args.folder, "transforms_test.json")
    if path.exists(testfile):
        tests = json.load(open(testfile))
        os.rename(testfile, testfile+".sav")
        frames += tests["frames"]
    frames = sorted(frames, key=lambda f: f["file_path"])
    if args.sparse_files:
        img_path = path.join(args.folder, frames[0]["file_path"])
        if not img_path.split('.')[-1].lower() in ["jpg", "jpeg", "png"]:
            img_path += ".png"
        W,H = Image.open(img_path).size
        cam_file = path.join(args.folder, "cameras.txt")
        if path.exists(cam_file):
            os.rename(cam_file, cam_file+".sav") 
        with open(cam_file, "w") as out:
            # 1 PINHOLE 1600 1037 1186.0108045370664 1188.6026057333354 800 518.5
            out.write("\n".join(["%d PINHOLE %d %d %f %f %f %f" % (i+1, W, H, fov2focal(f["camera_angle_x"]) * W / 2, fov2focal(f["camera_angle_y"]) * H / 2, W / 2, H / 2) for i,f in enumerate(frames)]))
        img_file = path.join(args.folder, "images.txt")
        if path.exists(img_file):
            os.rename(img_file, img_file+".sav") 
        with open(img_file, "w") as out:
            for i,f in enumerate(frames):
                c2w = np.array(f["transform_matrix"])
                c2w[:3, 1:3] *= -1
                w2c = np.linalg.inv(c2w)
                q = matrix_to_quaternion(w2c[:3,:3])
                t = w2c[:3,3]
                # 194 0.87789737416535651 0.17656105410483855 -0.39021410310223553 -0.21413861946663995 -0.18595263308466015 -0.29445680205269809 2.1699432280663196 1 _DSC8873.JPG
                out.write("%d %f %f %f %f %f %f %f %d %s\n\n" % ((i+1,) + tuple(q) + tuple(t) + (i+1, f["file_path"].split("/")[-1])))
    if args.test_images:
        os.rename(trainfile, trainfile+".sav")
        if path.exists(testfile):
            os.rename(testfile, testfile+".sav")
        jfile["frames"] = [f for f in frames if f["file_path"] in args.test_images]
        json.dump(jfile, open(testfile, "w"), indent=2)
        jfile["frames"] = [f for f in frames if f["file_path"] not in args.test_images]
        json.dump(jfile, open(trainfile, "w"), indent=2)

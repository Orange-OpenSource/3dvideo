import numpy as np
import json
from argparse import ArgumentParser
import os.path as path
import os

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--folder")
    parser.add_argument("--test_images", nargs='+', type=str)
    args = parser.parse_args()

    trainfile = path.join(args.folder, "transforms_train.json")
    jfile = json.load(open(trainfile))
    os.rename(trainfile, trainfile+".sav")
    frames = jfile["frames"]
    testfile = path.join(args.folder, "transforms_test.json")
    if path.exists(testfile):
        tests = json.load(open(testfile))
        os.rename(testfile, testfile+".sav")
        frames += tests["frames"]
    jfile["frames"] = [f for f in frames if f["file_path"] in args.test_images]
    json.dump(jfile, open(testfile, "w"), indent=2)
    jfile["frames"] = [f for f in frames if f["file_path"] not in args.test_images]
    json.dump(jfile, open(trainfile, "w"), indent=2)

import os, sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)
from models import Gaze
from models.resnet import resnet18
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2, yaml, copy
from easydict import EasyDict as edict
import ctools, gtools
import argparse
from torch.cuda.amp import autocast

def main(test):

    # =================================> Setup <=========================
    reader = importlib.import_module("reader." + test.reader)
    torch.cuda.set_device(0)

    data = test.data
    load = test.load

    # Can be overridden with --weight; see README "Evaluation".
    if weight_path:
        modelpath = weight_path
    if not os.path.exists(modelpath):
        raise FileNotFoundError(
            f"Checkpoint not found: {modelpath}\n"
            "Run `python scripts/download_weights.py` first, or pass --weight."
        )

    # ===============================> Read Data <=========================
    if data.isFolder:
        data, _ = ctools.readfolder(data)

    print(f"==> Test: {data.label} <==")
    dataset = reader.loader(data, 32, num_workers=4, shuffle=False)

    # =============================> Test <=============================
    net = Gaze.GazeModel()
    net2 = model.Model()
    net.load_state_dict(torch.load(modelpath), strict=True)
    # net2.load_state_dict(torch.load("evaluation/gaze360/checkpoint/best_Iter73_trans6.pt"), strict=True)

    net.cuda(); net.eval(); net2.cuda(); net2.eval()

    length = len(dataset); accs = 0; count = 0

    # logname = f"{saveiter}.log"

    # outfile = open(os.path.join(logpath, logname), 'w')
    # outfile.write("name results gts\n")


    with torch.no_grad():
        losses = []
        for j, (data, label) in enumerate(dataset):

            for key in data:
                if key != 'name': data[key] = data[key].cuda()

            gts = label.cuda()

            with autocast():
                gazes = net(data["face"])

            for k, gaze in enumerate(gazes):

                gaze = gaze.cpu().detach().numpy()
                gt = gts.cpu().numpy()[k]

                losses.append(nn.L1Loss()(torch.from_numpy(gaze), torch.from_numpy(gt)).item())

                count += 1
                accs += gtools.angular(
                            gtools.gazeto3d(gaze),
                            gtools.gazeto3d(gt)
                        )

                gaze = [str(u) for u in gaze]
                gt = [str(u) for u in gt]

        loger = f"Total Num: {count}, avg: {accs/count:.12f}"
        print(loger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='HFGAD gaze estimation evaluation')

    parser.add_argument('-t', '--target', type=str,
                        help = 'config path about test')

    parser.add_argument('-w', '--weight', type=str, default=None,
                        help = 'checkpoint to evaluate '
                               '(default: evaluation/gaze360/checkpoint/ours_10.46.pt)')

    args = parser.parse_args()

    # Read model from train config and Test data in test config.

    test_conf = edict(yaml.load(open(args.target), Loader=yaml.FullLoader))

    main(test_conf.test)


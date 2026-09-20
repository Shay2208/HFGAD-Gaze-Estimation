import sys,os
base_dir = os.getcwd()
sys.path.insert(0, base_dir)
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import yaml
import cv2
import ctools
from models import Gaze
from models.resnet import resnet18, resnet50
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler
import argparse
import matplotlib.pyplot as plt

def main(config):

    #  ===================>> Setup <<=================================

    dataloader = importlib.import_module("reader." + config.reader)

    torch.cuda.set_device(config.device) 
    cudnn.benchmark = True

    data = config.data
    save = config.save
    params = config.params
    stepLR = config.stepLR

    print("===> Read data <===")

    if data.isFolder:
        data, _ = ctools.readfolder(data)

    dataset = dataloader.loader(
                    data,
                    params.batch_size, 
                    shuffle=True, 
                    num_workers=8
                )


    print("===> Model building <===")
    net = Gaze.GazeModel()
    net.train(); net.cuda()

    # Pretrain (optional)
    # `pretrain.path` is loaded with strict=False, so any checkpoint whose keys
    # do not match the current architecture is simply ignored.
    pretrain = config.pretrain

    if getattr(pretrain, "enable", False):
        weight = pretrain.path
        if not os.path.exists(weight):
            print(f"[warn] pretrain weight not found, training from scratch: {weight}")
        else:
            map_location = None
            if pretrain.get("device", None) is not None:
                map_location = {
                    f"cuda:{pretrain.device}": f"cuda:{config.device}",
                    "cuda:0": f"cuda:{config.device}",
                }
            state = torch.load(weight, map_location=map_location)
            missing, unexpected = net.load_state_dict(state, strict=False)
            print(
                f"[info] loaded pretrain weight: {weight} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )


    print("===> optimizer building <===")
    optimizer = optim.Adam(
                    net.parameters(),
                    lr=params.lr, 
                    betas=(0.9,0.999)
                )
  
    scheduler = optim.lr_scheduler.StepLR( 
                    optimizer, 
                    step_size=stepLR.decay_step, 
                    gamma=stepLR.decay
                )

    if params.warmup:
        scheduler = GradualWarmupScheduler( 
                        optimizer, 
                        multiplier=1, 
                        total_epoch=params.warmup, 
                        after_scheduler=scheduler
                    )

    savepath = os.path.join(save.metapath, save.folder, f"checkpoint")

    if not os.path.exists(savepath):
        os.makedirs(savepath)
 
    # =====================================>> Training << ====================================
    print("===> Training <===")

    length = len(dataset); total = length * params.epoch
    timer = ctools.TimeCounter(total)

    losses = []

    optimizer.zero_grad()
    optimizer.step()
    scheduler.step()


    with open(os.path.join(savepath, "train_log"), 'w') as outfile:
        outfile.write(ctools.DictDumps(config) + '\n')

        with open(os.path.join(savepath, "train_test_log"), 'w') as train_test_log:
            for epoch in range(1, params.epoch+1):
                epoch_losses = []
                for i, (data, anno) in enumerate(dataset):

                    # -------------- forward -------------
                    for key in data:
                        if key != 'name': data[key] = data[key].cuda()

                    anno = anno.cuda() 
                    loss = net.loss(data, anno)
                    epoch_losses.append(loss.item())

                    # -------------- Backward ------------
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    rest = timer.step()/3600


                    if i % 20 == 0:
                        log = f"[{epoch}/{params.epoch}]: " + \
                            f"[{i}/{length}] " +\
                            f"loss:{loss} " +\
                            f"lr:{ctools.GetLR(optimizer)} " +\
                            f"rest time:{rest:.2f}h"

                        print(log); outfile.write(log + "\n")
                        sys.stdout.flush(); outfile.flush()

                scheduler.step()
                losses.append(np.mean(epoch_losses))

                epoch_log = f"[{epoch}] " +\
                            f"lr: {ctools.GetLR(optimizer):.6f} "+\
                            f"loss: {losses[-1]:.12f} "
                
                train_test_log.write(epoch_log + '\n')
                sys.stdout.flush(); train_test_log.flush()

                if epoch % save.step == 0:
                    torch.save(
                            net.state_dict(), 
                            os.path.join(
                                savepath, 
                                f"Iter_{epoch}_{save.model_name}.pt"
                                )
                            )

    plt.figure()
    plt.plot(range(1, params.epoch+1), losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(savepath, 'train_loss.png'))
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Pytorch Basic Model Training')

    parser.add_argument('-s', '--train', type=str,
                        help='The source config for training.')

    args = parser.parse_args()

    config = edict(yaml.load(open(args.train), Loader=yaml.FullLoader))

    print("=====================>> (Begin) Training params << =======================")
    print(ctools.DictDumps(config))
    print("=====================>> (End) Traning params << =======================")

    main(config.train)
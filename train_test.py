import argparse
import copy
import importlib
import os
import sys

import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from easydict import EasyDict as edict
from warmup_scheduler import GradualWarmupScheduler
from torch.cuda.amp import GradScaler, autocast

import ctools
import gtools
from models import Gaze, model, resnet, PVT, gaze360

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_and_evaluate(train_config, test_config):

    # ===================>> Setup <<=================================
    train_dataloader = importlib.import_module("reader." + train_config.reader)
    test_dataloader = importlib.import_module("reader." + test_config.reader)

    torch.cuda.set_device(train_config.device)
    cudnn.benchmark = True

    train_data = train_config.data
    test_data = test_config.data
    save = train_config.save
    params = train_config.params
    cosine = train_config.cosine
    stepLR = train_config.stepLR

    print("===> Read data <===")
    if train_data.isFolder:
        train_data, _ = ctools.readfolder(train_data)

    train_dataset = train_dataloader.loader(
        train_data,
        params.batch_size,
        shuffle=True,
        num_workers=10
    )

    if test_data.isFolder:
        test_data, _ = ctools.readfolder(test_data)

    test_dataset = test_dataloader.loader(
        test_data, 32, num_workers=10, shuffle=False)

    # =====================================>> Model Building << ====================================
    print("===> Model building <===")
    set_seed(42)
    net = gaze360.GazeStatic()
    # net = resnet.Model()
    # net = PVT.Model()
    # net = model.Model()
    net.train()
    net.cuda()

    # Pretrain
    # pretrain = train_config.pretrain
    # net.base_model.load_state_dict(
    #     torch.load('ckpts/GazeTR-H-ETH.pt'),
    #     strict=False
    # )
    # if pretrain.enable and pretrain.device:
    #     net.load_state_dict(
    #         torch.load(
    #             pretrain.pretrain_path,
    #             map_location={
    #                 f"cuda:{pretrain.device}": f"cuda:{train_config.device}"}
    #         ),
    #         strict=False
    #     )
    # elif pretrain.enable and not pretrain.device:
    #     net.load_state_dict(
    #         torch.load(pretrain.pretrain_path),
    #         strict=False
    #     )

    print("===> Optimizer building <===")
    optimizer = optim.Adam(
        net.parameters(),
        lr=params.lr,
        betas=(0.9, 0.999)
    )
    if cosine.enable:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine.T_max,
            eta_min=cosine.eta_min
        )
    else:
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
    savepath_eval = os.path.join(save.metapath, save.folder, f"evaluation")

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    if not os.path.exists(savepath_eval):
        os.makedirs(savepath_eval)

    # =====================================>> Training and Evaluation << ====================================
    print("===> Training <===")

    train_length = len(train_dataset)
    total = train_length * params.epoch
    timer = ctools.TimeCounter(total)

    train_losses = []
    test_losses = []

    optimizer.zero_grad()
    optimizer.step()
    scheduler.step()
    scaler = GradScaler()

    with open(os.path.join(savepath, "train_log"), 'w') as outfile:
        outfile.write(ctools.DictDumps(train_config) + '\n')

        with open(os.path.join(savepath_eval, "train_test_log"), 'w') as train_test_log:
            train_test_log.write(ctools.DictDumps(train_config) + '\n')
            train_test_log.write(
                "=====================>> Training and Testing << =======================\n")
            sys.stdout.flush()
            train_test_log.flush()
            
            test_loss_min = float("inf")
            best_loss_iter = 0
            avg_min = 0
            best_dict = []
            for epoch in range(1, params.epoch + 1):
                epoch_losses = []
                net.train()
                for i, (data, anno) in enumerate(train_dataset):
                    # -------------- forward -------------
                    for key in data:
                        if key != 'name':
                            data[key] = data[key].cuda()

                    anno = anno.cuda()
                    with autocast():
                        loss = net.loss(data["face"], anno)
                    epoch_losses.append(loss.item())

                    # -------------- Backward ------------
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    rest = timer.step() / 3600

                    if i % 20 == 0:
                        log = f"[{epoch}/{params.epoch}]: " + \
                              f"[{i}/{train_length}] " + \
                              f"loss:{loss} " + \
                              f"lr:{ctools.GetLR(optimizer)} " + \
                              f"rest time:{rest:.2f}h"

                        print(log)

                scheduler.step()
                train_losses.append(np.mean(epoch_losses))

                sys.stdout.flush()


                # Evaluate on test dataset
                net.eval()
                test_epoch_losses = []
                accs = 0
                count = 0
                with torch.no_grad():
                    for j, (data, label) in enumerate(test_dataset):

                        for key in data:
                            if key != 'name':
                                data[key] = data[key].cuda()

                        gts = label.cuda()
                        with autocast():
                            gazes, _ = net(data["face"])

                        for k, gaze in enumerate(gazes):

                            gaze = gaze.cpu().detach().numpy()
                            gt = gts.cpu().numpy()[k]

                            test_epoch_losses.append(nn.L1Loss()(
                                torch.from_numpy(gaze), torch.from_numpy(gt)).item())

                            count += 1
                            accs += gtools.angular(
                                gtools.gazeto3d(gaze),
                                gtools.gazeto3d(gt)
                            )

                test_losses.append(np.mean(test_epoch_losses))

                test_log = f"[{epoch}/{params.epoch}] " + \
                           f"lr: {ctools.GetLR(optimizer):.6f} " + \
                           f"avg: {accs/count:.6f} " + \
                           f"train_loss: {train_losses[-1]:.12f} " + \
                           f"test_loss: {test_losses[-1]:.12f}\n"

                if test_losses[-1] < test_loss_min:
                    test_loss_min = test_losses[-1]
                    best_loss_iter = epoch
                    best_dict = copy.deepcopy(net.state_dict())
                    avg_min = accs/count
                
                outfile.write(test_log)
                sys.stdout.flush()
                outfile.flush()
                
                if epoch % save.step == 0:
                    train_test_log.write(test_log)
                    sys.stdout.flush()
                    train_test_log.flush()
            
            train_test_log.write(f"Min_Avg: {avg_min:.6f}, iter: {best_loss_iter}\n")
            torch.save(best_dict, os.path.join(savepath, f"Iter{best_loss_iter}_{save.model_name}.pt"))

    # Plot training and testing loss
    plt.figure()
    plt.plot(range(1, params.epoch + 1), train_losses, label='Train Loss')
    plt.plot(range(1, params.epoch + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(savepath, 'train_test_loss.png'))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Pytorch Basic Model Training and Evaluation')

    parser.add_argument('-s', '--train', type=str,
                        help='The source config for training.')

    parser.add_argument('-t', '--test', type=str,
                        help='The source config for testing.')

    args = parser.parse_args()

    train_config = edict(yaml.load(open(args.train), Loader=yaml.FullLoader))
    test_config = edict(yaml.load(open(args.test), Loader=yaml.FullLoader))

    print("=====================>> (Begin) Training and Testing params << =======================")
    print(ctools.DictDumps(train_config))
    print(ctools.DictDumps(test_config))
    print("=====================>> (End) Training and Testing params << =======================")

    train_and_evaluate(train_config.train, test_config.test)

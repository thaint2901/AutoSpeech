# -*- coding: utf-8 -*-
# @Date    : 2019-08-09
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import shutil
import os
from pathlib import Path
from tensorboardX import SummaryWriter
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models.model import Network
from models.model_irse import IR_50
from config import cfg, update_config
from utils import set_path, create_logger, save_checkpoint, count_parameters, Genotype
from data_objects.DeepSpeakerDataset import DeepSpeakerDataset
from data_objects.VoxcelebTestset import VoxcelebTestset
from functions import train_from_scratch, validate_verification
from loss import FocalLoss


def parse_args():
    parser = argparse.ArgumentParser(description='Train energy network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--load_path',
                        help="The path to resumed dir",
                        default=None)

    parser.add_argument('--text_arch',
                        help="The text to arch",
                        default=None)

    args = parser.parse_args()

    return args


def schedule_lr(optimizer):
    for params in optimizer.param_groups:
        params['lr'] /= 10.


def main():
    args = parse_args()
    update_config(cfg, args)
    # assert args.text_arch

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # Set the random seed manually for reproducibility.
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)

    # Loss
    criterion = FocalLoss().cuda()

    # load arch
    genotype = eval("Genotype(normal=[('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('dil_conv_3x3', 2), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 1), ('dil_conv_5x5', 3), ('dil_conv_3x3', 2), ('dil_conv_5x5', 4), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))")

    model = IR_50(cfg.MODEL.NUM_CLASSES)
    # model = Network(cfg.MODEL.INIT_CHANNELS, cfg.MODEL.NUM_CLASSES, cfg.MODEL.LAYERS, genotype)
    model = model.cuda()

    # optimizer = optim.Adam(
    #     model.parameters(),
    #     lr=cfg.TRAIN.LR
    # )
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # resume && make log dir and logger
    if args.load_path and os.path.exists(args.load_path):
        checkpoint_file = args.load_path
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)

        # load checkpoint
        begin_epoch = checkpoint['epoch']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        best_eer = checkpoint['best_eer']
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.path_helper = checkpoint['path_helper']

        # begin_epoch = cfg.TRAIN.BEGIN_EPOCH
        # last_epoch = -1
        # best_eer = 1.0
        # del checkpoint['state_dict']['classifier.weight']
        # del checkpoint['state_dict']['classifier.bias']
        # model.load_state_dict(checkpoint['state_dict'], strict=False)
        # # best_eer = checkpoint['best_eer']
        # # optimizer.load_state_dict(checkpoint['optimizer'])
        # exp_name = args.cfg.split('/')[-1].split('.')[0]
        # args.path_helper = set_path('/content/drive/My Drive/zalo/AutoSpeech/logs_scratch', exp_name)

        logger = create_logger(args.path_helper['log_path'])
        logger.info("=> loaded checkloggpoint '{}'".format(checkpoint_file))
    else:
        exp_name = args.cfg.split('/')[-1].split('.')[0]
        args.path_helper = set_path('logs_scratch', exp_name)
        logger = create_logger(args.path_helper['log_path'])
        begin_epoch = cfg.TRAIN.BEGIN_EPOCH
        best_eer = 1.0
        last_epoch = -1
    logger.info(args)
    logger.info(cfg)
    logger.info(f"selected architecture: {genotype}")
    logger.info("Number of parameters: {}".format(count_parameters(model)))

    # dataloader
    train_dataset = DeepSpeakerDataset(
        Path(cfg.DATASET.DATA_DIR),  cfg.DATASET.SUB_DIR, cfg.DATASET.PARTIAL_N_FRAMES)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
    test_dataset_verification = VoxcelebTestset(
        Path(cfg.DATASET.DATA_DIR), cfg.DATASET.PARTIAL_N_FRAMES)
    test_loader_verification = torch.utils.data.DataLoader(
        dataset=test_dataset_verification,
        batch_size=1,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    # training setting
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'train_global_steps': begin_epoch * len(train_loader),
        'valid_global_steps': begin_epoch // cfg.VAL_FREQ,
    }

    # training loop
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, cfg.TRAIN.END_EPOCH, cfg.TRAIN.LR_MIN,
    #     last_epoch=last_epoch
    # )

    for epoch in tqdm(range(begin_epoch, cfg.TRAIN.END_EPOCH), desc='train progress'):
        model.train()
        model.drop_path_prob = cfg.MODEL.DROP_PATH_PROB * epoch / cfg.TRAIN.END_EPOCH

        train_from_scratch(cfg, model, optimizer, train_loader, criterion, epoch, writer_dict)
        
        if epoch == 210 or epoch == 240 or epoch == 270:
            schedule_lr(optimizer)

        if epoch % cfg.VAL_FREQ == 0 or epoch == cfg.TRAIN.END_EPOCH - 1:
            # eer = validate_verification(cfg, model, test_loader_verification)

            # # remember best acc@1 and save checkpoint
            # is_best = eer < best_eer
            # best_eer = min(eer, best_eer)

            # save
            logger.info('=> saving checkpoint to {}'.format(args.path_helper['ckpt_path']))
            print('=> saving checkpoint to {}'.format(args.path_helper['ckpt_path']))
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_eer': best_eer,
                'optimizer': optimizer.state_dict(),
                'path_helper': args.path_helper
            }, True, args.path_helper['ckpt_path'], 'checkpoint_{}.pth'.format(epoch))

        # lr_scheduler.step(epoch)


if __name__ == '__main__':
    main()

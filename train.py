# Package.
import time
from pathlib import Path
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, train_test_split
from skimage.io import imread
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
from lib.dataset import Dataset
from lib.models.model_factory import get_model
from lib.utils import *
from lib.metrics import *
from lib.losses import *
from lib.optimizers import *
from lib.preprocess import preprocess
from lib.sam import SAM
from lib.autoaugment import RNFLPolicy
from torchtools.optim import Ranger, RangerLars # Over9000
from RandAugment import RandAugment
# from lib import multiprocess as multiprocessing
import dill as pickle

# Args.
def parse_args():

    # Hyperparameters.
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='DR', help='model name: (default: arch+timestamp)')
    parser.add_argument('--print_model', default=False, type=str2bool)
    parser.add_argument('--arch', '-a', metavar='ARCH', default='efficientnet-b4a', help='model architecture: ' + ' (default: se_resnext50_32x4d)')
    parser.add_argument('--freeze_bn', default=False, type=str2bool)
    parser.add_argument('--dropout_p', default=0, type=float)
    parser.add_argument('--loss', default='FocalLoss', choices=['CrossEntropyLoss', 'FocalLoss', 'MSELoss', 'BCELoss', 'L1Loss', 'multitask'])
    parser.add_argument('--reg_coef', default=1.0, type=float)
    parser.add_argument('--cls_coef', default=0.1, type=float)
    parser.add_argument('--epochs', default=35, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=23, type=int, metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--img_size', default=236, type=int, help='processed image size (default: 288)')
    parser.add_argument('--input_size', default=224, type=int, help='cropped input image size (default: 256)')
    parser.add_argument('--optimizer', default='RangerLars', choices=['SGD', 'Adam', 'AdamW', 'RAdam', 'Ranger','RangerLars', 'SAM'])
    parser.add_argument('--pred_type', default='classification', choices=['classification', 'regression', 'multitask'])
    parser.add_argument('--scheduler', default='CosineAnnealingLR', choices=['CosineAnnealingLR', 'ReduceLROnPlateau'])
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--min_lr', default=5e-5, type=float, help='minimum learning rate')
    parser.add_argument('--factor', default=0.8, type=float)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool, help='nesterov')
    parser.add_argument('--num_workers', default=0, type=int)

    # Preprocessing.
    parser.add_argument('--red_free', default=False, type=str2bool)
    parser.add_argument('--scale_radius', default=True, type=str2bool)
    parser.add_argument('--normalize', default=False, type=str2bool)
    parser.add_argument('--padding', default=False, type=str2bool)
    parser.add_argument('--remove', default=False, type=str2bool)

    # Data Augmentation.
    parser.add_argument('--AAugment', default=False, type=str2bool)
    parser.add_argument('--RAugment', default=True, type=str2bool)
    parser.add_argument('--RAugment_N', default=3, type=int)
    parser.add_argument('--RAugment_M', default=1, type=int)
    parser.add_argument('--rotate', default=True, type=str2bool)
    parser.add_argument('--rotate_min', default=-15, type=int)
    parser.add_argument('--rotate_max', default=15, type=int)
    parser.add_argument('--rescale', default=True, type=str2bool)
    parser.add_argument('--rescale_min', default=0.97, type=float)
    parser.add_argument('--rescale_max', default=1.03, type=float)
    parser.add_argument('--shear', default=False, type=str2bool)
    parser.add_argument('--shear_min', default=-15, type=int)
    parser.add_argument('--shear_max', default=15, type=int)
    parser.add_argument('--translate', default=True, type=str2bool)
    parser.add_argument('--translate_min', default=0.03, type=float)
    parser.add_argument('--translate_max', default=0.03, type=float)
    parser.add_argument('--flip', default=True, type=str2bool)
    parser.add_argument('--brightness', default=True, type=str2bool)
    parser.add_argument('--brightness_min', default=0.97, type=float)
    parser.add_argument('--brightness_max', default=1.03, type=float)
    parser.add_argument('--contrast', default=False, type=str2bool)
    parser.add_argument('--contrast_min', default=0.97, type=float)
    parser.add_argument('--contrast_max', default=1.03, type=float)
    parser.add_argument('--saturation', default=False, type=str2bool)
    parser.add_argument('--saturation_min', default=0.97, type=float)
    parser.add_argument('--saturation_max', default=1.03, type=float)
    parser.add_argument('--random_erase', default=False, type=str2bool)
    parser.add_argument('--random_erase_prob', default=0.5, type=float)
    parser.add_argument('--random_erase_sl', default=0.02, type=float)
    parser.add_argument('--random_erase_sh', default=0.4, type=float)
    parser.add_argument('--random_erase_r', default=0.3, type=float)

    # Dataset-Split.
    parser.add_argument('--train_dataset', default='RNFL')
    parser.add_argument('--cv', default=True, type=str2bool)
    parser.add_argument('--n_splits', default=3, type=int)
    parser.add_argument('--remove_duplicate', default=False, type=str2bool)

    # Over-Sampling.
    parser.add_argument('--class_aware', default=False, type=str2bool)
    parser.add_argument('--N_ratio', default=1, type=float)
    parser.add_argument('--P_ratio', default=1, type=float)
    parser.add_argument('--num_train_sample', default=2655, type=int)
    parser.add_argument('--num_val_sample', default=885, type=int)

    # Pseudo label.
    parser.add_argument('--pretrained_model')
    parser.add_argument('--pseudo_labels')
    args = parser.parse_args()
    return args

def train(args, train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    scores = AverageMeter()
    model.train()
    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        if args.pred_type == 'classification':
            loss = criterion(output, target)
        elif args.pred_type == 'regression':
            loss = criterion(output.view(-1), target.float())
        elif args.pred_type == 'multitask':
            loss = args.reg_coef * criterion['regression'](output[:, 0], target.float()) + \
                   args.cls_coef * criterion['classification'](output[:, 1:], target)
            output = output[:, 0].unsqueeze(1)

        # Compute gradient and do optimizing step.
        optimizer.zero_grad()
        loss.backward()
        if args.optimizer == 'SAM':
            optimizer.first_step(zero_grad=True)
            criterion(model(input), target).backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.step()

        if args.pred_type == 'regression' or args.pred_type == 'multitask':
            thrs = [0.5, 1.5, 2.5, 3.5]
            output[output < thrs[0]] = 0
            output[(output >= thrs[0]) & (output < thrs[1])] = 1
            output[(output >= thrs[1]) & (output < thrs[2])] = 2
            output[(output >= thrs[2]) & (output < thrs[3])] = 3
            output[output >= thrs[3]] = 4
        if args.pseudo_labels is not None:
            thrs = [0.5, 1.5, 2.5, 3.5]
            output[output < thrs[0]] = 0
            output[(output >= thrs[0]) & (output < thrs[1])] = 1
            output[(output >= thrs[1]) & (output < thrs[2])] = 2
            output[(output >= thrs[2]) & (output < thrs[3])] = 3
            output[output >= thrs[3]] = 4
        score = my_metrics(output, target)
        losses.update(loss.item(), input.size(0))
        scores.update(score, input.size(0))
    return losses.avg, scores.avg

def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    scores = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            if args.pred_type == 'classification':
                loss = criterion(output, target)
            elif args.pred_type == 'regression':
                loss = criterion(output.view(-1), target.float())
            elif args.pred_type == 'multitask':
                loss = args.reg_coef * criterion['regression'](output[:, 0], target.float()) + \
                       args.cls_coef * criterion['classification'](output[:, 1:], target)
                output = output[:, 0].unsqueeze(1)
            if args.pred_type == 'regression' or args.pred_type == 'multitask':
                thrs = [0.5, 1.5, 2.5, 3.5]
                output[output < thrs[0]] = 0
                output[(output >= thrs[0]) & (output < thrs[1])] = 1
                output[(output >= thrs[1]) & (output < thrs[2])] = 2
                output[(output >= thrs[2]) & (output < thrs[3])] = 3
                output[output >= thrs[3]] = 4
            score = my_metrics(output, target)
            losses.update(loss.item(), input.size(0))
            scores.update(score, input.size(0))
    return losses.avg, scores.avg

def main():
    args = parse_args()
    if args.name is None:
        args.name = '%s_%s' % (args.arch, datetime.now().strftime('%m%d%H%M'))
    if not os.path.exists('models/%s' % args.name):
        os.makedirs('models/%s' % args.name)
    if args.print_model:
        print('Config -----')
        for arg in vars(args):
            print('- %s: %s' % (arg, getattr(args, arg)))
        print('------------')
    with open('models/%s/args.txt' % args.name, 'w') as f:
        for arg in vars(args):
            print('- %s: %s' % (arg, getattr(args, arg)), file=f)
    joblib.dump(args, 'models/%s/args.pkl' % args.name)

    if args.loss == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss().cuda()
    elif args.loss == 'FocalLoss':
        criterion = FocalLoss().cuda()
    elif args.loss == 'MSELoss':
        criterion = nn.MSELoss().cuda()
    elif args.loss == 'BCELoss':
        criterion = nn.BCELoss().cuda()
    elif args.loss == 'L1Loss':
        criterion = nn.L1Loss().cuda()
    elif args.loss == 'multitask':
        criterion = {
            'classification': nn.CrossEntropyLoss().cuda(),
            'regression': nn.MSELoss().cuda(),
        }
    else:
        raise NotImplementedError

    if args.pred_type == 'classification':
        num_outputs = 5
    elif args.pred_type == 'regression':
        num_outputs = 1
    elif args.loss == 'multitask':
        num_outputs = 6
    else:
        raise NotImplementedError

    cudnn.benchmark = True

    train_transform = []
    if args.RAugment:
        train_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomAffine(
                degrees=(args.rotate_min, args.rotate_max) if args.rotate else 0,
                translate=(args.translate_min, args.translate_max) if args.translate else None,
                scale=(args.rescale_min, args.rescale_max) if args.rescale else None,
                shear=(args.shear_min, args.shear_max) if args.shear else None,
            ),
            transforms.CenterCrop(args.input_size),
            transforms.RandomHorizontalFlip(p=0.5 if args.flip else 0),
            # transforms.RandomVerticalFlip(p=0.5 if args.flip else 0),
            transforms.ColorJitter(
                brightness=(args.brightness_min, args.brightness_max) if args.brightness else 0,
                contrast=(args.contrast_min, args.contrast_max) if args.contrast else 0,
                saturation=(args.saturation_min, args.saturation_max) if args.saturation else 0,
                hue=0),
            RandomErase(
                prob=args.random_erase_prob if args.random_erase else 0,
                sl=args.random_erase_sl,
                sh=args.random_erase_sh,
                r=args.random_erase_r),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train_transform.transforms.insert(0, RandAugment(args.RAugment_N, args.RAugment_M))
    else:
        train_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomAffine(
                degrees=(args.rotate_min, args.rotate_max) if args.rotate else 0,
                translate=(args.translate_min, args.translate_max) if args.translate else None,
                scale=(args.rescale_min, args.rescale_max) if args.rescale else None,
                shear=(args.shear_min, args.shear_max) if args.shear else None,
            ),
            transforms.CenterCrop(args.input_size),
            transforms.RandomHorizontalFlip(p=0.5 if args.flip else 0),
            # transforms.RandomVerticalFlip(p=0.5 if args.flip else 0),
            transforms.ColorJitter(
                brightness=(args.brightness_min, args.brightness_max) if args.brightness else 0,
                contrast=(args.contrast_min, args.contrast_max) if args.contrast else 0,
                saturation=(args.saturation_min, args.saturation_max) if args.saturation else 0,
                hue=0),
            RandomErase(
                prob=args.random_erase_prob if args.random_erase else 0,
                sl=args.random_erase_sl,
                sh=args.random_erase_sh,
                r=args.random_erase_r),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    # IF Class Aware, need to do Data Augumentation on Validation.
    if args.class_aware:
        val_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.input_size)),
            transforms.RandomAffine(
                degrees=(args.rotate_min, args.rotate_max) if args.rotate else 0,
                translate=(args.translate_min, args.translate_max) if args.translate else None,
                scale=(args.rescale_min, args.rescale_max) if args.rescale else None,
                shear=(args.shear_min, args.shear_max) if args.shear else None,),
            transforms.CenterCrop(args.input_size),
            transforms.RandomHorizontalFlip(p=0.5 if args.flip else 0),
            # transforms.RandomVerticalFlip(p=0.5 if args.flip else 0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        val_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    # Data loading.
    if 'diabetic_retinopathy' in args.train_dataset:
        diabetic_retinopathy_dir = preprocess(
            'diabetic_retinopathy',
            args.img_size,
            scale=args.scale_radius,
            norm=args.normalize,
            pad=args.padding,
            remove=args.remove)
        diabetic_retinopathy_df = pd.read_csv('inputs/diabetic-retinopathy-resized/trainLabels.csv')
        diabetic_retinopathy_img_paths = \
            diabetic_retinopathy_dir + '/' + diabetic_retinopathy_df['image'].values + '.jpeg'
        diabetic_retinopathy_labels = diabetic_retinopathy_df['level'].values

    if 'RNFL' in args.train_dataset:
        aptos2019_dir = preprocess(
            'RNFL',
            args.img_size,
            scale=args.scale_radius,
            norm=args.normalize,
            pad=args.padding,
            remove=args.remove,
            red_free=args.red_free)
        aptos2019_df = pd.read_csv(Path("./data/dataset/train/label.csv"))
        aptos2019_img_paths = aptos2019_dir + '/' + aptos2019_df['id_code'].values + '.png'
        aptos2019_labels = aptos2019_df['diagnosis'].values

    if args.train_dataset == 'RNFL':
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=41)
        img_paths = []
        labels = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(aptos2019_img_paths, aptos2019_labels)):
            img_paths.append((aptos2019_img_paths[train_idx], aptos2019_img_paths[val_idx]))
            labels.append((aptos2019_labels[train_idx], aptos2019_labels[val_idx]))



    elif args.train_dataset == 'diabetic_retinopathy':
        img_paths = [(diabetic_retinopathy_img_paths, aptos2019_img_paths)]
        labels = [(diabetic_retinopathy_labels, aptos2019_labels)]
    elif 'diabetic_retinopathy' in args.train_dataset and 'aptos2019' in args.train_dataset:
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=41)
        img_paths = []
        labels = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(aptos2019_img_paths, aptos2019_labels)):
            img_paths.append((np.hstack((aptos2019_img_paths[train_idx], diabetic_retinopathy_img_paths)), aptos2019_img_paths[val_idx]))
            labels.append((np.hstack((aptos2019_labels[train_idx], diabetic_retinopathy_labels)), aptos2019_labels[val_idx]))
    # else:
    #     raise NotImplementedError

    if args.pseudo_labels:
        test_df = pd.read_csv('probs/%s.csv' % args.pseudo_labels)
        test_dir = preprocess(
            'test',
            args.img_size,
            scale=args.scale_radius,
            norm=args.normalize,
            pad=args.padding,
            remove=args.remove)
        test_img_paths = test_dir + '/' + test_df['id_code'].values + '.png'
        test_labels = test_df['diagnosis'].values
        for fold in range(len(img_paths)):
            img_paths[fold] = (np.hstack((img_paths[fold][0], test_img_paths)), img_paths[fold][1])
            labels[fold] = (np.hstack((labels[fold][0], test_labels)), labels[fold][1])

    if 'messidor' in args.train_dataset:
        test_dir = preprocess(
            'messidor',
            args.img_size,
            scale=args.scale_radius,
            norm=args.normalize,
            pad=args.padding,
            remove=args.remove)

    folds = []
    best_losses = []
    best_scores = []

    for fold, ((train_img_paths, val_img_paths), (train_labels, val_labels)) in enumerate(zip(img_paths, labels)):
        print('Fold [%d/%d]' %(fold+1, len(img_paths)))

        if os.path.exists('models/%s/model_%d.pth' % (args.name, fold+1)):
            log = pd.read_csv('models/%s/log_%d.csv' %(args.name, fold+1))
            best_loss, best_score = log.loc[log['val_loss'].values.argmin(), ['val_loss', 'val_score']].values
            folds.append(str(fold + 1))
            best_losses.append(best_loss)
            best_scores.append(best_score)
            continue

        if args.remove_duplicate:
            md5_df = pd.read_csv('inputs/strMd5.csv')
            duplicate_img_paths = aptos2019_dir + '/' + md5_df[(md5_df.strMd5_count > 1) & (~md5_df.diagnosis.isnull())]['id_code'].values + '.png'
            print(duplicate_img_paths)
            for duplicate_img_path in duplicate_img_paths:
                train_labels = train_labels[train_img_paths != duplicate_img_path]
                train_img_paths = train_img_paths[train_img_paths != duplicate_img_path]
                val_labels = val_labels[val_img_paths != duplicate_img_path]
                val_img_paths = val_img_paths[val_img_paths != duplicate_img_path]

        # Train.
        train_set = Dataset(
            train_img_paths,
            train_labels,
            transform=train_transform)

        _, class_sample_counts = np.unique(train_labels, return_counts=True)
        _, vclass_sample_counts = np.unique(val_labels, return_counts=True)
        print('Train:', class_sample_counts)
        print('Validation:', vclass_sample_counts)
        if args.class_aware:
            print('Class Aware(N:P):', args.N_ratio, ':', args.P_ratio)
            print('Number of Train samples:', args.num_train_sample)
            print('Number of Validation samples:', args.num_val_sample)
            print('------------------------------------------')

        weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
        weights = np.array([1, 1, 1, 1, 1])
        samples_weights = weights[train_labels]
        vsamples_weights = weights[val_labels]
        tsampler = WeightedRandomSampler(
            weights=samples_weights,
            num_samples=args.num_train_sample,
            replacement=True)

        vsampler = WeightedRandomSampler(
            weights=vsamples_weights,
            num_samples=args.num_val_sample,
            replacement=True)

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=False if args.class_aware else True,
            num_workers=args.num_workers,
            sampler=tsampler if args.class_aware else None)

        val_set = Dataset(
            val_img_paths,
            val_labels,
            transform=val_transform)
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            sampler=vsampler if args.class_aware else None)

        # Create model.
        model = get_model(model_name=args.arch,
                          num_outputs=num_outputs,
                          freeze_bn=args.freeze_bn,
                          dropout_p=args.dropout_p)

        if args.print_model:
            print(model)
        model = model.cuda()
        if args.pretrained_model is not None:
            model.load_state_dict(torch.load('models/%s/model_%d.pth' % (args.pretrained_model, fold+1)))

        # Print the model.
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'AdamW':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'RAdam':
            optimizer = RAdam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                  momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        elif args.optimizer == 'Ranger':
            optimizer = Ranger(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'RangerLars':
            optimizer = RangerLars(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'SAM':
            base_optimizer = RangerLars
            optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, weight_decay=args.weight_decay)
        # if args.scheduler == 'CosineAnnealingLR':
        #     scheduler = lr_scheduler.CosineAnnealingLR(
        #         optimizer, T_max=args.epochs, eta_min=args.min_lr)
        if args.scheduler == 'CosineAnnealingLR':
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                  optimizer, T_0=5, T_mult=6, eta_min=args.min_lr)
        elif args.scheduler == 'ReduceLROnPlateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience,
                                                       verbose=1, min_lr=args.min_lr)

        log = pd.DataFrame(index=[], columns=[
            'epoch', 'loss', 'score', 'val_loss', 'val_score'
        ])
        log = {
            'epoch': [],
            'loss': [],
            'score': [],
            'val_loss': [],
            'val_score': [],
        }

        best_loss = float('inf')
        best_score = 0
        for epoch in range(args.epochs):
            print('Epoch [%d/%d]' % (epoch + 1, args.epochs))

            # Train for one epoch.
            train_loss, train_score = train(
                args, train_loader, model, criterion, optimizer, epoch)
            # Evaluate on validation set.
            val_loss, val_score = validate(args, val_loader, model, criterion)

            if args.scheduler == 'CosineAnnealingLR':
                scheduler.step()
            elif args.scheduler == 'ReduceLROnPlateau':
                scheduler.step(val_loss)

            print('loss %.4f - score %.4f - val_loss %.4f - val_score %.4f - best_score %.4f'
                  % (train_loss, train_score, val_loss, val_score, best_score))

            log['epoch'].append(epoch)
            log['loss'].append(train_loss)
            log['score'].append(train_score)
            log['val_loss'].append(val_loss)
            log['val_score'].append(val_score)

            pd.DataFrame(log).to_csv('models/%s/log_%d.csv' % (args.name, fold+1), index=False)

            if val_score >= best_score:
                if (val_score/train_score) <= 1.6:
                    torch.save(model.state_dict(), 'models/%s/model_%d.pth' % (args.name, fold+1))
                    best_loss = val_loss
                    best_score = val_score
                    print("=> saved best model")

        print('val_loss:  %f' % best_loss)
        print('val_score: %f' % best_score)

        folds.append(str(fold + 1))
        best_losses.append(best_loss)
        best_scores.append(best_score)

        results = pd.DataFrame({
            'fold': folds + ['mean'],
            'best_loss': best_losses + [np.mean(best_losses)],
            'best_score': best_scores + [np.mean(best_scores)],
        })

        print(results)
        results.to_csv('models/%s/results.csv' % args.name, index=False)

        torch.cuda.empty_cache()

        if not args.cv:
            break


if __name__ == '__main__':
    main()

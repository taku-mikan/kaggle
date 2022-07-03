import argparse
import datetime
import gc
import glob

import albumentations as A
import matplotlib.pyplot as plt
#
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from albumentations.pytorch import ToTensorV2
from IPython import display
from sklearn.model_selection import (KFold, StratifiedGroupKFold,
                                        StratifiedKFold)

# pd.options.plotting.backend = "plotly"
import os
import random
import shutil
from glob import glob

from tqdm import tqdm

tqdm.pandas()
import copy
import gc
import time
from collections import defaultdict

# Albumentations for augmentations
import albumentations as A
# visualization
import cv2
import joblib
import matplotlib.pyplot as plt
import rasterio
import timm
# PyTorch 
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
# For colored terminal text
from colorama import Back, Fore, Style
from IPython import display as ipd
from joblib import Parallel, delayed
from matplotlib.patches import Rectangle
# Sklearn
from sklearn.model_selection import (KFold, StratifiedGroupKFold,
                                     StratifiedKFold)
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset

c_  = Fore.GREEN
sr_ = Style.RESET_ALL

import warnings

warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#

from datasets import BuildDataset, prepare_loaders
from losses.losses import criterion, dice_coef, iou_coef
from metric.base import Metric
from models.models import build_model, load_model
from utils.utils import fix_seed, plot_batch


def wandb_log(loss: float, metrics:Metric, phase:str) -> None:
    """
    Args:
        loss (float): 損失関数の値
        metric (Metric): Metricの値
        phase (str): train | val | test
    """
    log_items = {f"{phase}_loss" : loss}

    for metric, value in metrics.score().items():
        log_items[f"{phase}_{metric}"] = value
    
    wandb.log(log_items)


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch, n_accumulate):
    model.train()
    scaler = amp.GradScaler()
    
    dataset_size = 0
    running_loss = 0.0
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ')
    for step, (images, masks) in pbar:         
        images = images.to(device, dtype=torch.float)
        masks  = masks.to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        with amp.autocast(enabled=True):
            y_pred = model(images)
            loss   = criterion(y_pred, masks)
            loss   = loss / n_accumulate
            
        scaler.scale(loss).backward()
    
        if (step + 1) % n_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
                        lr=f'{current_lr:0.5f}',
                        gpu_mem=f'{mem:0.2f} GB')
        torch.cuda.empty_cache()
        gc.collect()
    
    return epoch_loss

@torch.no_grad()
def valid_one_epoch(model, optimizer, dataloader, device, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    
    val_scores = []
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')
    for step, (images, masks) in pbar:        
        images  = images.to(device, dtype=torch.float)
        masks   = masks.to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        y_pred  = model(images)
        loss    = criterion(y_pred, masks)
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        y_pred = nn.Sigmoid()(y_pred)
        val_dice = dice_coef(masks, y_pred).cpu().detach().numpy()
        val_jaccard = iou_coef(masks, y_pred).cpu().detach().numpy()
        val_scores.append([val_dice, val_jaccard])
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                        lr=f'{current_lr:0.5f}',
                        gpu_memory=f'{mem:0.2f} GB')
    val_scores  = np.mean(val_scores, axis=0)
    torch.cuda.empty_cache()
    gc.collect()
    
    return epoch_loss, val_scores

def run_training(model, optimizer, scheduler, device, num_epochs, train_loader, valid_loader, run, fold, n_accumulate):
    # To automatically log gradients
    wandb.watch(model, log_freq=100)
    
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_dice      = -np.inf
    best_epoch     = -1
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        print(f'Epoch {epoch}/{num_epochs}', end='')
        train_loss = train_one_epoch(model, optimizer, scheduler, 
                                            dataloader=train_loader, 
                                            device=device, epoch=epoch,
                                            n_accumulate=n_accumulate)
        
        val_loss, val_scores = valid_one_epoch(model, optimizer, valid_loader, 
                                                    device=device, 
                                                    epoch=epoch,
                                                    )
        val_dice, val_jaccard = val_scores
    
        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(val_loss)
        history['Valid Dice'].append(val_dice)
        history['Valid Jaccard'].append(val_jaccard)
        
        # Log the metrics
        wandb.log({"Train Loss": train_loss, 
                    "Valid Loss": val_loss,
                    "Valid Dice": val_dice,
                    "Valid Jaccard": val_jaccard,
                    "LR":scheduler.get_last_lr()[0]})
        
        print(f'Valid Dice: {val_dice:0.4f} | Valid Jaccard: {val_jaccard:0.4f}')
        
        # deep copy the model
        if val_dice >= best_dice:
            print(f"{c_}Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
            best_dice    = val_dice
            best_jaccard = val_jaccard
            best_epoch   = epoch
            run.summary["Best Dice"]    = best_dice
            run.summary["Best Jaccard"] = best_jaccard
            run.summary["Best Epoch"]   = best_epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f"best_epoch-{fold:02d}.bin"
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            wandb.save(PATH)
            print(f"Model Saved{sr_}")
            
        last_model_wts = copy.deepcopy(model.state_dict())
        PATH = f"last_epoch-{fold:02d}.bin"
        torch.save(model.state_dict(), PATH)
            
        print(); print()
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Score: {:.4f}".format(best_jaccard))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history


def fetch_scheduler(optimizer, scheduler, min_lr, T_0, T_max):
    if scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=T_max, 
                                                    eta_min=min_lr)
    elif scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=T_0, 
                                                                eta_min=min_lr)
    elif scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                    mode='min',
                                                    factor=0.1,
                                                    patience=7,
                                                    threshold=0.0001,
                                                    min_lr=min_lr,)
    elif scheduler == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    elif scheduler == None:
        return None
        
    return scheduler

def main(args: argparse.Namespace):
    # time
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H%M%S")

    # seedの固定
    fix_seed(args.seed, args.no_deterministic)

    # wandbの設定
    try:
        wandb.init(project="Kaggle-UWM", name="kaggle_test")
        anonymous = None
    except:
        anonymous = "must"

    # データセットの作成
    path_df = pd.DataFrame(glob('./datasets/image_datasets/images/images/*'), columns=['image_path'])
    path_df['mask_path'] = path_df.image_path.str.replace('images','masks')
    path_df['id'] = path_df.image_path.map(lambda x: x.split('/')[-1].replace('.npy',''))

    df = pd.read_csv('./datasets/mask_datasets/train.csv')
    df['segmentation'] = df.segmentation.fillna('')
    df['rle_len'] = df.segmentation.map(len) # length of each rle mask

    df2 = df.groupby(['id'])['segmentation'].agg(list).to_frame().reset_index() # rle list of each id
    df2 = df2.merge(df.groupby(['id'])['rle_len'].agg(sum).to_frame().reset_index()) # total length of all rles of each id

    df = df.drop(columns=['segmentation', 'class', 'rle_len'])
    df = df.groupby(['id']).head(1).reset_index(drop=True)
    df = df.merge(df2, on=['id'])
    df['empty'] = (df.rle_len==0) # empty masks

    df = df.drop(columns=['image_path','mask_path'])
    df = df.merge(path_df, on=['id'])

    fault1 = 'case7_day0'
    fault2 = 'case81_day30'
    df = df[~df['id'].str.contains(fault1) & ~df['id'].str.contains(fault2)].reset_index(drop=True)

    # TODO: ここまではOK
    # create folds
    skf = StratifiedGroupKFold(n_splits=args.n_fold, shuffle=True, random_state=args.seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['empty'], groups = df["case"])):
        df.loc[val_idx, 'fold'] = fold
    # display(df.groupby(['fold','empty'])['id'].count())

    # augumentaiton
    data_transforms = {
        "train": A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            ], p=0.25),
            A.CoarseDropout(max_holes=8, max_height=args.img_size[0]//20, max_width=args.img_size[1]//20,
                                min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
            ], p=1.0),
        
        "valid": A.Compose([
            ], p=1.0)
    }

    # TODO: ここでではOK
    # dataloaderの作成
    train_loader, valid_loader = prepare_loaders(df, fold=0, debug=True, train_bs=args.train_bs, valid_bs=args.valid_bs, transforms=data_transforms)

    gc.collect()

    # モデルの作成
    model = build_model(backbone=args.backbone, num_classes=args.num_classes, device=device)
    # Optimizerの設定
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # schedulerの設定
    scheduler = fetch_scheduler(
        optimizer, scheduler=args.scheduler, 
        min_lr=args.min_lr, T_0 = args.T_0, T_max=args.T_max)

    # 損失関数 / Metricの設定
    JaccardLoss = smp.losses.JaccardLoss(mode='multilabel')
    DiceLoss    = smp.losses.DiceLoss(mode='multilabel')
    BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
    LovaszLoss  = smp.losses.LovaszLoss(mode='multilabel', per_image=False)
    TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)

    # run_name の設定

    # Early_stoppingの設定

    # wnadbの設定

    # 重み等のload

    # summaryの表示

    # train / test
    # TODO: ここまでOK
    for fold in tqdm(args.folds):
        print(f'#'*15)
        print(f'### Fold: {fold}')
        print(f'#'*15)
        run = wandb.init(project='uw-maddison-gi-tract', 
                        config={k:v for k, v in dict(vars(args)).items() if '__' not in k},
                        anonymous=anonymous,
                        name=f"fold-{fold}|dim-{args.img_size[0]}x{args.img_size[1]}|model-{args.model_name}",
                        group=args.comment,
                        )
        train_loader, valid_loader = prepare_loaders(df, fold=fold,
                                                        train_bs=args.train_bs,
                                                        valid_bs=args.valid_bs,
                                                        transforms=data_transforms,
                                                        debug=args.debug,
                                                        )
        model     = build_model(backbone=args.backbone, num_classes=args.num_classes, device=device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = fetch_scheduler(optimizer, scheduler=args.scheduler, min_lr=args.min_lr, T_0=args.min_lr, T_max=args.T_max)
        model, history = run_training(model, optimizer, scheduler,
                                    device=device,
                                    num_epochs=args.epochs,
                                    train_loader=train_loader,
                                    valid_loader=valid_loader,
                                    run=run,
                                    fold=fold,
                                    n_accumulate=args.n_accumulate)
        run.finish()
        display(ipd.IFrame(run.url, width=1000, height=720))
    

    # Prediction
    test_dataset = BuildDataset(df.query("fold==0 & empty==0").sample(frac=1.0), label=False, 
                            transforms=data_transforms['valid'])
    test_loader  = DataLoader(test_dataset, batch_size=5, 
                            num_workers=4, shuffle=False, pin_memory=True)
    imgs = next(iter(test_loader))
    imgs = imgs.to(args.device, dtype=torch.float)

    preds = []
    for fold in args.folds:
        model = load_model(f"best_epoch-{fold:02d}.bin")
        with torch.no_grad():
            pred = model(imgs)
            pred = (nn.Sigmoid()(pred)>0.5).double()
        preds.append(pred)
        
    imgs  = imgs.cpu().detach()
    preds = torch.mean(torch.stack(preds, dim=0), dim=0).cpu().detach()

    plot_batch(imgs, preds, size=5)
    # wandbのlogの作成

    # 各種設定

    # 重みの設定


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--no_deterministic", action="store_false")

    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--exp_name", type=str, default="2.50")
    parser.add_argument("--model", choices=["Unet"], default="Unet")
    parser.add_argument("--backbone", type=str, default="efficientnet-b4")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_bs", type=int, default=32)
    parser.add_argument("--valid_bs", type=int, default=64)
    parser.add_argument('--img_size', nargs="*", type=int, default=[160, 192])
    parser.add_argument("--img_h", type=int, default=160)
    parser.add_argument("--img_w", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--scheduler", choices=["CosineAnnealingLR"], default="CosineAnnealingLR")
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--T-max", type=int, default=143)
    parser.add_argument("--T_0", type=int, default=25)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--wd", type=float, default=1e-6)
    parser.add_argument("--n_accumulate", type=float, default=1)
    parser.add_argument("--n_fold", type=int, default=5)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument('--folds', nargs="*", type=int, default=[0])
    parser.add_argument("--model_name", type=str, default="UNet")
    parser.add_argument("--comment", type=str, default='unet-efficientnet_b0-160x192-ep=5')

    return parser.parse_args()

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    main(parse_args())

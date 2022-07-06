import gc
import os
import argparse
import warnings
from glob import glob
from numpy import argsort
from tqdm import tqdm

import matplotlib.pyplot as plt 
import albumentations as A
import pandas as pd
import torch
import torch.nn as nn
from   torch.utils.data import DataLoader
from colorama import Fore,  Style

from utils.utils_test import fix_seed, get_metadata, path2info, masks2rles
from data.datasets import TestDataset
from models.models import load_model
from utils.utils import parse_with_config

c_  = Fore.GREEN
sr_ = Style.RESET_ALL
warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
tqdm.pandas()


@torch.no_grad()
def inference(model_paths, test_loader, thr, device, debug, backbone, num_classes, img_size, config, num_log: int=1):
    msks = []; imgs = [];
    pred_strings = []; pred_ids = []; pred_classes = [];
    for idx, (img, ids, heights, widths) in enumerate(tqdm(test_loader, total=len(test_loader), desc='Infer ')):
        img = img.to(device, dtype=torch.float)
        size = img.size()
        msk = []
        msk = torch.zeros((size[0], 3, size[2], size[3]), device=device, dtype=torch.float32)
        for path in model_paths:
            model = load_model(path, backbone, num_classes, device, img_size, model, config)
            out   = model(img) # .squeeze(0) # removing batch axis
            out   = nn.Sigmoid()(out) # removing channel axis
            msk+=out/len(model_paths)
        msk = (msk.permute((0,2,3,1))>thr).to(torch.uint8).cpu().detach().numpy() # shape: (n, h, w, c)
        result = masks2rles(msk, ids, heights, widths)
        pred_strings.extend(result[0])
        pred_ids.extend(result[1])
        pred_classes.extend(result[2])
        if idx<num_log and debug:
            img = img.permute((0,2,3,1)).cpu().detach().numpy()
            imgs.append(img[::5])
            msks.append(msk[::5])
        del img, msk, out, model, result
        gc.collect()
        torch.cuda.empty_cache()
    return pred_strings, pred_ids, pred_classes, imgs, msks

def main(args: argparse.Namespace):
    # seed
    fix_seed(args.seed)

    # path
    base_path = "datasets/image_datasets"
    ckpt_dir = 'checkpoints'

    # data
    sub_df = pd.read_csv('datasets/test_datasets/sample_submission.csv')
    if not len(sub_df):
        debug = True
        sub_df = pd.read_csv('datasets/test_datasets/train.csv')
        sub_df = sub_df[~sub_df.segmentation.isna()][:1000*3]
        sub_df = sub_df.drop(columns=['class','segmentation']).drop_duplicates()
    else:
        debug = False
        sub_df = sub_df.drop(columns=['class','predicted']).drop_duplicates()
    sub_df = sub_df.progress_apply(get_metadata, axis=1)

    if debug:
        paths = glob(f'datasets/test_datasets/train/**/*png',recursive=True)
    else:
        paths = glob(f'datasets/test_datasets/test/**/*png',recursive=True)
    path_df = pd.DataFrame(paths, columns=['image_path'])
    path_df = path_df.progress_apply(path2info, axis=1)
    path_df.head()

    # merge data
    test_df = sub_df.merge(path_df, on=['case','day','slice'], how='left')

    # 2.5 MetaData
    channels=3
    stride=2
    for i in range(channels):
        test_df[f'image_path_{i:02}'] = test_df.groupby(['case','day'])['image_path'].shift(-i*stride).fillna(method="ffill")
    test_df['image_paths'] = test_df[[f'image_path_{i:02d}' for i in range(channels)]].values.tolist()
    if debug:
        test_df = test_df.sample(frac=1.0)
    test_df.image_paths[0]

    # Transform
    data_transforms = {
    "train": A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
        ], p=0.25),
    ], p=1.0),
    
    "valid": A.Compose([
        ], p=1.0)
    }

    # create data
    test_dataset = TestDataset(test_df, args.img_size, transforms=data_transforms['valid'])
    test_loader  = DataLoader(test_dataset, batch_size=args.valid_bs, 
                            num_workers=4, shuffle=False, pin_memory=False)
    model_paths  = glob(f'{ckpt_dir}/best_epoch*.bin')
    pred_strings, pred_ids, pred_classes, imgs, msks = inference(model_paths, test_loader, args.thr, device, debug, args.backbone, args.num_classes, img_size=args.img_size, config=args)

    # visualization
    if debug:
        for img, msk in zip(imgs[0][:5], msks[0][:5]):
            plt.figure(figsize=(12, 7))
            plt.subplot(1, 3, 1); plt.imshow(img, cmap='bone');
            plt.axis('OFF'); plt.title('image')
            plt.subplot(1, 3, 2); plt.imshow(msk*255); plt.axis('OFF'); plt.title('mask')
            plt.subplot(1, 3, 3); plt.imshow(img, cmap='bone'); plt.imshow(msk*255, alpha=0.4);
            plt.axis('OFF'); plt.title('overlay')
            plt.tight_layout()
            plt.show()

    del imgs, msks
    gc.collect()

    # --- submission ---
    pred_df = pd.DataFrame({
    "id":pred_ids,
    "class":pred_classes,
    "predicted":pred_strings
    })
    if not debug:
        sub_df = pd.read_csv('datasets/test_datasts/sample_submission.csv')
        del sub_df['predicted']
    else:
        sub_df = pd.read_csv('datasets/test_datasets/train.csv')[:1000*3]
        del sub_df['segmentation']
        
    sub_df = sub_df.merge(pred_df, on=['id','class'])
    sub_df.to_csv('submission.csv',index=False)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--no_deterministic", action="store_false")

    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--exp_name", type=str, default="v4")
    parser.add_argument("--comment", type=str, default='unet-efficientnet_b0-160x192-ep=5')
    parser.add_argument("--model_name", type=str, default="UNet")
    parser.add_argument("--model", choices=["Unet"], default="Unet")
    parser.add_argument("--backbone", type=str, default="efficientnet-b0")
    parser.add_argument("--train_bs", type=int, default=32)
    parser.add_argument("--valid_bs", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument('--img_size', nargs="*", type=int, default=[320, 384])
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--scheduler", choices=["CosineAnnealingLR"], default="CosineAnnealingLR")
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--T-max", type=int, default=143)
    parser.add_argument("--T_0", type=int, default=25)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--wd", type=float, default=1e-6)
    parser.add_argument("--n_accumulate", type=float, default=1)
    parser.add_argument("--n_fold", type=int, default=5)
    parser.add_argument('--folds', nargs="*", type=int, default=[0])
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument('--thr', type=float, default=0.40)
    
    return parse_with_config(parser)

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    main(parse_args())
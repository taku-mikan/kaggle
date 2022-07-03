import glob
import os
from typing import List

from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from tqdm import tqdm
import cv2

IMG_SIZE = [320, 384]

def im2np(root: str = "datasets/") -> None:
    """
    Args:
        root (str): datasetのpath
    """
    im_dir = root + "train/" # datasets/train/
    case_dir = sorted(glob.glob(im_dir+"case*"))
    save_path = root + "images/images/" # datasets/images/
    os.makedirs(save_path, exist_ok=True)

    for case in tqdm(case_dir):
        # case : datasets/train/caseXXX
        day_dir = sorted(glob.glob(case+"/case*"))

        for day in day_dir:
            # day : datasets/train/caseXXX/caseXXX_dayYY
            img_dir = glob.glob(day+"/scans/*.png")
            
            for img_path in img_dir:
                image = load_img(img_path)
                

                break
            
            break
        break

    
    # os.makedirs()

def load_img(path: str, size: List[int]=IMG_SIZE):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    shape0 = np.array(img.shape[:2])
    resize = np.array(size)

    if np.any(shape0!=resize):
        diff = resize - shape0
        pad0 = diff[0]
        pad1 = diff[1]
        pady = [pad0//2, pad0//2 + pad0%2]
        padx = [pad1//2, pad1//2 + pad1%2]
        img = np.pad(img, [pady, padx])
        img = img.reshape((*resize))
    
    return img

def load_msk(path: str, size=IMG_SIZE):
    msk = np.load(path)

    shape0 = np.array(msk.shape[:2])
    resize = np.array(size)
    if np.any(shape0!=resize):
        diff = resize - shape0
        pad0 = diff[0]
        pad1 = diff[1]
        pady = [pad0//2, pad0//2 + pad0%2]
        padx = [pad1//2, pad1//2 + pad1%2]
        msk = np.pad(msk, [pady, padx, [0,0]])
        msk = msk.reshape((*resize, 3))
    return msk

def show_img(img, mask=None):
    plt.imshow(img, cmap='bone')
    
    if mask is not None:
        # plt.imshow(np.ma.masked_where(mask!=1, mask), alpha=0.5, cmap='autumn')
        plt.imshow(mask, alpha=0.5)
        handles = [Rectangle((0,0),1,1, color=_c) for _c in [(0.667,0.0,0.0), (0.0,0.667,0.0), (0.0,0.0,0.667)]]
        labels = [ "Large Bowel", "Small Bowel", "Stomach"]
        plt.legend(handles,labels)
    plt.axis('off')
    
def load_imgs(img_paths, size=IMG_SIZE):
    imgs = np.zeros((*size, len(img_paths)), dtype=np.uint16)
    for i, img_path in enumerate(img_paths):
        img = load_img(img_path, size=size)
        imgs[..., i]+=img
    return imgs


def main():
    path = "datasets/train.csv"
    df = pd.read_csv(path)
    print("df : ", df.shape)
    print(df.head())
    
    df['segmentation'] = df.segmentation.fillna('')
    df['rle_len'] = df.segmentation.map(len) # length of each rle mask
    df['mask_path'] = df.mask_path.str.replace('/png/','/np').str.replace('.png','.npy')

    df2 = df.groupby(['id'])['segmentation'].agg(list).to_frame().reset_index() # rle list of each id
    df2 = df2.merge(df.groupby(['id'])['rle_len'].agg(sum).to_frame().reset_index()) # total length of all rles of each id

    df = df.drop(columns=['segmentation', 'class', 'rle_len'])
    df = df.groupby(['id']).head(1).reset_index(drop=True)
    df = df.merge(df2, on=['id'])
    df['empty'] = (df.rle_len==0) # empty masks

    print(df.head())

if __name__ == "__main__":
    main()
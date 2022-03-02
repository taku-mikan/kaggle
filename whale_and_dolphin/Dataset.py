"""
データセット用のファイル
"""
import os

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image

TRAIN_DIR = "train_images/"
TEST_DIR = "test_images/"


class WhaleDataset(Dataset):
    """
    whale_and_dolphinコンペ用データセット
    """
    def __init__(self, df, image_dir="./data", train=True):
        """
        コンストラクタ
        """
        super(WhaleDataset, self).__init__()
        self.df = df
        self.images = self.df["image"]
        self.train = train
        self.transforms = transforms.Compose([
            transforms.Resize((671, 804)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # train-testデータのダウンロード
        if train:
            self.image_path = image_dir + TRAIN_DIR
        else:
            self.image_path = image_dir + TEST_DIR
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_path, self.images.iloc[idx])
        image = read_image(image_path)
        image = self.transforms(image)
        label = self.df["label"].iloc[idx]
        return image, label


if __name__ == "__main__":
    df_train = pd.read_csv("./data/train.csv")
    df_train['label'] = df_train.groupby('individual_id').ngroup()
    print("df_train : ", df_train.shape)
    train_dataset = WhaleDataset(df_train, train=True)
    print("train_dataset : ", len(train_dataset))



        
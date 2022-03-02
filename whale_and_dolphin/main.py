import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from Dataset import WhaleDataset
from model import MODEL_CNN

def train(model, optimizer, train_dataloader, device, num_epochs):
    model.train()
    train_loss = []
    print("-----Training------")
    start_time = time.time()
    for epoch in range(num_epochs):
        train_loss.append(0)
        num_batches = 0
        for img_batch, _ in tqdm(train_dataloader):
            img_batch = img_batch.to(device)


def main():
    # 各種設定
    batch_size = 16
    learning_rate = 0.01
    num_epochs = 100
    valid_proportion = 0.1

    # dataのロード
    df_train = pd.read_csv("./data/train.csv")
    df_sample = pd.read_csv("./sample_submission.csv")
    df_train['label'] = df_train.groupby('individual_id').ngroup()

    valid_df = df_train.sample(frac=valid_proportion, replace=False, random_state=1).copy()
    train_df = df_train[~df_train['image'].isin(valid_df['image'])].copy()

    train_dataset = WhaleDataset(df=train_df, train=True)
    valid_dataset = WhaleDataset(valid_df, train=True)
    dataset_dict = {"train" : train_dataset, "val" : valid_dataset}
    


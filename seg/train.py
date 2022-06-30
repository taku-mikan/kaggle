import datetime
import wandb
import argparse

import torch

from metric.base import Metric
from utils.utils import fix_seed

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






def main(args: argparse.Namespace):
    # time
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H%M%S")

    # seedの固定
    fix_seed(args.seed, args.no_deterministic)

    # データセットの作成

    # モデルの作成

    # Optimizerの設定

    # schedulerの設定

    # 損失関数 / Metricの設定

    # run_name の設定

    # Early_stoppingの設定

    # wnadbの設定

    # 重み等のload

    # summaryの表示

    # train / test

    # wandbのlogの作成

    # 各種設定

    # 重みの設定
    pass


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--no_deterministic", action="store_false")

    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--exp_name", type=str, default="2.50")
    parser.add_argument("--model", choices=["Unet"], default="Unet")
    parser.add_argument("--backbone", type=str, default="efficientnet-b8")
    parser.add_argument("batch_size", type=int, default=64)
    parser.add_argument("--img_h", type=int, default=160)
    parser.add_argument("--img_w", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--scheduler", choices=["CosineAnnealingLR"], default="CosineAnnealingLR")
    parser.add_argument("min_lr", type=float, default=1e-6)
    parser.add_argument("T-max", type=int, default=143)
    parser.add_argument("--T_0", 25)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--wd", type=float, default=1e-6)
    parser.add_argument("--n_accumulate", type=float, default=1)
    parser.add_argument("--n_fold", type=int, default=5)
    parser.add_argument("num_classes", type=int, default=3)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    main(parse_args())
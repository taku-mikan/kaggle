import csv
import os

from PIL import Image
from torch.utils.data import Dataset

class UWMDataset(Dataset):
    """
    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, root, image_set:str = "train", transform=None):
        super().__init__()

        self.root = root
        self.image_set = image_set
        self.transform = transform

        self.dir = ""

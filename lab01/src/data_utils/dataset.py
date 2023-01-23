import os
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image


class CarsDataset(Dataset):
    def __init__(self,
                 img_path,
                 ann_path,
                 transforms=None,
                 idxs=None
                 ):
        self.img_path = img_path
        self.ann = pd.read_csv(ann_path)
        if idxs is not None:
            self.ann = self.ann.iloc[idxs]
        self.transforms = transforms

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        c, fname = self.ann.iloc[index]
        img = Image.open(os.path.join(self.img_path, fname))
        if self.transforms is not None:
            img = self.transforms(img)

        return img, c

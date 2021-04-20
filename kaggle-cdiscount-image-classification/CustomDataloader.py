# CustomDataLoader class
import torch
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt


def show_custom_image(df, idx):
    plt.imshow(df[df['prod_id'].values == idx]['img_arr'].values[0])


# show_custom_image(df, 0)


class CustomDataLoader(Dataset):
    def __init__(self, df, transform=None, train=True):
        self.df = df
        self.transform = transform
        self.train = train

        self.ids = df['prod_id'].count()

    def __len__(self):
        return self.ids

    def __getitem__(self, idx):
        # df[df['prod_id'] == idx]
        img = df[df['prod_id'].values == idx]['img_arr'].values[0]
        label = df[df['prod_id'].values == idx]['cat_id'].values[0]

        img = np.asarray(img)

        if self.transform:
            img = self.transform(img)

        return img.astype('float32'), label

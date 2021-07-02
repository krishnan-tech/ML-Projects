from io import BytesIO

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from io import BytesIO
import bson
import pandas as pd
import albumentations
import torch
import torchvision

train_example_bson = './train_example.bson'


def get_single_item(idx):
    data_bson = bson.decode_file_iter(open(train_example_bson, 'rb'))
    prod_id = []
    cat_id = []
    img_arr = []
    index = 0

    for c, d in enumerate(data_bson):
        product_id = d['_id']
        category_id = d['category_id']  # This won't be in Test data
    #     prod_to_category[product_id] = category_id
        for e, pic in enumerate(d['imgs']):
            # array of image
            # picture = imread(io.BytesIO(pic['picture']))

            # bytes of image
            picture = pic['picture']

            if idx == index:
                prod_id.append(product_id)
                cat_id.append(category_id)
                img_arr.append(picture)

            index += 1

    return (prod_id[0], cat_id[0], img_arr[0])


class CustomDataLoader(Dataset):
    def __init__(self, df, transform=None, train=True, resize=None):
        self.df = df
        self.transform = transform
        self.train = train

        self.ids = self.df.iloc[:, 0].values

        self.resize = resize
        self.aug = albumentations.Compose([
            torchvision.transforms.ToTensor(),
            # albumentations.Normalize()
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # print(idx)
        # df[df['prod_id'] == idx]
        # print(self.df[self.df.iloc[:, 0].values == idx]['cat_id'].values[0])
        # img = self.df[self.df.iloc[:, 0].values == idx]['img_arr'].values[0]
        # label = self.df[self.df.iloc[:, 0].values == idx]['cat_id'].values[0]
        # arr = [(val['imgs'][0]['picture'], val['category_id'])
        #        for i, val in enumerate(data_bson) if i == idx][0]

        # img, label = arr[0], arr[1]

        (prod_id, targets, img) = get_single_item(idx)

        image = Image.open(BytesIO(img)).convert("RGB")

        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )

        image = np.array(image)
        # augmented = self.aug(image=image)
        # image = augmented['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        print({
            torch.tensor(image, dtype=torch.float),
            torch.tensor(targets, dtype=torch.long),
        })
        return torch.tensor(image, dtype=torch.float), torch.tensor(targets, dtype=torch.long)


# df = pd.read_csv('./dataframe.csv')
# cds = CustomDataLoader(df)

# print(cds.__len__())
# print(cds.__getitem__(0))

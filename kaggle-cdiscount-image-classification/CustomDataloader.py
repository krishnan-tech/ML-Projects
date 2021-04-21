# CustomDataLoader class
from matplotlib.image import imread
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import io
import bson
from skimage.io import imread
from PIL import Image
from io import BytesIO

import torch
from torch.utils.data import Dataset
from torchvision import transforms

train_example_bson = './train_example.bson'
category_names = './category_names.csv'

data_bson = bson.decode_file_iter(open(train_example_bson, 'rb'))

data_transforms = {
    'train': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def make_df():
    prod_id = []
    cat_id = []
    img_arr = []

    for c, d in enumerate(data_bson):
        product_id = d['_id']
        category_id = d['category_id']  # This won't be in Test data
    #     prod_to_category[product_id] = category_id
        for e, pic in enumerate(d['imgs']):
            picture = imread(io.BytesIO(pic['picture']))

            prod_id.append(product_id)
            cat_id.append(category_id)
            img_arr.append(picture)

    df = pd.DataFrame(list(zip(img_arr, prod_id, cat_id)),
                      columns=['img_arr', 'prod_id', 'cat_id'])
    df.to_csv('./dataframe.csv')
    return df


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

    print(idx, index, prod_id, cat_id)

    return (prod_id[0], cat_id[0], img_arr[0])


def show_custom_image(df, idx):
    plt.imshow(df[df['prod_id'].values == idx]['img_arr'].values[0])


# show_custom_image(df, 0)


class CustomDataLoader(Dataset):
    def __init__(self, df, transform=None, train=True):
        self.df = df
        self.transform = transform
        self.train = train

        self.ids = self.df.iloc[:, 0].values

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        print(idx)
        # df[df['prod_id'] == idx]
        # print(self.df[self.df.iloc[:, 0].values == idx]['cat_id'].values[0])
        # img = self.df[self.df.iloc[:, 0].values == idx]['img_arr'].values[0]
        # label = self.df[self.df.iloc[:, 0].values == idx]['cat_id'].values[0]
        # arr = [(val['imgs'][0]['picture'], val['category_id'])
        #        for i, val in enumerate(data_bson) if i == idx][0]

        # img, label = arr[0], arr[1]
        (prod_id, label, img) = get_single_item(idx)

        img = Image.open(BytesIO(img))
        # print(label)

        # img = np.asarray(img)
        # img = torch.tensor(img)

        # label = torch.from_numpy(label)

        if self.transform:
            img = self.transform(img)

        return img, label


def main():
    # make_df()
    df = pd.read_csv('./dataframe.csv')
    train_loader = CustomDataLoader(
        df, transform=data_transforms['train'], train=True)
    dataloader = DataLoader(train_loader, batch_size=16, shuffle=True)

    print(next(iter(dataloader)))


main()

# df = pd.read_csv('./dataframe.csv')
# print(df)
# print(df.iloc[:, 0].values)
# arr = df[df['prod_id'].values == idx]['img_arr'].values[0]
# print(arr)
# df = make_df()
# cdl = CustomDataLoader(df)
# print(cdl.__len__())
# print(cdl.__getitem__(0))
# idx = 89

# arr = [val['imgs'][0]['picture']
#    for i, val in enumerate(data_bson) if i == idx]
# arr = [(val['imgs'][0]['picture'], val['category_id'])
#        for i, val in enumerate(data_bson) if i == idx][0]

# img, label = arr[0], arr[1]
# img = (Image.open(BytesIO(img)))
# img.save('greyscale.png')


# print(get_single_item(79))
# img = df[df.iloc[:, 0].values == idx]['img_arr'].values[0]
# img = np.asarray(img)
# print(type(img))
# image = np.array(Image.open(io.BytesIO(img)))
# print(image)

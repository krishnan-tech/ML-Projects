# CustomDataLoader class
import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import bson
from skimage.io import imread
from PIL import Image
from io import BytesIO

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms

from dataloader import CustomDataLoader

# CustomDataLoader = CustomDataLoader()

train_example_bson = './train_example.bson'
category_names = './category_names.csv'

save_model = "./train.torch"

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
            picture = imread(BytesIO(pic['picture']))

            prod_id.append(product_id)
            cat_id.append(category_id)
            img_arr.append(picture)

    df = pd.DataFrame(list(zip(img_arr, prod_id, cat_id)),
                      columns=['img_arr', 'prod_id', 'cat_id'])
    df.to_csv('./dataframe.csv')
    return df


def show_custom_image(df, idx):
    plt.imshow(df[df['prod_id'].values == idx]['img_arr'].values[0])


def make_model():
    model = torchvision.models.resnet34(num_classes=5270)
    return model


def main():
    make_df()
    df = pd.read_csv('./dataframe.csv')
    train_loader = CustomDataLoader(
        df, transform=data_transforms['train'], train=True)
    dataloader = DataLoader(train_loader, batch_size=16, shuffle=True)

    model = make_model()
    # print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    losses = []
    i = 0
    try:
        for i, data in enumerate(dataloader, 1):
            image, label = data
            # print(image, label)

            image, label = Variable(image), Variable(label)

            pred = model(image)
            print(pred)
            print(label)
            print(pred.shape)
            print(label.shape)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 25 == 0:
                print("Current loss: {:.4f}".format(loss_float))

    except KeyboardInterrupt:
        logging.info("Interrupted prematurely at iteration {}".format(i))

    logging.info("Saving state dict...")
    with open(save_model, "wb") as fh:
        torch.save({'state_dict': model.state_dict()}, fh)

    plt.plot(losses)
    plt.show()


main()

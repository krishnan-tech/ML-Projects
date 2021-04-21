import os
import glob

import torch

import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn import model_selection

import config
# import dataset


def run_training():
    image_files = glob.glob(os.path.join(config.DATA_DIR, "*.png"))
    # print(image_files)

    # "../../..\\abcde.png" -> "abced"
    targets_orig = [x.split("\\")[-1][:-4] for x in image_files]

    # "abcde" -> [a, b, c, d, e ]
    targets = [[c for c in x] for x in targets_orig]

    target_flat = [c for clist in targets for c in clist]

    lbl_enc = preprocessing.LabelEncoder()
    lbl_enc.fit(target_flat)

    targets_enc = [lbl_enc.transform(x) for x in targets]
    targets_enc = np.array(targets_enc) + 1

    print(targets_enc)


run_training()

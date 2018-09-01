import tensorflow as tf
# from tf.contrib.keras.utils.data_utils import Sequence
from tensorflow.contrib.keras.python.keras.utils.np_utils import to_categorical
import cv2 as cv
import numpy as np
import platform
import math
import os
import os.path
import platform


"""This module provides classes to generate data as keras.utils.Sequence
(noted as Seqence later) for training or validation from CASIA-WebFace dataset

Class:

WebFaceReader -- read CASIA-WebFace dataset in data_path and you can call the
  attribute train_seq to get the keras.utils.Sequence for training or val_seq
  to get the keras.utils.Sequence for validation
WebFaceSequence -- a class to generate data batch as type Sequence
"""


class WebFaceReader():
    def __init__(self, data_path,
                val_split,
                batch_size,
                image_size,
                classes):
        self.data_path = data_path
        self.val_split = val_split
        self.classes=classes
        self.x = []
        self.y = []
        self.preprocessing()
        self.total_num = len(self.x)
        self.split_index = math.ceil(self.total_num * self.val_split)

        self.train_seq = WebFaceSequence(x_set=self.x[self.split_index:],
                                         y_set=self.y[self.split_index:],
                                         batch_size=batch_size,
                                         image_size=image_size,
                                         classes=classes)
        self.val_seq = WebFaceSequence(x_set=self.x[:self.split_index],
                                       y_set=self.y[:self.split_index],
                                       batch_size=batch_size,
                                       image_size=image_size,
                                       classes=classes)

    def preprocessing(self,
                      pre_shuffle=True,
                      shuffle_seed=1024):
        """assign self.x and self.y according to the self.data_path"""
        # rename the directory in WebFace in order
        #if platform.system() == "Windows":
        #    for idx, filefoler in enumerate(os.listdir(self.data_path)):
        #        if int(filefoler) == 0:
        #            break
        #        os.rename(os.path.join(self.data_path, filefoler),
        #                  os.path.join(self.data_path, str(idx).zfill(7)))
        # add element to self.x and self.y
        #for filefolder in os.listdir(self.data_path):
        for filefolder in range(self.classes):
            for file in os.listdir(os.path.join(self.data_path, str(filefolder).zfill(7))):
                self.x.append(os.path.join(self.data_path, str(filefolder).zfill(7), file))
                self.y.append(int(filefolder))

        if pre_shuffle == True:
            self.x, self.y = self.shuffle_list(self.x, self.y, shuffle_seed)

    def shuffle_list(self, data, label, shuffle_seed=1024):
        idx = np.arange(len(label))
        np.random.seed(shuffle_seed)
        np.random.shuffle(idx)
        return [data[i] for i in idx], [label[i] for i in idx]


class WebFaceSequence():

    """A class to generate train batch from WebFace

    you may want to assign these value for your program:
        x_set: a list stored the image path
        y_set: a list stored the label of each image in x_set in order
        data_path: the absolute path of WebFace database
        batch_size: assign a proper number according to your GPU memeory
        img_size: the original size of images in WebFace is 250, you may want to scale the image
    """

    def __init__(self, x_set, y_set, batch_size, image_size, classes):
        # self.x will store the path of all images
        # self.y will store the label according to the order of directories
        # super(WebFaceSequence, self).__init__()
        self.x = x_set
        self.y = y_set
        self.classes=classes
        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        return len(self.x) // self.batch_size

    def __getitem__(self, idx):

        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
                        cv.resize(cv.imread(file_name),
                                 (self.image_size, self.image_size))
                        for file_name in batch_x
                        ]), to_categorical(np.array(batch_y), self.classes)

if __name__ == "__main__":
    if platform.system() == "Windows":
        data_path = "D:\\Database\\CASIA-WebFace_croped_20"
    elif platform.system() == "Linux":
        data_path = "/home/x000000/LK/Database/CASIA-WebFace_croped_20"
    classes = len(os.listdir(data_path))
    data_reader = WebFaceReader(data_path, val_split=0.1, batch_size=32, classes=classes)
    print(len(data_reader.train_seq))
    print(data_reader.train_seq[0][1].shape)
    # x, y = data_reader.train_seq.__getitem__(idx=0)
    # if x == [] or y is []:
    #     print("x or y is empty.")
    # print(x)
    # print(y)

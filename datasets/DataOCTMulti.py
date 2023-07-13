'''
Description: 
Author: Ming Liu (lauadam0730@gmail.com)
Date: 2021-05-07 19:59:24
'''
import os
import sys
import random
import numpy as np
import torch
import csv
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import datasets.utils_data as utils

class DatasetOCTMultiCLA(Dataset):
    def __init__(self, data_dir, Flags, mode='train'):
        super().__init__()
        self.mode = mode
        self.Flags = Flags
        self.inputsize = [Flags.img_size, Flags.img_size]

        self.image_dir = os.path.join(data_dir, '{}data'.format(mode))
        self.label_dir = os.path.join(data_dir, '{}labels.csv'.format(mode))
        self.listfile = os.path.join(data_dir, '{}files.txt'.format(mode))

        self.filenames = utils.txt2list(self.listfile)
        print("Num of {} images:  {}".format(mode, len(self.filenames)))

        self.to_tensor = transforms.ToTensor()

        # -------标签文件-------
        with open(self.label_dir, "r") as f:
            reader = csv.reader(f)
            self.label_file = list(reader)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        batch_name = self.filenames[index]  # only 1 image name return
        image_arr1, image_arr2 = self.get_img(batch_name, mode=self.mode)
        x_data1 = self.to_tensor(image_arr1.copy()).float()
        x_data2 = self.to_tensor(image_arr2.copy()).float()
        # if 'train' in self.mode:
        #     x_data = utils.random_erase(x_data) # random erase

        for row in self.label_file:
            if batch_name in row:
                label = torch.tensor(list(map(int, row[1:12]))).float()
                break

        return x_data1, x_data2, label, batch_name

    # load images and labels depend on filenames
    def get_img(self, file_name, mode='train'):
        if 'train' in self.mode:
            image_file_dir = os.path.join(self.image_dir, file_name)
            image_files = utils.all_files_under(image_file_dir, append_path=False)
            if len(image_files) < 2:
                print(image_file_dir)
            images_file_1 = os.path.join(image_file_dir, image_files[0])
            images_file_2 = os.path.join(image_file_dir, image_files[1])
            image_im1 = Image.open(images_file_1).resize(self.inputsize)
            image_im1 = utils.random_perturbation(image_im1)
            image_im1 = utils.random_geometric3(image_im1)
            image_im2 = Image.open(images_file_2).resize(self.inputsize)
            image_im2 = utils.random_perturbation(image_im2)
            image_im2 = utils.random_geometric3(image_im2)

            image_arr1 = np.array(image_im1, dtype=np.float32) / 255.0
            image_arr2 = np.array(image_im2, dtype=np.float32) / 255.0

            return image_arr1, image_arr2

        elif 'val' in self.mode:
            image_file_dir = os.path.join(self.image_dir, file_name)
            image_files = utils.all_files_under(image_file_dir, append_path=False)
            if len(image_files) < 2:
                print(image_file_dir)
            images_file_1 = os.path.join(image_file_dir, image_files[0])
            images_file_2 = os.path.join(image_file_dir, image_files[1])
            image_im1 = Image.open(images_file_1).resize(self.inputsize)
            image_im2 = Image.open(images_file_2).resize(self.inputsize)

            image_arr1 = np.array(image_im1, dtype=np.float32) / 255.0
            image_arr2 = np.array(image_im2, dtype=np.float32) / 255.0

            return image_arr1, image_arr2

        elif 'test' in self.mode:
            image_file_dir = os.path.join(self.image_dir, file_name)
            image_files = utils.all_files_under(image_file_dir, append_path=False)
            if len(image_files) < 2:
                print(image_file_dir)
            images_file_1 = os.path.join(image_file_dir, image_files[0])
            images_file_2 = os.path.join(image_file_dir, image_files[1])
            image_im1 = Image.open(images_file_1).resize(self.inputsize)
            image_im2 = Image.open(images_file_2).resize(self.inputsize)

            image_arr1 = np.array(image_im1, dtype=np.float32) / 255.0
            image_arr2 = np.array(image_im2, dtype=np.float32) / 255.0

            return image_arr1, image_arr2

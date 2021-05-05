import pandas as pd
import os
import cv2
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def get_all_data(self):
        images = []
        labels = []
        for i in range(0, self.__len__()):
            images.append(self.__getitem__(i)["image"])
            labels.append(self.__getitem__(i)["label"])
        images = torch.tensor(images).type(torch.FloatTensor)
        images = images.view(images.shape[0], -1)
        labels = torch.tensor(labels)
        ctx = {"images": images, "labels": labels}
        return ctx

    def show_image(self, idx, title):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        img = mpimg.imread(img_path)
        plt.title(title)
        plt.imshow(img)
        plt.show()

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv2.bitwise_not(cv2.imread(img_path, 0))
        label = self.img_labels.iloc[idx, 1]
        sample = {"image": image, "label": label}
        return sample

    def __str__(self):
        return 'ImageDataset: \nimg_labels:\n' + str(self.img_labels) + '\nimg_dir:\n' + str(self.img_dir)

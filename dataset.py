from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as F
import skimage.io as io
from PIL import Image
import glob
import os


class NYUv2(Dataset):
    def __init__(self, image_folder_in: str, image_folder_out: str, transform=None):
        self.image_folder_in = image_folder_in
        self.image_folder_out = image_folder_out
        self.transform = transform

        self.color_list = []
        self.normal_list = []

        print("Loading color images...")
        self.color_list = [f for f in os.listdir(self.image_folder_in)]
        self.color_list = sorted(self.color_list)
        print(f"Found {len(self.color_list)} images.")

        print("Loading normal images...")
        self.normal_list = [f for f in os.listdir(self.image_folder_out)]
        self.normal_list = sorted(self.normal_list)
        print(f"Found {len(self.normal_list)} images.")
        # print(list(zip(self.color_list,self.normal_list)))

    def __len__(self):
        return len(self.color_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(
            self.image_folder_in, self.color_list[index]))
        nrm = Image.open(os.path.join(
            self.image_folder_out, self.normal_list[index]))

        if self.transform:
            img = self.transform(img)
            nrm = self.transform(nrm)

        return img, nrm

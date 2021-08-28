import torch
import h5py
import io
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.dataloader import DataLoader


from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
import os


def path_to_pil_img(path, h5py_file_path):
    with h5py.File(h5py_file_path, 'r') as h5_file:
        binary_data = h5_file[path]["_"][...]
    return Image.open(io.BytesIO(binary_data)).convert("RGB")


class GoogleLandmarkDataset(Dataset):
    def __init__(self,
                 image_list,
                 class_ids,
                 resize_shape,
                 transform=None,
                 root_dir=None,
                 h5py_file_path=None):
        super().__init__()

        if h5py_file_path is not None:
            self.h5py_file_path = h5py_file_path
            self.use_h5 = True
        else:
            self.root_dir = root_dir
            self.use_h5 = False

        self.image_list = image_list
        self.class_ids = class_ids
        self.transform = transform
        self.base_transforms = transforms.Compose([
            transforms.Resize(resize_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),  # ImageNet Normalization
        ])

    def __getitem__(self, index):
        if self.use_h5:
            img_path = self.image_list[index] + '.jpg'
            img = path_to_pil_img(img_path, os.path.join(self.h5py_file_path, img_path[:3] + '.h5py'))
        else:
            img_path = str(self.image_list[index])
            img_path = os.path.join(self.root_dir, img_path[0], img_path[1], img_path[2], img_path + '.jpg')
            img = Image.open(img_path).convert("RGB")
        assert img is not None, f'path: {img_path} is invalid'
        if self.transform is not None:
            img = self.transform(img)
        img = self.base_transforms(img)  # Always apply the base_transforms
        label = torch.tensor(self.class_ids[index]).long()
        return img, label

    def __len__(self):
        return len(self.image_list)


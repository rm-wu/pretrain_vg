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


def path_to_pil_img(path, h5_file):
    binary_data = h5_file[path]["_"][...]
    return Image.open(io.BytesIO(binary_data)).convert("RGB")


class GoogleLandmarkDataset(Dataset):
    def __init__(self,
                 image_list,
                 class_ids,
                 resize_shape,
                 transform=None,
                 h5py_file_path=None):
        super().__init__()

        if h5py_file_path is not None:
            self.h5py_file_path = h5py_file_path
            self.h5_dataset = None

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
        if self.h5py_file_path is not None:
            img_path = self.image_list[index] + '.jpg'
            if self.h5_dataset is None:
                self.h5_dataset = h5py.File(self.h5py_file_path, "r")
            img = path_to_pil_img(img_path, self.h5_dataset)
        else:
            img_path = str(self.image_list[index])
            img = Image.open(img_path).convert("RGB")
        assert img is not None, f'path: {img_path} is invalid'
        if self.transform is not None:
            img = self.transform(img)
        img = self.base_transforms(img)  # Always apply the base_transforms
        label = torch.tensor(self.class_ids[index]).long()
        return img, label

    def __len__(self):
        return len(self.image_list)


# TODO: TOREMOVE
def prepare_dataloaders(dataset_name, data_path, train_batch_size=32, eval_batch_size=64,
                        test_size=0.0, seed=None, resize_shape=(224, 224), num_workers=0,
                        train_transforms=None, eval_transforms=None):
    if dataset_name not in ['gldv2', 'places']:
        raise ValueError(f"The dataset {dataset_name} is not implemented yet")
    if dataset_name == 'gldv2':
        # Read the pkl file that contains the paths and labels of each image and splits the training data into train and
        # validation sets
        df = pd.read_pickle(data_path)
        train_split, valid_split = train_test_split(df, test_size=test_size, random_state=seed)

        train_dataset = GoogleLandmarkDataset(
            paths=train_split['path'].values,
            class_ids=train_split['landmark_id'].values,
            resize_shape=resize_shape,
            transform=train_transforms)
        valid_dataset = GoogleLandmarkDataset(
            paths=valid_split['path'].values,
            class_ids=valid_split['landmark_id'].values,
            resize_shape=resize_shape,
            transform=eval_transforms)

        train_dl = DataLoader(dataset=train_dataset,
                              batch_size=train_batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)
        valid_dl = DataLoader(dataset=valid_dataset,
                              batch_size=eval_batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=True)

        num_classes = df['landmark_id'].max() + 1

    elif dataset_name == 'places':
        train_dir = os.path.join(data_path, 'train')
        valid_dir = os.path.join(data_path, 'val')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # Here the same augmentation techniques used in the official repository are employed
        # train_dataset = datasets.ImageFolder(root=train_dir,
        train_dataset = datasets.Places365(root=train_dir,
                                           split="train-standard",
                                           transform=transforms.Compose([
                                               transforms.RandomSizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               normalize
                                           ]))
        #valid_dataset = datasets.ImageFolder(root=valid_dir,
        valid_dataset = datasets.Places365(root=valid_dir,
                                           split="val",
                                           transform=transforms.Compose([
                                               transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               normalize]))

        train_dl = DataLoader(dataset=train_dataset,
                              batch_size=train_batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)
        valid_dl = DataLoader(dataset=valid_dataset,
                              batch_size=eval_batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=True)
        num_classes = 365  # Places365 contains 365 classes

    return train_dl, valid_dl, num_classes

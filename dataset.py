import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader


from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd

class LandmarkDataset(Dataset):
    def __init__(self,
                 paths,
                 class_ids,
                 transform=None):
        super().__init__()
        self.paths = paths
        self.class_ids = class_ids
        self.transform = transform
        self.base_transforms = transforms.Compose([
            transforms.Resize((224, 224)),  # ((640, 480)),  # TODO: this resize does not take into account different shapes of GLDv2 images
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),  # ImageNet Normalization
        ])

    def __getitem__(self, index):
        img_path = str(self.paths[index])
        img = Image.open(img_path).convert("RGB")
        assert img is not None, f'path: {img_path} is invalid'
        if self.transform is not None:  # TODO: controllare che fare l'augmentation in questo modo vada bene
            img = self.transform(img)
        img = self.base_transforms(img)  # Always apply the base_transforms
        label = torch.tensor(self.class_ids[index]).long()
        return img, label

    def __len__(self):
        return len(self.paths)


def prepare_dataloaders(dataset_name, df_path, train_batch_size=32, eval_batch_size=64,
                        test_size=0.0, seed=None, num_workers=0,
                        train_transforms=None, eval_transforms=None):
    # TODO: add Places
    if dataset_name not in ['gldv2', 'places']:
        raise ValueError(f"The dataset {dataset_name} is not implemented yet")
    if dataset_name == 'gldv2':
        # Read the pkl file that contains the paths and labels of each image and splits the training data into train and
        # validation sets
        df = pd.read_pickle(df_path)
        # TODO: stratify to get the same data distribution ?
        train_split, valid_split = train_test_split(df, test_size=test_size, random_state=seed)

        train_dataset = LandmarkDataset(
            paths=train_split['path'].values,
            class_ids=train_split['landmark_id'].values,
            transform=train_transforms)
        valid_dataset = LandmarkDataset(
            paths=valid_split['path'].values,
            class_ids=valid_split['landmark_id'].values,
            transform=eval_transforms)

        train_dl = DataLoader(dataset=train_dataset,
                              batch_size=train_batch_size,
                              shuffle=True,
                              num_workers=num_workers)

        valid_dl = DataLoader(dataset=valid_dataset,
                              batch_size=eval_batch_size,
                              shuffle=False,
                              num_workers=num_workers)
        num_classes = df['landmark_id'].max() + 1   # TODO: to check

    elif dataset_name == 'places':
        raise NotImplementedError("Places to implement")

    return train_dl, valid_dl, num_classes

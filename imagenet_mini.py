import os
import pandas as pd
# from torchvision.io import read_image
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

class ImageNetMini(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.train = train
        self.root=root+'/imnet100_full/'
        self.img_dir = 'train'
        self.annotations_file = 'trainLabels.csv'
        if not self.train:
            self.annotations_file = 'valLabels.csv'
            self.img_dir = 'val'
        self.img_labels = pd.read_csv(self.root+self.annotations_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root+self.img_dir, self.img_labels.iloc[idx, 0])
        image = default_loader(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
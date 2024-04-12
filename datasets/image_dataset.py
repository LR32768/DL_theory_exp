import os
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn

class Cifar10(Dataset):
    def __init__(self, datapath='./data',
                 trunc_len=None,
                 return_label=True,
                 random_flip=False,
                 random_crop=False,
                 class_filter=None,
                 is_train=True,
                 transform=None
                 ):

        self.data = []
        self.targets = []
        self.return_label = return_label
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip() if random_flip else nn.Identity(),
            transforms.RandomCrop(32, padding=4) if random_crop else nn.Identity(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]) if transform is None else transform

        # load torchvision cifar10 dataset
        raw_dataset = torchvision.datasets.CIFAR10(root=datapath, train=is_train, download=True, transform=self.transform)
        cnt = 0
        print("Loading CIFAR10 dataset...")
        for data, target in raw_dataset:
            if trunc_len is not None and cnt >= trunc_len:
                break
            if class_filter is None or target in class_filter:
                self.data.append(data)
                self.targets.append(target)
                cnt += 1
        print("Done!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.return_label:
            return self.data[idx], self.targets[idx]
        else:
            return self.data[idx]

    def get_data(self):
        """ return images as a tensor X and labels as a tensor y """
        return torch.stack(self.data).reshape(len(self.data), -1), torch.tensor(self.targets)

class ImageDataset(Dataset):
    def __init__(self, datapath,
                 image_size=224,
                 trunc_len=None,
                 return_label=True,
                 random_flip=False,
                 random_crop=False,
                 class_filter=None,
                 is_train=True,
                 transform=None):

        self.data = []
        self.targets = []
        self.return_label = return_label
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip() if random_flip else nn.Identity(),
            transforms.RandomCrop(image_size, padding=4) if random_crop else nn.Identity(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]) if transform is None else transform

        print("Loading ImageNet dataset...")
        raw_dataset = torchvision.datasets.ImageFolder(root=datapath, transform=self.transform, train=is_train)
        cnt = 0
        for data, target in raw_dataset:
            if trunc_len is not None and cnt >= trunc_len:
                break
            if class_filter is None or target in class_filter:
                self.data.append(data)
                self.targets.append(target)
                cnt += 1
        print("Done!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.return_label:
            return self.data[idx], self.targets[idx]
        else:
            return self.data[idx]

    def get_data(self):
        """ return images as a tensor X and labels as a tensor y """
        return torch.stack(self.data).reshape(len(self.data), -1), torch.tensor(self.targets)

class GeneralImageDataset(Dataset):
    def __init__(self, datapath,
                 image_size=224,
                 trunc_len=None,
                 return_label=True,
                 random_flip=False,
                 random_crop=False,
                 transform=None,
                 ):

        self.data = []
        self.targets = []
        self.return_label = return_label
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip() if random_flip else nn.Identity(),
            transforms.RandomCrop(image_size, padding=4) if random_crop else nn.Identity(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]) if transform is None else transform

        cnt = 0
        print(f"Loading general image dataset from {datapath}...")
        assert os.path.exists(datapath), f"Path {datapath} does not exist!"
        for root, dirs, files in os.walk(datapath):
            for file in files:
                if trunc_len is not None and cnt >= trunc_len:
                    break
                img = Image.open(os.path.join(root, file))
                img = self.transform(img)
                self.data.append(img)
                cnt += 1

        print("Done!")
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_data(self):
        """ return images as a tensor X and labels as a tensor y """
        return torch.stack(self.data).reshape(len(self.data), -1), torch.tensor(self.targets)

if __name__ == "__main__":
    # dataset = Cifar10()
    # print(len(dataset))
    # print(dataset[0])
    # print(dataset[0][0].shape)
    # print(dataset[0][1])
    # print("Done!")
    dataset = GeneralImageDataset(datapath='/cluster/home1/lurui/ffhq128',
                                  trunc_len=1000, image_size=64, random_flip=True)
    print(len(dataset))
    print(dataset[3].shape)
    print("Done!")

    print(dataset.get_data()[0].shape)
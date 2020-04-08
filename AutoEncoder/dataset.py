import numpy as np
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


def get_cifar_dataset():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomAffine(20),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    return trainset, testset


def get_pattern_dataset():
    transform_train = transforms.Compose([
        transforms.Pad(32),
        transforms.CenterCrop((32, 32)),

        transforms.RandomAffine(20),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Pad(32),
        transforms.CenterCrop((32, 32)),
        transforms.ToTensor(),
    ])

    trainset = ImageFolder('./data/pattern_imgs/train_data', transform=transform_train)
    testset = ImageFolder('./data/pattern_imgs/test_data', transform=transform_test)

    return trainset, testset


class CombineDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.offsets = np.cumsum(self.lengths)
        self.length = np.sum(self.lengths)

    def __getitem__(self, index):
        for i, offset in enumerate(self.offsets):
            if index < offset:
                if i > 0:
                    index -= self.offsets[i-1]
                return self.datasets[i][index]
        raise IndexError(f'{index} exceeds {self.length}')

    def __len__(self):
        return self.length
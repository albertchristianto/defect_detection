from cmath import e
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from utils import *
from torchvision import transforms

class ImageClassificationDataset(Dataset):
    def __init__(self, data, transform, train_mode=True):
        self.train_mode = train_mode
        self.data = data
        self.numSample = len(data)
        self.transformation = transform
        self.networks_w = self.transformation['input_width']
        self.networks_h = self.transformation['input_height']
        self.define_means_stds()
        self.torch_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(self.transformation['random_horizontal_flips']),
            transforms.RandomRotation(self.transformation['random_rotation']),
            transforms.Resize((self.networks_h, self.networks_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean = self.means, std = self.stds)
        ])

    def define_means_stds(self):
        if self.transformation['means_stds'] == "vgg_means_stds":
            self.means = vgg_means
            self.stds = vgg_stds
        elif self.transformation['means_stds'] == "mt_means_stds":
            self.means = magnetic_means
            self.stds = magnetic_stds

    def __getitem__(self, index):
        imgPath = self.data[index][0]
        label = int(self.data[index][1])
        try:
            if (self.train_mode):
                img = Image.open(imgPath)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img = self.torch_transform(img)
            else:
                img = cv2.imread(imgPath)
                img = cv2.resize(img, (self.networks_w, self.networks_h), interpolation=cv2.INTER_CUBIC)
                img = vgg_preprocess(img, self.means, self.stds)
                img = torch.from_numpy(img)
        except Exception as e:
            print('image path in error: {}, {}'.format(imgPath, e))

        return img, label

    def __len__(self):
        return self.numSample

def getLoader(dataset_root, transform, bsize = 16):
    trainData, valData, weight_samples, class_name = loadtxtfiles(dataset_root)

    trainDataset = ImageClassificationDataset(data=trainData, transform = transform)
    sampler = WeightedRandomSampler(weight_samples, len(weight_samples))
    trainDataloader = DataLoader(trainDataset, batch_size=bsize, sampler=sampler, num_workers=1)

    valDataset = ImageClassificationDataset(data=valData, transform = transform, train_mode=False)
    valDataloader = DataLoader(valDataset, batch_size = bsize, num_workers = 1, shuffle = True)

    return trainDataloader, valDataloader, class_name

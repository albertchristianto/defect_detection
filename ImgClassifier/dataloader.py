import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from utils import *
from torchvision import transforms
from loguru import logger

class ImageClassificationDataset(Dataset):
    def __init__(self, data, transform, train_mode=True):
        self.train_mode = train_mode
        self.data = data
        self.numSample = len(data)
        self.transformation = transform
        self.input_size = self.transformation['input_size']
        self.define_means_stds()
        self.torch_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(self.transformation['random_horizontal_flips']),
            transforms.RandomRotation(self.transformation['random_rotation']),
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean = self.means, std = self.stds)
        ])

    def define_means_stds(self):
        if self.transformation['use_vgg']:
            self.means = vgg_means
            self.stds = vgg_stds
            return
        self.means = self.transformation['means']
        self.stds = self.transformation['stds']

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
                img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC)
                img = vgg_preprocess(img, self.means, self.stds)
                img = torch.from_numpy(img)
        except Exception as e:
            logger.error(f'{imgPath}; {e}')

        return img, label

    def __len__(self):
        return self.numSample

def getLoader(dataset_root, transform, bsize = 16, use_vgg=False):
    trainData, valData, weight_samples = loadtxtfiles(dataset_root, transform['class_name'])
    transform['use_vgg'] = use_vgg
    trainDataset = ImageClassificationDataset(data=trainData, transform=transform)
    sampler = WeightedRandomSampler(weight_samples, len(weight_samples))
    trainDataloader = DataLoader(trainDataset, batch_size=bsize, sampler=sampler, num_workers=1)
    valDataset = ImageClassificationDataset(data=valData, transform=transform, train_mode=False)
    valDataloader = DataLoader(valDataset, batch_size = bsize, num_workers=1, shuffle=True)
    return trainDataloader, valDataloader

if __name__ == '__main__':
    transform = {}
    transform['random_horizontal_flips'] = 0.5
    transform['random_rotation'] = [-180, 180]
    transform['input_size'] = 224
    transform['means'] = None
    transform['stds'] = None
    dataset_root = "E:/Albert_Christianto/Project/defect_detection/dataset/magnetic_tile"
    class_name_path = os.path.join(dataset_root, 'classes_name.txt')
    transform['class_name'] = get_class_name(class_name_path)
    logger.trace('Get the dataloader')
    trainLoader, valDataloader = getLoader(dataset_root, transform, bsize=16, use_vgg=True)
    logger.info(f'train dataset len: {len(trainLoader.dataset)}')
    logger.info(f'validate dataset len: {len(valDataloader.dataset)}')
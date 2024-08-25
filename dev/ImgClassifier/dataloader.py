import os
import cv2
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from PIL import ImageEnhance
from torchvision.transforms import RandomAffine, RandomPerspective

from torchvision import transforms
import torchvision.transforms.functional as F

from loguru import logger

vgg_means = np.array([0.485, 0.456, 0.406], dtype=np.float32)
vgg_stds = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def vgg_preprocess(image, means=vgg_means, stds=vgg_stds):
    image = image.astype(np.float32) / 255.0
    preprocessed_img = image.copy()[:, :, ::-1]# swap bgr to rgb
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = preprocessed_img.transpose((2, 0, 1)).astype(np.float32)
    return preprocessed_img

def get_means_stds(means_stds_path):
    the_file = open(means_stds_path)
    the_file = the_file.readlines()
    means_list = the_file[0].replace('\n','').split(',')[0:3]
    stds_list = the_file[1].replace('\n','').split(',')[0:3]
    means = np.array(means_list, dtype=np.float32)
    stds = np.array(stds_list, dtype=np.float32)
    return means, stds

def get_class_name(class_name_path):
    class_name_file = open(class_name_path)
    class_name = []
    class_name_file = class_name_file.readlines()
    for theFileNow in class_name_file:
        theFileNow = theFileNow.replace('\n','').split(' ')[0]
        class_name.append(theFileNow)
    return class_name

def loadtxtfiles(dataset_root, class_name):
    train_data = []
    val_data = []
    len_data = {}

    for class_idx, each_class in enumerate(class_name):
        len_data[each_class] = 0
        class_folder_datalist_path = dataset_root + '/' + each_class + '/datalist'
        class_datalist_list = os.listdir(class_folder_datalist_path)
        for each_datalist in class_datalist_list:
            each_datalist_path = class_folder_datalist_path + '/' + each_datalist
            theFile = open(each_datalist_path)
            theFile = theFile.readlines()
            for theFileNow in theFile:
                theFileNow = dataset_root + '/' + theFileNow.replace('\n','').split(' ')[0]
                each_data = []
                each_data.append(theFileNow)
                each_data.append(class_idx)
                if (each_datalist.startswith('train')):
                    len_data[each_class] += 1 #compute the how many class inside the training data per class
                    train_data.append(each_data)
                    continue
                if (each_datalist.startswith('val')):
                    val_data.append(each_data)
                    continue

    all_data = 0
    for each_key in len_data:
        all_data = all_data + len_data[each_key]
    weight_value_default = 1.0 / all_data
    weight_value = {}
    for each_key in len_data:
        weight_value[each_key] = weight_value_default / float(len_data[each_key])
    weight_samples = np.zeros(all_data)
    last_idx = 0
    for each_class in class_name:
        weight_samples[last_idx:(last_idx + len_data[each_class])] = weight_value[each_class]
        last_idx += len_data[each_class]

    return train_data, val_data, weight_samples

def write_class_name(classes_name_path, class_name):
    classes_name_Txt = open(classes_name_path, 'w')
    for class_name_now in class_name:
        classes_name_Txt.write('{}\n'.format(class_name_now))
    classes_name_Txt.close()

def write_means_stds(means_stds_path, means, stds):
    mean_stds_Txt = open(means_stds_path, 'w')
    for each_mean in means:
        mean_stds_Txt.write('{},'.format(each_mean))
    mean_stds_Txt.write('\n')
    for each_std in stds:
        mean_stds_Txt.write('{},'.format(each_std))
    mean_stds_Txt.close()

class RandomBrightness(object):
    def __init__(self, brightness=0.2):
        self.brightness = brightness

    def __call__(self, img):
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        return enhancer.enhance(factor)

class MotionBlur(object):
    def __init__(self, kernel_sizes, angle_range=(-45, 45)):
        self.kernel_sizes = kernel_sizes
        self.angle_range = angle_range

    def __call__(self, img):
        kernel_size = np.random.choice(self.kernel_sizes)
        angle = np.random.uniform(self.angle_range[0], self.angle_range[1])

        # Creating a motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size

        # Rotating the kernel to the specified angle
        center = (kernel_size // 2, kernel_size // 2)
        center = (float(center[0]), float(center[1]))  # Convert to float
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))

        # Convert PIL image to numpy array
        img_np = np.array(img)
        # Apply the kernel using OpenCV
        img_blur = cv2.filter2D(img_np, -1, kernel)

        # Convert numpy array back to PIL image
        img = F.to_pil_image(img_blur)
        return img

class PadToSquare:
    def __init__(self):
        pass

    def __call__(self, image):
        # Get the size of the image
        width, height = image.size

        # Determine the padding for each side to make the image square
        if width > height:
            padding = (0, (width - height) // 2, 0, (width - height + 1) // 2)
        else:
            padding = ((height - width) // 2, 0, (height - width + 1) // 2, 0)
        # if (random.random() > 0.7):
        #     padding_color = tuple(random.randint(0, 255) for _ in range(3))
        # else:
        padding_color = tuple([0, 0, 0])
        # Apply padding
        padded_image = F.pad(image, padding, fill=padding_color)

        return padded_image

def pad_to_square(image):
    height, width = image.shape[:2]

    # Calculate padding for each side to make the image square
    if width > height:
        padding_top = (width - height) // 2
        padding_bottom = width - height - padding_top
        padding_left = 0
        padding_right = 0
    else:
        padding_left = (height - width) // 2
        padding_right = height - width - padding_left
        padding_top = 0
        padding_bottom = 0

    # Generate a random fill color if needed
    fill_color = [0, 0, 0]  # Default fill color (black)

    # Apply padding to make the image square
    padded_image = cv2.copyMakeBorder(
        image,
        padding_top,
        padding_bottom,
        padding_left,
        padding_right,
        cv2.BORDER_CONSTANT,
        value=fill_color
    )

    return padded_image

class ImageClassificationDataset(Dataset):
    def __init__(self, data, transform, train_mode=True):
        self.train_mode = train_mode
        self.data = data
        self.numSample = len(data)
        self.transformation = transform
        self.input_size = self.transformation['input_size']
        self.define_means_stds()
        self.torch_transform = transforms.Compose([
            PadToSquare(),           # Pad to make the image square
            MotionBlur(kernel_sizes=[5, 9, 15], angle_range=(-15, 15)),  # Add the custom MotionBlur transform with angle range
            RandomBrightness(brightness=0.2),
            RandomPerspective(distortion_scale=0.2, p=0.5),
            RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
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
                img = pad_to_square(img)
                img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC)
                img = vgg_preprocess(img, self.means, self.stds)
                img = torch.from_numpy(img)
        except Exception as e:
            logger.error(f'{imgPath}: {e}')

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
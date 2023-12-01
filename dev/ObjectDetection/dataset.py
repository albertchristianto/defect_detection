import os
import cv2
import math
import random
import torch.utils.data as data
import cv2
import numpy as np
import torch

import os

import torch.utils.data as data
import xml.etree.ElementTree as ET

# import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import torch
import torch.nn as nn

# from utils import find_peaks

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

## HeatMap Genrating Functions

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2
    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2
    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
  
    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1) 
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right] 
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right] 
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def load_set(dataset_root_path, datalist_folder, the_list):
    all_data = []
    for each in the_list:
        true_path = os.path.join(datalist_folder, each)
        theFile = open(true_path, 'r')
        theFile = theFile.readlines()
        for each_file in theFile:
            the_img_path = os.path.join(dataset_root_path, each_file.replace('\n',''))
            all_data.append(the_img_path)
    return all_data

def load_dataset(dataset_root_path):
    datalist_folder = os.path.join(dataset_root_path, 'SplitSets')
    train_set_list = list(filter(lambda x: x.startswith('train_'), os.listdir(datalist_folder)))
    val_set_list = list(filter(lambda x: x.startswith('val_'), os.listdir(datalist_folder)))
    train_data = load_set(dataset_root_path, datalist_folder, train_set_list)
    val_data = load_set(dataset_root_path, datalist_folder, val_set_list)
    return train_data, val_data

class ctDataset(data.Dataset):
    def __init__(self, datalist, input_size=512, stride=4, val_mode=False):
        # self.v2 = v2
        # self.val_mode = val_mode
        self._data = datalist
        self.input_size = input_size
        self.stride = stride
        # self.classes = ['product']
        self.classes = ['product', 'product-label', 'compartment', 'compartment-label']

    def __len__(self):
        return len(self._data)

    def _DetResize(self, img, input_size, use_random=False):
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size) / input_size
        if im_ratio > model_ratio:
            new_height = input_size
            new_width = int(new_height / im_ratio)
            start_y = 0
            start_x = 0
            if use_random:
                start_x = int((input_size - new_width - 1) * random.random())
        else:
            new_width = input_size
            new_height = int(new_width * im_ratio)
            start_y = 0
            start_x = 0
            if use_random:
                start_y = int((input_size - new_height - 1) * random.random())
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros( (input_size, input_size, 3), dtype=np.uint8 )
        det_img[start_y:new_height+start_y, start_x:new_width+start_x, :] = resized_img
        return det_img, det_scale, start_y, start_x

    def open_xml_annotation(self, input_path):
        with open(input_path, 'r') as f:
            tree = ET.parse(f)
            root = tree.getroot()
            all_data =[]
            for obj in root.iter('object'):
                cls = obj.find('name').text
                qty = 0
                if cls.find('_')!=-1:
                    str_tmp = cls.split('_')
                    cls = str_tmp[0]
                    qty = int(str_tmp[1])

                cls_id = self.classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
                        int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
                tmp =  [cls_id, b[0], b[1], b[2], b[3], qty]
                all_data.append(tmp)
        return np.array(all_data)

    def __getitem__(self, index):
        image_path = self._data[index]
        # print(image_path)
        xml_path = image_path.split('.')[0] + '.xml'
        img = cv2.imread(image_path)
        img, det_scale, start_y, start_x = self._DetResize(img, self.input_size, True)
        #disp=img.copy()
        img = vgg_preprocess(img)
        bboxes = self.open_xml_annotation(xml_path)

        output_h = self.input_size // self.stride
        output_w = self.input_size // self.stride
        draw_gaussian = draw_umich_gaussian
        hm = np.zeros((len(self.classes), output_h, output_w), dtype=np.float32)   
        reg_mask = np.zeros((1, output_h, output_w), dtype=np.float32)
        wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        cnt_off = np.zeros((2, output_h, output_w), dtype=np.float32)
        # if self.v2:
        qty = np.zeros((1, output_h, output_w), dtype=np.float32)
        qty_mask = np.zeros((1, output_h, output_w), dtype=np.float32)
        for i, c in enumerate(bboxes):
            bbox = c[1:5].astype(np.float32)

            bbox *= det_scale
            bbox[0] += start_x
            bbox[1] += start_y
            bbox[2] += start_x
            bbox[3] += start_y
            # print(bbox)
            bbox = bbox / self.stride
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h <= 0 or w <= 0:
                continue
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))  
            radius = max(0, int(radius))
            ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32) 
            ct_int = ct.astype(np.int32) 
            if reg_mask[0, ct_int[1], ct_int[0]] == 1.0: #and wh[0, ct_int[1], ct_int[0]] < h and wh[1, ct_int[1], ct_int[0]] < w:
                continue 
            # if (reg_mask[0, ct_int[1], ct_int[0]] == 1.0): must think a way on how to handle multiple object in one grid
            #     hm[:, ct_int[1], ct_int[0]] = 0.0
            draw_gaussian(hm[int(c[0])], ct_int, radius)
            wh[0, ct_int[1], ct_int[0]] = 1. * h
            wh[1, ct_int[1], ct_int[0]] = 1. * w
            cnt_off[0, ct_int[1], ct_int[0]] = ct[1] - ct_int[1]
            cnt_off[1, ct_int[1], ct_int[0]] = ct[0] - ct_int[0]
            reg_mask[0, ct_int[1], ct_int[0]] = 1.0
            # reg_mask[1, ct_int[1], ct_int[0]] = 1.0
            if int(c[0]) == 2:
                qty[0, ct_int[1], ct_int[0]] = c[5] / 100
                qty_mask[0, ct_int[1], ct_int[0]] = 1.0
            # print(ct_int,wh[:, ct_int[1], ct_int[0]],reg[:, ct_int[1], ct_int[0]])
            # print(ct_int)

        img = torch.from_numpy(img)
        hm = torch.from_numpy(hm)
        wh = torch.from_numpy(wh)
        cnt_off = torch.from_numpy(cnt_off)
        reg_mask = torch.from_numpy(reg_mask)

        # if self.v2 :
        qty = torch.from_numpy(qty)
        qty_mask = torch.from_numpy(qty_mask)

        return img, hm, wh, cnt_off, reg_mask, qty, qty_mask#, disp

def _nms(heat, conf_thresh):
    hmax = nn.functional.max_pool2d(heat, (3, 3), stride=1, padding=1)
    keep = (hmax == heat).float()
    thresh = (heat > conf_thresh).float()
    return heat * keep * thresh

def find_peaks(param, img):
    """
    Given a (grayscale) image, find local maxima whose value is above a given
    threshold (param['thre1'])
    :param img: Input image (2d array) where we want to find peaks
    :return: 2d np.array containing the [x,y] coordinates of each peak found
    in the image
    """

    peaks_binary = _nms(img, param)
    # print(peaks_binary)
    # Note reverse ([::-1]): we return [[x y], [x y]...] instead of [[y x], [y
    # x]...]
    return torch.nonzero(peaks_binary)

def reverse_res_to_bbox(hm, wh, off, stride=4):
    hm = hm[None, :, :, :]
    wh = wh[None, :, :, :]
    # off = wh[None, :, :, :]

    scores, classes_idx = torch.max(hm, 1)
    ycxc = find_peaks(0.3, scores)

    new_cls = classes_idx[:, ycxc[:,1], ycxc[:,2]]
    new_wh = wh[ycxc[:,0], :, ycxc[:,1], ycxc[:,2]]
    new_off = off[:, ycxc[:,1], ycxc[:,2]]

    bboxes = []
    x1 = (ycxc[:,2] - (new_wh[:,1]/2.0) + new_off[1, :]) * stride
    y1 = (ycxc[:,1] - (new_wh[:,0]/2.0) + new_off[0, :]) * stride
    x2 = (ycxc[:,2] + (new_wh[:,1]/2.0) + new_off[1, :]) * stride
    y2 = (ycxc[:,1] + (new_wh[:,0]/2.0) + new_off[0, :]) * stride
    bboxes = torch.stack((new_cls[0,:], x1, y1, x2, y2), 1)

    return bboxes

if __name__ == "__main__":
    # train_data, val_data =load_dataset('/media/dtv4070ti-1/WOOS/Dataset/Inventories')
    # torch.nonzero(peaks_binary)
    train_data, val_data =load_dataset('/media/dtv4070ti-1/WOOS/Dataset/RetailShelf')
    # im_idx = 10

    my_dataset = ctDataset(val_data, val_mode=True)
    # img, hm, wh, reg, reg_mask, = my_dataset.__getitem__(im_idx)

    # plt.title("Original Image")
    # plt.imshow(disp)
    # plt.show()

    # plt.title("Ground Truth Heat Map")
    # im = plt.imshow(hm[0])
    # plt.colorbar(im)
    # plt.show()

    LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
    LABEL_SCALE = 0.5
    LABEL_THICKNESS = 1

    val_loader = DataLoader(my_dataset, batch_size=1, shuffle=False, num_workers=0)

    for i, [img, hm, wh, reg, reg_mask, qty, qty_mask, det_scale, start_y, start_x] in enumerate(val_loader):
        heatmap = wh[0, 0,:,:].cpu().detach().numpy()
        hm = hm * reg_mask
        wh = wh * reg_mask
        reg = reg * reg_mask
        bboxes = reverse_res_to_bbox(img, hm, wh, reg, reg_mask)
        print(bboxes)
        # display = display[0].numpy()
        # classes = ['product', 'product-label', 'compartment', 'compartment-label']
        # for product in bboxes:
        #     print()
        #     product = np.array(product, dtype=np.int64)
        #     # calculate label size
        #     (label_width, label_height), _ = cv2.getTextSize(classes[int(product[0])], LABEL_FONT, LABEL_SCALE, LABEL_THICKNESS)
        #     # calculate label position
        #     label_x = product[1]
        #     label_y = product[2] - label_height - 5
        #     # draw filled rectangle behind label
        #     display = cv2.rectangle(display, (label_x, label_y), 
        #                             (label_x + label_width, product[2]), (50, 10, 70), -1)
        #     display = cv2.rectangle(display, (product[1], product[2]), 
        #                             (product[3], product[4]), (50, 10, 70), 2)
        #     display = cv2.putText(display, classes[int(product[0])], 
        #                         (label_x, label_y + label_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # display = cv2.resize(display, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)

        # cv2.imshow('test', display)
        cv2.imshow('heat', heatmap)
        cv2.waitKey(0)
        break
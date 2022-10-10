import os
import argparse
import cv2
import numpy as np

def create_folder(the_path):
    if not os.path.exists(the_path):
        os.makedirs(the_path)

N_TIMES_TO_CROP = 10
TARGET_SIZE = 224

parser = argparse.ArgumentParser(description='Process dataset to our format')
parser.add_argument('--dataset_path', default='E:/Dataset/NHA12D-Crack-Detection-Dataset-and-Comparison-Study-master/NHA12D_dataset', type=str, metavar='DIR', help='path to NHA12D road crack dataset')
parser.add_argument('--out_dataset_path', default='E:\Albert Christianto\Project\defect_detection\dataset', type=str, metavar='DIR', help='path to NHA12D road crack dataset')
args = parser.parse_args()

magnetic_tile_path = os.path.join(args.out_dataset_path, 'road_crack')
create_folder(magnetic_tile_path)
normal_class_path = os.path.join(magnetic_tile_path, 'Normal/NHA12D')
create_folder(normal_class_path)
defect_class_path = os.path.join(magnetic_tile_path, 'Defect/NHA12D')
create_folder(defect_class_path)

folders_list = os.listdir(args.dataset_path)
idx = 0
for each_folder in folders_list:
    folder_path = os.path.join(args.dataset_path, each_folder)
    images_path = os.path.join(folder_path, "Raw_images")
    gts_path = os.path.join(folder_path, "Masks")
    images_list = list(filter(lambda x: x.endswith('.jpg'), os.listdir(images_path)))
    for each_image in images_list:
        img_path = os.path.join(images_path, each_image)
        img_filename = each_image.split('.')[0]
        png_path = os.path.join(gts_path, (img_filename + '.png'))
        img = cv2.imread(img_path)
        gt = cv2.imread(png_path)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        gt = (gt > 0) * 255.0
        height, width, _ = img.shape
        w_movement = int((width-TARGET_SIZE) / (N_TIMES_TO_CROP-1))
        h_movement = int((height-TARGET_SIZE) / (N_TIMES_TO_CROP-1))
        for x in range(N_TIMES_TO_CROP):
            x1 = w_movement * x
            x2 = TARGET_SIZE + x1
            for y in range(N_TIMES_TO_CROP):
                y1 = h_movement * y
                y2 = TARGET_SIZE + y1
                cropped_image = img[y1:y2, x1:x2, :]
                cropped_gt = gt[y1:y2, x1:x2]
                normal_class = cropped_gt.sum() == 0 #normal
                out_image = img_filename + f'_{idx}.jpg'
                out_gt = img_filename + f'_{idx}.png'
                idx += 1
                if normal_class:
                    out_image_path = os.path.join(normal_class_path, out_image)
                    out_gt_path = os.path.join(normal_class_path, out_gt)
                else:
                    out_image_path = os.path.join(defect_class_path, out_image)
                    out_gt_path = os.path.join(defect_class_path, out_gt)
                cv2.imwrite(out_image_path, cropped_image)
                cv2.imwrite(out_gt_path, cropped_gt)
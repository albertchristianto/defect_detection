import os
import argparse
import cv2
import numpy as np

def create_folder(the_path):
    if not os.path.exists(the_path):
        os.makedirs(the_path)

N_TIMES_TO_CROP = 10

parser = argparse.ArgumentParser(description='Process dataset to our format')
parser.add_argument('--dataset_path', default='E:\Dataset\Magnetic-tile-defect-datasets', type=str, metavar='DIR', help='path to magnetic tile dataset')
parser.add_argument('--out_dataset_path', default='E:\Albert Christianto\Project\defect_detection\dataset', type=str, metavar='DIR', help='path to magnetic tile dataset')
args = parser.parse_args()

magnetic_tile_path = os.path.join(args.out_dataset_path, 'magnetic_tile')
create_folder(magnetic_tile_path)
normal_class_path = os.path.join(magnetic_tile_path, 'Normal/MT')
create_folder(normal_class_path)
defect_class_path = os.path.join(magnetic_tile_path, 'Defect/MT')
create_folder(defect_class_path)

folders_list = os.listdir(args.dataset_path)
for each_folder in folders_list:
    folder_path = os.path.join(args.dataset_path, each_folder)
    if os.path.isfile(folder_path) or each_folder == "debug":
        continue
    images_path = os.path.join(folder_path, "Imgs")
    images_list = list(filter(lambda x: x.endswith('.jpg'), os.listdir(images_path)))
    for each_image in images_list:
        img_path = os.path.join(images_path, each_image)
        img_filename = each_image.split('.')[0]
        png_path = os.path.join(images_path, (img_filename + '.png'))
        img = cv2.imread(img_path)
        gt = cv2.imread(png_path)
        height, width, channel = img.shape
        fixed_height = height < width
        out_size = min(height, width)
        movement = int(abs(height-width) / (N_TIMES_TO_CROP-1))
        for i in range(N_TIMES_TO_CROP):
            start_coordinate = movement * i
            end_coordinate = out_size + start_coordinate
            if fixed_height:
                cropped_image = img[:, start_coordinate:end_coordinate, :]
                cropped_gt = gt[:, start_coordinate:end_coordinate, :]
            else:
                cropped_image = img[start_coordinate:end_coordinate, :, :]
                cropped_gt = gt[start_coordinate:end_coordinate, :, :]
            normal_class = cropped_gt.sum() == 0 #normal
            out_image = img_filename + f'{i}.jpg'
            out_gt = img_filename + f'{i}.png'
            if normal_class:
                out_image_path = os.path.join(normal_class_path, out_image)
                out_gt_path = os.path.join(normal_class_path, out_gt)
            else:
                out_image_path = os.path.join(defect_class_path, out_image)
                out_gt_path = os.path.join(defect_class_path, out_gt)

            cv2.imwrite(out_image_path, cropped_image)
            cv2.imwrite(out_gt_path, cropped_gt)
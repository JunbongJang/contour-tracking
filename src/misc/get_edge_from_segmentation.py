'''
Junbong Jang
10/7/2021

To extract cellular edge from the raw image by using the segmented images
Generated data will be used to train Optical Flow model
'''

import numpy as np
import os
import cv2
import glob
from tqdm import tqdm


def extract_edge_region(mask, img, default_size):

    edge_gt = cv2.Canny(mask, 100, 200)
    kernel = np.ones((default_size, default_size), np.uint8)
    dilated_edge_gt = cv2.dilate(edge_gt, kernel, iterations=1)

    # cv2.bitwise_and(mask, dilated_edge_gt)
    edge_img = cv2.bitwise_and(img, dilated_edge_gt)

    # to get the sobel edge 
    # extracted_edge = dilated_edge_gt
    # sobel_x, sobel_y = get_sobel_edge(extracted_edge)

    return edge_gt, edge_img


def get_sobel_edge(img):
    sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    abs_sobel64f = np.absolute(sobelx64f)  # necessary to get both edges, white to black and black to white
    sobel_x = np.uint8(abs_sobel64f)

    sobely64f = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    abs_sobel64f = np.absolute(sobely64f)  # necessary to get both edges, white to black and black to white
    sobel_y = np.uint8(abs_sobel64f)

    return sobel_x, sobel_y


if __name__ == '__main__':
    generated_path = f'generated_mask_edge_on_image/'

    dataset_name_list = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03']

    for dataset_name in dataset_name_list:
        generated_root_path = generated_path + dataset_name
        generated_path = generated_root_path + '/img/'
        if not os.path.exists(generated_path):
            os.makedirs(generated_path)
        generated_path = generated_root_path + '/mask/'
        if not os.path.exists(generated_path):
            os.makedirs(generated_path)

        img_path = f'../assets/{dataset_name}/img/'
        mask_path = f'../assets/{dataset_name}/mask/'

        assert len(glob.glob(mask_path + '*.png')) == len(glob.glob(img_path + '*.png'))

        mask_filenames = glob.glob(mask_path + '*.png')
        mask_filenames.sort()
        # calculate the dice coefficient
        for img_index in tqdm(range(len(mask_filenames))):
            mask_filename = mask_filenames[img_index]
            mask_filename = mask_filename[len(mask_path):]
            mask = cv2.imread(mask_path + mask_filename, 0)

            img_filename = mask_filename
            img = cv2.imread(img_path + img_filename, 0)

            edge_mask, edge_img = extract_edge_region(mask, img, 32)

            cv2.imwrite(generated_root_path + f"/img/{img_filename}", edge_img)
            cv2.imwrite(generated_root_path + f"/mask/{mask_filename}", edge_mask)

'''
Author: Junbong Jang
Date: 2/7/2022

It loads PC, HACKS, and Jellyfish videos, GT labels, pseudo-labels from Mechanical model, predictions from various contour tracking algorithms. 
Then, it draws manuscript figures and evaluate models' performance by spatial and contour accuracy.
'''

import os
import cv2
from glob import glob
import matplotlib.pyplot as plt
import pylab
import numpy as np
from tqdm import tqdm
import shutil
from statistics import mean
import matplotlib as mpl
import scipy.io as sio
import ast
from matplotlib.colors import LinearSegmentedColormap

from visualization_utils import plot_tracking_points, display_image_in_actual_size, get_image_name
from contour_tracking_manuscript_figures import rainbow_contour_pred_only, manuscript_figure1_trajectory, manuscript_figure1, manuscript_figure4_for_jelly, manuscript_figure4, manuscript_figure4_no_GT, manuscript_figure5


def get_ordered_contour_points_from_mask(a_mask):
    # find contours without approx
    cnts = cv2.findContours(a_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    # get the max-area contour
    cnt = sorted(cnts, key=cv2.contourArea)[-1]
    ordered_contour_points = cnt[:, 0, :]

    return ordered_contour_points


def remove_points_touching_image_boundary(ordered_contour_points, a_image):
    remove_point_indices = []

    def is_point_touching_boundary(a_point, a_image):
        return a_point[0] == 0 or a_point[0] == a_image.shape[1] - 1 or \
                a_point[1] == 0 or a_point[1] == a_image.shape[0] - 1

    for point_index, a_point in enumerate(ordered_contour_points):
        # a_point is x and y coordinates or column and row
        if is_point_touching_boundary(a_point, a_image):
            remove_point_indices.append(point_index)
        elif point_index > 0 and point_index < ordered_contour_points.shape[0]-1:
            # special case where the point is not but left and right points are touching the image boundary
            left_point = ordered_contour_points[point_index - 1, :]
            right_point = ordered_contour_points[point_index + 1, :]
            if is_point_touching_boundary(left_point, a_image) and is_point_touching_boundary(right_point, a_image):
                remove_point_indices.append(point_index)
    processed_ordered_contour_points = np.delete(ordered_contour_points, remove_point_indices, axis=0)

    return processed_ordered_contour_points


def reorder_contour_points(processed_ordered_contour_points, height, width):
    # get left_anchor_bool
    leftmost_x = np.amin(processed_ordered_contour_points[:,0])
    rightmost_x = np.amax(processed_ordered_contour_points[:,0])
    bottom_y = np.amax(processed_ordered_contour_points[:,1])
    top_y = np.amin(processed_ordered_contour_points[:,1])

    if leftmost_x == 0:
        # print('left anchor')
        # find the index with the least x coordinate (left most point)
        least_x_index = np.argmin(processed_ordered_contour_points[:,0])
        # reorder by the x coordinate in increasing/decreasing order
        processed_ordered_contour_points = np.roll(processed_ordered_contour_points, -least_x_index, axis=0)
    elif rightmost_x == width-1:
        # print('right_anchor')
        max_x_index = np.argmax(processed_ordered_contour_points[:,0])
        processed_ordered_contour_points = np.roll(processed_ordered_contour_points, -max_x_index, axis=0)
    elif top_y == 0:
        # print('top anchor')
        min_y_index = np.argmin(processed_ordered_contour_points[:,1])
        processed_ordered_contour_points = np.roll(processed_ordered_contour_points, -min_y_index, axis=0)

    elif bottom_y == height-1:
        # print('bottom anchor')
        max_y_index = np.argmax(processed_ordered_contour_points[:,1])
        processed_ordered_contour_points = np.roll(processed_ordered_contour_points, -max_y_index, axis=0)

    return processed_ordered_contour_points


def save_sampled_tracking_points(contour_points, a_image_name, dataset_folder, save_folder):
    if not os.path.exists(f"{root_generated_path}{dataset_folder}/{save_folder}"):
        os.mkdir(f"{root_generated_path}{dataset_folder}/{save_folder}")

    save_string = ""
    for a_coordinate in contour_points:
        # saved as x & y coordinates
        save_string += str(a_coordinate[0]) + ' ' + str(a_coordinate[1]) + '\n'

    with open(f'{root_generated_path}{dataset_folder}/{save_folder}/{a_image_name}.txt', 'w') as f:
        f.write(save_string)


def sample_contour_points(root_assets_path, dataset_folder, image_folder, processed_mask_folder, image_format):
    '''
    Given a mask, get ordered contour points along the boundary of the segmentation mask, 
    '''

    def plot_points(a_img, ordered_contour_points, unit_normal_list, dataset_folder, save_folder, filename):

        if len(a_img.shape) == 2:
            three_channel_img = np.repeat(a_img[:, :, np.newaxis], 3, axis=2)
        fig, ax = display_image_in_actual_size(three_channel_img, cm, blank=False)
        NUM_COLORS = len(ordered_contour_points)
        for a_index, a_coord in enumerate(ordered_contour_points):
            # TODO: is it ok to just ignore it?
            if a_coord is not None:
                ax.scatter(x=a_coord[0], y=a_coord[1], c=np.array([cm(1. * a_index / NUM_COLORS)]), s=1)
        if unit_normal_list is not None:
            for a_index, (a_coord, a_normal) in enumerate(zip(ordered_contour_points, unit_normal_list)):
                ax.scatter(x=a_coord[0], y=a_coord[1], c=np.array([cm(1. * a_index / NUM_COLORS)]), s=1)
                ax.quiver(a_coord[0], a_coord[1], a_normal[0], a_normal[1], angles='xy', scale=15, units="width", width=0.005, color=np.array([cm(1. * a_index / NUM_COLORS)]))

        if not os.path.exists(f"{root_generated_path}{dataset_folder}/{save_folder}"):
            os.mkdir(f"{root_generated_path}{dataset_folder}/{save_folder}")

        fig.savefig(f"{root_generated_path}{dataset_folder}/{save_folder}/{filename}.png", bbox_inches="tight",
                    pad_inches=0)
        plt.close()

    if not os.path.exists(root_generated_path + dataset_folder):
        os.mkdir(root_generated_path + dataset_folder)

    mask_path_list = sorted(glob(f"{root_assets_path}/{dataset_folder}/{processed_mask_folder}/*{image_format}"))
    img_path_list = sorted(glob(f"{root_assets_path}/{dataset_folder}/{image_folder}/*{image_format}"))

    cm = pylab.get_cmap('gist_rainbow')
    number_of_edge_pixels_list = []
    total_num_points_list = []
    for img_index, (mask_path, img_path) in tqdm(enumerate(zip(mask_path_list, img_path_list))):
        a_image_name = get_image_name(mask_path, image_format )
        a_image_name = a_image_name.replace('refined_', '')  # to make the name of mask the same as the name of image
        # a_image_name = a_image_name[-3:]

        assert a_image_name == get_image_name(img_path, image_format )

        a_img = plt.imread(img_path)
        a_mask = plt.imread(mask_path).astype('uint8')
        # sample all points along the contour with order
        ordered_contour_points = get_ordered_contour_points_from_mask(a_mask)
        ordered_contour_points = reorder_contour_points(ordered_contour_points, height=a_mask.shape[0], width=a_mask.shape[1])
        processed_ordered_contour_points = remove_points_touching_image_boundary(ordered_contour_points, a_mask)
        total_num_points_list.append(processed_ordered_contour_points.shape[0])
        plot_points(a_img, processed_ordered_contour_points, None, dataset_folder, save_folder='contour_points_visualize', filename=f"{a_image_name}")
        save_sampled_tracking_points(processed_ordered_contour_points, a_image_name, dataset_folder, save_folder='contour_points')

    total_num_points_array = np.asarray(total_num_points_list)
    total_num_point_diff_array = total_num_points_array[1:] - total_num_points_array[:-1]
    print('max contour points:', np.amax(total_num_points_array))
    print('num_point_diff max', np.amax(total_num_point_diff_array), 'min', np.amin(total_num_point_diff_array))


def convert_GT_tracking_points_to_contour_indices(root_generated_path, dataset_folder):
    '''
    Convert the ground truth tracking points in x and y coordinates to contour indices along the boundary of the segmentation mask
    '''
    
    contour_points_path_list = sorted(glob(f"{root_generated_path}{dataset_folder}/contour_points/*.txt"))

    # ------------------------------------------------------------------------
    # GT_tracking_points_path_list = glob(f"{root_generated_path}{dataset_folder}/MATLAB_tracked_points/*.txt")  # MATLAB pseudo-labeled GT points
    GT_tracking_points_path_list = sorted(glob(f"{root_generated_path}{dataset_folder}/points/*.txt"))  # my manual GT points

    # if there is no MATLAB pseudo-labels or manual GT labels, create dummy labels
    if len(GT_tracking_points_path_list) == 0:
        print('No MATLAB pseudo-labels or manual GT labels. Creating dummy labels.')
        os.makedirs(f"{root_generated_path}{dataset_folder}/points/", exist_ok=True)

        save_string = ""
        for i in range(4):
            # reorder row & column coordinate to x & y coordinate
            save_string += '1 7\n'

        for a_contour_point_path in contour_points_path_list:
            a_image_name = get_image_name(a_contour_point_path, '.txt')
            GT_tracking_points_path = f'{root_generated_path}{dataset_folder}/points/{a_image_name}.txt'
            with open(GT_tracking_points_path, 'w') as f:
                f.write(save_string)

            GT_tracking_points_path_list.append(GT_tracking_points_path)

    # Make dense frames sparse
    # GT_tracking_points_path_list = [GT_tracking_points_path_list[0]] + GT_tracking_points_path_list[4::5]
    # contour_points_path_list = [contour_points_path_list[0]] + contour_points_path_list[4::5]
    # ------------------------------------------------------------------------

    assert len(GT_tracking_points_path_list) == len(contour_points_path_list)

    for a_index, (a_tracking_points_path, a_contour_points_path) in enumerate(tqdm(zip(GT_tracking_points_path_list, contour_points_path_list))):
        a_image_name = get_image_name(a_tracking_points_path, '.txt')

        gt_tracking_points = np.loadtxt(a_tracking_points_path)
        gt_tracking_points = gt_tracking_points.astype('int32')

        contour_points = np.loadtxt(a_contour_points_path)
        contour_points = contour_points.astype('int32')

        # find index of matching contour point for every gt points
        contour_indices = []
        for a_gt_point in gt_tracking_points:
            min_dist = None
            match_index = None
            for a_contour_index, a_contour_point in enumerate(contour_points):
                a_dist = np.linalg.norm(a_gt_point - a_contour_point)
                if min_dist is None or min_dist > a_dist:
                    min_dist = a_dist
                    match_index = a_contour_index
            contour_indices.append(match_index)

        assert gt_tracking_points.shape[0] == len(contour_indices)
        sorted_contour_indices = sorted(contour_indices)  # !!! sort indices in ascending order

        save_path = f"{root_generated_path}{dataset_folder}/tracked_points_in_contour_indices"
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        save_string = ""
        for a_contour_index in sorted_contour_indices:
            save_string += str(a_contour_index) + '\n'

        with open(f'{save_path}/{a_image_name}.txt', 'w') as f:
            f.write(save_string)


def copy_paste_images_to_generated_path(root_assets_path, root_generated_path, image_folder, image_format, dataset_folder):
    img_path_list = glob(f"{root_assets_path}/{dataset_folder}/{image_folder}/*{image_format}")
    dst_root_path = f"{root_generated_path}/{dataset_folder}/images"
    os.makedirs(dst_root_path, exist_ok=True)

    for src_img_path in img_path_list:
        src_img_name = os.path.basename(src_img_path)
        shutil.copy(src_img_path, f"{root_generated_path}/{dataset_folder}/images/{src_img_name}")
    

def evaluate_Matlab_tracking_points_on_my_GT_tracking_points(dataset_folder, image_folder, image_format):
    '''
    load GT points
    load contour points
    load Matlab prediction protrusion

    :param dataset_folder:
    :return:
    '''

    Matlab_GT_tracking_points_path_list = sorted(glob(f"{root_generated_path}{dataset_folder}/MATLAB_tracked_points/*.txt"))
    my_GT_tracking_points_path_list = sorted(glob(f"{root_generated_path}{dataset_folder}/points/*.txt"))  # my manual GT points
    contour_points_path_list = sorted(glob(f"{root_generated_path}{dataset_folder}/contour_points/*.txt"))
    img_path_list = sorted(glob(f"{root_assets_path}/{dataset_folder}/{image_folder}/*{image_format}"))

    # Matlab_GT_tracking_points_path_list = Matlab_GT_tracking_points_path_list[:41]
    # my_GT_tracking_points_path_list = my_GT_tracking_points_path_list[:41]
    # contour_points_path_list = contour_points_path_list[:41]
    # img_path_list = img_path_list[:41]

    if len(Matlab_GT_tracking_points_path_list) == 41 and len(contour_points_path_list) == 200:
        contour_points_path_list = [contour_points_path_list[0]] + contour_points_path_list[4::5]
    assert len(Matlab_GT_tracking_points_path_list) == len(contour_points_path_list)
    assert len(my_GT_tracking_points_path_list) == 41

    sa_list = []
    rsa_list = []
    ta_list = []
    ca_list = []
    selected_matlab_GT_tracking_points_indices = []
    for a_index, (matlab_GT_tracking_points_path, contour_points_path, image_path) in enumerate(zip(Matlab_GT_tracking_points_path_list, contour_points_path_list, img_path_list)):
        if len(Matlab_GT_tracking_points_path_list) == 200:
            my_GT_tracking_points_path = my_GT_tracking_points_path_list[(a_index+1)//5]
        else:
            my_GT_tracking_points_path = my_GT_tracking_points_path_list[a_index]

        # if len(Matlab_GT_tracking_points_path_list) == 200 and (a_index == 0 or (a_index+1) % 5 == 0):
        matlab_GT_tracking_points = np.loadtxt(matlab_GT_tracking_points_path)
        matlab_GT_tracking_points = matlab_GT_tracking_points.astype('int32')

        my_gt_tracking_points = np.loadtxt(my_GT_tracking_points_path)
        my_gt_tracking_points = my_gt_tracking_points.astype('int32')

        contour_points = np.loadtxt(contour_points_path)
        contour_points = contour_points.astype('int32')

        # -------------------------------------------------------------------------------
        # Put my_gt_tracking_points along the contour points
        gt_tracking_points_on_contour_indices = []
        for my_gt_point in my_gt_tracking_points:
            min_dist = None
            match_index = None
            for index, contour_point in enumerate(contour_points):
                a_dist = np.linalg.norm(contour_point - my_gt_point)
                if min_dist is None or min_dist > a_dist:
                    min_dist = a_dist
                    match_index = index

            gt_tracking_points_on_contour_indices.append(match_index)
        assert len(gt_tracking_points_on_contour_indices) == len(my_gt_tracking_points)
        gt_tracking_points_on_contour = contour_points[gt_tracking_points_on_contour_indices]

        # -------------------------------------------------------------------------------
        if a_index == 0:
            # find corresponding index of Matlab tracking points for every GT tracking points
            for gt_point in gt_tracking_points_on_contour:
                min_dist = None
                match_index = None
                for matlab_index, matlab_gt_point in enumerate(matlab_GT_tracking_points):
                    a_dist = np.linalg.norm(matlab_gt_point - gt_point)
                    if min_dist is None or min_dist > a_dist:
                        min_dist = a_dist
                        match_index = matlab_index

                selected_matlab_GT_tracking_points_indices.append(match_index)
            assert len(selected_matlab_GT_tracking_points_indices) == len(gt_tracking_points_on_contour)

        elif a_index > 0:
            a_image = plt.imread(image_path)
            image_height = a_image.shape[0]
            image_width = a_image.shape[1]
            spatial_accuracy_threshold = 0.02
            rsa_threshold = 0.01
            ca_threshold = 0.01

            selected_matlab_GT_tracking_points = matlab_GT_tracking_points[selected_matlab_GT_tracking_points_indices]
            absolute_sa = metrics.spatial_accuracy(gt_tracking_points_on_contour, selected_matlab_GT_tracking_points, image_width, image_height, spatial_accuracy_threshold)
            sa_list.append( absolute_sa )

            # get contour points closest to the selected_matlab_GT_tracking_points
            selected_matlab_contour_indices = []
            for a_matlab_point in selected_matlab_GT_tracking_points:
                min_dist = None
                match_index = None
                for contour_index, contour_point in enumerate(contour_points):
                    a_dist = np.linalg.norm(contour_point - a_matlab_point)
                    if min_dist is None or min_dist > a_dist:
                        min_dist = a_dist
                        match_index = contour_index
                selected_matlab_contour_indices.append(match_index)

            ca_list.append( metrics.contour_accuracy(gt_tracking_points_on_contour_indices, selected_matlab_contour_indices, matlab_GT_tracking_points.shape[0], ca_threshold) )

            # ------------------------------------------------------------------------------
            # get corresponding matlab indices for prev_gt_tracking_points
            prev_overfit_selected_matlab_GT_tracking_points_indices = []
            # find corresponding index of Matlab tracking points for every GT tracking points
            for gt_point in prev_gt_tracking_points:
                min_dist = None
                match_index = None
                for matlab_index, matlab_gt_point in enumerate(prev_matlab_GT_tracking_points):
                    a_dist = np.linalg.norm(matlab_gt_point - gt_point)
                    if min_dist is None or min_dist > a_dist:
                        min_dist = a_dist
                        match_index = matlab_index

                prev_overfit_selected_matlab_GT_tracking_points_indices.append(match_index)
            assert len(prev_overfit_selected_matlab_GT_tracking_points_indices) == len(gt_tracking_points_on_contour)
            overfit_selected_matlab_GT_tracking_points = matlab_GT_tracking_points[prev_overfit_selected_matlab_GT_tracking_points_indices]

            relative_sa = metrics.relative_spatial_accuracy(gt_tracking_points_on_contour, overfit_selected_matlab_GT_tracking_points, prev_gt_tracking_points, prev_gt_tracking_points, image_width, image_height, rsa_threshold)
            rsa_list.append( relative_sa )

            prev_selected_matlab_GT_tracking_points = prev_matlab_GT_tracking_points[selected_matlab_GT_tracking_points_indices]
            ta = metrics.temporal_accuracy(gt_tracking_points_on_contour, selected_matlab_GT_tracking_points, prev_gt_tracking_points, prev_selected_matlab_GT_tracking_points, image_width, image_height, spatial_accuracy_threshold)
            ta_list.append( ta )

            #----------------------------------------------------------------------------------

            # visualize points
            plot_tracking_points(f"{root_generated_path}{dataset_folder}/", 'matlab_gt_my_gt_compare', a_index, gt_tracking_points_on_contour, selected_matlab_GT_tracking_points, a_image)

        # save previous results for relative spatial accuracy
        prev_matlab_GT_tracking_points = matlab_GT_tracking_points
        prev_gt_tracking_points = gt_tracking_points_on_contour

    print('SA: ', round(mean(sa_list), 4))
    # print('Relative: ', round(mean(rsa_list), 4))
    # print('Temporal: ', round(mean(ta_list), 4))
    print('CA: ', round(mean(ca_list), 4))

    print('sa_list', sa_list)
    print('ca_list', ca_list)

    return sa_list, rsa_list, ta_list, ca_list


def plot_colar_bar(root_generated_path, dataset_folder, image_path, cmap):
    plot_dir = f"{root_generated_path}{dataset_folder}/"
    save_name = 'colorbar'

    # prepare save folder
    if not os.path.exists(f"{plot_dir}/{save_name}/"):
        os.makedirs(f"{plot_dir}/{save_name}/")

    gt_cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["lightcoral", "red"])
    pred_cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["greenyellow", "green"])
    matlab_cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["skyblue", "blue"])
    contour_cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["dimgrey", "snow"])
    tracking_point_cmap = pylab.get_cmap('gist_rainbow')

    a_image = plt.imread(image_path)
    plt.imshow(a_image, cmap=cmap)
    cbar = plt.colorbar()
    cbar.set_ticks([])

    plt.savefig(f"{plot_dir}/{save_name}/hi.svg", bbox_inches="tight", pad_inches=0, dpi=400)
    plt.close()


if __name__ == "__main__":
    dataset_name = 'PC' #PC # HACKS # JELLY 
    image_format = '.png'
        
    # for processing SDC dataset
    if dataset_name == 'HACKS':
        root_assets_path = "assets/Computer Vision/HACKS_live/"
        root_generated_path = "generated/Computer Vision/HACKS_live/"
        processed_mask_folder = 'masks_png'
        image_folder = 'images_png'

        # All folders
        # dataset_path_list = glob(f"{root_assets_path}/*")
        # dataset_folders = []
        # for dataset_path in dataset_path_list:
        #     dataset_folders.append( dataset_path.split('\\')[-1] )

        # folder container top or bottom anchored images
        # dataset_folders = ['3_122217_S02_DMSO_09', '3_122217_S02_DMSO_14', '3_122217_S02_DMSO_19', '4_Cell_5', '5_120217_S02_CK689_50uM_03',
        #                    '7_120217_S02_CK689_50uM_14', '8_120217_S02_CK689_50uM_09', '9_TFM-08122012-3']

        # Dense frames dataset
        # dataset_folders = ['1_050818_DMSO_09', '2_052818_S02_none_08', '3_120217_S02_CK689_50uM_08',
        #                    '4_122217_S02_DMSO_04', '5_120217_S02_CK689_50uM_07', '6_052818_S02_none_02',
        #                    '7_120217_S02_CK689_50uM_13', '8_TFM-08122012-5', '9_052818_S02_none_12']

        # Sparse frames dataset (used this for manuscript)
        dataset_folders = ['1_050818_DMSO_09_sparse', '2_052818_S02_none_08_sparse', '3_120217_S02_CK689_50uM_08_sparse',
                           '4_122217_S02_DMSO_04_sparse', '5_120217_S02_CK689_50uM_07_sparse', '6_052818_S02_none_02_sparse',
                           '7_120217_S02_CK689_50uM_13_sparse', '8_TFM-08122012-5_sparse', '9_052818_S02_none_12_sparse']

        # current setting
        # dataset_folders = ['2_052818_S02_none_08_sparse']
    # ---------------------------------------------------------------
    # for processing MARS-Net phase contrast dataset
    elif dataset_name == 'PC':
        root_assets_path = '/data/junbong/optical_flow/assets/data_processing_pc/'
        root_generated_path = "/data/junbong/optical_flow/assets/data_processed_pc/"
        processed_mask_folder = 'refined_masks/refined_masks_for_channel_1'
        image_folder = 'img'
        # dataset_folders =  ['040119_PtK1_S01_01_phase', '040119_PtK1_S01_01_phase_ROI2', '040119_PtK1_S01_01_phase_2_DMSO_nd_01', "040119_PtK1_S01_01_phase_3_DMSO_nd_03"]
        # image_folder = 'img_all'

        # for MATLAB pseudo-label dataset
        # dataset_folders = ['matlab_040119_PtK1_S01_01_phase', 'matlab_040119_PtK1_S01_01_phase_ROI2', 'matlab_040119_PtK1_S01_01_phase_2_DMSO_nd_01','matlab_040119_PtK1_S01_01_phase_2_DMSO_nd_02','matlab_040119_PtK1_S01_01_phase_3_DMSO_nd_03']
        # image_folder = 'images'

        # for Sparse frames dataset (used this for manuscript)
        # dataset_folders = ['040119_PtK1_S01_01_phase_sparse', '040119_PtK1_S01_01_phase_ROI2_sparse', '040119_PtK1_S01_01_phase_2_DMSO_nd_01_sparse', '040119_PtK1_S01_01_phase_3_DMSO_nd_03_sparse']
        # image_folder = 'img'

        # current setting
        dataset_folders = ["040119_PtK1_S01_01_phase_sparse"]
    # -----------------------------------------------
    elif dataset_name == 'JELLY':
        root_assets_path = 'generated/Computer Vision/Jellyfish/'
        root_generated_path = "generated/Computer Vision/Jellyfish/"
        image_folder = 'cropped'
        image_folder = 'cropped_color'
        processed_mask_folder = 'refined_masks/refined_masks_for_channel_1'
        
        dataset_folders = ["First"]

    else:
        raise ValueError('Unknown dataset_name', dataset_name)

    #------------------------------------------------
    # for sampling contour points and converting GT points to contour indices

    for dataset_folder in dataset_folders:
        print('dataset', dataset_folder)
        # --------------------------- Data preprocessing --------------------------------------
        copy_paste_images_to_generated_path(root_assets_path, root_generated_path, image_folder, image_format, dataset_folder)
        sample_contour_points(root_assets_path, dataset_folder, image_folder, processed_mask_folder, image_format)
        convert_GT_tracking_points_to_contour_indices(root_generated_path, dataset_folder)





        # --------------------------- Data Loading for manuscript drawing --------------------------------------------
        # Matlab_GT_tracking_points_path_list = glob(f"{root_generated_path}{dataset_folder}/MATLAB_tracked_points/*.txt")
        # my_GT_tracking_points_path_list = glob(f"{root_generated_path}{dataset_folder}/points/*.txt")  # my manual GT points
        
        # pred_tracking_points_np = np.load(f"{root_generated_path}{dataset_folder}/saved_tracking_points.npy", allow_pickle=True)
        # pred_tracking_points_contour_indices = np.load(f"{root_generated_path}{dataset_folder}/tracked_contour_points.npy", allow_pickle=True)
        
        # contour_points_path_list = glob(f"{root_generated_path}{dataset_folder}/contour_points/*.txt")
        # img_path_list = glob(f"{root_assets_path}/{dataset_folder}/{image_folder}/*{image_format}")
        
        # if len(contour_points_path_list) == 200:
        #     contour_points_path_list = [contour_points_path_list[0]] + contour_points_path_list[4::5]
        #     assert pred_tracking_points_contour_indices.shape[0] == 40
        #     assert pred_tracking_points_np.shape[0] == 40
        #     assert len(Matlab_GT_tracking_points_path_list) == len(contour_points_path_list)
        #     assert len(Matlab_GT_tracking_points_path_list) == len(my_GT_tracking_points_path_list)
        #     assert len(my_GT_tracking_points_path_list) == 41
        # if  len(contour_points_path_list) == 199:
        #     contour_points_path_list = [contour_points_path_list[0]] + contour_points_path_list[4::5]
        # if len(contour_points_path_list) == 40 and pred_tracking_points_contour_indices.shape[0] == 40:
        #     pred_tracking_points_contour_indices = pred_tracking_points_contour_indices[:-1]
        # assert len(img_path_list) == len(contour_points_path_list)
        # assert len(img_path_list) == pred_tracking_points_contour_indices.shape[0] + 1

        # ---------------------------- MATLAB ---------------------
        # first dimension is column, x
        # second dimension is row, y
        # loaded_matlab_data = sio.loadmat(f'{root_generated_path}{dataset_folder}/WindowingPackage/protrusion/protrusion_vectors.mat')
        # movie_smoothedEdge = loaded_matlab_data['smoothedEdge']
        
        # # read matlab_correspondence dict
        # track_id_dict_list = []
        # for a_dict_index in range((movie_smoothedEdge.shape[0] - 1)):
        #     with open(f'{root_generated_path}{dataset_folder}/matlab_correspondence/{a_dict_index}.txt', "r") as f:
        #         contents = f.read()
        #         track_id_dict_list.append(ast.literal_eval(contents))

        # --------------------------- Data Loading Ends --------------------------------------------

        # --------------------------- Draw Manuscript Figures --------------------------------------------
        # plot_colar_bar(img_path_list[0])
        # manuscript_figure4_for_jelly(root_generated_path, dataset_folder, img_path_list, contour_points_path_list, pred_tracking_points_contour_indices,
        #                    Matlab_GT_tracking_points_path_list, track_id_dict_list, movie_smoothedEdge, my_GT_tracking_points_path_list, arrow_plot=False )
        # manuscript_figure4(root_generated_path, dataset_folder, img_path_list, contour_points_path_list, pred_tracking_points_contour_indices, Matlab_GT_tracking_points_path_list, my_GT_tracking_points_path_list, arrow_plot=False )
        # manuscript_figure4_no_GT(root_generated_path, dataset_folder, img_path_list, contour_points_path_list, pred_tracking_points_contour_indices, Matlab_GT_tracking_points_path_list)
        # manuscript_figure1(root_generated_path, dataset_folder, img_path_list, contour_points_path_list, pred_tracking_points_contour_indices)
        # manuscript_figure1_trajectory(root_generated_path, dataset_folder, img_path_list, contour_points_path_list, pred_tracking_points_contour_indices)
        # manuscript_figure5(root_generated_path, dataset_folder, img_path_list, contour_points_path_list, pred_tracking_points_contour_indices)

        # rainbow_contour_pred_only(root_generated_path, dataset_folder, img_path_list, contour_points_path_list, pred_tracking_points_contour_indices)

        # --------------------------- For Rebuttal ---------------------------
        # rebuttal_error_study(root_generated_path)
        # cmap = mpl.colors.LinearSegmentedColormap.from_list("", [(173/255,255/255,47/255), (0, 50/255,0)], N=6)
        # plot_colar_bar(root_generated_path, dataset_folder, img_path_list[0], cmap)
        # rebuttal_labeling_figure(root_generated_path, dataset_folder, img_path_list, contour_points_path_list)

    # for evaluation of Mechanical model on my GT points (used this for manuscript comparison table)
    # all_sa = []
    # all_rsa = []
    # all_ta = []
    # all_ca = []
    # for dataset_folder in dataset_folders:
    #     print('dataset', dataset_folder)
    #     sa_list, rsa_list, ta_list, ca_list = evaluate_Matlab_tracking_points_on_my_GT_tracking_points(dataset_folder, image_folder, image_format)
    #     all_sa = all_sa + sa_list
    #     all_rsa = all_rsa + rsa_list
    #     all_ta = all_ta + ta_list
    #     all_ca = all_ca + ca_list
    # print('Average SA: ', round(mean(all_sa), 4))
    # print('Average RSA: ', round(mean(all_rsa), 4))
    # print('Average TA: ', round(mean(all_ta), 4))
    # print('Average CA: ', round(mean(all_ca), 4))


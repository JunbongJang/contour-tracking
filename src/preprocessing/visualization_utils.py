'''
Author: Junbong Jang
Date: 8/8/2022

Given an input image and tracking points of that image,
Draw tracking points over the image.
'''

import os
import cv2
from glob import glob
import matplotlib.pyplot as plt
import pylab
import numpy as np
from tqdm import tqdm


def get_image_name(image_path, image_format):
    return os.path.basename(image_path).replace(image_format, '')


def display_image_in_actual_size(img, cmap, dpi = 100, blank=False):

    if len(img.shape) == 2:
        height, width = img.shape
    elif len(img.shape) == 3:
        height, width, depth = img.shape
    else:
        raise ValueError('incorrect im_data shape')

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    if blank:
        img[:, :, :] = 0
        # img[:, :, 0] =` 222/255
        # img[:, :, 1] = 235/255
        # img[:, :, 2] = 247/255
    ax.imshow(img, cmap=cmap)

    return fig, ax


def visualize_points(root_path, points_path, cmap, image_format, a_folder, save_path):
    if os.path.exists(save_path) is not True:
        os.makedirs(save_path)

    # images = glob(f"{root_path}/{a_folder}/images/*{image_format}")
    points_textfiles = glob(f"{root_path}/{points_path}/*.txt")

    # keep image paths that have corresponding textfiles (for visualizing PoST)
    image_paths = []
    for textfile_path in points_textfiles:
        image_name = os.path.basename(textfile_path).replace('.txt', image_format)
        corresponding_image_path = f"assets/Computer Vision/PoST_gt/{a_folder}/images/{image_name}"
        image_paths.append(corresponding_image_path)

    assert len(image_paths) == len(points_textfiles)
    image_paths = sorted(image_paths)
    points_textfiles = sorted(points_textfiles)

    tracking_points = {}
    last_image_path = ""

    # get the max number of points among all point set files to have the same color map for points
    MAX_NUM_POINTS = 0
    for image_path in image_paths:
        image_name = get_image_name(image_path, image_format)
        textfile_path = f"{root_path}/{points_path}/{image_name}.txt"
        if os.path.exists(textfile_path):
            ndarray_points = np.loadtxt(textfile_path)

            if MAX_NUM_POINTS < ndarray_points.shape[0]:
                MAX_NUM_POINTS = ndarray_points.shape[0]

    # find the image that has the corresponding points_textfile
    for image_index, (image_path, textfile_path)  in enumerate(zip(image_paths, points_textfiles)):
        image_name = get_image_name(image_path, image_format)
        if os.path.exists(textfile_path):
            print('image_name', image_name)
            ndarray_points = np.loadtxt(textfile_path)

            # set up image
            last_image_path = image_path
            im = plt.imread(image_path)
            plt.imshow(im, cmap=cmap)
            cm = pylab.get_cmap('gist_rainbow')

            for a_index, a_point in enumerate(ndarray_points):
                # initialize tracking_points dict
                if not tracking_points:
                    tracking_points[a_index] = [[], []]
                elif a_index not in tracking_points:
                    tracking_points[a_index] = [[], []]

                # draw all points
                x, y = a_point
                # TODO: is it ok to ignore these points going outside the image?
                if x >= 0 and y >= 0 and x < im.shape[1] and y < im.shape[0]:
                    plt.scatter(x=x, y=y, c=np.array([cm(1. * a_index / MAX_NUM_POINTS)]), s=10)
                tracking_points[a_index][0].append(x)
                tracking_points[a_index][1].append(y)

            plt.axis('off')
            plt.savefig(f"{save_path}/{image_name}.png", bbox_inches="tight", pad_inches=0)
            plt.close()

        # --------- Visualize contour points as mask -----------------
        tracking_points_np = np.zeros((len(tracking_points), 2))
        for a_key in tracking_points:
            tracking_points_np[a_key, 0] = tracking_points[a_key][0][image_index]
            tracking_points_np[a_key, 1] = tracking_points[a_key][1][image_index]

        # a_mask = cnt2mask(tracking_points_np, im.shape)
        # cv2.imwrite(f"{save_path}/cnt2mask_{image_name}.png", a_mask*255)

    # -------------- draw trajectory of tracking points in the last image --------------------
    last_image_name = get_image_name(last_image_path, image_format)
    ndarray_points = np.loadtxt(f"{root_path}/{points_path}/{last_image_name}.txt")

    # set up image
    a_img = plt.imread(last_image_path) # .astype('uint8')
    fig, ax = display_image_in_actual_size(a_img, cmap)
    NUM_COLORS = len(ndarray_points)
    cm = pylab.get_cmap('gist_rainbow')

    for a_key in tracking_points:
        # draw all points
        x, y = tracking_points[a_key]
        ax.plot(x, y, c=np.array([cm(1. * a_key / NUM_COLORS)]), marker='o', linewidth=1, markersize=1)

    fig.savefig(f"{save_path}/trajectory_{last_image_name}.png", bbox_inches="tight", pad_inches=0)
    plt.close()

def visualize_points_in_frame(a_image_path, ordered_contour_points, save_path, save_name):
    cur_image = plt.imread(a_image_path)
    plt.imshow(cur_image, cmap='gray')

    cmap = pylab.get_cmap('gist_rainbow')
    NUM_COLORS = len(ordered_contour_points)
    for a_index, a_coord in enumerate(ordered_contour_points):
        if a_coord is not None:
            plt.scatter(x=a_coord[0], y=a_coord[1], c=np.array([cmap(1. * a_index / NUM_COLORS)]), s=1)

    if not os.path.exists(f"{save_path}"):
        os.mkdir(f"{save_path}")

    # save figure
    plt.savefig(f"{save_path}/{save_name}.png", bbox_inches="tight", pad_inches=0)
    plt.close()


def plot_tracking_points(plot_dir, save_name, cur_index, gt_tracking_points, pred_tracking_points, a_image):
    '''
    visualize and save moved tracking points

    :param plot_dir:
    :param save_name:
    :param image:
    :return:
    '''
    cm = pylab.get_cmap('gist_rainbow')

    # prepare save folder
    if not os.path.exists(f"{plot_dir}/{save_name}/"):
        os.makedirs(f"{plot_dir}/{save_name}/")

    MAX_NUM_POINTS = pred_tracking_points.shape[0]
    # -------------- draw predicted points on an image --------------
    for point_index, (pred_tracking_point, gt_tracking_point) in enumerate(zip(pred_tracking_points, gt_tracking_points)):
        pred_col, pred_row = pred_tracking_point
        plt.scatter(x=pred_col, y=pred_row, c=np.array([cm(1. * point_index / MAX_NUM_POINTS)]), s=5)

        gt_col, gt_row = gt_tracking_point
        plt.scatter(x=gt_col, y=gt_row, s=5, facecolors='none', linewidths=0.5, edgecolors=np.array([cm(1. * point_index / MAX_NUM_POINTS)]))

    plt.imshow(a_image, cmap='gray')
    plt.axis('off')
    plt.savefig(f"{plot_dir}/{save_name}/matlab_gt_vs_my_gt{cur_index}.png", bbox_inches="tight", pad_inches=0)
    plt.close()

# --------------------------------------------------------------------------------
# for PoST
def visualize_points_in_POST(root_assets_path, root_generated_path):
    '''
    Date: 2/7/2022

    Given an input image and the label about the location of contour points from PoST dataset,
    Draw the contour points over the image.
    '''

    image_format = '.jpg'
    all_folders = glob(f"{root_assets_path}/PoST_gt/*")
    # all_folders = ['floating_polygon'] # bear', 'blackswan', 'freeway', 'helicopter
    cmap = None

    for a_folder in all_folders:
        a_folder = os.path.basename(a_folder)
        print('-----------------------', a_folder, '-----------------------')

        save_path = f"{root_generated_path}/PoST_gt/{a_folder}/"
        gt_points_path = f"{a_folder}/points/"
        visualize_points(f"{root_assets_path}/PoST_gt", gt_points_path, cmap, image_format, a_folder, save_path)

        # save_path = f"{root_generated_path}/PoST_predicted/{a_folder}/"
        # predicted_points_path = f"{a_folder}/"
        # visualize_points(f"{root_assets_path}/PoST_predicted", predicted_points_path, cmap, image_format, a_folder, save_path)


def visualize_points_in_live_cell(root_assets_path, root_generated_path):
    '''
    Given an input image and the label about the location of contour points from PoST dataset,
    Draw the contour points over the image.
    '''

    image_format = '.png'
    all_folders = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03']
    cmap = 'gray'

    for a_folder in all_folders:
        print('-----------------------', a_folder, '-----------------------')
        a_folder = os.path.basename(a_folder)

        save_path = f"{root_generated_path}/{a_folder}/gt/"
        points_path = f"{a_folder}/points/"
        visualize_points(root_assets_path, points_path, cmap, image_format, a_folder, save_path)

        # save_path = f"{root_generated_path}/{a_folder}/generated/"
        # points_path = f"{root_path}/{a_folder}/generated_points/"
        # visualize_points(root_path, points_path, cmap, image_format, a_folder, save_path)



if __name__ == "__main__":
    visualize_points_in_POST(root_assets_path='assets/Computer Vision/', root_generated_path='generated/Computer Vision/')
    
    # root_assets_path = 'assets/Computer Vision/PC_live/'
    # root_generated_path = "generated/Computer Vision/PC_live/"
    # visualize_points_in_live_cell(root_assets_path, root_generated_path)
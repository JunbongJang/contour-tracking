'''
Author: Junbong Jang
Date: 2/7/2022

Given an input image and the label about the location of contour points from PoST dataset,

'''

import os
import cv2
from glob import glob
import matplotlib.pyplot as plt
import pylab
import numpy as np
from tqdm import tqdm
import collections

import geopandas as gpd
import shapely
import rasterio
from rasterio import features
from shapely.geometry import Polygon
from statistics import mean
import metrics
import matplotlib as mpl
import scipy.io as sio
import ast
from matplotlib.colors import LinearSegmentedColormap

from visualization_utils import plot_tracking_points, display_image_in_actual_size, get_image_name
from contour_tracking_manuscript_figures import rainbow_contour_pred_only, manuscript_figure1_trajectory, manuscript_figure1, manuscript_figure4_for_jelly, manuscript_figure4, manuscript_figure4_no_GT, manuscript_figure5, rebuttal_labeling_figure, rebuttal_error_study


def overlay_edge_over_img(img, canny_edge):
    # overlay with the original image
    colorful_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    colorful_canny_edge = cv2.cvtColor(canny_edge, cv2.COLOR_GRAY2RGB)
    colorful_canny_edge[:, :, 1:2] = 0

    overlaid_img = cv2.addWeighted(colorful_img, 0.5, colorful_canny_edge, 1, 0)

    return overlaid_img


def cnt2mask(cnt, size):
    '''
    Range of the intensity of the mask is 0 ~ 1
    '''
    mask = np.zeros((size[0], size[1], 3))
    abs_cnt_np = cnt.astype('int32')
    mask = cv2.drawContours(mask, [abs_cnt_np], -1, (1, 1, 1), cv2.FILLED)

    return mask


def mask_to_polygon(a_mask):
    '''
    refer to https://rocreguant.com/convert-a-mask-into-a-polygon-for-images-using-shapely-and-rasterio/1786/
    https://rasterio.readthedocs.io/en/latest/api/rasterio.features.html#rasterio.features.shapes
    https://rasterio.readthedocs.io/en/stable/topics/georeferencing.html

    :param a_mask:
    :return:
    '''
    #
    all_polygons = []
    for shape, value in features.shapes(a_mask.astype(np.uint8), mask=(a_mask > 0), transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
        all_polygons.append(shapely.geometry.shape(shape))

    all_polygons = gpd.GeoSeries(shapely.ops.unary_union(all_polygons))[0]

    return all_polygons


def sample_contour_points_from_mask(a_mask):
    # This samples the most contour points to approximate the contour by a polygon
    polygon_mask = mask_to_polygon(a_mask)

    if polygon_mask.boundary.geom_type == 'LineString':
        polygon_boundaries = polygon_mask.boundary
    else:
        # hueristic: get the longest boundary
        longest_boundary_index = None
        longest_boundary_length = 0
        for poylgon_mask_boundary_index in range(len(polygon_mask.boundary)):
            cur_boundary_length = polygon_mask.boundary[poylgon_mask_boundary_index].length
            if longest_boundary_length < cur_boundary_length:
                longest_boundary_index = poylgon_mask_boundary_index
                longest_boundary_length = cur_boundary_length

        polygon_boundaries = polygon_mask.boundary[longest_boundary_index]

    # combine two columns of x and y into tuples of x and y
    coords = np.dstack(polygon_boundaries.coords.xy)[0]
    coords = coords.astype('int32')
    # convert list of lists to list of tuples
    # coords = np.dstack(polygon_boundaries.coords.xy).tolist()
    # coords = coords[0]
    # coords = [tuple(x) for x in coords]

    return coords


def draw_spline_along_contour(root_path, thresh_img, img_index):
    '''
    Refer to https://stackoverflow.com/questions/47936474/is-there-a-function-similar-to-opencv-findcontours-that-detects-curves-and-repla
    다양한 외곽선 관련 함수 https://deep-learning-study.tistory.com/232
    :return:
    '''

    thresh_img = thresh_img.astype('uint8') * 255

    # find contours without approx
    cnts = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2]

    # contours_image = cv2.drawContours(thresh_img, cnts, -1, (0, 0, 255), 2, cv2.LINE_AA)
    # cv2.imwrite(f"{root_path}/contour_{img_index}.png", contours_image)

    # get the max-area contour
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    # calc arclentgh
    arclen = cv2.arcLength(cnt, True)

    print('arclen', arclen)
    print('contour area', cv2.contourArea(cnt))
    print('isConvex', cv2.isContourConvex(cnt))
    contour_moments = cv2.moments(cnt)
    cx = int(contour_moments['m10'] / contour_moments['m00'])
    cy = int(contour_moments['m01'] / contour_moments['m00'])
    print('centroid', cx, cy)

    # approx
    eps = 0.0005
    epsilon = arclen * eps
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # draw the result
    canvas = thresh_img.copy()
    if len(canvas.shape) == 2:
        canvas = np.repeat(canvas[:, :, np.newaxis], 3, axis=2)

    for pt in approx:
        cv2.circle(canvas, (pt[0][0], pt[0][1]), 2, (0, 255, 0), -1)

    canvas = cv2.drawContours(canvas, [approx], -1, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(f"{root_path}/spline_contour_{img_index}.png", canvas)


def sample_points_between_two_endpoints(ordered_contour_points, a_image_name, dataset_folder, NUM_SAMPLE_POINTS):
    '''
    Given GT cell body segmentation and GT tracking points,
    sample points evenly along the mask contour between GT tracking points which will be used as training set for contour tracking model

    TODO: there is an error that the two endpoints are not actual endpoints in the mask
    :return:
    '''
    # load gt points
    # gt_points_path = f"{root_assets_path}{dataset_folder}/points/{a_image_name}.txt"
    gt_points_path = f"{root_generated_path}{dataset_folder}/MATLAB_tracked_points/{a_image_name}.txt"
    if os.path.exists(gt_points_path):
        gt_points = np.loadtxt(gt_points_path)
    gt_points = gt_points.astype('int32')

    # get the closest contour point from the GT tracking point
    gt_contour_points = {}
    for gt_point_index, gt_point in enumerate(gt_points):
        min_dist = 100000
        closest_point = []
        closest_point_index = None  # necessary for ordering gt points
        for point_index, a_point in enumerate(ordered_contour_points):
            x_diff = a_point[0] - gt_point[0]
            y_diff = a_point[1] - gt_point[1]
            dist = (x_diff**2 + y_diff**2)**(1/2)
            if dist < min_dist:
                min_dist = dist
                closest_point = a_point
                closest_point_index = point_index
        gt_contour_points[closest_point_index] = closest_point

    # find orders among GT contour points
    gt_ordered_contour_points = collections.OrderedDict(sorted(gt_contour_points.items()))

    # sample points between two adjacent GT tracking points
    # WARNING: potential incosistent ordered points among frames if ordered_contour_points are ordered from the middle of the contour, not the corner
    NUM_GT_POINTS = gt_points.shape[0]
    contour_point_indices = list(gt_ordered_contour_points.keys())
    NUM_CONTOUR_POINTS_INTERVAL = contour_point_indices[-1] - contour_point_indices[0]

    if NUM_SAMPLE_POINTS is None:
        NUM_SAMPLE_POINTS = NUM_CONTOUR_POINTS_INTERVAL

    # ------------------------------
    # Case 1: even sampling
    sampled_contour_indices = np.linspace(contour_point_indices[0]+1, contour_point_indices[-1], num=NUM_SAMPLE_POINTS, endpoint=False, retstep=False, dtype=None, axis=0)
    sampled_contour_indices = np.round(sampled_contour_indices).astype(int)

    # Case 2: even sampling when ordered_contour_points are ordered a bit differently
    # contour_point_indices[1] ~ ordered_contour_points.shape[0] and 0 ~ contour_point_indices[0]
    # sampled_contour_indices = np.linspace(contour_point_indices[1]+1, ordered_contour_points.shape[0]+contour_point_indices[0], num=NUM_SAMPLE_POINTS, endpoint=False, retstep=False, dtype=None, axis=0)
    # sampled_contour_indices = np.round(sampled_contour_indices).astype(int)
    # sampled_contour_indices_one = sampled_contour_indices[sampled_contour_indices < ordered_contour_points.shape[0]]
    # sampled_contour_indices_second = sampled_contour_indices[sampled_contour_indices >= ordered_contour_points.shape[0]] - ordered_contour_points.shape[0]
    # sampled_contour_indices = np.concatenate((sampled_contour_indices_one, sampled_contour_indices_second))
    # ------------------------------

    ordered_contour_points = np.array(ordered_contour_points)
    sampled_contour_points = ordered_contour_points[sampled_contour_indices]

    return sampled_contour_points, sampled_contour_indices


def get_unit_normal_on_contour(a_contour, sampled_contour_indices):
    '''

    :param a_contour: represented by the piecewise straight lines or splines
    :param sampled_contour_indices:
    :return: unit_normal_list
    '''

    def get_unit_normal(a_contour, a_sampled_contour_index):
        '''
        p(s) returns an np.array of size 2. A point on the spline.
        s + e is a different point for all s within the spline, and nonzero e.

        Compute tangent by central differences. You can use a closed form tangent if you have it.
        referred https://stackoverflow.com/questions/66676502/how-can-i-move-points-along-the-normal-vector-to-a-curve-in-python

        '''

        tangent_at_s = (a_contour[a_sampled_contour_index+1] - a_contour[a_sampled_contour_index-1] ) / 2
        normal_at_s = np.array([-tangent_at_s[1], tangent_at_s[0]])
        unit_normal_at_s = normal_at_s / np.linalg.norm(normal_at_s)

        return unit_normal_at_s

    unit_normal_list = []
    for a_sampled_contour_index in sampled_contour_indices:
        if a_sampled_contour_index is not None:
            unit_normal_list.append( get_unit_normal(a_contour, a_sampled_contour_index) )
        else:
            unit_normal_list.append( None )


    return unit_normal_list


def get_corresponding_points_along_normal_vector(sampled_contour_points, unit_normal_list, next_ordered_contour_points):
    '''

    :param sampled_contour_points:
    :param unit_normal_list:
    :param next_ordered_contour_points: first index is the height, second index is the width
    :return:
    '''

    def along_normal_point_intersect_next_ordered_contour(along_normal_point, next_ordered_contour_points):

        # find potential matches
        corresponding_point_list = []
        corresponding_index_list = []
        dist_list = []
        x_comp = np.ceil(along_normal_point[0]).astype(dtype='int32')
        y_comp = np.ceil(along_normal_point[1]).astype(dtype='int32')

        corresponding_index = np.where((next_ordered_contour_points[:, 0] == x_comp) * (next_ordered_contour_points[:, 1] == y_comp))[0]
        if len(corresponding_index) > 0:
            corresponding_point_list.append([x_comp, y_comp])
            corresponding_index_list.append(corresponding_index[0])
            a_dist = (along_normal_point[0] - x_comp) ** 2 + (along_normal_point[1] - y_comp) ** 2
            dist_list.append(a_dist)

        x_comp = np.floor(along_normal_point[0]).astype(dtype='int32')
        y_comp = np.ceil(along_normal_point[1]).astype(dtype='int32')
        corresponding_index = np.where((next_ordered_contour_points[:, 0] == x_comp) * (next_ordered_contour_points[:, 1] == y_comp))[0]
        if len(corresponding_index) > 0:
            corresponding_point_list.append([x_comp, y_comp])
            corresponding_index_list.append(corresponding_index[0])
            a_dist = (along_normal_point[0] - x_comp) ** 2 + (along_normal_point[1] - y_comp) ** 2
            dist_list.append(a_dist)

        x_comp = np.ceil(along_normal_point[0]).astype(dtype='int32')
        y_comp = np.floor(along_normal_point[1]).astype(dtype='int32')
        corresponding_index = np.where((next_ordered_contour_points[:, 0] == x_comp) * (next_ordered_contour_points[:, 1] == y_comp))[0]
        if len(corresponding_index) > 0:
            corresponding_point_list.append([x_comp, y_comp])
            corresponding_index_list.append(corresponding_index[0])
            a_dist = (along_normal_point[0] - x_comp) ** 2 + (along_normal_point[1] - y_comp) ** 2
            dist_list.append(a_dist)

        x_comp = np.floor(along_normal_point[0]).astype(dtype='int32')
        y_comp = np.floor(along_normal_point[1]).astype(dtype='int32')
        corresponding_index = np.where((next_ordered_contour_points[:, 0] == x_comp) * (next_ordered_contour_points[:, 1] == y_comp))[0]
        if len(corresponding_index) > 0:
            corresponding_point_list.append([x_comp, y_comp])
            corresponding_index_list.append(corresponding_index[0])
            a_dist = (along_normal_point[0] - x_comp) ** 2 + (along_normal_point[1] - y_comp) ** 2
            dist_list.append(a_dist)

        # find the closest match
        if len(corresponding_point_list) > 0:
            min_index = 0
            min_dist = dist_list[min_index]
            for a_index, a_dist in enumerate(dist_list):
                if min_dist > a_dist:
                    min_dist = a_dist
                    min_index = a_index

            corresponding_point = corresponding_point_list[min_index]
            corresponding_index = corresponding_index_list[min_index]
        else:
            corresponding_point = None
            corresponding_index = None

        return corresponding_point, corresponding_index

    # ----------------------------------------------
    corresponding_point_list = []
    corresponding_index_list = []
    for sampled_point_index, sampled_point in enumerate(sampled_contour_points):
        corresponding_point = None
        corresponding_index = None
        if sampled_point is not None:
            a_unit_normal = unit_normal_list[sampled_point_index]

            # when dist_from_sampled_point = 0, it checks whether the sampled point did not move
            for dist_from_sampled_point in range(10):
                along_pos_normal_point = sampled_point + dist_from_sampled_point * a_unit_normal
                along_neg_normal_point = sampled_point - dist_from_sampled_point * a_unit_normal

                # floor, ceil the point to get 4 neighboring points
                corresponding_point, corresponding_index = along_normal_point_intersect_next_ordered_contour(along_pos_normal_point, next_ordered_contour_points)
                if corresponding_point is None:
                    corresponding_point, corresponding_index = along_normal_point_intersect_next_ordered_contour(along_neg_normal_point, next_ordered_contour_points)

                if corresponding_point is not None:
                    break

            if corresponding_point is None:
                # TODO: For now, just set it to adjacent point
                corresponding_point = corresponding_point_list[sampled_point_index-1]
                corresponding_index = corresponding_index_list[sampled_point_index-1]
                # raise Exception("dist_from_sampled_point is greater than 10!")

        corresponding_point_list.append(corresponding_point)
        corresponding_index_list.append(corresponding_index)
    print( 'number of None', corresponding_point_list.count(None) )
    return corresponding_point_list, corresponding_index_list


def get_ordered_contour_points_from_mask(a_mask):
    # find contours without approx
    cnts = cv2.findContours(a_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    # get the max-area contour
    cnt = sorted(cnts, key=cv2.contourArea)[-1]
    ordered_contour_points = cnt[:, 0, :]

    return ordered_contour_points


def find_uturn_index(ordered_contour_points):
    uturn_index = None

    for point_index in range(ordered_contour_points.shape[0]-2):
        if (ordered_contour_points[point_index,:] == ordered_contour_points[point_index+2,:]).all():
            uturn_index = point_index + 1

    # assert uturn_index is not None
    print('uturn_index', uturn_index)

    return uturn_index

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


def sample_contour_points(dataset_folder, image_folder, processed_mask_folder, image_format):
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

        # fig = plt.gcf()
        # fig.set_size_inches(20, 20)
        fig.savefig(f"{root_generated_path}{dataset_folder}/{save_folder}/{filename}.png", bbox_inches="tight",
                    pad_inches=0)
        plt.close()

    if not os.path.exists(root_generated_path + dataset_folder):
        os.mkdir(root_generated_path + dataset_folder)

    mask_path_list = glob(f"{root_assets_path}/{dataset_folder}/WindowingPackage/{processed_mask_folder}/*{image_format}")
    img_path_list = glob(f"{root_assets_path}/{dataset_folder}/{image_folder}/*{image_format}")

    cm = pylab.get_cmap('gist_rainbow')
    number_of_edge_pixels_list = []
    total_num_points_list = []
    for img_index, (mask_path, img_path) in tqdm(enumerate(zip(mask_path_list, img_path_list))):

        a_img = plt.imread(img_path)
        a_mask = plt.imread(mask_path).astype('uint8')
        a_image_name = get_image_name(mask_path, image_format )
        a_image_name = a_image_name.replace('refined_', '')
        a_image_name = a_image_name[-3:]

        # ------------------------
        # sample a few points to approximate the contour by a polygon
        # draw_spline_along_contour(root_generated_path + dataset_folder, a_mask, img_index)

        # ------------------------
        # sample the most contour points to approximate the contour by a polygon
        # ordered_contour_points = sample_contour_points_from_mask(a_mask)

        # -------------------------
        # simply sample all points along the contour with order
        ordered_contour_points = get_ordered_contour_points_from_mask(a_mask)
        ordered_contour_points = reorder_contour_points(ordered_contour_points, height=a_mask.shape[0], width=a_mask.shape[1])
        processed_ordered_contour_points = remove_points_touching_image_boundary(ordered_contour_points, a_mask)
        total_num_points_list.append(processed_ordered_contour_points.shape[0])
        plot_points(a_img, processed_ordered_contour_points, None, dataset_folder, save_folder='contour_points_visualize', filename=f"{a_image_name}")
        save_sampled_tracking_points(processed_ordered_contour_points, a_image_name, dataset_folder, save_folder='contour_points')

        # ------------------------
        # sampled_contour_points, sampled_contour_indices = sample_points_between_two_endpoints(ordered_contour_points, a_image_name, dataset_folder, NUM_SAMPLE_POINTS=None)  # NUM_SAMPLE_POINTS=128
        # plot_points(a_img, sampled_contour_points, None, dataset_folder, save_folder='contour_points_visualize', filename=f"{a_image_name}")
        # save_sampled_tracking_points(sampled_contour_points, a_image_name, dataset_folder, save_folder='contour_points')

        # -------------------------
        # iterative normal line sampling
        # if img_index < len(mask_path_list)-1:
        #     # sample all points along the contour with order
        #     ordered_contour_points = get_ordered_contour_points_from_mask(a_mask)
        #     # plot_points(a_img, ordered_contour_points, None, dataset_folder, save_folder='all_points', filename=f"{a_image_name}")
        #
        #     if img_index == 0:
        #         sampled_contour_points, sampled_contour_indices = sample_points_between_two_endpoints(ordered_contour_points, a_image_name, dataset_folder, NUM_SAMPLE_POINTS=128)
        #         save_sampled_tracking_points(sampled_contour_points, a_image_name, dataset_folder, 'sampled_normal_points')
        #     # plot_points(a_img, sampled_contour_points, None, dataset_folder, save_folder='contour_points', filename=f"{a_image_name}")
        #     else:
        #         # use corresponding points from the previous iteration
        #         sampled_contour_points = corresponding_contour_points
        #         sampled_contour_indices = corresponding_contour_indices
        #         # find the closest ordered contour point to the corresponding points
        #
        #     unit_normal_list = get_unit_normal_on_contour(ordered_contour_points, sampled_contour_indices)
        #     # plot_points(a_img, sampled_contour_points, unit_normal_list, dataset_folder, save_folder='normal_vectors', filename=f"{a_image_name}")  # plot normal
        #
        #     # get next frame
        #     next_img = plt.imread(mask_path_list[img_index+1]).astype('uint8')
        #     next_image_name = get_image_name(mask_path_list[img_index+1], image_format )
        #     next_ordered_contour_points = get_ordered_contour_points_from_mask(next_img)
        #     corresponding_contour_points, corresponding_contour_indices = get_corresponding_points_along_normal_vector(sampled_contour_points, unit_normal_list, next_ordered_contour_points)
        #     plot_points(a_img, corresponding_contour_points, None, dataset_folder, save_folder='corresponding_points', filename=f"{a_image_name}")  # plot normal
        #
        #     # save sampled tracking points ( synthetic training set )
        #     save_sampled_tracking_points(corresponding_contour_points, next_image_name, dataset_folder, 'sampled_normal_points')

        # -------------------
        # sample all points along the contour without order!
        # a_edge = cv2.Canny(a_mask * 255, 100, 200, 3, L2gradient=True)
        # # cv2.imwrite(f"{root_generated_path}{a_image_name}.png", a_edge)
        #
        # # get coords of the contour points
        # y_coords, x_coords = np.where(a_edge > 0)
        #
        # # assign color to each contour point in order
        # NUM_COLORS = len(x_coords)
        # number_of_edge_pixels_list.append(NUM_COLORS)
        # print('Total # of edge pixels', NUM_COLORS)
        #
        # color_edge = np.zeros((a_edge.shape[0], a_edge.shape[1], 3), dtype=np.float64)
        # for a_index, (x_coord, y_coord) in tqdm(enumerate(zip(x_coords, y_coords))):
        #     color_edge[y_coord, x_coord, :] = cm(1. * a_index / NUM_COLORS)[:3]
        # cv2.imwrite(f"{root_generated_path}{a_image_name}_contour.png", color_edge*255)

        # dillation_size = 32
        # kernel = np.ones((dillation_size, dillation_size), np.uint8)
        # dilated_edge = cv2.dilate(a_edge, kernel, iterations=1)

        # overlaid_img = overlay_edge_over_img(a_mask * 255, a_edge)
        # cv2.imwrite(f"{root_generated_path}{a_image_name}_overlaid.png", overlaid_img)

        # assign id to each pixel of the contour in order
        # contours, _ = cv2.findContours(a_edge.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # find contour of the lines
        # contours_image = cv2.drawContours(a_mask, contours, 0, (255, 0, 0), 1)
        # cv2.imwrite(f"{root_generated_path}{a_image_name}_contour2.png", contours_image)

        # print('------------------------------')

    # plt.plot(number_of_edge_pixels_list)
    # plt.xlabel("frame #")
    # plt.ylabel("Total number of edge pixels")
    # plt.show()

    total_num_points_array = np.asarray(total_num_points_list)
    total_num_point_diff_array = total_num_points_array[1:] - total_num_points_array[:-1]
    print('max contour points:', np.amax(total_num_points_array))
    print('num_point_diff max', np.amax(total_num_point_diff_array), 'min', np.amin(total_num_point_diff_array))


def convert_GT_tracking_points_to_contour_indices(dataset_folder):
    '''
    Convert the ground truth tracking points in x and y coordinates to contour indices along the boundary of the segmentation mask
    '''
    
    contour_points_path_list = glob(f"{root_generated_path}{dataset_folder}/contour_points/*.txt")

    # ------------------------------------------------------------------------
    GT_tracking_points_path_list = glob(f"{root_generated_path}{dataset_folder}/MATLAB_tracked_points/*.txt")  # MATLAB pseudo-labeled GT points
    # GT_tracking_points_path_list = glob(f"{root_generated_path}{dataset_folder}/points/*.txt")  # my manual GT points

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


def evaluate_Matlab_tracking_points_on_my_GT_tracking_points(dataset_folder, image_folder, image_format):
    '''
    load GT points
    load contour points
    load Matlab prediction protrusion

    :param dataset_folder:
    :return:
    '''

    Matlab_GT_tracking_points_path_list = glob(f"{root_generated_path}{dataset_folder}/MATLAB_tracked_points/*.txt")
    my_GT_tracking_points_path_list = glob(f"{root_generated_path}{dataset_folder}/points/*.txt")  # my manual GT points
    contour_points_path_list = glob(f"{root_generated_path}{dataset_folder}/contour_points/*.txt")
    img_path_list = glob(f"{root_assets_path}/{dataset_folder}/{image_folder}/*{image_format}")

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
        root_assets_path = 'assets/Computer Vision/PC_live/'
        root_generated_path = "generated/Computer Vision/PC_live/"
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
        sample_contour_points(dataset_folder, image_folder, processed_mask_folder, image_format)
        convert_GT_tracking_points_to_contour_indices(dataset_folder)

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


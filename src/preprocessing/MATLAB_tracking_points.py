'''
Author Junbong Jang
Date: 4/18/2022

Load protrusion results from MATLAB windowing & Protrusion package
Overlay protrusion results on the raw image
Create the edge correspondence dataset for our AI tracking model
'''

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pylab
from tqdm import tqdm
from glob import glob
import ast
import os
from visualization_utils import visualize_points, visualize_points_in_frame

np.random.seed(10)

GLOBAL_left_offset = 10
GLOBAL_right_offset = 10
GLOBAL_total_offset = GLOBAL_left_offset + GLOBAL_right_offset

def get_image_name(image_path, image_format):
    return image_path.split('\\')[-1].replace(image_format, '')


def show_figure(cur_image, movie_index, title_string):
    plt.imshow(cur_image, cmap='gray')
    # plt.axis('off')
    # plt.gca().invert_yaxis()
    plt.title(f'frame {movie_index}. {title_string}')
    plt.show()
    # plt.savefig(f"{save_path}/{image_name}.png", bbox_inches="tight", pad_inches=0)
    plt.close()


def save_figure(cur_image, movie_index, title_string, save_path):
    plt.imshow(cur_image, cmap='gray')
    plt.title(f'frame {movie_index}. {title_string}')
    plt.savefig(f"{save_path}/{movie_index}_{title_string}.png", bbox_inches="tight", pad_inches=0)
    plt.close()


def move_prev_tracked_point_and_align_to_smoothEdge(cur_smoothedEdge, next_smoothedEdge, cur_protrusion, cur_smoothedEdge_index):
    # move the previous tracked points
    x, y = cur_smoothedEdge[cur_smoothedEdge_index, :]
    u, v = cur_protrusion[cur_smoothedEdge_index, :]
    new_x = x + u
    new_y = y + v

    # find each moved point with the closest smoothedEdge point in the current frame
    min_dist = 185808  # cur_image.shape[0] * cur_image.shape[1]
    min_index = None
    for next_smoothedEdge_i, (x, y) in enumerate(next_smoothedEdge):
        dist = (new_x - x) ** 2 + (new_y - y) ** 2
        if dist < min_dist:
            min_index = next_smoothedEdge_i
            min_dist = dist

    return min_index


def visualize_MATLAB_protrusion_vectors(cur_image, movie_index, rounded_cur_smoothedEdge, cur_protrusion, cur_normal):
    '''
    visualize the smoothed edge, protrusion, normal vectors

    :param rounded_cur_smoothedEdge:
    :param cur_protrusion:
    :param cur_normal:
    :return:
    '''
    MAX_NUM_POINTS = rounded_cur_smoothedEdge.shape[0]

    for a_index, (x,y) in enumerate(rounded_cur_smoothedEdge):
        plt.scatter(x=x, y=y, c=np.array([cm(1. * a_index / MAX_NUM_POINTS)]), s=1)
    show_figure(cur_image, movie_index, 'cur_boundary')

    for a_index, (x,y) in enumerate(rounded_cur_smoothedEdge):
        u,v = cur_protrusion[a_index, :]
        plt.quiver(x, y, u, v, angles='xy', scale=15, units="width", width=0.003, color=np.array([cm(1. * a_index / MAX_NUM_POINTS)]))
    show_figure(cur_image, movie_index, 'protrusion')

    # visualize the normal vector
    for a_index, (x,y) in enumerate(rounded_cur_smoothedEdge):
        u,v = cur_normal[a_index, :]
        plt.quiver(x, y, u, v, angles='xy', scale=15, units="width", width=0.003, color=np.array([cm(1. * a_index / MAX_NUM_POINTS)]))
    show_figure(cur_image, movie_index, 'normal')

    # move each edge pixel in the smoothedEdge according the protrusion vector
    for a_index, (x,y) in enumerate(rounded_cur_smoothedEdge):
        u,v = cur_protrusion[a_index, :]
        new_x = x + u
        new_y = y + v
        new_x = round(new_x)
        new_y = round(new_y)
        plt.scatter(x=new_x, y=new_y, c=np.array([cm(1. * a_index / MAX_NUM_POINTS)]), s=1)
    show_figure(cur_image, movie_index, 'next_boundary')


# def visualize_tracked_points(track_id_dict_list, image_path_list, movie_protrusion, movie_smoothedEdge, save_path):
#     '''
#     show how each point's coordinate changes throughout the movie
#     To deal with the expansion, simply start visualization at the frame where expansion happens
#
#
#     :param track_id_dict_list:
#     :param image_path_list:
#     :param movie_protrusion:
#     :param movie_smoothedEdge:
#     :param save_path:
#     :return:
#     '''
#
#     # Choose which tracking points to visualize
#     point_inteval = 10
#     TOTAL_NUM_POINTS = len(track_id_dict_list[0]) // point_inteval
#     cur_track_id_list = []
#     for i in range(1, TOTAL_NUM_POINTS+1):
#         cur_track_id_list.append(i*point_inteval)
#     next_track_id_list = cur_track_id_list
#
#     # Draw each point and its protrusion velocity
#     tracking_point_dict_trajectory = {}  # {cur_track_id: [x_list, y_list]}
#     for movie_index, a_track_id_dict_list in enumerate(tqdm(track_id_dict_list)):
#         if movie_index < len(track_id_dict_list)-1:
#             # At each frame of the movie, get associated rounded coordinate for each smoothEdge index
#             cur_image = plt.imread(image_path_list[movie_index])
#             cur_smoothedEdge = movie_smoothedEdge[movie_index][0]
#             cur_protrusion = movie_protrusion[movie_index][0]
#
#             for cur_track_index, cur_track_id in enumerate(cur_track_id_list):
#                 if movie_index > 0:
#                     cur_track_id = next_track_id_list[cur_track_index]
#
#                 x, y = cur_smoothedEdge[cur_track_id, :]
#                 u,v = cur_protrusion[cur_track_id, :]
#                 # plt.quiver(x, y, u, v, angles='xy', scale=15, units="width", width=0.005, color=np.array([cm(1. * cur_track_index / len(cur_track_id_list))]))
#                 plt.scatter(x=x, y=y, c=np.array([cm(1. * cur_track_index / len(cur_track_id_list) )]), s=2)
#                 next_track_id_list[cur_track_index] = a_track_id_dict_list[cur_track_id]
#
#                 if movie_index == 0:
#                     tracking_point_dict_trajectory[cur_track_index] = [[x], [y]]
#                 else:
#                     tracking_point_dict_trajectory[cur_track_index][0].append(x)
#                     tracking_point_dict_trajectory[cur_track_index][1].append(y)
#
#             save_figure(cur_image, movie_index, 'tracked point', save_path)
#
#     # Draw trajectory of all points
#     for cur_track_index, cur_track_id in enumerate(cur_track_id_list):
#         plt.plot(tracking_point_dict_trajectory[cur_track_index][0], tracking_point_dict_trajectory[cur_track_index][1],
#                  c=np.array([cm(1. * cur_track_index / len(cur_track_id_list) )]), marker='o', linestyle='solid', linewidth=1, markersize=1)
#
#     save_figure(cur_image, movie_index, 'tracked trajectory', save_path)


def save_correspondence_as_tracking_points_per_frame(track_id_dict_list, image_path_list, movie_smoothedEdge, NUM_SAMPLE_POINTS, save_path, visualize_save_path=''):
    '''
    show how each point's coordinate changes throughout the movie
    Side note: to deal with the expansion, simply start visualization at the frame where expansion happens

    :param track_id_dict_list:
    :param image_path_list:
    :param movie_smoothedEdge:
    :param save_path:
    :return:
    '''

    if os.path.exists(save_path) is not True:
        os.makedirs(save_path)

    if visualize_save_path != '':
        if os.path.exists(visualize_save_path) is not True:
            os.makedirs(visualize_save_path)

    TOTAL_NUM_POINTS = len(track_id_dict_list[0])

    # for every interval, reset to new tracking points to prevent point convergence
    MOVIE_INDEX_INTERVAL = 100000   # if you don't want reset, set it to 100000

    for movie_index, a_track_id_dict_list in enumerate(tqdm(track_id_dict_list)):
        cur_smoothedEdge = movie_smoothedEdge[movie_index][0]
        contour_points = []  # shape = (Number of Points, 2)
        if movie_index % MOVIE_INDEX_INTERVAL == 0:
        # if movie_index == 0:
            next_track_id_list = []
            TOTAL_NUM_POINTS = len(track_id_dict_list[movie_index])

            for point_index in range(TOTAL_NUM_POINTS):
                next_track_id_list.append(track_id_dict_list[movie_index][point_index])
                x, y = cur_smoothedEdge[point_index, :]
                contour_points.append([round(x),round(y)])

        else:
            # At each frame of the movie, get corresponding coordinate for each smoothEdge index
            for point_index in range(TOTAL_NUM_POINTS):
                cur_track_id = next_track_id_list[point_index]
                x, y = cur_smoothedEdge[cur_track_id, :]
                contour_points.append([round(x),round(y)])
                next_track_id_list[point_index] = a_track_id_dict_list[cur_track_id]

        contour_points = sample_few_tracking_points(contour_points, NUM_SAMPLE_POINTS)
        a_image_name = get_image_name(image_path_list[movie_index], image_format)
        save_coordinates_to_textfile(contour_points, a_image_name, save_path)
        if visualize_save_path != '':
            visualize_points_in_frame(image_path_list[movie_index], ordered_contour_points=contour_points, save_path=visualize_save_path, save_name=a_image_name)


    # for the last frame
    contour_points = []
    movie_index = movie_index + 1
    cur_smoothedEdge = movie_smoothedEdge[movie_index][0]
    for point_index in range(TOTAL_NUM_POINTS):
        cur_track_id = next_track_id_list[point_index]
        x, y = cur_smoothedEdge[cur_track_id, :]
        contour_points.append([round(x), round(y)])

    # convert coordinates to text file
    contour_points = sample_few_tracking_points(contour_points, NUM_SAMPLE_POINTS)
    a_image_name = get_image_name(image_path_list[movie_index], image_format)
    save_coordinates_to_textfile(contour_points, a_image_name, save_path)
    if visualize_save_path != '':
        visualize_points_in_frame(image_path_list[movie_index], ordered_contour_points=contour_points, save_path=visualize_save_path, save_name=a_image_name)


def sample_few_tracking_points(contour_points, NUM_SAMPLE_POINTS):

    # # sample evenly spaced contour points
    # total_contour_points = len(contour_points)
    # point_interval = total_contour_points // (NUM_SAMPLE_POINTS + GLOBAL_total_offset)
    # sampled_contour_points = []
    # for a_index, contour_point in enumerate(contour_points):
    #     if a_index % point_interval == 0:
    #         sampled_contour_points.append(contour_point)
    #
    # # remove first and last contour points
    # sampled_contour_points = sampled_contour_points[left_offset:-right_offset]
    sampled_contour_points = contour_points[GLOBAL_left_offset:-GLOBAL_right_offset]

    return sampled_contour_points

    # randomly select points that exceed the number_of_points_to_sample
    # num_exceed = len(sampled_contour_points) - NUM_SAMPLE_POINTS
    # if num_exceed > 0:
    #     exceed_interval = len(sampled_contour_points) // num_exceed
    #
    #     reduced_sampled_contour_points = []
    #     for a_index, contour_point in enumerate(sampled_contour_points):
    #         if a_index % exceed_interval == (exceed_interval-1) and a_index < (exceed_interval * num_exceed):
    #             continue
    #         else:
    #             reduced_sampled_contour_points.append(contour_point)
    #
    #     assert len(reduced_sampled_contour_points) == NUM_SAMPLE_POINTS
    # else:
    #     reduced_sampled_contour_points = sampled_contour_points
    #
    # return reduced_sampled_contour_points


def save_coordinates_to_textfile(contour_points, a_image_name, save_path):
    save_string = ""
    for a_coordinate in contour_points:
        # saved as x & y coordinates
        save_string += str(a_coordinate[0]) + ' ' + str(a_coordinate[1]) + '\n'

    with open(f'{save_path}/{a_image_name}.txt', 'w') as f:
        f.write(save_string)



if __name__ == "__main__":
    # for MARS-Net dataset
    # root_folder = 'Tracking'
    # root_path = f'../../../assets/Computer Vision/{root_folder}/'
    # for dense frames
    # image_folder = 'img_all'
    # dataset_folders =  ['040119_PtK1_S01_01_phase', '040119_PtK1_S01_01_phase_ROI2', '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_3_DMSO_nd_03']
    # dataset_folders = ['matlab_040119_PtK1_S01_01_phase', 'matlab_040119_PtK1_S01_01_phase_ROI2', 'matlab_040119_PtK1_S01_01_phase_2_DMSO_nd_01','matlab_040119_PtK1_S01_01_phase_2_DMSO_nd_02','matlab_040119_PtK1_S01_01_phase_3_DMSO_nd_03']
    # ----
    # for sparse frames
    # image_folder = 'img'
    # dataset_folders = ['040119_PtK1_S01_01_phase_sparse', '040119_PtK1_S01_01_phase_ROI2_sparse', '040119_PtK1_S01_01_phase_3_DMSO_nd_03_sparse']

    # -------------------------
    # for SDC dataset
    # root_folder = 'HACKS_live'
    # image_folder = 'images_png'
    # root_path = f'../../../assets/Computer Vision/{root_folder}/'
    # dataset_path_list = glob(f"{root_path}/*")
    # convert to folder_list
    # dataset_folders = []
    # for dataset_path in dataset_path_list:
    #     dataset_folders.append( dataset_path.split('\\')[-1] )
    # only GT labeled folders
    # dataset_folders = ['1_050818_DMSO_09_sparse', '2_052818_S02_none_08_sparse', '3_120217_S02_CK689_50uM_08_sparse',
    #                    '4_122217_S02_DMSO_04_sparse', '5_120217_S02_CK689_50uM_07_sparse', '6_052818_S02_none_02_sparse',
    #                    '7_120217_S02_CK689_50uM_13_sparse', '8_TFM-08122012-5_sparse', '9_052818_S02_none_12_sparse']
    # -------------------------
    # for jelly dataset
    root_folder = 'Jellyfish'
    image_folder = 'cropped'
    root_path = f'../../../generated/Computer Vision/{root_folder}/'
    dataset_folders = ['First']

    # -------------------------

    cm = pylab.get_cmap('gist_rainbow')

    for a_folder in dataset_folders:
        print(a_folder)
        image_format = '.png'
        image_path_list = glob(f'{root_path}{a_folder}/{image_folder}/*{image_format}')

        loaded_matlab_data = scipy.io.loadmat(f'{root_path}{a_folder}/WindowingPackage/protrusion/protrusion_vectors.mat')

        # first dimension is column, x
        # second dimension is row, y
        movie_protrusion = loaded_matlab_data['protrusion']
        movie_normals = loaded_matlab_data['normals']
        movie_smoothedEdge = loaded_matlab_data['smoothedEdge']

        # ------------------------------------- Convert MATLAB into correspondence format --------------------------------------------------------
        # for every pixels along the current frame's contour (smoothedEdge), find its next position in the next frame's contour
        # keys: ids of all points
        # values: [(x1,y1), (x2,y2), ...] which are the (row, column) coordinates of a point for 200 frames

        track_id_dict_list = [{}]

        for movie_index in tqdm(range(movie_smoothedEdge.shape[0] - 1)):
            cur_image_name = get_image_name(image_path_list[movie_index], image_format)
            cur_image = plt.imread(image_path_list[movie_index])

            # automatically parse each string array into ndarray
            cur_protrusion = movie_protrusion[movie_index][0]
            cur_normal = movie_normals[movie_index][0]
            cur_smoothedEdge = movie_smoothedEdge[movie_index][0]

            assert cur_smoothedEdge.shape[0] == cur_normal.shape[0] == cur_protrusion.shape[0]

            # -------------------------------------------------------
            # rounded_cur_smoothedEdge = np.around(cur_smoothedEdge, decimals=0).astype('uint16')
            # visualize_MATLAB_protrusion_vectors(cur_image, movie_index, rounded_cur_smoothedEdge, cur_protrusion, cur_normal)
            # ---------------------------------------------------------------

            # initialize contour id to each pixel in the first frame's smoothedEdge
            next_smoothedEdge = movie_smoothedEdge[movie_index + 1][0]
            # print('next smoothed edge points: ', next_smoothedEdge.shape[0])
            if movie_index == 0:
                for track_id, (x,y) in enumerate(cur_smoothedEdge):
                    next_track_id = move_prev_tracked_point_and_align_to_smoothEdge(cur_smoothedEdge, next_smoothedEdge, cur_protrusion, track_id)
                    track_id_dict_list[movie_index][track_id] = next_track_id
                print('Total tracking contour points: ', len(track_id_dict_list[movie_index]))

            # track the same point no matter where it goes based on protrusion
            # For each iteration, move the tracking points
            # find each moved point with the closest smoothedEdge point in the current frame
            # assign the index of the smoothedEdge to track_id_dict_list such that one tracking id has a list of smoothedEdge index
            # expansion case: find the smoothedEdge index that does not belong to current track_id_dict_list

            else:
                track_id_dict_list.append({})
                for track_id in track_id_dict_list[movie_index-1].values():
                    next_track_id = move_prev_tracked_point_and_align_to_smoothEdge(cur_smoothedEdge, next_smoothedEdge, cur_protrusion, track_id)
                    track_id_dict_list[movie_index][track_id] = next_track_id

                # expansion case: find cur_smoothedEdge index that does not belong to current track_id_dict_list
                tracked_id_list = track_id_dict_list[movie_index].keys()
                for track_id, (x,y) in enumerate(cur_smoothedEdge):
                    if track_id not in tracked_id_list:
                        next_track_id = move_prev_tracked_point_and_align_to_smoothEdge(cur_smoothedEdge, next_smoothedEdge, cur_protrusion,
                                                                                    track_id)
                        track_id_dict_list[movie_index][track_id] = next_track_id

                assert len(track_id_dict_list[movie_index]) == cur_smoothedEdge.shape[0]

        # save to create the edge correspondence dataset
        if os.path.exists(f'{root_path}{a_folder}/matlab_correspondence/') is not True:
            os.makedirs(f'{root_path}{a_folder}/matlab_correspondence/')
        for a_dict_index in range(len(track_id_dict_list)):
            with open(f'{root_path}{a_folder}/matlab_correspondence/{a_dict_index}.txt', 'w') as f:
                print(track_id_dict_list[a_dict_index], file=f)

        # ----------------------------------------- Save MATLAB tracked points in x,y coordinate and visualize them ---------------------------------------------------------

        # read matlab_correspondence dict
        track_id_dict_list = []
        for a_dict_index in range((movie_smoothedEdge.shape[0] - 1)):
            with open(f'{root_path}{a_folder}/matlab_correspondence/{a_dict_index}.txt', "r") as f:
                contents = f.read()
                track_id_dict_list.append(ast.literal_eval(contents))

        initial_NUM_SAMPLE_POINTS = movie_smoothedEdge[0][0].shape[0] - GLOBAL_total_offset
        print('initial_NUM_SAMPLE_POINTS', initial_NUM_SAMPLE_POINTS)
        save_correspondence_as_tracking_points_per_frame(track_id_dict_list, image_path_list, movie_smoothedEdge,
                                                         NUM_SAMPLE_POINTS=initial_NUM_SAMPLE_POINTS,
                                                         save_path=f'../../../generated/Computer Vision/{root_folder}/{a_folder}/MATLAB_tracked_points/',
                                                         visualize_save_path=f'../../../generated/Computer Vision/{root_folder}/{a_folder}/MATLAB_tracked_points_visualize/')

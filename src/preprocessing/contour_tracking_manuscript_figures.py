'''
Author: Junbong Jang
Date: 1/29/2023

Generate figures for contour tracking manuscript

'''

import os
import matplotlib.pyplot as plt
import pylab
import numpy as np
from tqdm import tqdm

from statistics import mean
import matplotlib as mpl
import scipy.io as sio
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# ----------------------------------- Utility functions -----------------------------------


def get_closest_indices_from_gt_and_pred(init_gt_points, pred_points):
    closest_pred_point_indices = []
    for gt_point in init_gt_points:
        min_dist = None
        match_index = None
        for index, pred_point in enumerate(pred_points):
            a_dist = np.linalg.norm(pred_point - gt_point)
            if min_dist is None or min_dist > a_dist:
                min_dist = a_dist
                match_index = index

        closest_pred_point_indices.append(match_index)
    return closest_pred_point_indices



def get_red_to_green_cmap():
    # https://stackoverflow.com/questions/38246559/how-to-create-a-heat-map-in-python-that-ranges-from-green-to-red
    # c = ["red", "tomato", "green", "darkgreen"]
    # v = [0, .33, 0.66, 1.]
    # l = list(zip(v, c))
    # cmap = LinearSegmentedColormap.from_list('rg', l, N=256)
    cmap = ListedColormap(["red", "tomato", "green", "darkgreen"])

    return cmap

def draw_arrow_plot(cur_x, cur_y, prev_x, prev_y, arrow_color='w'):
    delta_x = cur_x - prev_x
    delta_y = cur_y - prev_y
    # subtract delta by epsilon since arrows go outside the circle
    # epsilon = 2
    # if abs(delta_y) > epsilon:
    #     if delta_y > 0:
    #         delta_y = delta_y - epsilon
    #     else:
    #         delta_y = delta_y + epsilon
    # if abs(delta_x) > epsilon:
    #     if delta_x > 0:
    #         delta_x = delta_x - epsilon
    #     else:
    #         delta_x = delta_x + epsilon

    plt.arrow(prev_x, prev_y, delta_x, delta_y, linewidth=0.1, head_width=2, edgecolor=arrow_color, facecolor=arrow_color, antialiased=True)


def prevent_trajectory_cross(pred_tracking_points_contour_indices):
    # order preservation constraint
    # TODO: improve it by refering to the contour correspondence via ant colony optimization paper

    max_contour_index = np.amax(pred_tracking_points_contour_indices)

    # go the other way also
    # for for_index, contour_index in enumerate(pred_tracking_points_contour_indices[::-1]):
    #     # descending case
    #     if for_index > 0:
    #         prev_contour_index = pred_tracking_points_contour_indices[pred_tracking_points_contour_indices.shape[0]-for_index]
    #         if max_contour_index//4 > (contour_index - prev_contour_index) and prev_contour_index < contour_index:
    #             # current point at i <-- prev point at i + 1 index
    #             pred_tracking_points_contour_indices[pred_tracking_points_contour_indices.shape[0] - for_index - 1] = pred_tracking_points_contour_indices[pred_tracking_points_contour_indices.shape[0] - for_index]

    for for_index, contour_index in enumerate(pred_tracking_points_contour_indices):
        # ascending case
        if for_index > 0:
            prev_contour_index = pred_tracking_points_contour_indices[for_index - 1]
            if max_contour_index//4 > (prev_contour_index - contour_index) and prev_contour_index > contour_index:
                # current point at i <-- prev point at i - 1 index
                pred_tracking_points_contour_indices[for_index] = prev_contour_index

    return pred_tracking_points_contour_indices


def load_protrusion_colormap():
    matlab_file = sio.loadmat('protrusion_map_colormap.mat')

    return matlab_file['velocity_cmap']

# ----------------------------------- Draw Manuscript Figures -----------------------------------

def manuscript_figure1_trajectory(root_generated_path, dataset_folder, img_path_list, contour_points_path_list, pred_tracking_points_contour_indices):

    plot_dir = f"{root_generated_path}{dataset_folder}/"
    save_name = 'manuscript_figure1'

    # prepare save folder
    if not os.path.exists(f"{plot_dir}/{save_name}/"):
        os.makedirs(f"{plot_dir}/{save_name}/")

    # -------------------------- Combine many predictions ------------------------------------------------
    tracking_point_cmap = pylab.get_cmap('gist_rainbow')
    contour_cmap = get_red_to_green_cmap()

    initial_frame = 31
    final_frame = 35
    a_image = plt.imread(img_path_list[initial_frame])
    plt.imshow(a_image, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')

    # initial_tracked_contour_point_indices_dict = {455:[],460:[],465:[],470:[],480:[],485:[],486:[],487:[],488:[],495:[],500:[],505:[],520:[],530:[],535:[],540:[],545:[]}
    # initial_tracked_contour_point_indices_dict = {740:[],746:[],750:[],755:[],760:[],765:[],770:[],775:[],780:[],785:[], 790:[],795:[],796:[]}
    initial_tracked_contour_point_indices_dict = {}
    for a_index in range(0, 483, 7):
        initial_tracked_contour_point_indices_dict[a_index] = []
    for cur_frame in tqdm(range(initial_frame, final_frame+1)):
        # Draw contours of multiple frames in one image
        a_contour_points_path = contour_points_path_list[cur_frame]
        contour_points = np.loadtxt(a_contour_points_path)
        contour_points = contour_points.astype('int32')

        frame_pred_tracking_points_contour_indices = pred_tracking_points_contour_indices[cur_frame - 1, :]

        for for_index, a_tracked_contour_point_index in enumerate(initial_tracked_contour_point_indices_dict.keys()):

            if cur_frame > initial_frame:
                prev_tracked_contour_point_index = initial_tracked_contour_point_indices_dict[a_tracked_contour_point_index][-1]
            else:
                prev_tracked_contour_point_index = a_tracked_contour_point_index

            next_tracked_contour_point_index = frame_pred_tracking_points_contour_indices[prev_tracked_contour_point_index]
            initial_tracked_contour_point_indices_dict[a_tracked_contour_point_index].append(next_tracked_contour_point_index)

            # plot arrow
            if cur_frame > initial_frame:
                cur_contour_point_index = initial_tracked_contour_point_indices_dict[a_tracked_contour_point_index][-1]
                prev_contour_point_index = initial_tracked_contour_point_indices_dict[a_tracked_contour_point_index][-2]
                cur_x, cur_y = contour_points[cur_contour_point_index]
                prev_x, prev_y = prev_contour_points[prev_contour_point_index]
                draw_arrow_plot(cur_x, cur_y, prev_x, prev_y, arrow_color=np.array([tracking_point_cmap(1. * for_index / len(initial_tracked_contour_point_indices_dict.keys()) ) ]) )

        prev_contour_points = contour_points


    plt.savefig(f"{plot_dir}/{save_name}/trajectory{initial_frame}_{final_frame}.png", bbox_inches="tight", pad_inches=0, dpi=400)
    plt.close()



def manuscript_figure1(root_generated_path, dataset_folder, img_path_list, contour_points_path_list, pred_tracking_points_contour_indices):
    '''
    load my deep learning model's predicted points
    load contour points
    load images

    Draw arrows from the contour

    :param dataset_folder:
    :return:
    '''

    plot_dir = f"{root_generated_path}{dataset_folder}/"
    save_name = 'manuscript_figure1'

    # prepare save folder
    if not os.path.exists(f"{plot_dir}/{save_name}/"):
        os.makedirs(f"{plot_dir}/{save_name}/")

    # -------------------------- Combine many predictions ------------------------------------------------
    # contour_cmap = get_red_to_green_cmap()
    #
    # initial_frame = 31
    # final_frame = 35
    # a_image = plt.imread(img_path_list[initial_frame])
    # plt.imshow(a_image, cmap='gray', vmin=0, vmax=1)
    # plt.axis('off')
    # for cur_frame in tqdm(range(initial_frame, final_frame+1)):
    #     # Draw contours of multiple frames in one image
    #     a_contour_points_path = contour_points_path_list[cur_frame]
    #     contour_points = np.loadtxt(a_contour_points_path)
    #     contour_points = contour_points.astype('int32')
    #     a_color = np.array([contour_cmap(1. * (final_frame-cur_frame) / (final_frame-initial_frame) )])
    #     for point_index, a_point in enumerate(contour_points):
    #         if point_index % 2 == 0:
    #             a_col, a_row = a_point
    #             plt.scatter(x=a_col, y=a_row, c=a_color, s=0.1, marker='o', antialiased=True)
    #
    #     if cur_frame > initial_frame:
    #         # draw points from each frame on an image
    #         frame_pred_tracking_points_contour_indices = pred_tracking_points_contour_indices[cur_frame-1, :]
    #         if prev_contour_points.shape[0] > frame_pred_tracking_points_contour_indices.shape[0]:
    #             min_valid_index = frame_pred_tracking_points_contour_indices.shape[0]
    #         else:
    #             min_valid_index = np.argmax(frame_pred_tracking_points_contour_indices >= frame_pred_tracking_points_contour_indices[-1])
    #             # assert prev_contour_points.shape[0] == min_valid_index
    #         for point_index in range(min_valid_index):
    #             if point_index % 2 == 0:
    #                 contour_point_index = frame_pred_tracking_points_contour_indices[point_index]
    #                 cur_x, cur_y = contour_points[contour_point_index]
    #                 prev_x, prev_y = prev_contour_points[point_index]
    #                 draw_arrow_plot(cur_x, cur_y, prev_x, prev_y)
    #     prev_contour_points = contour_points
    #
    # plt.savefig(f"{plot_dir}/{save_name}/combine{initial_frame}_{final_frame}.png", bbox_inches="tight", pad_inches=0, dpi=400)
    # plt.close()

    # -------------------------- Every two frames ------------------------------------------------

    contour_cmap = pylab.get_cmap('gist_rainbow')
    for cur_frame in tqdm(range(0, len(contour_points_path_list)-1)):

        a_image = plt.imread(img_path_list[cur_frame])
        img_height, img_width = a_image.shape[:2]
        plt.imshow(a_image, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')

        two_list = [contour_points_path_list[cur_frame], contour_points_path_list[cur_frame+1]]
        # Draw contours of two frames in one image
        for two_frame, a_contour_points_path in enumerate(two_list):
            contour_points = np.loadtxt(a_contour_points_path)
            contour_points = contour_points.astype('int32')

            # for point_index, a_point in enumerate(contour_points):
            #     if point_index % 2 == 0:
            #         a_col, a_row = a_point
            #         if two_frame == 0:
            #             plt.scatter(x=a_col, y=a_row, c='g', s=0.1, marker='o', antialiased=True)
            #         else:
            #             plt.scatter(x=a_col, y=a_row, c='r', s=0.1, marker='o', antialiased=True)
            if two_frame == 1:
                # draw points from each frame on an image
                frame_pred_tracking_points_contour_indices = pred_tracking_points_contour_indices[cur_frame, :]
                frame_pred_tracking_points_contour_indices = prevent_trajectory_cross(frame_pred_tracking_points_contour_indices)
                if prev_contour_points.shape[0] > frame_pred_tracking_points_contour_indices.shape[0]:
                    min_valid_index = frame_pred_tracking_points_contour_indices.shape[0]
                else:
                    min_valid_index = np.argmax(frame_pred_tracking_points_contour_indices >= frame_pred_tracking_points_contour_indices[-1])
                    # assert prev_contour_points.shape[0] == min_valid_index
                for point_index in range(min_valid_index):
                    a_color = np.array([contour_cmap(1. * point_index / min_valid_index)])
                    if point_index % 2 == 0:
                        contour_point_index = frame_pred_tracking_points_contour_indices[point_index]
                        cur_x, cur_y = contour_points[contour_point_index]
                        prev_x, prev_y = prev_contour_points[point_index]
                        # to prevent white padding due to arrows going outside the image
                        if cur_x >= 3 and cur_x < (img_width-3) and cur_y >= 3 and cur_y < (img_height-3) and \
                            prev_x >= 3 and prev_x < (img_width-3) and prev_y >= 3 and prev_y < (img_height-3):
                            draw_arrow_plot(cur_x, cur_y, prev_x, prev_y, a_color)

            prev_contour_points = contour_points

        fig = plt.gcf()
        fig.set_size_inches(10, 10)
        plt.savefig(f"{plot_dir}/{save_name}/{cur_frame}.png", bbox_inches="tight", pad_inches=0, dpi=400)
        plt.close()


def manuscript_figure4_for_jelly(root_generated_path, dataset_folder, img_path_list, contour_points_path_list, pred_tracking_points_contour_indices,
                                 Matlab_GT_tracking_points_path_list, track_id_dict_list, movie_smoothedEdge, my_GT_tracking_points_path_list, arrow_plot=False):
    '''
    load my deep learning model's predicted points
    load contour points
    load images

    :param dataset_folder:
    :return:
    '''

    # --------------------------------------------- load GT and MATLAB prediction data ---------------------------------------------
    Matlab_GT_tracking_points_all_frames = []
    gt_tracking_points_all_frames = []
    selected_matlab_GT_tracking_points_indices = []
    first_gt_tracking_points_on_contour_indices = []
    for a_index, (matlab_GT_tracking_points_path, contour_points_path, image_path) in enumerate( zip(Matlab_GT_tracking_points_path_list, contour_points_path_list, img_path_list)):
        if len(Matlab_GT_tracking_points_path_list) == 200:
            my_GT_tracking_points_path = my_GT_tracking_points_path_list[(a_index + 1) // 5]
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
        gt_tracking_points_on_contour_indices = get_closest_indices_from_gt_and_pred(my_gt_tracking_points, contour_points)
        assert len(gt_tracking_points_on_contour_indices) == len(my_gt_tracking_points)
        if a_index == 0:
            first_gt_tracking_points_on_contour_indices = gt_tracking_points_on_contour_indices
        gt_tracking_points_on_contour = contour_points[gt_tracking_points_on_contour_indices]
        if a_index == 0:
            init_gt_tracking_points_on_contour = gt_tracking_points_on_contour

        gt_tracking_points_all_frames.append(gt_tracking_points_on_contour)
        # -------------------------------------------------------------------------------
        # get Matlab points along the contour points
        if a_index == 0:
            # find corresponding index of Matlab tracking points for every GT tracking points
            selected_matlab_GT_tracking_points_indices = get_closest_indices_from_gt_and_pred(gt_tracking_points_on_contour, matlab_GT_tracking_points)
            assert len(selected_matlab_GT_tracking_points_indices) == len(gt_tracking_points_on_contour)

        elif a_index > 0:
            # a_image = plt.imread(image_path)
            selected_matlab_GT_tracking_points = matlab_GT_tracking_points[selected_matlab_GT_tracking_points_indices]
            Matlab_GT_tracking_points_all_frames.append(selected_matlab_GT_tracking_points)

    # ----------------------------- From save_correspondence_as_tracking_points_per_frame, load MATLAB points differently --------------------------------------------
    TOTAL_NUM_POINTS = len(track_id_dict_list[0])
    # for every interval, reset to new tracking points to prevent point convergence
    All_MATLAB_contour_points = []
    for movie_index, a_track_id_dict_list in enumerate(tqdm(track_id_dict_list)):
        cur_smoothedEdge = movie_smoothedEdge[movie_index][0]
        MATLAB_contour_points = []  # shape = (Number of Points, 2)
        if movie_index == 0:
            next_track_id_list = []
            TOTAL_NUM_POINTS = len(track_id_dict_list[movie_index])

            for point_index in range(TOTAL_NUM_POINTS):
                next_track_id_list.append(track_id_dict_list[movie_index][point_index])
                x, y = cur_smoothedEdge[point_index, :]
                MATLAB_contour_points.append([round(x), round(y)])

        else:
            # At each frame of the movie, get corresponding coordinate for each smoothEdge index
            for point_index in range(TOTAL_NUM_POINTS):
                cur_track_id = next_track_id_list[point_index]
                x, y = cur_smoothedEdge[cur_track_id, :]
                MATLAB_contour_points.append([round(x), round(y)])
                next_track_id_list[point_index] = a_track_id_dict_list[cur_track_id]
        All_MATLAB_contour_points.append(MATLAB_contour_points)

    # for the last frame
    MATLAB_contour_points = []
    movie_index = movie_index + 1
    cur_smoothedEdge = movie_smoothedEdge[movie_index][0]
    for point_index in range(TOTAL_NUM_POINTS):
        cur_track_id = next_track_id_list[point_index]
        x, y = cur_smoothedEdge[cur_track_id, :]
        MATLAB_contour_points.append([round(x), round(y)])
    All_MATLAB_contour_points.append(MATLAB_contour_points)

    closest_matlab_point_indices = get_closest_indices_from_gt_and_pred(init_gt_tracking_points_on_contour, All_MATLAB_contour_points[0])
    closest_matlab_point_indices = np.asarray(closest_matlab_point_indices)
    All_MATLAB_contour_points = All_MATLAB_contour_points[4::5]

    # ----------------------------------- Process prediction -----------------------------------------

    # create a dict of points to track
    initial_tracked_contour_point_indices_dict = {}
    for a_contour_index in first_gt_tracking_points_on_contour_indices:
        initial_tracked_contour_point_indices_dict[a_contour_index] = []

    pred_tracking_points_np = np.zeros(shape=(39, len(initial_tracked_contour_point_indices_dict.keys()), 2))

    # create prediction np from dense correspondences
    for cur_frame in tqdm(range(39)):
        a_contour_points_path = contour_points_path_list[cur_frame +1]
        contour_points = np.loadtxt(a_contour_points_path)
        contour_points = contour_points.astype('int32')

        frame_pred_tracking_points_contour_indices = pred_tracking_points_contour_indices[cur_frame, :]

        for for_index, a_tracked_contour_point_index in enumerate(initial_tracked_contour_point_indices_dict.keys()):
            if cur_frame > 0:
                prev_tracked_contour_point_index = initial_tracked_contour_point_indices_dict[a_tracked_contour_point_index][-1]
            else:
                prev_tracked_contour_point_index = a_tracked_contour_point_index

            next_tracked_contour_point_index = frame_pred_tracking_points_contour_indices[prev_tracked_contour_point_index]
            initial_tracked_contour_point_indices_dict[a_tracked_contour_point_index].append(next_tracked_contour_point_index)
            pred_tracking_points_np[cur_frame, for_index, :] = contour_points[next_tracked_contour_point_index]

    # -----------------------------------------------------------------------------------
    # --------------------------- Visualize points --------------------------------------
    # -----------------------------------------------------------------------------------

    Matlab_GT_tracking_points_np = np.asarray(Matlab_GT_tracking_points_all_frames)
    gt_tracking_points_np = np.asarray(gt_tracking_points_all_frames[1:])
    first_gt_tracking_points_np = np.asarray(gt_tracking_points_all_frames[0])

    # sort points
    sort_index = np.argsort(first_gt_tracking_points_np[: ,1], axis=0)
    first_gt_tracking_points_np = first_gt_tracking_points_np[sort_index]
    gt_tracking_points_np = gt_tracking_points_np[: ,sort_index]
    Matlab_GT_tracking_points_np = Matlab_GT_tracking_points_np[: ,sort_index]

    sort_index = np.argsort(pred_tracking_points_np[0 ,:, 1], axis=0)
    pred_tracking_points_np = pred_tracking_points_np[:, sort_index]

    assert pred_tracking_points_np.shape == Matlab_GT_tracking_points_np.shape
    assert pred_tracking_points_np.shape == gt_tracking_points_np.shape

    gt_cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["lightcoral", "red"])
    pred_cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["greenyellow", "green"])
    matlab_cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["skyblue", "blue"])
    contour_cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["dimgrey", "snow"])
    initial_frame = 0
    final_frame = 39 # pred_tracking_points_np.shape[0]

    sa_list = []
    ca_list = []

    first_contour_index = 0
    last_contour_index = 1000
    total_frames = final_frame - initial_frame

    a_image = plt.imread(img_path_list[initial_frame])
    image_height, image_width = a_image.shape[:2]
    plt.imshow(a_image, cmap='gray')
    plt.axis('off')

    plot_dir = f"{root_generated_path}{dataset_folder}/"
    save_name = 'manuscript_figure4'

    # prepare save folder
    if not os.path.exists(f"{plot_dir}/{save_name}/"):
        os.makedirs(f"{plot_dir}/{save_name}/")

    for cur_frame in tqdm(range(initial_frame, final_frame)):
        # Draw contours of multiple frames in one image
        # if cur_frame == initial_frame:
        #     a_contour_points_path = contour_points_path_list[cur_frame]
        #     contour_points = np.loadtxt(a_contour_points_path)
        #     contour_points = contour_points.astype('int32')
        #     contour_points = contour_points[first_contour_index:last_contour_index, :]
        #     a_color = np.array([contour_cmap(1. * (cur_frame - initial_frame) / total_frames)])
        #     for point_index, a_point in enumerate(contour_points):
        #         a_col, a_row = a_point
        #         plt.scatter(x=a_col, y=a_row, c=a_color, s=1, marker='s', linewidths=1, antialiased=True)
        #
        a_contour_points_path = contour_points_path_list[cur_frame +1]
        contour_points = np.loadtxt(a_contour_points_path)
        contour_points = contour_points.astype('int32')
        # contour_points = contour_points[first_contour_index:last_contour_index,:]
        # a_color = np.array([contour_cmap(1. * (cur_frame-initial_frame) / total_frames)])
        # for point_index, a_point in enumerate(contour_points):
        #     a_col, a_row = a_point
        #     plt.scatter(x=a_col, y=a_row, c=a_color, s=1, marker='s',linewidths=1, antialiased=True)

        if not arrow_plot:
            a_image = plt.imread(img_path_list[cur_frame +1])
            plt.imshow(a_image, cmap='gray')
            plt.axis('off')

        # -----------------------------------------------------------------------------------
        for a_point_index in range(pred_tracking_points_np.shape[1]):
            if arrow_plot:
                cur_x, cur_y = gt_tracking_points_np[cur_frame, a_point_index, :]
                if cur_frame == 0:
                    prev_x, prev_y = first_gt_tracking_points_np[a_point_index, :]
                else:
                    prev_x, prev_y = gt_tracking_points_np[cur_frame -1, a_point_index, :]
                draw_arrow_plot(cur_x, cur_y, prev_x, prev_y, arrow_color=np.array(gt_cmap(1. * (cur_frame -initial_frame) / total_frames)))

                cur_x, cur_y = Matlab_GT_tracking_points_np[cur_frame, a_point_index, :]
                if cur_frame > 0:
                    prev_x, prev_y = Matlab_GT_tracking_points_np[cur_frame -1, a_point_index, :]
                draw_arrow_plot(cur_x, cur_y, prev_x, prev_y, arrow_color= np.array(matlab_cmap(1. * (cur_frame -initial_frame) / total_frames)))

                cur_x, cur_y = pred_tracking_points_np[cur_frame, a_point_index, :]
                if cur_frame > 0:
                    prev_x, prev_y = pred_tracking_points_np[cur_frame -1, a_point_index, :]
                draw_arrow_plot(cur_x, cur_y, prev_x, prev_y, arrow_color=np.array(pred_cmap(1. * (cur_frame -initial_frame) / total_frames)) )
            else:
                # np.asarray(All_MATLAB_contour_points[cur_frame])[closest_matlab_point_indices][a_point_index]
                cur_x, cur_y = Matlab_GT_tracking_points_np[cur_frame, a_point_index, :]
                plt.scatter(x=cur_x, y=cur_y, c='b', s=1, marker='o', antialiased=True)
                cur_x, cur_y = pred_tracking_points_np[cur_frame, a_point_index, :]
                plt.scatter(x=cur_x, y=cur_y, c='g', s=1, marker='o', antialiased=True)
                cur_x, cur_y = gt_tracking_points_np[cur_frame, a_point_index, :]
                plt.scatter(x=cur_x, y=cur_y, c='r', s=1, marker='o', antialiased=True)

        # if not arrow_plot:
        #     fig = plt.gcf()
        #     # fig.set_size_inches(10, 10)
        #     plt.savefig(f"{plot_dir}/{save_name}/gt_mech_ours_{cur_frame}.png", bbox_inches="tight", pad_inches=0, dpi=300)
        #     plt.close()

        # ---------------------------- Evaluate ----------------------------------
        # spatial_accuracy_threshold = 0.25
        # ca_threshold = 50
        #
        # selected_matlab_GT_tracking_points = np.asarray(All_MATLAB_contour_points[cur_frame])[closest_matlab_point_indices]  # pred_tracking_points_np[cur_frame]
        # absolute_sa = metrics.spatial_accuracy(gt_tracking_points_on_contour, selected_matlab_GT_tracking_points, image_width, image_height, spatial_accuracy_threshold)
        # sa_list.append(absolute_sa)
        #
        # # get contour points closest to the selected_matlab_GT_tracking_points
        # selected_matlab_contour_indices = get_closest_indices_from_gt_and_pred(selected_matlab_GT_tracking_points, contour_points)
        # ca_list.append(metrics.contour_accuracy(gt_tracking_points_on_contour_indices, selected_matlab_contour_indices, ca_threshold))


    # -------------- Draw line plot of a few predicted points on an image --------------
    # tracking_point_cmap = pylab.get_cmap('gist_rainbow')
    # MAX_NUM_POINTS = pred_tracking_points_np.shape[1]  # shape is [#frames, #points, 2]
    #
    # for a_point_index in tqdm(range(pred_tracking_points_np.shape[1])):
    #     col_list = pred_tracking_points_np[:, a_point_index, 0]
    #     row_list = pred_tracking_points_np[:, a_point_index, 1]
    #     plt.plot(col_list, row_list, c='g', marker='o', linewidth=1, markersize=1, antialiased=True) # np.array(tracking_point_cmap(1. * a_point_index / MAX_NUM_POINTS))
    #
    #     col_list = Matlab_GT_tracking_points_np[:, a_point_index, 0]
    #     row_list = Matlab_GT_tracking_points_np[:, a_point_index, 1]
    #     plt.plot(col_list, row_list, c='b', marker='o', linewidth=1, markersize=1, antialiased=True)
    #
    #     col_list = gt_tracking_points_np[:, a_point_index, 0]
    #     row_list = gt_tracking_points_np[:, a_point_index, 1]
    #     plt.plot(col_list, row_list, c='r', marker='o', linewidth=1, markersize=1, antialiased=True)

    if arrow_plot:
        fig = plt.gcf()
        fig.set_size_inches(10, 10)
        plt.savefig(f"{plot_dir}/{save_name}/all_frames_trajectory_{initial_frame}_{final_frame}.png", bbox_inches="tight", pad_inches=0, dpi=400)
        plt.close()


    print('SA: ', round(mean(sa_list), 4))
    # print('Relative: ', round(mean(rsa_list), 4))
    # print('Temporal: ', round(mean(ta_list), 4))
    print('CA: ', round(mean(ca_list), 4))



def manuscript_figure4(root_generated_path, dataset_folder, img_path_list, contour_points_path_list, pred_tracking_points_contour_indices, Matlab_GT_tracking_points_path_list, my_GT_tracking_points_path_list, arrow_plot=False):
    '''
    load my deep learning model's predicted points
    load contour points
    load images

    Can draw contours of multiple frames on the first frame
    Draw GT points, our contour tracker's predicted points, and mechanical model's predicted points for comparison

    :param dataset_folder:
    :return:
    '''

    # --------------------------------------------- load GT and MATLAB prediction data ---------------------------------------------
    Matlab_GT_tracking_points_all_frames = []
    gt_tracking_points_all_frames = []
    selected_matlab_GT_tracking_points_indices = []
    first_gt_tracking_points_on_contour_indices = []
    for a_index, (matlab_GT_tracking_points_path, contour_points_path, image_path) in enumerate( zip(Matlab_GT_tracking_points_path_list, contour_points_path_list, img_path_list)):
        if len(Matlab_GT_tracking_points_path_list) == 200:
            my_GT_tracking_points_path = my_GT_tracking_points_path_list[(a_index + 1) // 5]
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
        gt_tracking_points_on_contour_indices = get_closest_indices_from_gt_and_pred(my_gt_tracking_points, contour_points)
        assert len(gt_tracking_points_on_contour_indices) == len(my_gt_tracking_points)
        if a_index == 0:
            first_gt_tracking_points_on_contour_indices = gt_tracking_points_on_contour_indices
        gt_tracking_points_on_contour = contour_points[gt_tracking_points_on_contour_indices]
        if a_index == 0:
            init_gt_tracking_points_on_contour = gt_tracking_points_on_contour

        gt_tracking_points_all_frames.append(gt_tracking_points_on_contour)
        # -------------------------------------------------------------------------------
        # get Matlab points along the contour points
        if a_index == 0:
            # find corresponding index of Matlab tracking points for every GT tracking points
            selected_matlab_GT_tracking_points_indices = get_closest_indices_from_gt_and_pred(gt_tracking_points_on_contour, matlab_GT_tracking_points)
            assert len(selected_matlab_GT_tracking_points_indices) == len(gt_tracking_points_on_contour)

        elif a_index > 0:
            selected_matlab_GT_tracking_points = matlab_GT_tracking_points[selected_matlab_GT_tracking_points_indices]
            Matlab_GT_tracking_points_all_frames.append(selected_matlab_GT_tracking_points)


    # ----------------------------------- Process prediction -----------------------------------------

    # create a dict of points to track
    initial_tracked_contour_point_indices_dict = {}
    for a_contour_index in first_gt_tracking_points_on_contour_indices:
        initial_tracked_contour_point_indices_dict[a_contour_index] = []

    pred_tracking_points_np = np.zeros(shape=(40, len(initial_tracked_contour_point_indices_dict.keys()), 2))

    # create prediction np from dense correspondences
    for cur_frame in tqdm(range(40)):
        a_contour_points_path = contour_points_path_list[cur_frame +1]
        contour_points = np.loadtxt(a_contour_points_path)
        contour_points = contour_points.astype('int32')

        frame_pred_tracking_points_contour_indices = pred_tracking_points_contour_indices[cur_frame, :]

        for for_index, a_tracked_contour_point_index in enumerate(initial_tracked_contour_point_indices_dict.keys()):
            if cur_frame > 0:
                prev_tracked_contour_point_index = initial_tracked_contour_point_indices_dict[a_tracked_contour_point_index][-1]
            else:
                prev_tracked_contour_point_index = a_tracked_contour_point_index

            next_tracked_contour_point_index = frame_pred_tracking_points_contour_indices[prev_tracked_contour_point_index]
            initial_tracked_contour_point_indices_dict[a_tracked_contour_point_index].append(next_tracked_contour_point_index)
            pred_tracking_points_np[cur_frame, for_index, :] = contour_points[next_tracked_contour_point_index]

    # -----------------------------------------------------------------------------------
    # --------------------------- Visualize points --------------------------------------
    # -----------------------------------------------------------------------------------

    Matlab_GT_tracking_points_np = np.asarray(Matlab_GT_tracking_points_all_frames)
    gt_tracking_points_np = np.asarray(gt_tracking_points_all_frames[1:])
    first_gt_tracking_points_np = np.asarray(gt_tracking_points_all_frames[0])

    # sort points
    sort_index = np.argsort(first_gt_tracking_points_np[: ,1], axis=0)
    first_gt_tracking_points_np = first_gt_tracking_points_np[sort_index]
    gt_tracking_points_np = gt_tracking_points_np[: ,sort_index]
    Matlab_GT_tracking_points_np = Matlab_GT_tracking_points_np[: ,sort_index]

    sort_index = np.argsort(pred_tracking_points_np[0 ,:, 1], axis=0)
    pred_tracking_points_np = pred_tracking_points_np[:, sort_index]

    assert pred_tracking_points_np.shape == Matlab_GT_tracking_points_np.shape
    assert pred_tracking_points_np.shape == gt_tracking_points_np.shape

    gt_cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["lightcoral", "red"])
    pred_cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["greenyellow", "green"])
    matlab_cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["skyblue", "blue"])
    contour_cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["dimgrey", "snow"])
    initial_frame = 0
    final_frame = 40 # pred_tracking_points_np.shape[0]

    first_contour_index = 0
    last_contour_index = 1000
    total_frames = final_frame - initial_frame

    a_image = plt.imread(img_path_list[initial_frame])
    plt.imshow(a_image, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')

    plot_dir = f"{root_generated_path}{dataset_folder}/"
    save_name = 'manuscript_figure4'

    # prepare save folder
    if not os.path.exists(f"{plot_dir}/{save_name}/"):
        os.makedirs(f"{plot_dir}/{save_name}/")

    for cur_frame in tqdm(range(initial_frame, final_frame)):
        # Draw contours of multiple frames in one image
        # if cur_frame == initial_frame:
        #     a_contour_points_path = contour_points_path_list[cur_frame]
        #     contour_points = np.loadtxt(a_contour_points_path)
        #     contour_points = contour_points.astype('int32')
        #     contour_points = contour_points[first_contour_index:last_contour_index, :]
        #     a_color = np.array([contour_cmap(1. * (cur_frame - initial_frame) / total_frames)])
        #     for point_index, a_point in enumerate(contour_points):
        #         a_col, a_row = a_point
        #         plt.scatter(x=a_col, y=a_row, c=a_color, s=1, marker='s', linewidths=1, antialiased=True)
        #
        # a_contour_points_path = contour_points_path_list[cur_frame+1]
        # contour_points = np.loadtxt(a_contour_points_path)
        # contour_points = contour_points.astype('int32')
        # contour_points = contour_points[first_contour_index:last_contour_index,:]
        # a_color = np.array([contour_cmap(1. * (cur_frame-initial_frame) / total_frames)])
        # for point_index, a_point in enumerate(contour_points):
        #     a_col, a_row = a_point
        #     plt.scatter(x=a_col, y=a_row, c=a_color, s=1, marker='s',linewidths=1, antialiased=True)

        if not arrow_plot:
            a_image = plt.imread(img_path_list[cur_frame +1])
            plt.imshow(a_image, cmap='gray', vmin=0, vmax=1)
            plt.axis('off')

        # -----------------------------------------------------------------------------------
        for a_point_index in range(pred_tracking_points_np.shape[1]):
            if arrow_plot:
                cur_x, cur_y = gt_tracking_points_np[cur_frame, a_point_index, :]
                if cur_frame == 0:
                    prev_x, prev_y = first_gt_tracking_points_np[a_point_index, :]
                else:
                    prev_x, prev_y = gt_tracking_points_np[cur_frame -1, a_point_index, :]
                draw_arrow_plot(cur_x, cur_y, prev_x, prev_y, arrow_color=np.array(gt_cmap(1. * (cur_frame -initial_frame) / total_frames)))

                cur_x, cur_y = Matlab_GT_tracking_points_np[cur_frame, a_point_index, :]
                if cur_frame > 0:
                    prev_x, prev_y = Matlab_GT_tracking_points_np[cur_frame -1, a_point_index, :]
                draw_arrow_plot(cur_x, cur_y, prev_x, prev_y, arrow_color= np.array(matlab_cmap(1. * (cur_frame -initial_frame) / total_frames)))

                cur_x, cur_y = pred_tracking_points_np[cur_frame, a_point_index, :]
                if cur_frame > 0:
                    prev_x, prev_y = pred_tracking_points_np[cur_frame -1, a_point_index, :]
                draw_arrow_plot(cur_x, cur_y, prev_x, prev_y, arrow_color=np.array(pred_cmap(1. * (cur_frame -initial_frame) / total_frames)) )
            else:
                cur_x, cur_y = Matlab_GT_tracking_points_np[cur_frame, a_point_index, :]
                plt.scatter(x=cur_x, y=cur_y, c='b', s=1, marker='o', antialiased=True)
                cur_x, cur_y = pred_tracking_points_np[cur_frame, a_point_index, :]
                plt.scatter(x=cur_x, y=cur_y, c='g', s=1, marker='o', antialiased=True)
                cur_x, cur_y = gt_tracking_points_np[cur_frame, a_point_index, :]
                plt.scatter(x=cur_x, y=cur_y, c='r', s=1, marker='o', antialiased=True)

        if not arrow_plot:
            fig = plt.gcf()
            # fig.set_size_inches(10, 10)
            plt.savefig(f"{plot_dir}/{save_name}/gt_mech_ours_{cur_frame}.png", bbox_inches="tight", pad_inches=0, dpi=300)
            plt.close()

    if arrow_plot:
        fig = plt.gcf()
        fig.set_size_inches(10, 10)
        plt.savefig(f"{plot_dir}/{save_name}/all_frames_trajectory_{initial_frame}_{final_frame}.png", bbox_inches="tight", pad_inches=0, dpi=400)
        plt.close()


def manuscript_figure4_no_GT(root_generated_path, dataset_folder, img_path_list, contour_points_path_list, pred_tracking_points_contour_indices, Matlab_GT_tracking_points_path_list):
    '''
    load my deep learning model's predicted points
    load contour points
    load images

    :param dataset_folder:
    :return:
    '''

    # ----------------------------------- Process my prediction -----------------------------------------

    pred_cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["greenyellow", "green"])
    matlab_cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["skyblue", "blue"])
    contour_cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["dimgrey", "snow"])

    total_frames = 5
    for initial_frame in tqdm(range(40 - total_frames)):
        final_frame = total_frames + initial_frame

        init_contour_points = np.loadtxt(contour_points_path_list[initial_frame])
        init_contour_points = init_contour_points.astype('int32')

        # create a dict of points to track
        initial_tracked_contour_point_indices_dict = {}
        max_contour_points = init_contour_points.shape[0]
        if init_contour_points.shape[0] > 1640:
            max_contour_points = 1640
        for a_contour_index in range(0, max_contour_points, 3):
            initial_tracked_contour_point_indices_dict[a_contour_index] = []

        pred_tracking_points_np = np.zeros(shape=(total_frames, len(initial_tracked_contour_point_indices_dict.keys()), 2))

        # create prediction np from dense correspondences
        for cur_frame in range(total_frames):
            a_contour_points_path = contour_points_path_list[initial_frame + cur_frame +1]
            contour_points = np.loadtxt(a_contour_points_path)
            contour_points = contour_points.astype('int32')

            frame_pred_tracking_points_contour_indices = pred_tracking_points_contour_indices[initial_frame + cur_frame, :]

            for for_index, a_tracked_contour_point_index in enumerate(initial_tracked_contour_point_indices_dict.keys()):
                if cur_frame > 0:
                    prev_tracked_contour_point_index = initial_tracked_contour_point_indices_dict[a_tracked_contour_point_index][-1]
                else:
                    prev_tracked_contour_point_index = a_tracked_contour_point_index

                next_tracked_contour_point_index = frame_pred_tracking_points_contour_indices[prev_tracked_contour_point_index]
                initial_tracked_contour_point_indices_dict[a_tracked_contour_point_index].append(next_tracked_contour_point_index)
                pred_tracking_points_np[cur_frame, for_index, :] = contour_points[next_tracked_contour_point_index]

        # --------------------------------------------- load MATLAB prediction data ---------------------------------------------
        # Matlab_GT_tracking_points_all_frames = []
        # selected_matlab_GT_tracking_points_indices = []
        # for a_index, (matlab_GT_tracking_points_path, contour_points_path, image_path) in enumerate( zip(Matlab_GT_tracking_points_path_list, contour_points_path_list, img_path_list)):
        #
        #     matlab_GT_tracking_points = np.loadtxt(matlab_GT_tracking_points_path)
        #     matlab_GT_tracking_points = matlab_GT_tracking_points.astype('int32')
        #
        #     contour_points = np.loadtxt(contour_points_path)
        #     contour_points = contour_points.astype('int32')
        #
        #     # -------------------------------------------------------------------------------
        #     if a_index == 0:
        #
        #         for a_tracked_contour_point_index in initial_tracked_contour_point_indices_dict.keys():
        #             a_contour_point = contour_points[a_tracked_contour_point_index]
        #
        #             # find corresponding index of Matlab tracking points for every tracked contour points
        #             min_dist = None
        #             match_index = None
        #             for matlab_index, matlab_gt_point in enumerate(matlab_GT_tracking_points):
        #                 a_dist = np.linalg.norm(matlab_gt_point - a_contour_point)
        #                 if min_dist is None or min_dist > a_dist:
        #                     min_dist = a_dist
        #                     match_index = matlab_index
        #
        #             selected_matlab_GT_tracking_points_indices.append(match_index)
        #         assert len(selected_matlab_GT_tracking_points_indices) == len(initial_tracked_contour_point_indices_dict.keys())
        #
        #     elif a_index > 0:
        #
        #         selected_matlab_GT_tracking_points = matlab_GT_tracking_points[selected_matlab_GT_tracking_points_indices]
        #         Matlab_GT_tracking_points_all_frames.append(selected_matlab_GT_tracking_points)
        #
        # Matlab_GT_tracking_points_np = np.asarray(Matlab_GT_tracking_points_all_frames)

        # -----------------------------------------------------------------------------------
        # --------------------------- Visualize points --------------------------------------
        # -----------------------------------------------------------------------------------

        # sort points
        # sort_index = np.argsort(Matlab_GT_tracking_points_np[0,:,0], axis=0)
        # Matlab_GT_tracking_points_np = Matlab_GT_tracking_points_np[:,sort_index]

        sort_index = np.argsort(pred_tracking_points_np[0 ,: ,0], axis=0)
        pred_tracking_points_np = pred_tracking_points_np[:, sort_index]

        # sort_index = np.argsort(init_contour_points[:,0], axis=0)
        contour_point_indices = list(initial_tracked_contour_point_indices_dict.keys())
        init_contour_points_np = init_contour_points[contour_point_indices, :]
        init_contour_points_np = init_contour_points_np[sort_index]

        # assert pred_tracking_points_np.shape == Matlab_GT_tracking_points_np.shape

        first_contour_index = 0
        last_contour_index = 1000

        a_image = plt.imread(img_path_list[final_frame])
        plt.imshow(a_image, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')

        plot_dir = f"{root_generated_path}{dataset_folder}/"
        save_name = 'manuscript_figure4_no_GT'

        # prepare save folder
        if not os.path.exists(f"{plot_dir}/{save_name}/"):
            os.makedirs(f"{plot_dir}/{save_name}/")

        for cur_frame in range(total_frames):
            # Draw contours of multiple frames in one image
            # if cur_frame == initial_frame:
            #     a_contour_points_path = contour_points_path_list[cur_frame]
            #     contour_points = np.loadtxt(a_contour_points_path)
            #     contour_points = contour_points.astype('int32')
            #     contour_points = contour_points[first_contour_index:last_contour_index, :]
            #     a_color = np.array([contour_cmap(1. * (cur_frame - initial_frame) / total_frames)])
            #     for point_index, a_point in enumerate(contour_points):
            #         a_col, a_row = a_point
            #         plt.scatter(x=a_col, y=a_row, c=a_color, s=1, marker='s', linewidths=1, antialiased=True)

            # a_contour_points_path = contour_points_path_list[initial_frame+cur_frame+1]
            # contour_points = np.loadtxt(a_contour_points_path)
            # contour_points = contour_points.astype('int32')
            # contour_points = contour_points[first_contour_index:last_contour_index,:]
            # a_color = np.array([contour_cmap(1. * (cur_frame-initial_frame) / total_frames)])
            # for point_index, a_point in enumerate(contour_points):
            #     a_col, a_row = a_point
            #     plt.scatter(x=a_col, y=a_row, c=a_color, s=1, marker='s',linewidths=1, antialiased=True)

            # -----------------------------------------------------------------------------------
            for a_point_index in range(pred_tracking_points_np.shape[1]):
                cur_x, cur_y = pred_tracking_points_np[cur_frame, a_point_index, :]
                if cur_frame == 0:
                    prev_x, prev_y = init_contour_points_np[a_point_index, :]
                else:
                    prev_x, prev_y = pred_tracking_points_np[cur_frame -1, a_point_index, :]
                draw_arrow_plot(cur_x, cur_y, prev_x, prev_y, arrow_color=np.array(pred_cmap(1. * (cur_frame) / total_frames)))

                # cur_x, cur_y = Matlab_GT_tracking_points_np[cur_frame, a_point_index, :]
                # if cur_frame > 0:
                #     prev_x, prev_y = Matlab_GT_tracking_points_np[cur_frame-1, a_point_index, :]
                # draw_arrow_plot(cur_x, cur_y, prev_x, prev_y, arrow_color= np.array(matlab_cmap(1. * (cur_frame) / total_frames)))

        fig = plt.gcf()
        fig.set_size_inches(10, 10)
        plt.savefig(f"{plot_dir}/{save_name}/trajectory_{initial_frame}_{final_frame}.png", bbox_inches="tight", pad_inches=0, dpi=400)
        plt.close()


# --------------------------------

def rainbow_contour_pred_only(root_generated_path, dataset_folder, img_path_list, contour_points_path_list, pred_tracking_points_contour_indices):
    '''
    load my deep learning model's predicted points
    load contour points
    load images

    :param dataset_folder:
    :return:
    '''

    # ----------------------------------- Process my prediction -----------------------------------------

    pred_cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["greenyellow", "green"])
    matlab_cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["skyblue", "blue"])
    contour_cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["dimgrey", "snow"])

    total_frames = 1
    for initial_frame in tqdm(range(40 - total_frames)):
        final_frame = total_frames + initial_frame

        init_contour_points = np.loadtxt(contour_points_path_list[initial_frame])
        init_contour_points = init_contour_points.astype('int32')

        # create a dict of points to track
        initial_tracked_contour_point_indices_dict = {}
        max_contour_points = init_contour_points.shape[0]
        if init_contour_points.shape[0] > 1640:
            max_contour_points = 1640
        for a_contour_index in range(0, max_contour_points, 3):
            initial_tracked_contour_point_indices_dict[a_contour_index] = []

        pred_tracking_points_np = np.zeros(shape=(total_frames, len(initial_tracked_contour_point_indices_dict.keys()), 2))

        # create prediction np from dense correspondences
        for cur_frame in range(total_frames):
            a_contour_points_path = contour_points_path_list[initial_frame + cur_frame +1]
            contour_points = np.loadtxt(a_contour_points_path)
            contour_points = contour_points.astype('int32')

            frame_pred_tracking_points_contour_indices = pred_tracking_points_contour_indices[initial_frame + cur_frame, :]

            for for_index, a_tracked_contour_point_index in enumerate(initial_tracked_contour_point_indices_dict.keys()):
                if cur_frame > 0:
                    prev_tracked_contour_point_index = initial_tracked_contour_point_indices_dict[a_tracked_contour_point_index][-1]
                else:
                    prev_tracked_contour_point_index = a_tracked_contour_point_index

                next_tracked_contour_point_index = frame_pred_tracking_points_contour_indices[prev_tracked_contour_point_index]
                initial_tracked_contour_point_indices_dict[a_tracked_contour_point_index].append(next_tracked_contour_point_index)
                pred_tracking_points_np[cur_frame, for_index, :] = contour_points[next_tracked_contour_point_index]

      
        # -----------------------------------------------------------------------------------
        # --------------------------- Visualize points --------------------------------------
        # -----------------------------------------------------------------------------------

        # sort points
        sort_index = np.argsort(pred_tracking_points_np[0 ,: ,0], axis=0)
        pred_tracking_points_np = pred_tracking_points_np[:, sort_index]

        # sort_index = np.argsort(init_contour_points[:,0], axis=0)
        contour_point_indices = list(initial_tracked_contour_point_indices_dict.keys())
        init_contour_points_np = init_contour_points[contour_point_indices, :]
        init_contour_points_np = init_contour_points_np[sort_index]

        a_image = plt.imread(img_path_list[final_frame])
        plt.imshow(a_image, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')

        plot_dir = f"{root_generated_path}{dataset_folder}/"
        save_name = 'rainbow_contour_pred_only_figure'
        cm = pylab.get_cmap('gist_rainbow')

        # prepare save folder
        if not os.path.exists(f"{plot_dir}/{save_name}/"):
            os.makedirs(f"{plot_dir}/{save_name}/")

        first_contour_index = 0
        last_contour_index = 1000

        for cur_frame in range(total_frames):
            # Draw contours of multiple frames in one image
            if cur_frame == 0:
                a_contour_points_path = contour_points_path_list[initial_frame+cur_frame]
                contour_points = np.loadtxt(a_contour_points_path)
                contour_points = contour_points.astype('int32')
                contour_points = contour_points[first_contour_index:last_contour_index, :]
                a_color = np.array([contour_cmap(1. * (cur_frame) / total_frames)])
                for point_index, a_point in enumerate(contour_points):
                    a_col, a_row = a_point
                    plt.scatter(x=a_col, y=a_row, c=a_color, s=1, marker='s', linewidths=1, antialiased=True)

            # a_contour_points_path = contour_points_path_list[initial_frame+cur_frame+1]
            # contour_points = np.loadtxt(a_contour_points_path)
            # contour_points = contour_points.astype('int32')
            # contour_points = contour_points[first_contour_index:last_contour_index,:]
            # a_color = np.array([contour_cmap(1. * (cur_frame+1) / total_frames)])
            # for point_index, a_point in enumerate(contour_points):
            #     a_col, a_row = a_point
            #     plt.scatter(x=a_col, y=a_row, c=a_color, s=1, marker='s',linewidths=1, antialiased=True)

            # -----------------------------------------------------------------------------------
            if cur_frame == (total_frames-1):
                for a_point_index in range(pred_tracking_points_np.shape[1]):
                    cur_x, cur_y = pred_tracking_points_np[cur_frame, a_point_index, :]
                    if cur_frame == 0:
                        prev_x, prev_y = init_contour_points_np[a_point_index, :]
                    else:
                        prev_x, prev_y = pred_tracking_points_np[cur_frame -1, a_point_index, :]

                    # -----
                    # a_color = np.array([cm(1. * a_point_index / pred_tracking_points_np.shape[1])])
                    # plt.scatter(x=cur_x, y=cur_y, c=a_color, s=1, marker='s',linewidths=1, antialiased=True)
                    # plt.scatter(x=prev_x, y=prev_y, c=a_color, s=1, marker='s',linewidths=1, antialiased=True)
                    # ------
                    draw_arrow_plot(cur_x, cur_y, prev_x, prev_y, arrow_color=np.array(pred_cmap(1. * (cur_frame) / total_frames)))

        fig = plt.gcf()
        fig.set_size_inches(10, 10)
        plt.savefig(f"{plot_dir}/{save_name}/trajectory_{initial_frame}_{final_frame}.png", bbox_inches="tight", pad_inches=0, dpi=400)
        plt.close()



# -----------------------------------------------------------------------------------------------------

def manuscript_figure5(root_generated_path, dataset_folder, img_path_list, contour_points_path_list, pred_tracking_points_contour_indices):
    '''

    Draw trajectory arrows for all contour points between two end-points
    Draw morphodynamics protrusion map

    :param dataset_folder:
    :param contour_points_path_list:
    :param pred_tracking_points_contour_indices:
    :return:
    '''
    plot_dir = f"{root_generated_path}{dataset_folder}/"
    save_name = 'manuscript_figure5'

    # prepare save folder
    if not os.path.exists(f"{plot_dir}/{save_name}/"):
        os.makedirs(f"{plot_dir}/{save_name}/")

    # -------------------------- Combine many predictions -----------------------
    tracking_point_cmap = pylab.get_cmap('gist_rainbow')
    initial_frame = 0
    final_frame = 4
    total_frames = final_frame - initial_frame

    plt.rcParams["font.family"] = "Times New Roman"
    a_image = plt.imread(img_path_list[initial_frame])
    plt.imshow(a_image, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    # plt.xticks(np.arange(0,total_frames + 1, 5), fontsize=16)
    # plt.yticks(fontsize=16)
    # plt.xlabel('Number of Frame')
    # plt.ylabel('Index of Tracked Point')

    # ----------------------- initialize -----------------------
    contour_points = np.loadtxt(contour_points_path_list[0])
    contour_points = contour_points.astype('int32')
    initial_tracked_contour_point_indices_dict = {}
    # choose the range of contour points to track here
    for a_index in range(contour_points.shape[0]):
        if a_index > 375 and a_index < 635:
            initial_tracked_contour_point_indices_dict[a_index] = []
    # initial_tracked_contour_point_indices_dict = {740:[],746:[],750:[],755:[],760:[],765:[],770:[],775:[],780:[],785:[], 790:[],795:[],796:[]}
    heatmap_np = np.zeros(shape=(len(initial_tracked_contour_point_indices_dict), total_frames) )

    for cur_frame in tqdm(range(initial_frame, final_frame+1)):
        # Draw contours of multiple frames in one image
        a_contour_points_path = contour_points_path_list[cur_frame]
        contour_points = np.loadtxt(a_contour_points_path)
        contour_points = contour_points.astype('int32')

        if cur_frame > initial_frame:
            # calculate normal vector of contour points
            # Compute tangent by central differences. You can use a closed form tangent if you have it.
            # referred https://stackoverflow.com/questions/66676502/how-can-i-move-points-along-the-normal-vector-to-a-curve-in-python
            two_left_contour_points = np.roll(prev_contour_points, shift=-2, axis=0)
            tangent_vectors = (two_left_contour_points - prev_contour_points) / 2

            normal_vectors = np.stack([-tangent_vectors[:-2, 1], tangent_vectors[:-2, 0]], axis=-1)  # ignore last two indices due to shift above
            # repeated_normal_vectors = tf.repeat( normal_vectors, pred_offsets.shape[0], axis=0)
            unit_normal_vectors = normal_vectors / (np.expand_dims(np.linalg.norm(normal_vectors, axis=-1), -1) + 0.0000001)
            # put back first and last index that were shifted
            unit_normal_vectors = np.append(unit_normal_vectors, [[0,0]], axis=0)
            unit_normal_vectors = np.insert(unit_normal_vectors, 0, [0,0], axis=0)

            # plot normal vectors for verification
            # if cur_frame == initial_frame:
            #     a_contour_points_path = contour_points_path_list[cur_frame]
            #     contour_points = np.loadtxt(a_contour_points_path)
            #     contour_points = contour_points.astype('int32')
            #     for point_index, a_point in enumerate(contour_points):
            #         a_color = np.array([tracking_point_cmap(1. * point_index / len(contour_points))])
            #         a_col, a_row = a_point
            #         # plt.scatter(x=a_col, y=a_row, c=a_color, s=1, marker='s', linewidths=1, antialiased=True)
            #         delta_x, delta_y = unit_normal_vectors[point_index,:]
            #         draw_arrow_plot(a_col+delta_x, a_row+delta_y, a_col, a_row, arrow_color=a_color )

        frame_pred_tracking_points_contour_indices = pred_tracking_points_contour_indices[cur_frame - 1, :]

        for for_index, a_tracked_contour_point_index in enumerate(initial_tracked_contour_point_indices_dict.keys()):

            if cur_frame > initial_frame:
                prev_tracked_contour_point_index = initial_tracked_contour_point_indices_dict[a_tracked_contour_point_index][-1]
            else:
                prev_tracked_contour_point_index = a_tracked_contour_point_index

            next_tracked_contour_point_index = frame_pred_tracking_points_contour_indices[prev_tracked_contour_point_index]
            initial_tracked_contour_point_indices_dict[a_tracked_contour_point_index].append(next_tracked_contour_point_index)

            # calculate velocity with respect to the normal vector
            if cur_frame > initial_frame:
                cur_contour_point_index = initial_tracked_contour_point_indices_dict[a_tracked_contour_point_index][-1]
                prev_contour_point_index = initial_tracked_contour_point_indices_dict[a_tracked_contour_point_index][-2]
                cur_x, cur_y = contour_points[cur_contour_point_index]
                prev_x, prev_y = prev_contour_points[prev_contour_point_index]

                delta_x = cur_x - prev_x
                delta_y = cur_y - prev_y

                if for_index % 4 == 0:
                    draw_arrow_plot(cur_x, cur_y, prev_x, prev_y, arrow_color=np.array([tracking_point_cmap(1. * for_index / len(initial_tracked_contour_point_indices_dict.keys()))]))

                # normalize delta
                # a_norm = np.linalg.norm([delta_x, delta_y])
                # norm_delta_x = delta_x / (a_norm  + 0.0000001)
                # norm_delta_y = delta_y / (a_norm  + 0.0000001)

                delta_along_normal_vector = np.dot([delta_x, delta_y] , unit_normal_vectors[prev_contour_point_index])

                cutoff = 5
                delta_along_normal_vector = np.clip(delta_along_normal_vector, -cutoff, cutoff)

                heatmap_np[for_index, cur_frame-1] = delta_along_normal_vector # protrusion/retraction velocity

        prev_contour_points = contour_points


    # protrusion_map_np = load_protrusion_colormap()
    protrusion_map_cmap = sns.color_palette("coolwarm", as_cmap=True) # ListedColormap(protrusion_map_np)
    # plt.imshow(heatmap_np, cmap=protrusion_map_cmap, interpolation='nearest', aspect='auto')

    # cbar = plt.colorbar()
    # cbar.ax.tick_params(labelsize=16)

    fig = plt.gcf()
    fig.set_size_inches(20, 15)
    plt.savefig(f"{plot_dir}/{save_name}/morphodynamics_{initial_frame}_{final_frame}.png", bbox_inches="tight", pad_inches=0, dpi=400)
    plt.close()


def rebuttal_error_study(root_generated_path):
    test1_normal = [0.447, 1.342, 3.0, 0.0, 1.0, -1.342, 3.13, -1.0, -4.472, 35.0, -1.0, 1.0, -0.447, -4.95, 2.0, 0.0, 1.789, 0.894, -4.919, 12.522, 0.447, 2.828, 0.447, -3.536, 11.314, 0.447, 3.0, -1.342, -4.95, 8.497, 1.789, 2.683, -2.0, -4.025, 9.0, 2.236, 1.789, -2.0, -2.828, 4.243, 4.025, 0.894, -1.0, -2.121, 4.472, 8.05, -0.894, -4.95, -0.894, 5.0, 1.342, -0.447, -3.0, -2.828, 6.0, -4.0, 6.0, -3.578, -2.828, 10.733, -6.0, 3.13, -5.367, -2.683, 8.497, -6.708, 2.683, -5.367, -4.025, -2.0, -2.121, 4.472, -7.603, -1.0, -0.894, 0.0, 4.243, -9.0, 4.95, -21.466, -0.707, 2.828, -11.628, 0.0, -19.0, 1.414, 4.0, -11.0, 2.0, -29.0, 1.414, 0.894, -5.0, 3.0, -48.746, 0.707, 5.814, -9.839, 9.899, 1.414, 2.236, 0.0, -7.071, 12.021, -1.0, 1.789, -1.342, -8.05, 9.391, 4.0, -0.707, 1.789, -2.683, 12.021, -2.0, 1.342, 5.367, -4.025, 4.0, -1.342, -0.707, -3.0, -1.342, 12.021, 2.121, 0.0, 1.342, 4.919, 10.286, 7.0, 0.447, 6.364, 1.789, 11.18, 4.472, 0.447, 1.0, 5.814, 14.311, 3.578, 0.447, -1.342, -0.894, 4.95, 1.789, 0.447, 1.0, -0.894, 2.236, 2.0, 0.0, -1.0, 0.894, 6.364, 0.0, 0.447, 2.0, 12.0, 8.05, -0.447, -0.447, 1.0, 8.0, 5.367, -2.121, -0.894, 0.0, 6.708, 5.814, -2.121, 1.414, 0.0, 0.894, 2.236, -2.121, 0.447, -1.0, 1.342, 4.919, -3.578, -0.894, -3.0, -3.0, -0.894, -2.121, 0.707, -6.0, 8.0, -6.0, -7.155, -0.447, 0.0, 11.0, 3.578, -4.243, 0.447, 6.0, 7.155, -15.205, -3.536]
    test1_SA = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0]
    test2_normal = [0.447, 0.0, 2.0, 0.0, 0.0, 1.789, 1.789, 2.0, -0.707, 1.0, 6.0, -2.0, 3.578, -0.894, 1.789, 2.0, 0.0, 3.0, -0.447, 2.0, 0.0, 4.025, 4.0, 0.447, 2.683, 0.0, 3.0, 2.236, 2.121, 1.414, 2.236, -1.414, 0.894, -4.0, 3.13, 4.025, -2.121, 4.0, 2.121, 4.025, 3.13, -0.894, 1.342, -1.342, 2.236, 1.0, 2.0, -0.447, -12.728, 3.13, 0.0, 3.0, 0.447, -15.652, 4.0, -0.707, 2.236, 3.0, 7.603, 2.683, 0.447, 1.342, -2.683, -6.708, 0.707, 1.414, 1.0, 0.0, -4.025, 2.121, 2.121, 0.0, 0.0, -2.683, 2.121, 2.0, 1.414, -5.0, -3.578, -1.414, 2.236, -0.447, -4.025, 7.155, -2.828, 0.707, 0.0, -4.243, 4.243, 2.683, 3.536, -2.0, -5.0, 4.95, 0.894, 8.05, -1.342, -16.0, 7.155, 3.536, 2.0, -1.0, 5.814, 7.603, 1.342, 3.0, 1.0, -4.243, 5.657, 0.894, -1.0, 0.894, -0.447, 6.708, -1.342, 0.894, -0.447, 20.125, 4.472, -1.414, -5.0, 0.0, 21.92, 3.578, 0.0, -6.0, 0.0, 16.971, 3.13, 2.121, -2.236, -2.683, 12.969, 0.0, 0.0, -0.894, -1.0, 8.944, -2.0, -2.828, 5.367, 3.0, 5.367, -1.0, -0.707, 3.578, 0.0, 1.414, 4.025, 1.342, 4.472, -0.894, -3.578, 1.0, -0.894, 7.155, -2.828, 6.261, -3.0, -3.13, -1.0, 2.0, -7.0, 0.0, -15.0, 6.261, 0.0, 4.243, -1.789, 0.0, -2.0, 2.0, 1.414, -0.707, -4.472, 0.894, 0.0, -2.121, 0.894, 4.472, -2.236, 4.919, -5.814, 0.447, 2.683, -2.236, 4.0, -0.894, 2.683, -3.578, -1.789, -2.683, 2.236, -1.789, -8.497, 0.894, -1.342, 6.261, -1.789, -10.0]
    test2_SA = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1]
    test3_normal = [-5.367, 4.472, 5.0, -1.0, 0.894, -0.447, -6.261, 4.0, -2.828, -4.025, 7.778, 0.894, -0.894, -1.109, 0.447, 6.708, -2.0, 2.0, -2.0, -2.121, -3.13, -4.0, 0.0, -1.0, -8.497, 0.0, -6.0, 3.0, 0.0, 0.447, -1.342, -6.0, 0.0, 2.236, 8.497, 0.707, -6.0, 1.0, -4.919, 7.155, 0.0, -8.0, 0.447, -4.472, -8.222, -0.707, -5.0, -2.0, 0.447, -0.316, -1.414, -7.0, 0.0, 2.0, -3.0, -1.664, -5.0, 0.0, 1.0, 2.496, 8.05, 1.0, -5.0, 1.342, -2.0, 4.0, 5.376, -3.578, -2.0, 5.367, 4.919, 11.068, -1.0, 3.536, 2.121, 9.839, 14.0, 4.472, -0.707, 7.0, 5.367, 13.864, 4.919, 2.236, -1.789, -7.603, 0.277, 4.715, 5.367, 7.778, -6.261, 0.0, -4.95, 5.814, -10.0, -2.683, -0.447, -3.0, 1.0, 1.342, -3.578, 5.0, 0.0, 1.0, 9.391, -0.707, 0.447, 1.414, 2.0, -13.0, 2.236, -8.05, -1.789, 4.0, 9.391, -4.243, 5.06, -0.894, 10.733, 5.0, 9.803, 8.944, 2.683, 5.0, -6.261, 3.578, 4.0, 2.236, -1.0, -16.994, -12.0, 1.0, -0.894, 3.578, -7.603, 1.789, -8.05, 2.683, 1.0, 2.0, -2.0, -13.416, 4.025, 2.0, 5.367, -2.683, -15.205, 3.13, 1.0, 8.0, -4.025, -10.286, 10.733, 2.0, 15.652, 1.789, 1.0, -5.367, 4.0, 7.603, 2.828, -1.789, 1.789, 15.205, 6.261, 11.628, 9.0, 0.447, 5.0, 0.0, 3.13, 19.092, -4.0, 7.0, -24.15, 0.0, 21.187, -4.0, -2.236, -25.044, -5.814, 33.941, -11.628, -0.894, -17.0, -8.485, -32.0, -5.0, 7.0, -5.367, -16.547, -33.0, -12.522, 12.969, 2.121, -9.192, -18.336, -13.864, 16.1, 3.536]
    test3_SA = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    test4_normal = [1.0, -1.789, -1.414, -3.13, 0.0, -4.919, -0.447, 0.447, -2.236, -5.0, -4.243, -0.447, -13.435, -5.0, 7.155, 0.0, -0.707, -0.447, -1.414, -19.677, 1.789, -3.0, -0.707, 0.447, 2.683, 1.789, -2.683, 0.0, -1.0, -2.828, -1.342, 0.707, 6.261, 1.0, 0.894, 3.0, 1.414, -2.0, 4.0, 5.0, 2.121, -9.0, 1.342, 0.0, -6.0, -8.0, -3.13, 7.071, -6.261, 0.447, 0.447, -4.025, 0.894, -2.0, 0.447, -7.155, -4.95, 0.447, 2.683, -1.0, -11.628, 2.828, -4.919, 1.789, -0.447, -2.0, -3.13, -12.522, 3.13, -6.0, 4.0, -3.536, -6.708, 1.0, -12.021, -1.342, -0.447, 8.05, -1.342, -6.708, -3.0, -3.578, 1.342, 5.657, 2.683, -0.894, -6.261, -1.0, 0.0, -1.414, 7.603, 4.95, -2.121, -6.261, 1.342, 0.0, -4.243, 4.243, 1.342, -2.0, -4.243, 1.265, 1.342, -6.708, 5.367, -1.789, 0.0, 0.707, -5.0, 3.0, -6.708, 0.894, -4.472, 1.414, -2.0, -0.447, 2.0, -8.05, 0.0, 2.0, 4.243, 6.708, 6.708, 5.657, -8.497, -1.0, 0.0, 1.0, 7.0, 3.13, -3.536, -0.894, -2.0, 1.0, 1.789, 6.708, 3.536, -6.261, -0.447, -1.789, -4.0, 2.0, -7.603, 7.603, 3.0, 1.342, 3.0, -2.236, 5.0, -6.0, 1.789, 2.0, 1.789, 0.894, -2.236, -2.0, -5.0, -2.121, -3.0, 5.367, 0.0, -2.236, 4.919, -14.311, -4.025, -4.0, 4.0, -2.236, -2.236, 7.071, -15.205, -1.789, 8.485, 4.919, 0.707, 1.789, 7.071, -10.0, -1.0, 1.342, 10.286, 8.0, 3.13, 8.944, -10.0, -1.0, 6.0, 7.778, 5.0, 4.0, 2.121, -5.0, 0.447, 6.0, 1.414, 0.0, -3.0, -3.578, 1.789, 1.0, 3.0, 0.447, 0.0, 4.472, 2.683, 4.919, -0.447, -2.0, -3.536, -1.789, 3.536, -4.95, 8.944, 0.0, 5.814, 2.683, -0.707, 5.367, -8.485, 6.0, 0.0, -4.0, 4.025, -1.342, -2.0, -5.814, 8.0, 0.447, -3.0, -4.243, 0.447, 0.0, 1.0, 3.13, 1.789, -5.0, -4.95, 0.0, -4.472, 7.155, 0.894, 4.95, -5.0, -12.075, 1.0, 0.0, 8.497, 0.0, 1.342, 7.778, 0.0, 0.0, 2.236, -4.243, 1.789, 1.0, -7.0, -18.0, -1.0, -4.243, -7.155, 0.447, 0.447, -12.0, -28.622, -1.342, -5.657, -10.733, -0.447, 3.578, -2.683, -32.527, 1.0, -7.778, -9.839, -2.683, 0.894, -4.472, -35.777, -1.789]
    test4_SA = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]

    # offset_frames = 0
    # test_normal = test1_normal[offset_frames:] + test2_normal[offset_frames:] + test3_normal[offset_frames:] + test4_normal[offset_frames:]
    # test_SA = test1_SA[offset_frames:] + test2_SA[offset_frames:] + test3_SA[offset_frames:] + test4_SA[offset_frames:]

    def analyze_normal_SA(test_normal, test_SA):
        test_normal = np.array(test_normal)
        test_SA = np.array(test_SA)

        print('----------------------------------------------')
        print('count positive normal', np.sum((test_normal > 0)*1))
        print('count negative normal', np.sum((test_normal < 0)*1))
        # protrusion/expansion
        expansion_mean = np.mean(test_SA[test_normal > 0])
        small_expansion_mean = np.mean(test_SA[(test_normal > 0)*(test_normal<1)])

        # retraction/contraction
        contraction_mean = np.mean(test_SA[test_normal < 0])
        small_contract_mean = np.mean(test_SA[(test_normal < 0)*(test_normal>-1)])

        abs_test_normal = np.abs(test_normal)

        still_mean = np.mean(test_SA[abs_test_normal == 0])
        print('still_mean', still_mean)
        for a_offset in range(9):
            # print('offset', a_offset, 'mean:', np.mean(test_SA[ (abs_test_normal <= (a_offset+1) ) * (abs_test_normal > (a_offset) ) ]))
            print('offset', a_offset, 'expansion mean:', np.mean(test_SA[ (test_normal <= (a_offset+1) ) * (test_normal > 0 ) ]))
        for a_offset in range(9):
            a_offset = a_offset*-1
            # print('offset', a_offset, 'mean:', np.mean(test_SA[ (abs_test_normal <= (a_offset+1) ) * (abs_test_normal > (a_offset) ) ]))
            print('offset', a_offset, 'contraction mean:', np.mean(test_SA[ (test_normal >= (a_offset-1) ) * (test_normal < 0 ) ]))
        for a_offset in range(9):
            # print('offset', a_offset, 'mean:', np.mean(test_SA[ (abs_test_normal <= (a_offset+1) ) * (abs_test_normal > (a_offset) ) ]))
            print('offset', a_offset, 'abs mean:', np.mean(test_SA[ (abs_test_normal <= (a_offset+1) ) ]))


        print('expansion_mean', expansion_mean)
        print('contraction_mean', contraction_mean)
        print('----------------------------------------------')

        return expansion_mean, contraction_mean

    # expansion_mean1, contraction_mean1 = analyze_normal_SA(test1_normal, test1_SA)
    # expansion_mean2, contraction_mean2 = analyze_normal_SA(test2_normal, test2_SA)
    # expansion_mean3, contraction_mean3 = analyze_normal_SA(test3_normal, test3_SA)
    # expansion_mean4, contraction_mean4 = analyze_normal_SA(test4_normal, test4_SA)
    # print( 'Total Average Expansion', (expansion_mean1 + expansion_mean2 + expansion_mean3 + expansion_mean4) / 4 )
    # print( 'Total Average Contraction', (contraction_mean1 + contraction_mean2 + contraction_mean3 + contraction_mean4) / 4 )

    test_normal = test1_normal + test2_normal + test3_normal + test4_normal
    test_SA = test1_SA + test2_SA + test3_SA + test4_SA
    expansion_mean, contraction_mean = analyze_normal_SA(test_normal, test_SA)


def rebuttal_labeling_figure(root_generated_path, dataset_folder, img_path_list, contour_points_path_list):
    '''
    Draw overlaid contour points for 6 frame duration. e.g. 5,6,7,8,9,10 frame

    :param root_generated_path:
    :param dataset_folder:
    :param img_path_list:
    :param contour_points_path_list:
    :return:
    '''

    plot_dir = f"{root_generated_path}{dataset_folder}/"
    save_name = 'rebuttal_labeling_figure'

    # prepare save folder
    if not os.path.exists(f"{plot_dir}/{save_name}/"):
        os.makedirs(f"{plot_dir}/{save_name}/")

    initial_frame = 84
    final_frame = 110
    frame_duration = 6

    plt.rcParams["font.family"] = "Times New Roman"
    # lightgreen rgb(173,255,47)
    # darkgreen (0,100/255,0)
    # greenyellow (173/255,255/255,47/255)
    contour_cmap = mpl.colors.LinearSegmentedColormap.from_list("", [(173/255,255/255,47/255), (0, 50/255,0)])
    # contour_cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["dimgrey", "snow"])

    for cur_frame in tqdm(range(initial_frame, final_frame-frame_duration)):
        # Draw contours of multiple frames in one image
        for frame_offset in range(frame_duration):
            if frame_offset == 0:

                a_image = plt.imread(img_path_list[cur_frame])
                plt.imshow(a_image, cmap='gray')
                plt.axis('off')

            a_contour_points_path = contour_points_path_list[cur_frame+frame_offset]
            contour_points = np.loadtxt(a_contour_points_path)
            contour_points = contour_points.astype('int32')

            a_color = np.array([contour_cmap(1. * frame_offset / frame_duration)])

            for point_index, a_point in enumerate(contour_points):
                a_col, a_row = a_point
                plt.scatter(x=a_col, y=a_row, c=a_color, s=1, antialiased=True)

        fig = plt.gcf()
        fig.set_size_inches(20, 20)
        plt.savefig(f"{plot_dir}/{save_name}/{cur_frame}_{cur_frame+frame_duration}.png", bbox_inches="tight", pad_inches=0, dpi=400)
        plt.close()


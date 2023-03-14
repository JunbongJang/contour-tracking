# coding=utf-8
# Copyright 2023 Junbong Jang.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Some plotting functionality for contour tracking inference.
"""

import io
import os
import time

import matplotlib
matplotlib.use('Agg')  # None-interactive plots do not need tk
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
import tensorflow as tf
import cv2
import pylab
from tqdm import tqdm

from src import uflow_utils  # for resampling the warped image

# How much to scale motion magnitude in visualization.
_FLOW_SCALING_FACTOR = 50.0

# pylint:disable=g-long-lambda


def print_log(log, epoch=None, mean_over_num_steps=1):
  """Print log returned by UFlow.train(...)."""

  if epoch is None:
    status = ''
  else:
    status = '{} -- '.format(epoch)
      
  status += 'total-loss: {:.6f}'.format(
      np.mean(log['total-loss'][-mean_over_num_steps:]))

  for key in sorted(log):
    if key not in ['total-loss']:
      loss_mean = np.mean(log[key][-mean_over_num_steps:])
      status += ', {}: {:.6f}'.format(key, loss_mean)
  print(status)

def print_eval(eval_dict):
  """Prints eval_dict to console."""

  status = ''.join(
      ['{}: {:.6f}, '.format(key, eval_dict[key]) for key in sorted(eval_dict)])
  print(status[:-2])


def plot_log(log, plot_dir):
  plt.figure(1)
  plt.clf()

  keys = ['total-loss'
         ] + [key for key in sorted(log) if key not in ['total-loss']]
  for key in keys:
    plt.plot(log[key], '--' if key == 'total-loss' else '-', label=key)
  plt.legend()
  save_and_close(os.path.join(plot_dir, 'log.png'))


def log_to_tensorboard(summary_writer, log, saved_offset_dict, epoch=None, mean_over_num_steps=1):
    with summary_writer.as_default():
        for key in sorted(log):
            tf.summary.scalar(key, np.mean(log[key][-mean_over_num_steps:]), step=epoch)

        with tf.name_scope("saved_offset_dict"):
            for seq_key in saved_offset_dict.keys():
                for layer_key in saved_offset_dict[seq_key].keys():
                    a_string_tensor = tf.strings.as_string(saved_offset_dict[seq_key][layer_key])
                    tf.summary.text(f"seq_pair {seq_key}, layer {layer_key}:", a_string_tensor, step=epoch)

def save_and_close(filename):
  """Save figures."""

  # Create a python byte stream into which to write the plot image.
  buf = io.BytesIO()

  # Save the image into the buffer.
  plt.savefig(buf, format='png')

  # Seek the buffer back to the beginning, then either write to file or stdout.
  buf.seek(0)
  with tf.io.gfile.GFile(filename, 'w') as f:
    f.write(buf.read(-1))
  plt.close('all')


def time_data_it(data_it, simulated_train_time_ms=100.0):
  print('Timing training iterator with simulated train time of {:.2f}ms'.format(
      simulated_train_time_ms))
  for i in range(100):
    start = time.time()
    _ = data_it.get_next()
    end = time.time()
    print(i, 'Time to get one batch (ms):', (end - start) * 1000)
    if simulated_train_time_ms > 0.0:
      plt.pause(simulated_train_time_ms / 1000.)


def save_image_as_png(image, filename):
  image_uint8 = tf.image.convert_image_dtype(image, tf.uint8, saturate=True)
  image_png = tf.image.encode_png(image_uint8)
  tf.io.write_file(filename, image_png)


def plot_data(data_it, plot_dir, num_plots):
  print('Saving images from the dataset to', plot_dir)
  for i, (image_batch, _) in enumerate(data_it):
    if i >= num_plots:
      break
    for j, image_sequence in enumerate(image_batch):
      for k, image in enumerate(image_sequence):
        save_image_as_png(
            image, os.path.join(plot_dir, '{}_{}_{}.png'.format(i, j, k)))


def flow_to_rgb(flow):
  """Computes an RGB visualization of a flow field."""
  shape = flow.shape
  is_graph_mode = False
  if not isinstance(shape[0], int):  # In graph mode, this is a Dimension object
    is_graph_mode = True
    shape = [s.value for s in shape]
  height, width = [float(s) for s in shape[-3:-1]]
  scaling = _FLOW_SCALING_FACTOR / (height**2 + width**2)**0.5

  # Compute angles and lengths of motion vectors.
  if is_graph_mode:
    motion_angle = tf.atan2(flow[Ellipsis, 1], flow[Ellipsis, 0])
  else:
    motion_angle = np.arctan2(flow[Ellipsis, 1], flow[Ellipsis, 0])
  motion_magnitude = (flow[Ellipsis, 1]**2 + flow[Ellipsis, 0]**2)**0.5
  # print('motion_angle', motion_angle * (180 / np.pi))
  # print('motion_magnitude', motion_magnitude)

  # Visualize flow using the HSV color space, where angles are represented by
  # hue and magnitudes are represented by saturation.
  if is_graph_mode:
    flow_hsv = tf.stack([((motion_angle / np.math.pi) + 1.) / 2.,
                         tf.clip_by_value(motion_magnitude * scaling, 0.0, 1.0),
                         tf.ones_like(motion_magnitude)],
                        axis=-1)
  else:
    flow_hsv = np.stack([((motion_angle / np.math.pi) + 1.) / 2.,
                         np.clip(motion_magnitude * scaling, 0.0, 1.0),
                         np.ones_like(motion_magnitude)],
                        axis=-1)
  # Transform colors from HSV to RGB color space for plotting.
  if is_graph_mode:
    return tf.image.hsv_to_rgb(flow_hsv)
  return matplotlib.colors.hsv_to_rgb(flow_hsv)


def flow_tensor_to_rgb_tensor(motion_image):
  """Visualizes flow motion image as an RGB image.

  Similar as the flow_to_rgb function, but with tensors.

  Args:
    motion_image: A tensor either of shape [batch_sz, height, width, 2] or of
      shape [height, width, 2]. motion_image[..., 0] is flow in x and
      motion_image[..., 1] is flow in y.

  Returns:
    A visualization tensor with same shape as motion_image, except with three
    channels. The dtype of the output is tf.uint8.
  """
  # sqrt(a^2 + b^2)
  hypot = lambda a, b: (tf.cast(a, tf.float32)**2.0 + tf.cast(b, tf.float32)**
                        2.0)**0.5
  height, width = motion_image.get_shape().as_list()[-3:-1]
  scaling = _FLOW_SCALING_FACTOR / hypot(height, width)
  x, y = motion_image[Ellipsis, 0], motion_image[Ellipsis, 1]
  motion_angle = tf.atan2(y, x)
  motion_angle = (motion_angle / np.math.pi + 1.0) / 2.0
  motion_magnitude = hypot(y, x)
  motion_magnitude = tf.clip_by_value(motion_magnitude * scaling, 0.0, 1.0)
  value_channel = tf.ones_like(motion_angle)
  flow_hsv = tf.stack([motion_angle, motion_magnitude, value_channel], axis=-1)
  flow_rgb = tf.image.convert_image_dtype(
      tf.image.hsv_to_rgb(flow_hsv), tf.uint8)
  return flow_rgb


def post_imshow(label=None, height=None, width=None):
  plt.xticks([])
  plt.yticks([])
  if label is not None:
    plt.xlabel(label)
  if height is not None and width is not None:
    plt.xlim([0, width])
    plt.ylim([0, height])
    plt.gca().invert_yaxis()


def plot_flow(image1, image2, flow, filename, plot_dir):
  """Overlay images, plot those and flow, and save the result to file."""
  num_rows = 2
  num_columns = 1

  def subplot_at(column, row):
    plt.subplot(num_rows, num_columns, 1 + column + row * num_columns)

  height, width = [float(s) for s in image1.shape[-3:-1]]
  plt.figure('plot_flow', [10. * width / (2 * height), 10.])
  plt.clf()

  subplot_at(0, 0)
  plt.imshow((image1 + image2) / 2.)
  post_imshow()

  subplot_at(0, 1)
  plt.imshow(flow_to_rgb(flow))
  post_imshow()

  plt.subplots_adjust(
      left=0.001, bottom=0.001, right=1, top=1, wspace=0.01, hspace=0.01)

  save_and_close(os.path.join(plot_dir, filename))


def plot_movie_frame(plot_dir, index, image, flow_uv, frame_skip=None):
  """Plots a frame suitable for making a movie."""

  def save_fig(name, plot_dir):
    plt.xticks([])
    plt.yticks([])
    if frame_skip is not None:
      filename = str(index) + '_' + str(frame_skip) + '_' + name
      plt.savefig(os.path.join(plot_dir, filename), bbox_inches='tight')
    else:
      filepath = '{:06d}_{}'.format(index, name)
      plt.savefig(os.path.join(plot_dir, filepath), bbox_inches='tight')
    plt.clf()

  flow_uv = -flow_uv[:, :, ::-1]
  plt.figure()
  plt.clf()

  minimal_frame = np.concatenate([image, flow_to_rgb(flow_uv)], axis=0)
  plt.imshow(minimal_frame)
  save_fig('minimal_video_frame', plot_dir)

  plt.close('all')


def plot_masks(image, masks, filename, plot_dir):
  """Overlay images, plot those and flow, and save the result to file."""
  num_rows = 2
  num_columns = 1

  def subplot_at(column, row):
    plt.subplot(num_rows, num_columns, 1 + column + row * num_columns)

  def ticks():
    plt.xticks([])
    plt.yticks([])

  height, width = [float(s) for s in image.shape[-3:-1]]
  plt.figure('plot_flow', [10. * width / (2 * height), 10.])
  plt.clf()

  subplot_at(0, 0)
  plt.imshow(image)
  ticks()

  subplot_at(0, 1)
  plt.imshow(masks)
  ticks()

  plt.subplots_adjust(
      left=0.001, bottom=0.001, right=1, top=1, wspace=0.01, hspace=0.01)

  save_and_close(os.path.join(plot_dir, filename))


def complete_paper_plot(plot_dir,
                        index,
                        image1,
                        image2,
                        flow_uv,
                        ground_truth_flow_uv,
                        flow_valid_occ,
                        predicted_occlusion,
                        ground_truth_occlusion,
                        frame_skip=None):
  """Plots rgb image, flow, occlusions, ground truth, all as separate images."""

  def save_fig(name, plot_dir):
    plt.xticks([])
    plt.yticks([])
    if frame_skip is not None:
      filename = str(index) + '_' + str(frame_skip) + '_' + name
      plt.savefig(os.path.join(plot_dir, filename), bbox_inches='tight')
    else:
      filepath = str(index) + '_' + name
      plt.savefig(os.path.join(plot_dir, filepath), bbox_inches='tight')
    plt.clf()

  flow_uv = -flow_uv[:, :, ::-1]
  ground_truth_flow_uv = -ground_truth_flow_uv[:, :, ::-1]
  plt.figure()
  plt.clf()

  plt.imshow((image1 + image2) / 2.)
  save_fig('image_rgb', plot_dir)

  plt.imshow(flow_to_rgb(flow_uv))
  save_fig('predicted_flow', plot_dir)

  plt.imshow(flow_to_rgb(ground_truth_flow_uv * flow_valid_occ))
  save_fig('ground_truth_flow', plot_dir)

  endpoint_error = np.sum(
      (ground_truth_flow_uv - flow_uv)**2, axis=-1, keepdims=True)**0.5
  plt.imshow(
      (endpoint_error * flow_valid_occ)[:, :, 0],
      cmap='viridis',
      vmin=0,
      vmax=40)
  save_fig('flow_error', plot_dir)

  plt.imshow((predicted_occlusion[:, :, 0]) * 255, cmap='gray')
  save_fig('predicted_occlusion', plot_dir)

  plt.imshow((ground_truth_occlusion[:, :, 0]) * 255, cmap='gray')
  save_fig('ground_truth_occlusion', plot_dir)


def plot_selfsup(key, images, flows, teacher_flow, student_flow, error,
                 teacher_mask, student_mask, mask, selfsup_transform_fns,
                 plot_dir):
  """Plots some data relevant to self-supervision."""
  num_rows = 3
  num_columns = 3

  def subplot_at(row, column):
    plt.subplot(num_rows, num_columns, 1 + column + row * num_columns)

  i, j, _ = key
  height, width = [float(s.value) for s in images[i].shape[-3:-1]]
  plt.figure('plot_flow',
             [10. * num_columns * width / (num_rows * height), 10.])
  plt.clf()

  subplot_at(0, 0)
  plt.imshow((images[i][0] + images[j][0]) / 2., interpolation='nearest')
  post_imshow('Teacher images')

  subplot_at(0, 1)

  transformed_image_i = selfsup_transform_fns[0](
      images[i], i_or_ij=i, is_flow=False)
  transformed_image_j = selfsup_transform_fns[0](
      images[j], i_or_ij=j, is_flow=False)
  plt.imshow(
      (transformed_image_i[0] + transformed_image_j[0]) / 2.,
      interpolation='nearest')
  post_imshow('Student images')

  subplot_at(0, 2)
  plt.imshow(
      teacher_mask[0, Ellipsis, 0],
      interpolation='nearest',
      vmin=0.,
      vmax=1.,
      cmap='viridis')
  post_imshow('Teacher mask')

  subplot_at(1, 0)
  plt.imshow(
      flow_to_rgb(flows[(i, j, 'original-teacher')][0][0].numpy()),
      interpolation='nearest')
  post_imshow('Teacher flow')

  subplot_at(1, 1)
  plt.imshow(flow_to_rgb(student_flow[0].numpy()), interpolation='nearest')
  post_imshow('Student flow')

  subplot_at(1, 2)
  plt.imshow(
      student_mask[0, Ellipsis, 0],
      interpolation='nearest',
      vmin=0.,
      vmax=1.,
      cmap='viridis')
  post_imshow('Student mask')

  subplot_at(2, 0)
  plt.imshow(flow_to_rgb(teacher_flow[0].numpy()), interpolation='nearest')
  post_imshow('Teacher flow (projected)')

  subplot_at(2, 1)
  plt.imshow(
      error[0, Ellipsis, 0],
      interpolation='nearest',
      vmin=0.,
      vmax=3.,
      cmap='viridis')
  post_imshow('Error')

  subplot_at(2, 2)
  plt.imshow(
      mask[0, Ellipsis, 0],
      interpolation='nearest',
      vmin=0.,
      vmax=1.,
      cmap='viridis')
  post_imshow('Combined mask')

  plt.subplots_adjust(
      left=0.001, bottom=0.05, right=1, top=1, wspace=0.01, hspace=0.1)

  filename = '{}.png'.format(time.time())
  save_and_close(os.path.join(plot_dir, filename))


def plot_smoothness(key, images, weights_xx, weights_yy, flow_gxx_abs,
                    flow_gyy_abs, flows, plot_dir):
  """Plots data relevant to smoothness."""
  num_rows = 3
  num_columns = 3

  def subplot_at(row, column):
    plt.subplot(num_rows, num_columns, 1 + column + row * num_columns)

  i, j, c = key
  height, width = [float(s.value) for s in images[i].shape[-3:-1]]
  plt.figure('plot_flow',
             [10. * num_columns * width / (num_rows * height), 10.])
  plt.clf()

  subplot_at(0, 0)
  plt.imshow(images[i][0], interpolation='nearest')
  post_imshow('Image')

  subplot_at(1, 0)
  plt.imshow(
      weights_xx[0, Ellipsis, 0],
      interpolation='nearest',
      cmap='viridis',
      vmin=0.0,
      vmax=1.0)
  post_imshow('Weights dxx {}'.format(np.mean(weights_xx[0, Ellipsis, 0])))

  subplot_at(2, 0)
  plt.imshow(
      weights_yy[0, Ellipsis, 0],
      interpolation='nearest',
      cmap='viridis',
      vmin=0.0,
      vmax=1.0)
  post_imshow('Weights dyy {}'.format(np.mean(weights_yy[0, Ellipsis, 0])))

  subplot_at(0, 1)
  plt.imshow(
      flow_to_rgb(flows[(i, j, c)][0][0].numpy()), interpolation='nearest')
  post_imshow('Flow')

  subplot_at(1, 1)
  plt.imshow(
      flow_gxx_abs[0, Ellipsis, 0],
      interpolation='nearest',
      cmap='viridis',
      vmin=0.0,
      vmax=1.0)
  post_imshow('FLow dxx')

  subplot_at(2, 1)
  plt.imshow(
      flow_gyy_abs[0, Ellipsis, 0],
      interpolation='nearest',
      cmap='viridis',
      vmin=0.0,
      vmax=1.0)
  post_imshow('Flow dyy')

  subplot_at(1, 2)
  plt.imshow(
      weights_xx[0, Ellipsis, 0] * flow_gxx_abs[0, Ellipsis, 0],
      interpolation='nearest',
      cmap='viridis',
      vmin=0.0,
      vmax=1.0)
  post_imshow('Loss dxx')

  subplot_at(2, 2)
  plt.imshow(
      weights_yy[0, Ellipsis, 0] * flow_gyy_abs[0, Ellipsis, 0],
      interpolation='nearest',
      cmap='viridis',
      vmin=0.0,
      vmax=1.0)
  post_imshow('Loss dyy')

  plt.subplots_adjust(
      left=0.001, bottom=0.05, right=1, top=1, wspace=0.01, hspace=0.1)

  filename = '{}.png'.format(time.time())
  save_and_close(os.path.join(plot_dir, filename))


# --------------------------------------------------------------------------------------------

def save_tracked_contour_indices(plot_dir, cur_index, num_plots, np_all_cur_id_assign):
    file_path = f"{plot_dir}/tracked_contour_points.npy"

    # save previous predicted tracking points
    if cur_index == 1:
        saved_tracking_points = np.zeros(shape=(num_plots, np_all_cur_id_assign.shape[0]), dtype=np.int32)
    else:
        if os.path.exists(file_path):
            saved_tracking_points = np.load(file_path)

    saved_tracking_points[cur_index - 1, :] = np_all_cur_id_assign
    np.save(file_path, saved_tracking_points)


def predict_tracking_plot_manuscript(plot_dir,
                                      cur_index,
                                      image1,
                                      image2,
                                      seg_point1,
                                      seg_point2,
                                      pred_tracking_points,
                                      num_plots):
    '''
    Plot tracking points on the corresponding frame
    '''

    a_image = image2
    cm = pylab.get_cmap('gist_rainbow')

    # prepare save folder
    if not os.path.exists(f"{plot_dir}/"):
        os.makedirs(f"{plot_dir}/")
    file_path = f"{plot_dir}/saved_pred_offset_points.npy"

    # save previous predicted tracking points
    if cur_index == 1:
        saved_tracking_points = np.zeros(shape=(num_plots, pred_tracking_points.shape[0], pred_tracking_points.shape[1]), dtype=np.float32)
    else:
        if os.path.exists(file_path):
            saved_tracking_points = np.load(file_path)
    saved_tracking_points[cur_index - 1, :, :] = pred_tracking_points
    np.save(file_path, saved_tracking_points)

    # MAX_NUM_POINTS = pred_tracking_points.shape[0]
    # # -------------- draw predicted points on an image --------------
    # for point_index, pred_tracking_point in enumerate(pred_tracking_points):
    #     pred_col, pred_row = pred_tracking_point
    #
    #     # TODO: is it ok to ignore these points going outside the image?
    #     if pred_col >= 0 and pred_row >= 0 and pred_col < a_image.shape[1] and pred_row < a_image.shape[0]:
    #         plt.scatter(x=pred_col, y=pred_row, c=np.array([cm(1. * point_index / MAX_NUM_POINTS)]), s=5)
    #
    # plt.imshow(a_image, cmap='gray')
    # plt.axis('off')
    # plt.savefig(f"{plot_dir}/pred_{cur_index}.png", bbox_inches="tight", pad_inches=0)
    # plt.close()
    #
    # if cur_index == num_plots:  # e.g) final frame's index is 50
    #     # -------------- draw trajectory of tracking points in the last image --------------------
    #     plt.imshow(a_image, cmap='gray')
    #
    #     # draw points from each frame on an image
    #     # saved_tracking_points shape is [#frames, #points, 2]
    #     for a_point_index in range(saved_tracking_points.shape[1]):
    #         col_list = saved_tracking_points[:, a_point_index, 0]
    #         row_list = saved_tracking_points[:, a_point_index, 1]
    #         plt.plot(col_list, row_list, c=np.array(cm(1. * a_point_index / MAX_NUM_POINTS)), marker='o', linewidth=2, markersize=2)
    #
    #     plt.savefig(f"{plot_dir}/trajectory.png", bbox_inches="tight", pad_inches=0)
    #     plt.close()


def predict_tracking_plot(plot_dir,
                         cur_index,
                         image1,
                         image2,
                         segmentation1,
                         segmentation2,
                         seg_point1,
                         seg_point2,
                         gt_tracking_points,
                         pred_tracking_points,
                         num_plots,
                         frame_skip=None):
    '''
    Plot tracking points on the corresponding frame
    Modified by Junbong Jang
    6/22/2022
    '''
    
    # --------------- Plot Original Image -------------------------
    # save_fig(plot_dir, cur_index, frame_skip, name='image_rgb', cv2_imwrite_data=image1)
    # ---------------- Plot Segmentation ------------------
    # save_fig(plot_dir, cur_index, frame_skip, name='segmentation1', cv2_imwrite_data=segmentation1)

    # ---------------- Plot 2D correlation matrix ----------------
    # prepare save folder
    # save_folder_name = 'corr_2d_matrix'
    # if not os.path.exists(f"{plot_dir}/{save_folder_name}/"):
    #     os.makedirs(f"{plot_dir}/{save_folder_name}/")
    #
    # cv2.imwrite(f"{plot_dir}/{save_folder_name}/pred_{cur_index}.png", np.expand_dims(corr_2d_matrix[0], axis=-1) * 255)

    # --------------- Plot Tracking Points ---------------
    def plot_tracking_points(plot_dir, num_plots, cur_index, gt_tracking_points, pred_tracking_points, a_image):
        '''
        visualize and save moved tracking points

        :return:
        '''
        cm = pylab.get_cmap('gist_rainbow')

        # prepare save folder
        if not os.path.exists(f"{plot_dir}/"):
            os.makedirs(f"{plot_dir}/")
        file_path = f"{plot_dir}/saved_tracking_points.npy"

        # save previous predicted tracking points
        if cur_index == 1:
            saved_tracking_points = np.zeros(shape=(num_plots, pred_tracking_points.shape[0], pred_tracking_points.shape[1]), dtype=np.int32 )
        else:
            if os.path.exists(file_path):
                saved_tracking_points = np.load(file_path)

        saved_tracking_points[cur_index-1,:,:] = np.round(pred_tracking_points)
        np.save(file_path, saved_tracking_points)

        MAX_NUM_POINTS = pred_tracking_points.shape[0]
        # -------------- draw predicted points on an image --------------
        for point_index, (pred_tracking_point, gt_tracking_point) in enumerate(zip(pred_tracking_points, gt_tracking_points)):
            pred_col, pred_row = pred_tracking_point

            # TODO: is it ok to ignore these points going outside the image?
            if pred_col >= 0 and pred_row >= 0 and pred_col < a_image.shape[1] and pred_row < a_image.shape[0]:
                plt.scatter(x=pred_col, y=pred_row, c=np.array([cm(1. * point_index / MAX_NUM_POINTS)]), s=5)

            gt_col, gt_row = gt_tracking_point
            plt.scatter(x=gt_col, y=gt_row, s=5, facecolors='none', linewidths=0.5, edgecolors=np.array([cm(1. * point_index / MAX_NUM_POINTS)]))
            
        plt.imshow(a_image, cmap='gray')
        plt.axis('off')
        plt.savefig(f"{plot_dir}/pred_{cur_index}.png", bbox_inches="tight", pad_inches=0)
        plt.close()

        # -------------- draw GT points on an image --------------
        # for point_index, gt_tracking_point in enumerate(gt_tracking_points):
        #     gt_col, gt_row = gt_tracking_point
        #     plt.scatter(x=gt_col, y=gt_row, s=5, facecolors='none', linewidths=0.5, edgecolors=np.array([cm(1. * point_index / MAX_NUM_POINTS)]))
        #
        # plt.imshow(a_image, cmap='gray')
        # plt.axis('off')
        # plt.savefig(f"{plot_dir}/gt_{cur_index}.png", bbox_inches="tight", pad_inches=0)
        # plt.close()
        
        if cur_index == num_plots:  # e.g) final frame's index is 50
            # -------------- draw trajectory of tracking points in the last image --------------------
            plt.imshow(a_image, cmap='gray')

            # draw points from each frame on an image
            # saved_tracking_points shape is [#frames, #points, 2]
            for a_point_index in range(saved_tracking_points.shape[1]):
                col_list = saved_tracking_points[:, a_point_index, 0]
                row_list = saved_tracking_points[:, a_point_index, 1]
                plt.plot(col_list, row_list, c=np.array(cm(1. * a_point_index / MAX_NUM_POINTS)), marker='o', linewidth=2, markersize=2)

            plt.savefig(f"{plot_dir}/trajectory.png", bbox_inches="tight", pad_inches=0)
            plt.close()

    # to reduce the number of tracking points
    if gt_tracking_points.shape[0] > 20:
        gt_tracking_points = gt_tracking_points[::3, :]
        pred_tracking_points = pred_tracking_points[::3, :]

    plot_tracking_points(plot_dir, num_plots=num_plots, cur_index=cur_index, gt_tracking_points=gt_tracking_points, pred_tracking_points=pred_tracking_points, a_image=image2)


def predict_plot(plot_dir,
                index,
                image1,
                image2,
                segmentation1,
                segmentation2,
                tracking_point1,
                tracking_point2,
                flow_uv,
                predicted_occlusion,
                predicted_range_map,
                forward_warp,
                forward_valid_warp_mask,
                num_plots,
                frame_skip=None):
  '''
  Plots rgb image, flow, occlusions, all as separate images.
  Modified by Junbong Jang
  5/12/2022
  '''

  # --------------- Plot Original Image -------------------------
  save_fig(plot_dir, index, frame_skip, name='image_rgb', cv2_imwrite_data=image1)

  # --------------- Convert flow map to arrow map --------------
  flow_uv = -flow_uv[:, :, ::-1]
  quiver_U = flow_uv[Ellipsis, 0]
  quiver_V = flow_uv[Ellipsis, 1]

  # maxpooling to make arrows more visible
  resize_factor = 8
  reshaped_quiver_U = tf.reshape(quiver_U, (1, quiver_U.shape[0], quiver_U.shape[1], 1))
  quiver_U = tf.nn.max_pool(reshaped_quiver_U, ksize=resize_factor, strides=resize_factor, padding='VALID')
  quiver_U = quiver_U[0,:,:,0]
  reshaped_quiver_V = tf.reshape(quiver_V, (1, quiver_V.shape[0], quiver_V.shape[1], 1))
  quiver_V = tf.nn.max_pool(reshaped_quiver_V, ksize=resize_factor, strides=resize_factor, padding='VALID')
  quiver_V = quiver_V[0,:,:,0]

  # set up the figure
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_aspect('equal')
  # ax.set_xlim(left=0, right=flow_uv.shape[1] / resize_factor)
  # ax.set_ylim(bottom=0, top=flow_uv.shape[0] / resize_factor)

  # draw arrows
  ax.quiver(-1*np.flip(quiver_U,0), np.flip(quiver_V,0), units="width", alpha=0.8)
  save_fig(plot_dir, index, frame_skip, name='predicted_flow_arrow', fig=fig)

  # --------------------------------------------------------------------------------------

  # --- this is only works if image1 is the segmented image because the canny edge works well on segmented images ---
  # blurred_image1 = cv2.blur(np.uint8(image1), (7, 7))
  # edge_image = cv2.Canny(blurred_image1, 0.5, 1, 3, L2gradient=True)  # because the image1's intensity is in range [0, 1]
  # flow_rgb = flow_to_rgb(flow_uv)
  # flow_rgb[edge_image>0] = 0

  # ----------------- Plot Optical Flow ------------------
  flow_rgb = flow_to_rgb(flow_uv)
  save_fig(plot_dir, index, frame_skip, name='predicted_flow', cv2_imwrite_data=flow_rgb)
  print("unique flow values: ", np.unique(flow_uv))

  # ---------------- Plot Occlusion ---------------------
  # occluded region is white
  # save_fig(plot_dir, index, frame_skip, name='predicted_occlusion', cv2_imwrite_data=predicted_occlusion[:, :, 0])

  # ---------------- Plot Range Map ---------------------
  # save_fig(plot_dir, index, frame_skip, name='predicted_range_map', cv2_imwrite_data=predicted_range_map[:, :, 0])

  # ---------------- Plot Warped image and contour ------------------
  a_warp = uflow_utils.flow_to_warp_np(flow_uv)

  def save_result_to_npy(a_result, save_path):
      if index == 1:
        list_of_result = np.expand_dims(a_result, axis=0)
      else:
        list_of_result = np.load(save_path, allow_pickle=True)
        list_of_result = np.append([a_result], list_of_result, axis=0)
      np.save(save_path, list_of_result, allow_pickle=True)

  # save warp to plot dense correspondence later
  warp_path = f"{plot_dir}/warp_list.npy"
  save_result_to_npy(a_warp, warp_path)
  
  # Warp Image
  warped_image1 = uflow_utils.resample_np(image2, a_warp)
  save_fig(plot_dir, index, frame_skip, name='predicted_warped_image', cv2_imwrite_data=warped_image1.numpy()[0])

  # Warp contour
  # tf.unique_with_counts(tf.reshape(warped_contour1, -1))
  warped_contour1 = uflow_utils.resample_np(segmentation2, a_warp)   # uint8 [ 474, 392, 1] -> uint8 [474, 392, 1]
  warped_contour1 = warped_contour1.numpy()[0]
  save_fig(plot_dir, index, frame_skip, name='predicted_warped_contour', cv2_imwrite_data=warped_contour1)

  error1 = warped_contour1.astype(np.int32) - segmentation1.astype(np.int32)   # e.g. 127 - 255  or 127 - 0
  error2 = segmentation1.astype(np.int32) - warped_contour1.astype(np.int32)   # e.g. 255 - 127  or 0 - 127

  # clipping to ignore negative values
  cliped_error1 = np.clip(error1, 0, 255)
  cliped_error2 = np.clip(error2, 0, 255)
  error = cliped_error1 + cliped_error2

  save_fig(plot_dir, index, frame_skip, name='contour_warp_error', cv2_imwrite_data=error.astype(np.uint8))

  # ---------------- Plot Segmentation ------------------
  save_fig(plot_dir, index, frame_skip, name='segmentation1', cv2_imwrite_data=segmentation1)

  # --------------- Plot Tracking Points ---------------
  if index>1:
      tracking_point1=None
  tracking_points = plot_tracking_points_from_optical_flow(plot_dir, save_name='optical_flow_tracking_points', num_plots=num_plots, cur_index=index, a_optical_flow=flow_uv, a_tracking_point=tracking_point1, a_image=image1)

  # if index == 50:
    # plot_contour_correspondence_with_color(plot_dir, final_frame_index=index, save_name='correspondence_warped_contour', final_frame_contour=segmentation2[:, :, 0]) # segmentation2 only has one channel

  return warped_contour1, tracking_points


def plot_tracking_points_from_optical_flow(plot_dir, save_name, num_plots, cur_index, a_optical_flow, a_tracking_point, a_image):
    '''
    get tracking point from the previous frame
    move tracking points using the current frame's optical flow
    save moved tracking point

    :param plot_dir:
    :param save_name:
    :param image:
    :return:
    '''
    cm = pylab.get_cmap('gist_rainbow')

    # prepare save folder
    if not os.path.exists(f"{plot_dir}/{save_name}/"):
      os.makedirs(f"{plot_dir}/{save_name}/")

    # get tracking points from the previous frame
    if a_tracking_point is None:
        file_path = f"{plot_dir}/{save_name}/saved_tracking_points.npy"
        if os.path.exists(file_path):
            loaded_tracking_points = np.load(file_path)
    else:
        loaded_tracking_points = np.zeros(shape=(num_plots+1, a_tracking_point.shape[0], a_tracking_point.shape[1]), dtype=np.int32 )
        loaded_tracking_points[0,:,:] = a_tracking_point

    MAX_NUM_POINTS = loaded_tracking_points.shape[1]

    # move tracking points using the optical flow
    for point_index, a_tracking_point in enumerate(loaded_tracking_points[cur_index-1]):

        col, row = a_tracking_point
        col_dif = round(a_optical_flow[row, col, 0])
        row_dif = round(a_optical_flow[row, col, 1])
        new_col = col + col_dif
        new_row = row + row_dif

        # TODO: is it ok to ignore these points going outside the image?
        # draw points on an image
        if new_col >= 0 and new_row >= 0 and new_col < a_image.shape[1] and new_row < a_image.shape[0]:
            plt.scatter(x=new_col, y=new_row, c=np.array([cm(1. * point_index / MAX_NUM_POINTS)]), s=10)
            loaded_tracking_points[cur_index, point_index, 0] = new_col
            loaded_tracking_points[cur_index, point_index, 1] = new_row

    plt.imshow(a_image, cmap='gray')
    plt.axis('off')
    plt.savefig(f"{plot_dir}/{save_name}/{cur_index}.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    # save moved tracking point
    np.save(f"{plot_dir}/{save_name}/saved_tracking_points.npy", loaded_tracking_points)

    if cur_index == num_plots:  # e.g) final frame's index is 50
        # -------------- draw trajectory of tracking points in the last image --------------------
        plt.imshow(a_image, cmap='gray')

        # draw points from each frame on an image
        for a_frame in loaded_tracking_points:
            for tracking_point_index, a_tracking_point in enumerate(a_frame):
                col, row = a_tracking_point
                plt.plot(col, row, c=np.array(cm(1. * tracking_point_index / MAX_NUM_POINTS)), marker='o', linewidth=2, markersize=2)

        plt.savefig(f"{plot_dir}/{save_name}/trajectory.png", bbox_inches="tight", pad_inches=0)
        plt.close()

    return loaded_tracking_points


def plot_contour_correspondence_with_color(plot_dir, final_frame_index, save_name, final_frame_contour):
    '''
    Visualize correspondence between contours in the video by color
    Using the saved optical flow at each time step, warp the 200th frame contour back to 1st frame contour

    For colormap inspiration, Refer to https://github.com/microsoft/DIF-Net/blob/main/generate.py

    :return:
    '''
    # ------------------- Color the contour -----------------------------
    cm = pylab.get_cmap('gist_rainbow')  # get color map

    y_coords, x_coords = np.where(final_frame_contour > 0)  # get coords of the contour points

    # assign color to each contour point in order
    NUM_COLORS = len(x_coords)
    print('Total # of edge pixels', NUM_COLORS)

    color_edge = np.zeros((final_frame_contour.shape[0], final_frame_contour.shape[1], 3), dtype=np.float32)
    for a_index, (x_coord, y_coord) in enumerate(zip(x_coords, y_coords)):
        color_edge[y_coord, x_coord, :] = cm(1. * a_index / NUM_COLORS)[:3]

    save_fig(plot_dir, final_frame_index+1, frame_skip=None, name=save_name, cv2_imwrite_data=color_edge)

    # ----------------- Warp the colored contour continuously -------
    # get list of warps saved in the computer
    warp_path = f"{plot_dir}/warp_list.npy"
    list_of_warps = np.load(warp_path, allow_pickle=True)

    print('Original Contour: ', np.sum(np.sum(color_edge, axis=-1) > 0), '   Unique Colors:', np.unique(np.reshape(color_edge, (-1, 3)), axis=0).shape[0] - 1 )
    # i.e. Warps 200th frame to 199th frame and then from 199th frame to 198th frame and so on...
    for a_index, a_warp in enumerate(list_of_warps):
        if a_index == 0:
            warped_contour = uflow_utils.resample_np(color_edge, a_warp)
        else:
            warped_contour = uflow_utils.resample_np(warped_contour, a_warp)

        warped_contour = warped_contour.numpy()[0]
        print('Warped Contour: ', np.sum(np.sum(warped_contour, axis=-1) > 0), '  Unique Colors:', np.unique(np.reshape(warped_contour, (-1, 3)), axis=0).shape[0] - 1 )
        
        save_fig(plot_dir, final_frame_index-a_index, frame_skip=None, name=save_name, cv2_imwrite_data=warped_contour)


def save_fig(plot_dir, index, frame_skip, name, fig=None, cv2_imwrite_data=None):

    plot_dir = f"{plot_dir}/{name}/"
    if not os.path.exists(plot_dir):
      os.makedirs(plot_dir)

    if frame_skip is not None:
        filename = f"{str(index)}_{str(frame_skip)}_{name}.png"
    else:
        filename = f"{str(index)}_{name}.png"
    full_path = os.path.join(plot_dir, filename)

    if cv2_imwrite_data is not None:
        if cv2_imwrite_data.dtype != np.uint8:
            cv2_imwrite_data = (cv2_imwrite_data*255).astype(np.uint8)
        cv2_imwrite_data = cv2.cvtColor(cv2_imwrite_data, cv2.COLOR_BGR2RGB)
        cv2.imwrite(full_path, cv2_imwrite_data)

    else:
        plt.xticks([])
        plt.yticks([])
        if fig is None:
          fig = plt
        fig.savefig(full_path, bbox_inches='tight')
        plt.close('all')


def plot_cum_accuracy_from_first_to_last_frame(plot_dir, list_of_sa_ours, list_of_ca_ours):
    num_frames = len(list_of_sa_ours)
    x_range = range(num_frames)
    np_frames = np.linspace(1, num_frames, num=num_frames)

    # ---------------------------------------------------------------------------------
    # rebuttal - on phase contrast videos
    list_of_sa_ours1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.6, 0.6, 0.8, 0.8, 0.8, 0.8, 0.6, 0.6, 0.4, 0.4, 0.8, 0.8, 0.8, 0.8, 0.4, 0.4, 0.4, 0.6, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.6, 0.6, 0.4, 0.4, 0.6]
    list_of_sa_ours2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.6, 0.8, 0.6]
    list_of_sa_ours3 = [1.0, 1.0, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 0.8, 1.0, 0.8, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]
    list_of_sa_ours4 = [1.0, 1.0, 1.0, 0.8571428571428571, 0.8571428571428571, 0.5714285714285714, 0.7142857142857143, 0.5714285714285714, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.8571428571428571, 1.0, 0.8571428571428571, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.5714285714285714, 0.5714285714285714, 0.7142857142857143, 0.5714285714285714, 0.5714285714285714, 0.5714285714285714, 0.5714285714285714, 0.5714285714285714, 0.5714285714285714, 0.42857142857142855, 0.42857142857142855, 0.42857142857142855, 0.42857142857142855, 0.42857142857142855, 0.42857142857142855, 0.2857142857142857, 0.2857142857142857]
    list_of_sa_ours = ( np.array(list_of_sa_ours1) + np.array(list_of_sa_ours2) + np.array(list_of_sa_ours3) + np.array(list_of_sa_ours4) ) / 4
    
    list_of_ca_ours1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.6, 0.6, 0.8, 0.8, 0.8, 0.8, 0.6, 0.6, 0.6, 0.6, 0.8, 0.8, 0.8, 0.8, 0.4, 0.4, 0.6, 0.8, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.6]
    list_of_ca_ours2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8]
    list_of_ca_ours3 = [1.0, 1.0, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 1.0, 0.8, 1.0, 1.0, 0.8, 1.0, 1.0, 0.8, 0.4, 0.4, 0.4, 0.6, 0.4, 0.4, 0.4, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]
    list_of_ca_ours4 = [1.0, 1.0, 1.0, 0.8571428571428571, 0.8571428571428571, 0.7142857142857143, 0.7142857142857143, 0.5714285714285714, 0.7142857142857143, 0.5714285714285714, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.8571428571428571, 1.0, 0.8571428571428571, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.5714285714285714, 0.5714285714285714, 0.7142857142857143, 0.5714285714285714, 0.5714285714285714, 0.42857142857142855, 0.5714285714285714, 0.5714285714285714, 0.5714285714285714, 0.42857142857142855, 0.5714285714285714, 0.42857142857142855, 0.42857142857142855, 0.42857142857142855, 0.42857142857142855, 0.42857142857142855, 0.42857142857142855]
    list_of_ca_ours = ( np.array(list_of_ca_ours1) + np.array(list_of_ca_ours2) + np.array(list_of_ca_ours3) + np.array(list_of_ca_ours4) ) / 4

    list_of_sa_mechanical1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.6, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2, 0.4, 0.2, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    list_of_sa_mechanical2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.6, 0.6, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.6, 0.6, 0.8, 0.8]
    list_of_sa_mechanical3 = [1.0, 1.0, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.6, 0.8, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    list_of_sa_mechanical4 = [1.0, 1.0, 1.0, 1.0, 0.8571428571428571, 1.0, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.7142857142857143, 0.8571428571428571, 0.8571428571428571, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.8571428571428571, 0.5714285714285714, 0.5714285714285714, 0.5714285714285714, 0.5714285714285714, 0.5714285714285714, 0.5714285714285714, 0.5714285714285714, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.5714285714285714, 0.5714285714285714, 0.5714285714285714, 0.5714285714285714, 0.5714285714285714, 0.7142857142857143, 0.5714285714285714, 0.5714285714285714, 0.42857142857142855, 0.5714285714285714, 0.5714285714285714, 0.42857142857142855, 0.42857142857142855]
    list_of_sa_mechanical = ( np.array(list_of_sa_mechanical1) + np.array(list_of_sa_mechanical2) + np.array(list_of_sa_mechanical3) + np.array(list_of_sa_mechanical4) ) / 4
    
    list_of_ca_mechanical1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.6, 0.6, 0.6, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    list_of_ca_mechanical2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    list_of_ca_mechanical3 = [1.0, 1.0, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.6, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    list_of_ca_mechanical4 = [1.0, 1.0, 1.0, 1.0, 0.8571428571428571, 1.0, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 1.0, 1.0, 1.0, 0.8571428571428571, 1.0, 0.7142857142857143, 0.7142857142857143, 0.8571428571428571, 0.8571428571428571, 0.5714285714285714, 0.5714285714285714, 0.5714285714285714, 0.5714285714285714, 0.5714285714285714, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.5714285714285714, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.5714285714285714, 0.7142857142857143, 0.5714285714285714, 0.5714285714285714, 0.7142857142857143, 0.7142857142857143, 0.42857142857142855]
    list_of_ca_mechanical = ( np.array(list_of_ca_mechanical1) + np.array(list_of_ca_mechanical2) + np.array(list_of_ca_mechanical3) + np.array(list_of_ca_mechanical4) ) / 4

    # list_of_sa_mechanical = [1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.8, 0.8, 0.8, 0.8, 0.8, 0.6, 0.6, 0.8, 0.6, 0.8, 0.6, 0.6, 0.8, 0.8, 0.8, 0.6, 0.6, 0.6, 0.6, 0.8, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
    # list_of_ca_mechanical = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.8, 0.6, 0.8, 0.8, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.8, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
    # ---------------------------------------------------------------------------------
    # get cumulative accuracy
    np_cum_sa = np.cumsum(list_of_sa_ours)
    np_cum_mean_sa = np_cum_sa/np_frames

    np_cum_ca = np.cumsum(list_of_ca_ours)
    np_cum_mean_ca = np_cum_ca/np_frames

    np_cum_sa_mechanical = np.cumsum(list_of_sa_mechanical)
    np_cum_mean_sa_mechanical = np_cum_sa_mechanical/np_frames

    np_cum_ca_mechanical = np.cumsum(list_of_ca_mechanical)
    np_cum_mean_ca_mechanical = np_cum_ca_mechanical/np_frames

    plt.figure(figsize=(6, 2), dpi=300)
    plt.plot(x_range, np_cum_mean_sa, label='Ours (SA$_{.02}$)', color='lawngreen')
    plt.plot(x_range, np_cum_mean_ca, label='Ours (CA$_{.01}$)', color='lawngreen', linestyle='dashed')
    plt.plot(x_range, np_cum_mean_sa_mechanical, label='Mechanical (SA$_{.02}$)', color='darkorange')
    plt.plot(x_range, np_cum_mean_ca_mechanical, label='Mechanical (CA$_{.01}$)', color='darkorange', linestyle='dashed')
    plt.xlabel('Frames')
    plt.ylabel('CMA')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/cumulative_accuracy.svg')

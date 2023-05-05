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

# Lint as: python3

"""
Data loading and evaluation utilities shared across multiple datasets.

Some datasets are very similar, so to prevent code duplication, shared utilities
are put into this class.

refer to https://www.tensorflow.org/guide/data for data pipeline
"""
# pylint:disable=g-importing-member
from collections import defaultdict
import sys
import time
from statistics import mean

import numpy as np
import tensorflow as tf

from src import tracking_utils
from src import uflow_plotting
from src import uflow_utils
from src import metrics
from datetime import datetime
from src import implicit_utils


def parse_data(proto,
               include_flow,
               height=None,
               width=None,
               include_occlusion=False,
               include_invalid=False,
               include_segmentations=False,
               include_seg_points=False,
               include_tracking_points=False,
               resize_gt_flow=True,
               gt_flow_shape=None):
  """
  Parses data proto and
  resizes images and tracking points

  Args:
    proto: path to data proto file
    include_flow: bool, whether or not to include flow in the output
    height: int or None height to resize image to
    width: int or None width to resize image to
    include_occlusion: bool, whether or not to also return occluded pixels (will
      throw error if occluded pixels are not present)
    include_invalid: bool, whether or not to also return invalid pixels (will
      throw error if invalid pixels are not present)
    resize_gt_flow: bool, wether or not to resize flow ground truth as the image
    gt_flow_shape: list, shape of the original ground truth flow (only required
      to set a fixed ground truth flow shape for tensorflow estimator in case of
      supervised training at full resolution resize_gt_flow=False)

  # 11/3/2021 modified by Junbong Jang
  Returns:
    images, flow: A tuple of (image1, image2), flow
  """

  # Parse context and image sequence from protobuffer.
  context_features = {
      'height': tf.io.FixedLenFeature([], tf.int64),
      'width': tf.io.FixedLenFeature([], tf.int64),
  }
  sequence_features = {
      'images': tf.io.FixedLenSequenceFeature([], tf.string),
  }

  if include_invalid:
    sequence_features['invalid_masks'] = tf.io.FixedLenSequenceFeature([], tf.string)

  if include_flow:
    context_features['flow_uv'] = tf.io.FixedLenFeature([], tf.string)

  if include_occlusion:
    context_features['occlusion_mask'] = tf.io.FixedLenFeature([], tf.string)

  if include_segmentations:
    sequence_features['segmentations'] = tf.io.FixedLenSequenceFeature([], tf.string)

  if include_seg_points:
    sequence_features['segmentation_points'] = tf.io.FixedLenSequenceFeature([], tf.string)

  if include_tracking_points:
    sequence_features['tracking_points'] = tf.io.FixedLenSequenceFeature([], tf.string)

  context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
      proto,
      context_features=context_features,
      sequence_features=sequence_features,
  )

  def deserialize(s, dtype, dims):
    return tf.reshape(tf.io.decode_raw(s, dtype), [context_parsed['height'], context_parsed['width'], dims])

  def deserialize_points(s, dtype, dims):
      return tf.reshape(tf.io.decode_raw(s, dtype), [-1, dims])

  images = tf.map_fn(
      lambda s: deserialize(s, tf.uint8, 3),
      sequence_parsed['images'],
      dtype=tf.uint8)

  images = tf.image.convert_image_dtype(images, tf.float32)
  orig_height = tf.shape(input=images)[1]
  orig_width = tf.shape(input=images)[2]

  # check if pre-specified size is given
  if height is None or width is None:
      # If specified size is not feasible with the model, change it to a feasible resolution (used to be ceil, instead of round)
      _num_levels = 5
      divisible_by_num = int(pow(2.0, _num_levels))
      height = tf.math.round(orig_height / divisible_by_num) * divisible_by_num
      width = tf.math.round(orig_width / divisible_by_num) * divisible_by_num
      height = tf.cast(height, 'float32')
      width = tf.cast(width, 'float32')

  images = uflow_utils.resize(images, height, width, is_flow=False)
  output = [images]

  if include_flow:
    flow_uv = deserialize(context_parsed['flow_uv'], tf.float32, 2)
    flow_uv = flow_uv[Ellipsis, ::-1]
    if height is not None and width is not None and resize_gt_flow:
      flow_uv = uflow_utils.resize(flow_uv, height, width, is_flow=True)
    else:
      if gt_flow_shape is not None:
        flow_uv.set_shape(gt_flow_shape)
    # To be consistent with uflow internals, we flip the ordering of flow.
    output.append(flow_uv)
    # create valid mask
    flow_valid = tf.ones_like(flow_uv[Ellipsis, :1], dtype=tf.float32)
    output.append(flow_valid)

  if include_occlusion:
    occlusion_mask = deserialize(context_parsed['occlusion_mask'], tf.uint8, 1)
    if height is not None and width is not None:
      occlusion_mask = uflow_utils.resize_uint8(
          occlusion_mask, height, width)
    output.append(occlusion_mask)

  if include_invalid:
    invalid_masks = tf.map_fn(
        lambda s: deserialize(s, tf.uint8, 1),
        sequence_parsed['invalid_masks'],
        dtype=tf.uint8)
    if height is not None and width is not None:
      invalid_masks = uflow_utils.resize_uint8(
          invalid_masks, height, width)
    output.append(invalid_masks)

  if include_segmentations:
    segmentations = tf.map_fn(
        lambda s: deserialize(s, tf.uint8, 1),
        sequence_parsed['segmentations'],
        dtype=tf.uint8)

    if height is not None and width is not None:
      segmentations = uflow_utils.resize_uint8(segmentations, height, width)
    output.append(segmentations)

  if include_seg_points:
    seg_points = tf.map_fn(
        lambda s: deserialize_points(s, tf.int32, 3),
        sequence_parsed['segmentation_points'],
        dtype=tf.int32)

    # move segmentation points based on the resized images and segmentation
    resized_seg_points = tracking_utils.resize_seg_points(orig_height, orig_width, height, width, seg_points)
    output.append(resized_seg_points)

  if include_tracking_points:
    tracking_points = tf.map_fn(
        lambda s: deserialize_points(s, tf.int32, 1),
        sequence_parsed['tracking_points'],
        dtype=tf.int32)

    output.append(tracking_points)

  # Only put the output in a list if there are more than one items in there.
  if len(output) == 1:
    output = output[0]

  return output


def compute_f_metrics(mask_prediction, mask_gt, num_thresholds=40):
  """Return a dictionary of the true positives, etc. for two binary masks."""
  results = defaultdict(dict)
  mask_prediction = tf.cast(mask_prediction, tf.float32)
  mask_gt = tf.cast(mask_gt, tf.float32)
  for threshold in np.linspace(0, 1, num_thresholds):
    mask_thresh = tf.cast(
        tf.math.greater(mask_prediction, threshold), tf.float32)
    true_pos = tf.cast(tf.math.count_nonzero(mask_thresh * mask_gt), tf.float32)
    true_neg = tf.math.count_nonzero((mask_thresh - 1) * (mask_gt - 1))
    false_pos = tf.cast(
        tf.math.count_nonzero(mask_thresh * (mask_gt - 1)), tf.float32)
    false_neg = tf.cast(
        tf.math.count_nonzero((mask_thresh - 1) * mask_gt), tf.float32)
    results[threshold]['tp'] = true_pos
    results[threshold]['fp'] = false_pos
    results[threshold]['fn'] = false_neg
    results[threshold]['tn'] = true_neg
  return results


def get_fmax_and_best_thresh(results):
  """Select which threshold produces the best f1 score."""
  fmax = -1.
  best_thresh = -1.
  for thresh, metrics in results.items():
    precision = metrics['tp'] / (metrics['tp'] + metrics['fp'] + 1e-6)
    recall = metrics['tp'] / (metrics['tp'] + metrics['fn'] + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    if f1 > fmax:
      fmax = f1
      best_thresh = thresh
  return fmax, best_thresh


def predict(
    inference_fn,
    dataset,
    height,
    width,
    progress_bar=False,
    plot_dir='',
    num_plots=0,
    include_segmentations=True,
    include_seg_points=True,
    include_tracking_points=True,
    evaluate_bool=False
):
  """inference function for flow.

  Args:
    inference_fn: An inference function that produces a flow_field from two
      images, e.g. the infer method of UFlow. 
      please look at infer() function in uflow_net.py
    dataset: A dataset produced by the method above with for_eval=True.
    height: int, the height to which the images should be resized for inference.
    width: int, the width to which the images should be resized for inference.
    progress_bar: boolean, flag to indicate whether the function should print a
      progress_bar during evaluaton.
    plot_dir: string, optional path to a directory in which plots are saved (if
      num_plots > 0).
    num_plots: int, maximum number of qualitative results to plot for the
      evaluation.

  Returns:
    None
  """
  eval_start_in_s = time.time()
  
  it = tf.compat.v1.data.make_one_shot_iterator(dataset)
  inference_times = []
  list_of_sa_at_thresholds = {0.02:[], 0.04:[], 0.06:[]}
  list_of_ca_at_thresholds = {0.01:[], 0.02:[], 0.03:[]}

  list_of_points_spatial_accuracy = []
  list_of_delta_along_normal_vector = []

  list_of_forward_occ_cycle_consistency_loss = []
  list_of_backward_occ_cycle_consistency_loss = []

  plot_count = 0
  eval_count = -1

  # these three variables are iteratively updated
  pred_tracking_points = None
  prev_id_assign = None
  tracking_pos_emb = None

  for cur_index, test_batch in enumerate(it):

    if progress_bar:
      sys.stdout.write(':')
      sys.stdout.flush()

    image_batch = test_batch[0]

    # if len(test_batch) == 1:
    #     segmentation_batch = test_batch[0]
    #     tracking_points_batch = test_batch[0]
    # if len(test_batch) == 2:
    #     if include_segmentations:
    #         segmentation_batch = test_batch[1]
    #         tracking_points_batch = test_batch[1]  # to prevent error in inference_fn
    #     elif include_tracking_points:
    #         segmentation_batch = test_batch[0]  # to prevent error in inference_fn
    #         tracking_points_batch = test_batch[1]
    # elif len(test_batch) == 3:
    #     segmentation_batch = test_batch[1]
    #     tracking_points_batch = test_batch[2]

    if len(test_batch) == 3:
        segmentation_batch = test_batch[0]
        seg_points_batch = test_batch[1]
        tracking_points_batch = test_batch[2]
    else:
        raise ValueError('test_batch size is not 3', len(test_batch))

    if evaluate_bool:
        f = lambda: inference_fn(
            image_batch[0],
            image_batch[1],
            segmentation_batch[0],
            segmentation_batch[1],
            tf.expand_dims(seg_points_batch[0], axis=0),
            tf.expand_dims(seg_points_batch[1], axis=0),
            tracking_point1= tf.expand_dims(tracking_points_batch[0], axis=0),
            tracking_pos_emb=tracking_pos_emb,
            input_height=height,
            input_width=width,
            infer_occlusion=True)

    else:
        # for the first frame, pass the initial tracking points
        if pred_tracking_points is None:
            # for debugging inference on tracking points
            # pred_tracking_points = tf.constant([[[100.0, 100.0], [110.0, 110.0], [120.0, 120.0], [130.0, 130.0], [140.0, 140.0], [150.0, 150.0], [160.0, 160.0]]])
            pred_tracking_points = tf.expand_dims(tracking_points_batch[0], axis=0)

        # refer to infer() in contour_flow_net.py
        f = lambda: inference_fn(
            image_batch[0],
            image_batch[1],
            segmentation_batch[0],
            segmentation_batch[1],
            tf.expand_dims(seg_points_batch[0], axis=0),
            tf.expand_dims(seg_points_batch[1], axis=0),
            tracking_point1=pred_tracking_points,
            tracking_pos_emb=tracking_pos_emb,
            input_height=height,
            input_width=width,
            infer_occlusion=True)

    # for contour tracking point
    inference_time_in_ms, (forward_occ_cycle_consistency_loss, backward_occ_cycle_consistency_loss, predicted_cur_contour_indices_tensor, tracking_pos_emb, resized_height, resized_width) = uflow_utils.time_it(f, execute_once_before=eval_count == 1)
    inference_times.append(inference_time_in_ms)

    # ---------------------- process forward_id_assign by contour length ----------------------
    prev_seg_points = tf.expand_dims(seg_points_batch[0,:,:2], axis=0)
    cur_seg_points = tf.expand_dims(seg_points_batch[1,:,:2], axis=0)
    prev_seg_points_mask = tf.expand_dims(seg_points_batch[0, :, -1], axis=0)
    cur_seg_points_mask = tf.expand_dims(seg_points_batch[1, :, -1], axis=0)

    prev_seg_points_limit = tracking_utils.get_first_occurrence_indices(prev_seg_points[:,:,0], -0.1)[0]  # index 0 is fine since batch size is 1
    cur_seg_points_limit = tracking_utils.get_first_occurrence_indices(cur_seg_points[:,:,0], -0.1)[0]
    
    # ----------------------

    # pred_back_spatial_tracking_points = cur_seg_points + backward_spatial_offset
    # pred_spatial_tracking_points = prev_seg_points + forward_spatial_offset
    # ----------------------- Metrics -----------------------
    if evaluate_bool:  # evaluation on validation set during training
        # perform validation for every epoch
        list_of_forward_occ_cycle_consistency_loss.append(forward_occ_cycle_consistency_loss)
        list_of_backward_occ_cycle_consistency_loss.append(backward_occ_cycle_consistency_loss)

    else:  # prediction on test set after training
        # convert id_assign to x,y points in image space
        id_assign1 = tracking_points_batch[0].numpy()
        id_assign2 = tracking_points_batch[1].numpy()
        seg_point1 = seg_points_batch[0].numpy()
        seg_point2 = seg_points_batch[1].numpy()
        pred_cur_contour_indices = predicted_cur_contour_indices_tensor[0].numpy()

        gt_tracking_points = seg_point2[id_assign2[:, 0], :2]  # index 0 to remove the last dimension

        # For Forward contour tracking
        if prev_id_assign is None:  # If this is the first frame, use the GT points in id_assign1
          np_cur_id_assign = pred_cur_contour_indices[id_assign1[:, 0]]
        else:
          np_cur_id_assign = pred_cur_contour_indices[prev_id_assign]

        # use the closest contour id to get the pred_tracking_points
        pred_tracking_points = seg_point2[np_cur_id_assign, :2]
        prev_id_assign = np_cur_id_assign

        # ------------------------- get normal vectors -------------------------
        def get_protrusion_along_normal_vector(prev_contour_points, cur_contour_points, id_assign1, np_cur_id_assign):
            two_left_contour_points = np.roll(prev_contour_points, shift=-2, axis=0)
            tangent_vectors = (two_left_contour_points - prev_contour_points) / 2

            normal_vectors = np.stack([-tangent_vectors[:-2, 1], tangent_vectors[:-2, 0]], axis=-1)  # ignore last two indices due to shift above
            # repeated_normal_vectors = tf.repeat( normal_vectors, pred_offsets.shape[0], axis=0)
            unit_normal_vectors = normal_vectors / (np.expand_dims(np.linalg.norm(normal_vectors, axis=-1), -1) + 0.0000001)
            # put back first and last index that were shifted
            unit_normal_vectors = np.append(unit_normal_vectors, [[0, 0]], axis=0)
            unit_normal_vectors = np.insert(unit_normal_vectors, 0, [0, 0], axis=0)

            delta_along_normal_vector_list = []
            for prev_contour_point_index, cur_contour_point_index in zip(id_assign1, np_cur_id_assign):
                prev_contour_point_index = prev_contour_point_index[0]
                cur_contour_point = cur_contour_points[cur_contour_point_index]
                cur_x, cur_y, cur_mask = cur_contour_point[0], cur_contour_point[1], cur_contour_point[2]
                prev_contour_point = prev_contour_points[prev_contour_point_index]
                prev_x, prev_y, prev_mask = prev_contour_point[0], prev_contour_point[1], prev_contour_point[2]

                assert cur_mask == 1
                assert prev_mask == 1

                delta_x = cur_x - prev_x
                delta_y = cur_y - prev_y

                delta_along_normal_vector = np.dot([delta_x, delta_y], unit_normal_vectors[prev_contour_point_index])
                delta_along_normal_vector = np.round(delta_along_normal_vector, decimals=3)
                delta_along_normal_vector_list.append(delta_along_normal_vector)

            # for each point, get delta
            return delta_along_normal_vector_list

        delta_along_normal_vector = get_protrusion_along_normal_vector(seg_point1, seg_point2, id_assign1, np_cur_id_assign)
        if cur_index == 0:
            list_of_delta_along_normal_vector = delta_along_normal_vector
        else:
            list_of_delta_along_normal_vector = list_of_delta_along_normal_vector + delta_along_normal_vector
        # --------------------------- Case2 Ends -----------------------------------
        # if (cur_index+2) % 5 == 0:
        if plot_dir and plot_count < num_plots:
          plot_count += 1
          save_dir = f'{plot_dir}/pred_tracking_points'

          uflow_plotting.predict_tracking_plot(save_dir,
                                               plot_count,
                                               image_batch[0].numpy(),
                                               image_batch[1].numpy(),
                                               segmentation_batch[0].numpy(),
                                               segmentation_batch[1].numpy(),
                                               seg_points_batch[0].numpy(),
                                               seg_points_batch[1].numpy(),
                                               gt_tracking_points,
                                               pred_tracking_points,
                                               num_plots,
                                               frame_skip=None)

          # Dense correspondence figure for manuscript
          # uflow_plotting.save_tracked_contour_indices(save_dir, plot_count, num_plots, np_all_cur_id_assign[:,0])
          # uflow_plotting.predict_tracking_plot_manuscript(save_dir,
          #                                                 plot_count,
          #                                                 image_batch[0].numpy(),
          #                                                 image_batch[1].numpy(),
          #                                                 seg_points_batch[0].numpy(),
          #                                                 seg_points_batch[1].numpy(),
          #                                                 pred_spatial_tracking_points[0],
          #                                                 num_plots)

        # ---------------- Evaluate F Metric, warp error, and spatial accuracy ------------------
        image_height, image_width = image_batch.shape[1:3]

        for sa_thershold in list_of_sa_at_thresholds.keys():
          a_spatial_accuracy = metrics.spatial_accuracy(gt_tracking_points, pred_tracking_points, image_width, image_height, thresh=sa_thershold)
          list_of_sa_at_thresholds[sa_thershold].append(a_spatial_accuracy)


        # SA depending on the contour expansion and contraction
        points_spatial_accuracy = metrics.point_wise_spatial_accuracy(gt_tracking_points, pred_tracking_points, image_width, image_height, thresh=0.02)
        if cur_index == 0:
            list_of_points_spatial_accuracy = points_spatial_accuracy.tolist()
        else:
            list_of_points_spatial_accuracy = list_of_points_spatial_accuracy + points_spatial_accuracy.tolist()

        if seg_point2[-1,2] == 1:  # entire vector is filled with valid contour points
            total_contour_length = seg_point2.shape[0]
        else:
            total_contour_length = np.where(seg_point2[:,2] == 0)[0][0]
        
        for ca_thershold in list_of_ca_at_thresholds.keys():
          a_contour_accuracy = metrics.contour_accuracy(id_assign2[:, 0], np_cur_id_assign, total_contour_length, thresh=ca_thershold)
          list_of_ca_at_thresholds[ca_thershold].append(a_contour_accuracy)

  if progress_bar:
    sys.stdout.write('\n')
    sys.stdout.flush()

  if evaluate_bool:
      # For Training, evaluate on validation
      eval_stop_in_s = time.time()
      results = {
          'val_backward_occ_cycle_consistency_loss': my_tf_round(tf.math.reduce_mean(list_of_backward_occ_cycle_consistency_loss), 4).numpy(),
          'val_forward_occ_cycle_consistency_loss': my_tf_round(tf.math.reduce_mean(list_of_forward_occ_cycle_consistency_loss), 4).numpy(),
          'val-inf-time(ms)': np.mean(inference_times),
          'val-eval-time(s)': eval_stop_in_s - eval_start_in_s
      }
      return results

  else:
      # For Prediction: Save Evaluation
      # uflow_plotting.plot_cum_accuracy_from_first_to_last_frame(plot_dir, list_of_sa, list_of_ca)

      save_filename = 'eval_results'
      print('-----------', save_filename, '-----------')
      
      for sa_thershold in list_of_sa_at_thresholds.keys():
       print(f'Spatial Accuracy_{sa_thershold}: ', round(mean(list_of_sa_at_thresholds[sa_thershold]), 4))
      # print('Relative Spatial Accuracy: ', round(mean(list_of_rsa), 4))
      # print('Temporal Accuracy: ', round(mean(list_of_ta), 4))
      for ca_thershold in list_of_ca_at_thresholds.keys():
       print(f'Contour Accuracy_{ca_thershold}: ', round(mean(list_of_ca_at_thresholds[ca_thershold]), 4))
      save_dict = {'spatial_accuracy_02': list_of_sa_at_thresholds[0.02],
                   'spatial_accuracy_04': list_of_sa_at_thresholds[0.04],
                   'spatial_accuracy_06': list_of_sa_at_thresholds[0.06],
                  #  'relative_spatial_accuracy': list_of_rsa,
                  #  'temporal accuracy': list_of_ta,
                   'contour_accuracy_01': list_of_ca_at_thresholds[0.01],
                   'contour_accuracy_02': list_of_ca_at_thresholds[0.02],
                   'contour_accuracy_03': list_of_ca_at_thresholds[0.03],
                   'list_of_delta_along_normal_vector': list_of_delta_along_normal_vector,
                   'list_of_points_spatial_accuracy':list_of_points_spatial_accuracy}

      with open(f'{plot_dir}/{save_filename}.txt', 'w') as f:
          print(save_dict, file=f)

      # save_eval_results(list_of_results, plot_dir, 'eval_results')

      eval_stop_in_s = time.time()
      print('---------------------------------------')
      print('inf-time(ms): ', np.mean(inference_times))
      print('eval-time(s): ', eval_stop_in_s - eval_start_in_s)


def my_tf_round(x, decimals=0):
  multiplier = tf.constant(10 ** decimals, dtype=x.dtype)
  return tf.round(x * multiplier) / multiplier


def save_eval_results(list_of_results, save_dir, save_filename):
    list_of_f = [a_tuple[0] for a_tuple in list_of_results]
    list_of_precision = [a_tuple[1] for a_tuple in list_of_results]
    list_of_recall = [a_tuple[2] for a_tuple in list_of_results]

    print('-----------', save_filename, '-----------')
    print('F: ', round(mean(list_of_f), 4))
    print('Precision: ', round(mean(list_of_precision), 4))
    print('Recall: ', round(mean(list_of_recall), 4))
    save_dict = {'f': list_of_f, 'precision': list_of_precision, 'recall': list_of_recall}

    if len(list_of_results[0]) >= 4:
        list_of_warp_error = [a_tuple[3] for a_tuple in list_of_results]
        print('Warp Error: ', round(mean(list_of_warp_error), 0))
        save_dict['warp_error'] = list_of_warp_error

        if len(list_of_results[0]) >= 5:
            list_of_spatial_accuracy = [a_tuple[4] for a_tuple in list_of_results]
            print('Spatial Accuracy: ', round(mean(list_of_spatial_accuracy), 4))
            save_dict['spatial_accuracy'] = list_of_spatial_accuracy

    with open(f'{save_dir}/{save_filename}.txt', 'w') as f:
        print(save_dict, file=f)


def list_eval_keys(prefix=''):
  """List the keys of the dictionary returned by the evaluate function."""
  keys = [
      'EPE', 'ER', 'inf-time(ms)', 'eval-time(s)', 'occl-f-max',
      'best-occl-thresh'
  ]
  if prefix:
    return [prefix + '-' + k for k in keys]
  return keys

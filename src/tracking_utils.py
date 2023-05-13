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

'''
Tensorflow implementation of PoST/utils/proc.py
Also includes some of my custom loss functions
'''

from PIL import Image, ImageStat
import cv2
import numpy as np
import math
import tensorflow as tf
import collections


# ------------------------------ Loss ------------------------------------

def robust_l2(x):
    return (x ** 2 + 0.001 ** 2)


def robust_l1(x):
    return (x ** 2 + 0.001 ** 2) ** 0.5


def occ_cycle_consistency_loss(prev_gt_occ, pred_occ):
    '''
    prev_gt_occ.shape TensorShape([8, sample_points, nearby_points])
    pred_occ.shape  TensorShape([8, sample_points, nearby_points])
    '''
    cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    error = cce(prev_gt_occ, pred_occ)   # robust_l2(prev_gt_occ - pred_occ)
    a_loss = tf.reduce_mean(error)
    
    return a_loss


def corr_2d_loss( gt_prev_id_assignments, gt_cur_id_assignments, corr_2d_matrix):
    # classification loss only for tracked points
    scce = tf.keras.losses.SparseCategoricalCrossentropy()

    sampled_corr_2d_matrix = sample_data_by_index(gt_prev_id_assignments, corr_2d_matrix)

    a_loss = scce(gt_cur_id_assignments[:,:,0], sampled_corr_2d_matrix)

    return a_loss

def corr_cycle_consistency(forward_corr_2d_matrix, backward_corr_2d_matrix):
    '''

    :param forward_corr_2d_matrix: shape (B, 1150, 1150), dtype float32
    :param backward_corr_2d_matrix: shape (B, 1150, 1150), dtype float32
    :return:
    '''
    scce = tf.keras.losses.SparseCategoricalCrossentropy()

    trans_backward_corr_2d_matrix = tf.transpose(backward_corr_2d_matrix, perm=[0, 2, 1])

    forawrd_loss = scce(tf.math.argmax(forward_corr_2d_matrix, axis=-1), trans_backward_corr_2d_matrix)
    backward_loss = scce(tf.math.argmax(trans_backward_corr_2d_matrix, axis=-1), forward_corr_2d_matrix)

    batch_size = tf.shape(forward_corr_2d_matrix)[0]
    num_seg_points = tf.shape(forward_corr_2d_matrix)[1]
    diagnoal_one_increasing_tensor = tf.repeat( tf.expand_dims( tf.range( num_seg_points ), axis=0), repeats=batch_size, axis=0)
    forward_diagnoal_loss = scce(diagnoal_one_increasing_tensor, forward_corr_2d_matrix)
    backward_diagnoal_loss = scce(diagnoal_one_increasing_tensor, backward_corr_2d_matrix)

    return forawrd_loss + backward_loss + forward_diagnoal_loss + backward_diagnoal_loss

# -------------------------------------------------------------------------------------------------------

def sample_data_by_index(index_tensor, data_tensor):
    '''
    sample from data_tensor by the indices inside index_tensor

    :param index_tensor: tensor of shape (B, num_tracked_points, 1), dtype int32
    :param data_tensor: tensor of shape (B, num_seg_points, channels), dtype float32

    e.g)
    index_tensor = tf.constant( [[[-1]]] , dtype='int32')
    sample_data_by_index(index_tensor, data_tensor)

    If the index in index_tensor is outside the bound of data_tensor as above,
    the output is tf.Tensor: shape=(1, 1, 1), numpy=array([[[0.]]]


    :return: tensor of shape (B, num_tracked_points, 1), dtype float32
    '''

    batch_size = tf.shape(index_tensor)[0]
    num_tracked_points = tf.shape(index_tensor)[1]
    gt_prev_id_assignment_vector = tf.reshape(index_tensor, shape=[-1])
    batch_index_list = tf.repeat(tf.range(batch_size), repeats=num_tracked_points)
    indices = tf.stack([batch_index_list, gt_prev_id_assignment_vector], axis=1)  # shape is (number of tracked points, 2)

    # sample from data_tensor by the index values inside index_tensor
    sampled_data_tensor = tf.gather_nd(data_tensor, indices)
    num_channels = tf.shape(data_tensor)[2]
    reshaped_sampled_data_tensor = tf.reshape(sampled_data_tensor, shape=[batch_size, num_tracked_points, num_channels])

    return reshaped_sampled_data_tensor


def matching_contour_points_loss(gt_prev_id_assignments, gt_cur_id_assignments, pred_id_assignments):
    """

    :param gt_prev_id_assignments: tensor of shape (B, num_tracked_points, 1), dtype int32
    :param gt_cur_id_assignments: tensor of shape (B, num_tracked_points, 1), dtype int32
    :param pred_id_assignments: tensor of shape (B, num_seg_points, 1), dtype float32

    :return: loss
    """
    # total_loss = 0
    # for batch_index in range(gt_prev_id_assignments.shape[0]):
    #     for gt_point_index in range(gt_prev_id_assignments.shape[1]):
    #         prev_seg_point_index = gt_prev_id_assignments[batch_index, gt_point_index, 0]
    #         pred_cur_seg_point_index = pred_id_assignments[batch_index, prev_seg_point_index, 0]
    #
    #         a_diff = tf.cast(gt_cur_id_assignments[batch_index, gt_point_index, 0] - pred_cur_seg_point_index, tf.float32)
    #         a_loss = robust_l1(a_diff)
    #         total_loss = total_loss + a_loss
    #
    # # average
    # a_loss = total_loss / (gt_prev_id_assignments.shape[0]*gt_prev_id_assignments.shape[1])

    # for debugging
    # index_tensor = tf.constant( [[[-1]],[[-1]],[[-1]],[[-1]],[[-1]],[[-1]],[[-1]],[[-1]]] , dtype='int32')
    # sample_data_by_index(index_tensor, pred_id_assignments)

    reshaped_sampled_pred_id_assignments = sample_data_by_index(gt_prev_id_assignments, pred_id_assignments)
    error = robust_l1( reshaped_sampled_pred_id_assignments - tf.cast( gt_cur_id_assignments, tf.float32) )
    a_loss = tf.reduce_mean(error)

    return a_loss


def matching_contour_points_spatial_metric(cur_seg_points, gt_prev_id_assignments, gt_cur_id_assignments, pred_id_assignments):
    """
    :param cur_seg_points: tensor of shape (B, num_seg_points, 2), dtype float32
    :param gt_prev_id_assignments: tensor of shape (B, num_tracked_points, 1), dtype int32
    :param gt_cur_id_assignments: tensor of shape (B, num_tracked_points, 1), dtype int32
    :param pred_id_assignments: tensor of shape (B, num_seg_points, 1), dtype float32

    :return: loss
    """
    sampled_pred_id_assignments = sample_data_by_index(gt_prev_id_assignments, pred_id_assignments)
    rounded_sampled_pred_id_assignments = tf.cast(tf.round(sampled_pred_id_assignments), dtype=tf.int32)

    a_loss = matching_spatial_points_loss(rounded_sampled_pred_id_assignments, gt_cur_id_assignments, cur_seg_points, cur_seg_points)

    return a_loss


def matching_spatial_points_loss(gt_prev_id_assignments, gt_cur_id_assignments, cur_seg_points, pred_points):

    sampled_gt_points = sample_data_by_index(gt_cur_id_assignments, cur_seg_points)
    sampled_pred_points = sample_data_by_index(gt_prev_id_assignments, pred_points)

    error = robust_l2(sampled_gt_points - sampled_pred_points)
    a_loss = tf.reduce_mean(error)

    return a_loss

# -------------------------------- Unsupervised Learning Losses -----------------------------------------------------------
def get_closest_contour_id(seg_points_xy, pred_tracking_points):
    '''
    TODO: get rid of for loop
    get the closest point in seg_points_xy with respect to pred_tracking_points

    :param seg_points_xy: shape [batch_size, num_seg_points, 2], dtype tf.float32
    :param seg_points_mask:  shape [batch_size, num_seg_points, 1], dtype tf.float32
    :param pred_tracking_points: shape [batch_size, num_sampled_points, 2], dtype tf.float32

    :return: shape [batch_size, num_seg_points, 1], dtype tf.int32
    '''
    closest_contour_id_list = []
    for a_point_index in range(pred_tracking_points.shape[1]):
        temp_pred_tracking_points = tf.expand_dims(pred_tracking_points[:, a_point_index, :], axis=1)
        diff_dist = seg_points_xy - tf.repeat(temp_pred_tracking_points, repeats=seg_points_xy.shape[1], axis=1)
        l2_dist = tf.math.reduce_euclidean_norm(diff_dist, axis=-1)
        cur_id_assign = tf.math.argmin(l2_dist, axis=1)  # shape (Batch_size) and dtype int64
        closest_contour_id_list.append(cur_id_assign)

    # (1150, 8) --> (8, 1150) --> (8, 1150, 1)
    closest_contour_id_tensor = tf.expand_dims( tf.transpose( tf.convert_to_tensor(closest_contour_id_list), perm=[1,0]), axis=-1)
    closest_contour_id_tensor = tf.cast(closest_contour_id_tensor, dtype=tf.int32)

    return closest_contour_id_tensor


def cycle_consistency_spatial_loss(prev_seg_points_xy, prev_seg_points_mask, cur_seg_points_xy, cur_seg_points_mask, forward_spatial_offset, backward_spatial_offset):
    prev_seg_points_mask = tf.expand_dims(prev_seg_points_mask, axis=-1)
    cur_seg_points_mask = tf.expand_dims(cur_seg_points_mask, axis=-1)

    pred_cur_seg_points = prev_seg_points_xy + forward_spatial_offset
    pred_prev_seg_points = cur_seg_points_xy + backward_spatial_offset

    # get the closest contour point id in the second contour
    closest_cur_seg_points_id = get_closest_contour_id(cur_seg_points_xy, prev_seg_points_mask, pred_cur_seg_points)

    # use the closest contour id above to get the pred_tracking_points
    pred_prev_tracking_points = sample_data_by_index(closest_cur_seg_points_id, pred_prev_seg_points)

    return_to_first_contour_error = robust_l2( prev_seg_points_xy - pred_prev_tracking_points ) * prev_seg_points_mask

    # to make the gradient flow in both directions,
    # get the closest contour point id in the first contour
    closest_prev_seg_points_id = get_closest_contour_id(prev_seg_points_xy, cur_seg_points_mask, pred_prev_seg_points)

    # use the closest contour id above to get the pred_tracking_points
    pred_cur_tracking_points = sample_data_by_index(closest_prev_seg_points_id, pred_cur_seg_points)

    return_to_second_contour_error = robust_l2( cur_seg_points_xy - pred_cur_tracking_points ) * cur_seg_points_mask

    # get_first_occurrence_indices(prev_seg_points_mask[:,:,0], 0.1)
    # get_first_occurrence_indices(cur_seg_points_mask[:,:,0], 0.1)
    a_loss = tf.reduce_mean(return_to_first_contour_error + return_to_second_contour_error)

    return a_loss


def cycle_consistency_assign_spatial_loss(prev_seg_points, cur_seg_points, forward_id_assignments, forward_spatial_offset):
    sampled_cur_seg_points = sample_data_by_index(tf.cast(tf.round(forward_id_assignments), dtype=tf.int32), cur_seg_points)

    pred_seg_points = prev_seg_points + forward_spatial_offset
    sampled_pred_seg_points = sample_data_by_index(tf.cast(tf.round(forward_id_assignments), dtype=tf.int32), pred_seg_points)

    error = robust_l1(sampled_cur_seg_points - sampled_pred_seg_points)
    a_loss = tf.reduce_mean(error)

    return a_loss


def cycle_consistency_assign_loss(forward_id_assignments, backward_id_assignments, prev_seg_mask, cur_seg_mask):
    """
    :param forward_id_assignments: tensor of shape (B, num_seg_points, 1), dtype float32
    :param backward_id_assignments: tensor of shape (B, num_seg_points, 1), dtype float32
    :param prev_seg_mask: tensor of shape (B, num_seg_points, 1), dtype float32
    :param cur_seg_mask: tensor of shape (B, num_seg_points, 1), dtype float32
    :return:
    """
    epsilon = 0.000000001
    contour_indices = make_contour_order_indices_from_offsets(forward_id_assignments)

    forward_id_offsets = ( forward_id_assignments - contour_indices ) * prev_seg_mask
    backward_id_offsets = ( backward_id_assignments - contour_indices ) * cur_seg_mask

    # Compute both ways so that gradients flow for both forward and backward assignments (gradient doesn't flow through tf.round)
    # index backward_id_offsets by forward_id_assignment
    sampled_backward_id_offsets = sample_data_by_index( tf.cast(tf.round(forward_id_assignments), dtype=tf.int32), backward_id_offsets)  # shape is (B, num_seg_points1, 1)
    # index forword_offsets by backward_id_assignments
    sampled_forward_id_offsets = sample_data_by_index(tf.cast(tf.round(backward_id_assignments), dtype=tf.int32), forward_id_offsets)  # shape is (B, num_seg_points2, 1)
    # normalize this loss by their magnitude to prevent the magnitude going to 0
    error_one = robust_l1( (forward_id_offsets + sampled_backward_id_offsets) / ( tf.math.abs(forward_id_offsets) + tf.math.abs(sampled_backward_id_offsets) + epsilon ) )
    error_two = robust_l1( (backward_id_offsets + sampled_forward_id_offsets) / ( tf.math.abs(backward_id_offsets) + tf.math.abs(sampled_forward_id_offsets) + epsilon) )

    a_loss = tf.reduce_mean(error_one + error_two)
    
    return a_loss


def contour_points_order_loss(pred_id_assignments):
    contour_order_indices = tf.range(tf.shape(pred_id_assignments)[1], delta=1, dtype=pred_id_assignments.dtype, name='range')
    contour_order_indices = tf.expand_dims( tf.expand_dims( contour_order_indices, axis=0), axis=-1)
    repeated_contour_order_indices = tf.repeat(contour_order_indices, repeats=tf.shape(pred_id_assignments)[0], axis=0)

    error = robust_l1(repeated_contour_order_indices - pred_id_assignments)
    a_loss = tf.reduce_mean(error)

    return a_loss

# ---------
def tracker_photometric_loss(cur_image, gt_cur_tracking_points, pred_cur_tracking_points):
    sampled_gt_image = exact_point_sampler(cur_image, gt_cur_tracking_points[:, :, 0], gt_cur_tracking_points[:, :, 1], abs_scale=True)  # (B, N, 3)
    sampled_pred_image = bilinear_sampler_1d(cur_image, pred_cur_tracking_points[:, :, 0], pred_cur_tracking_points[:, :, 1], abs_scale=True)  # (B, N, 3)

    error = robust_l1( (sampled_gt_image - sampled_pred_image) )
    a_loss = tf.reduce_mean(error)

    return a_loss


def tracker_unsupervised_photometric_loss(prev_image, cur_image, gt_prev_tracking_points, gt_prev_tracking_points_mask, pred_cur_tracking_points):
    # since the number of valid gt_prev_tracking_points and number of valid pred_cur_tracking_points are different, we need mask
    gt_prev_tracking_points_mask = tf.repeat( tf.expand_dims(gt_prev_tracking_points_mask, axis=-1), repeats=3, axis=-1)

    sampled_gt_image = exact_point_sampler(prev_image, gt_prev_tracking_points[:, :, 0], gt_prev_tracking_points[:, :, 1], abs_scale=True)  # (B, N, 3)
    sampled_pred_image = bilinear_sampler_1d(cur_image, pred_cur_tracking_points[:, :, 0], pred_cur_tracking_points[:, :, 1], abs_scale=True)  # (B, N, 3)

    error = robust_l1( (sampled_gt_image - sampled_pred_image) ) * gt_prev_tracking_points_mask
    a_loss = tf.reduce_mean(error)

    return a_loss


# -----------------------------------------------------------------------
def abs_robust_loss(diff, eps=0.01, q=0.4):
    """The so-called robust loss used by DDFlow."""
    return tf.pow((tf.abs(diff) + eps), q)

def census_transform(image, patch_size):
    """The census transform as described by DDFlow.

    See the paper at https://arxiv.org/abs/1902.09145

    Args:
      image: tensor of shape (b, h, w, c)
      patch_size: int
    Returns:
      image with census transform applied
    """
    intensities = tf.image.rgb_to_grayscale(image) * 255

    # 49x49 identity matrix to kernel [filter_height, filter_width, in_channels, out_channels]
    kernel = tf.reshape(
        tf.eye(patch_size * patch_size),
        (patch_size, patch_size, 1, patch_size * patch_size))

    neighbors = tf.nn.conv2d(
        input=intensities, filters=kernel, strides=[1, 1, 1, 1], padding='SAME')
    diff = neighbors - intensities
    # Coefficients adopted from DDFlow.
    diff_norm = diff / tf.sqrt(.81 + tf.square(diff))

    return diff_norm

def soft_hamming(a_bhwk, b_bhwk, thresh=.1):
    """A soft hamming distance between tensor a_bhwk and tensor b_bhwk.

    Args:
      a_bhwk: tf.Tensor of shape (batch, N, features)
      b_bhwk: tf.Tensor of shape (batch, N, features)
      thresh: float threshold

    Returns:
      a tensor with approx. 1 in location that are significantly
      more different than thresh and approx. 0 if significantly less
      different than thresh.
      (batch, N, 1)
    """
    sq_dist_bhwk = tf.square(a_bhwk - b_bhwk)
    soft_thresh_dist_bhwk = sq_dist_bhwk / (thresh + sq_dist_bhwk)
    return tf.reduce_sum(input_tensor=soft_thresh_dist_bhwk, axis=2, keepdims=True)
# -----------------------------------------------------------------------

def tracker_census_loss(cur_image, gt_cur_tracking_points, pred_cur_tracking_points, patch_size=5):
    """Compares the similarity of the census transform of two points"""

    census_cur_image = census_transform(cur_image, patch_size)  # B x H x W x patch_size^2

    sampled_census_gt_points = exact_point_sampler(census_cur_image, gt_cur_tracking_points[:, :, 0], gt_cur_tracking_points[:, :, 1], abs_scale=True)  # (B, N, patch_size^2)
    sampled_census_pred_points = bilinear_sampler_1d(census_cur_image, pred_cur_tracking_points[:, :, 0], pred_cur_tracking_points[:, :, 1], abs_scale=True)  # (B, N, patch_size^2)

    # soft hamming between two points
    hamming_bhw = soft_hamming(sampled_census_gt_points, sampled_census_pred_points)

    diff = abs_robust_loss(hamming_bhw)
    a_loss = tf.reduce_mean(input_tensor=diff)

    return a_loss


def tracker_unsupervised_census_loss(prev_image, cur_image, gt_prev_tracking_points, gt_prev_tracking_points_mask, pred_cur_tracking_points, patch_size=5):
    """Compares the similarity of the census transform of two points"""
    census_prev_image = census_transform(prev_image, patch_size)  # B x H x W x patch_size^2
    census_cur_image = census_transform(cur_image, patch_size)  # B x H x W x patch_size^2

    sampled_census_gt_points = exact_point_sampler(census_prev_image, gt_prev_tracking_points[:, :, 0],
                                                   gt_prev_tracking_points[:, :, 1],
                                                   abs_scale=True)  # (B, N, patch_size^2)
    sampled_census_pred_points = bilinear_sampler_1d(census_cur_image, pred_cur_tracking_points[:, :, 0],
                                                     pred_cur_tracking_points[:, :, 1],
                                                     abs_scale=True)  # (B, N, patch_size^2)

    # soft hamming between two points
    # since the number of valid gt_prev_tracking_points and number of valid pred_cur_tracking_points are different, we need mask
    hamming_bhw = soft_hamming(sampled_census_gt_points, sampled_census_pred_points)

    diff = abs_robust_loss(hamming_bhw) * tf.expand_dims(gt_prev_tracking_points_mask, axis=-1)
    a_loss = tf.reduce_mean(diff)

    return a_loss


# -------------------------------------------------------------------------------------------------------

def snake_loss_from_offsets(gt_prev_points, gt_prev_points_mask, pred_offsets):
    '''
    :param gt_prev_points: (B, N, 2), float32
    :param gt_prev_points_mask: (B,N), float32
    :param pred_offsets: (B, N, 2), float32
    :return:
    '''
    pred_points = gt_prev_points + pred_offsets

    left_pred_points = tf.roll(pred_points, shift=-1, axis=1)  # 0th index element goes to the last index
    point_diff_left = left_pred_points - pred_points  # x2 - x1, x3 - x2, ..., xn - x(n-1)
    point_diff_left = point_diff_left[:, :-1, :]  # trim shifted point
    point_diff_left_norm = tf.math.reduce_sum(point_diff_left ** 2, axis=-1)  # tf.maximum to prevent nan due to tf.math.sqrt

    valid_point_diff_left_norm = point_diff_left_norm * gt_prev_points_mask[:, :-1]  # to remove padded points at the end, ignore last one index due to shift above

    # this is to remove the large negative value padded points in the middle of the index
    last_valid_index_tensor = tf.cast( tf.argmax(tf.math.abs(valid_point_diff_left_norm), axis=-1) )
    batch_size = valid_point_diff_left_norm.shape[0]

    total_tension = 0
    total_stiffness = 0
    for batch_index in range(batch_size):
        if tf.reduce_sum(gt_prev_points_mask[batch_index]) < gt_prev_points_mask.shape[1]:
            last_valid_index = last_valid_index_tensor[batch_index]
        else:
            last_valid_index = gt_prev_points_mask.shape[1] - 1

        total_tension = total_tension + tf.reduce_mean(valid_point_diff_left_norm[batch_index, :last_valid_index])
        a_stiffness = tf.reduce_mean( tf.math.reduce_sum(( point_diff_left[batch_index, 1:last_valid_index] - point_diff_left[batch_index, :(last_valid_index-1)]) ** 2, axis=-1) )
        total_stiffness = total_stiffness + a_stiffness

    tension_loss = total_tension / batch_size  # continuity/tension term
    total_stiffness = total_stiffness / batch_size  # smoothness/stiffness term

    return tension_loss, total_stiffness


def mechanical_loss_from_offsets(gt_prev_points, gt_prev_points_mask, pred_offsets):
    '''

    :param gt_prev_points: (B, N, 2), float32
    :param gt_prev_points_mask: (B,N), float32
    :param pred_offsets: (B, N, 2), float32
    :return:
    '''
    # ------------ Linear Spring Force ------------
    pred_points = gt_prev_points + pred_offsets

    left_pred_points = tf.roll(pred_points, shift=-1, axis=1)
    point_diff = robust_l1(pred_points - left_pred_points)
    point_diff = point_diff[:, :-1,:] # trim shifted point
    point_diff_norm = tf.math.sqrt( tf.maximum(tf.math.reduce_sum(point_diff ** 2, axis=-1), 1e-9) )  # tf.maximum to prevent nan due to tf.math.sqrt

    average_length = tf.math.reduce_mean(point_diff_norm, axis=-1)  # shape (batch_size), dtype float32

    valid_point_diff_norm = point_diff_norm * gt_prev_points_mask[:,:-1]  # to remove padded points at the end, ignore last one index due to shift above

    # this is to remove the large negative value padded points in the middle of the index
    last_valid_index_tensor = tf.cast( tf.argmax(tf.math.abs(valid_point_diff_norm), axis=-1), dtype=tf.int32)

    total_linear_spring_force = 0
    batch_size = valid_point_diff_norm.shape[0]
    for batch_index in range(batch_size):
        if tf.reduce_sum(gt_prev_points_mask[batch_index]) < gt_prev_points_mask.shape[1]:
            last_valid_index = last_valid_index_tensor[batch_index]
        else:
            last_valid_index = gt_prev_points_mask.shape[1]
        total_linear_spring_force = total_linear_spring_force + tf.reduce_mean( robust_l1( valid_point_diff_norm[batch_index, :last_valid_index] - average_length[batch_index] ) )

    linear_spring_loss = total_linear_spring_force / batch_size
    # ------------ Normal Torsion Force -----------
    # Compute tangent by central differences. You can use a closed form tangent if you have it.
    # referred https://stackoverflow.com/questions/66676502/how-can-i-move-points-along-the-normal-vector-to-a-curve-in-python
    two_left_gt_prev_points = tf.roll(gt_prev_points, shift=-2, axis=1)
    tangent_vectors = (two_left_gt_prev_points - gt_prev_points) / 2

    normal_vectors = tf.stack([-tangent_vectors[:, :-2, 1], tangent_vectors[:, :-2, 0]], axis=-1)  # ignore last two indices due to shift above
    # repeated_normal_vectors = tf.repeat( normal_vectors, pred_offsets.shape[0], axis=0)
    unit_normal_vectors = normal_vectors / ( tf.expand_dims(tf.norm(normal_vectors, axis=-1), -1) + 0.0000001)

    unit_pred_offsets = pred_offsets[:, 1:-1, :] / ( tf.expand_dims(tf.norm(pred_offsets[:, 1:-1, :], axis=-1), -1) + 0.0000001)

    error = robust_l1( unit_normal_vectors - unit_pred_offsets ) * tf.expand_dims(gt_prev_points_mask[:,:-2], axis=-1) # to remove padded points at the end, ignore last two indices due to shift above
    normal_tendancy_loss = tf.reduce_mean( error )

    return linear_spring_loss, normal_tendancy_loss


# def mechanical_loss(gt_prev_points, pred_points):
#     '''
#
#     :param gt_prev_points: (B, N, 2)
#     :param pred_points: (B, N, 2)
#     :return:
#     '''
#     # ------------ Side Spring Force ------------
#     left_pred_points = tf.roll(pred_points, shift=1, axis=1)
#     trimed_left_pred_points = left_pred_points[0, 1:, :]  # trim left boundary point after roll
#
#     point_diff = tf.math.abs(pred_points[0, 1:, :] - trimed_left_pred_points)
#     distance_diff = tf.math.sqrt(tf.math.reduce_sum(point_diff ** 2, axis=1))
#     average_length = tf.math.reduce_mean(distance_diff)
#
#     linear_spring_force = (distance_diff - average_length)
#     left_linear_spring_force = linear_spring_force[1:]
#     right_linear_spring_force = linear_spring_force[:-1]
#
#     linear_spring_loss = tf.math.abs(tf.reduce_mean(left_linear_spring_force - right_linear_spring_force))
#
#     # ------------ Normal Tendancy Force -----------
#     # unit_w_vector = tf.linalg.normalize(gt_prev_points - pred_points, axis=1)[0]
#     w_vector = gt_prev_points - pred_points
#     w_vector = w_vector[0, 1:-1, :]
#
#     # Compute tangent by central differences. You can use a closed form tangent if you have it.
#     # referred https://stackoverflow.com/questions/66676502/how-can-i-move-points-along-the-normal-vector-to-a-curve-in-python
#     two_left_gt_prev_points = tf.roll(gt_prev_points, shift=2, axis=1)
#     tangent_vectors = (two_left_gt_prev_points - gt_prev_points) / 2
#
#     normal_vectors = tf.stack([-tangent_vectors[0, 2:, 1], tangent_vectors[0, 2:, 0]], axis=1)
#     # unit_normal_vectors = tf.linalg.normalize(normal_vectors, axis=1)[0]
#
#     normal_tendancy_loss = tf.math.abs(tf.reduce_mean(normal_vectors - w_vector))
#
#     return linear_spring_loss, normal_tendancy_loss

# ------------------------------------------------------------------------

def bayesian_points_loss(gt_points, pred_points, uncertainty):
    error = robust_l1(gt_points - pred_points) * tf.math.exp(-1*uncertainty) + uncertainty
    a_loss = tf.reduce_mean(error)

    return a_loss


def multi_level_points_loss(gt_points, saved_offset):
    total_loss = 0
    gamma = 0.5
    total_level = len(saved_offset)
    for cur_level, a_offset in saved_offset.items():
        pred_points = a_offset[:, :, :2]
        a_uncertainty = a_offset[:, :, -1]
        a_uncertainty = tf.expand_dims(a_uncertainty, axis=-1)

        total_loss = total_loss + bayesian_points_loss(gt_points, pred_points, a_uncertainty) * gamma**(total_level-cur_level)

        # total_loss = total_loss + matching_points_loss(gt_points, pred_points) * gamma**(total_level-cur_level)

    return total_loss


def pixel_matching_loss(gt_image, pred_image, global_transformed_points):
    # for global alignment in PoST
    image_size = gt_image.shape[-2:-4:-1]
    rel_global_transformed_points = abs2rel(global_transformed_points, image_size)
    a_mask = cnt2mask(rel_global_transformed_points, image_size)
    a_mask = tf.cast(a_mask > 0, tf.float32)
    total_area_of_mask = tf.reduce_sum(a_mask)

    a_loss_map = tf.reduce_sum(tf.math.multiply(robust_l2(gt_image - pred_image), a_mask) / total_area_of_mask)
    a_loss = tf.reduce_sum(a_loss_map) * 255

    return a_loss


def offset_regularization_loss(a_offset):
    # penalize offset all pointing to the same direction
    offset_loss = tf.math.reduce_sum(a_offset, axis=1)
    offset_loss = tf.math.abs(offset_loss)
    offset_loss = tf.math.reduce_sum(offset_loss)

    return offset_loss

# -----------------------------------------------------------------------
# ------------------------ Utility functions ----------------------------
# -----------------------------------------------------------------------

def make_contour_order_indices_from_offsets(id_assign_offsets):
    contour_order_indices = tf.range(tf.shape(id_assign_offsets)[1], delta=1, dtype=id_assign_offsets.dtype)
    contour_order_indices = tf.expand_dims(tf.expand_dims(contour_order_indices, axis=0), axis=-1)
    repeated_contour_order_indices = tf.repeat(contour_order_indices, repeats=tf.shape(id_assign_offsets)[0], axis=0)

    return repeated_contour_order_indices


def linear_seq_increase(iteration, max_seq):
    '''
    Rewrite below more succintly

    # cur_seq = 2
    # if iteration > 400:
    #     cur_seq = 3
    # elif iteration > 800:
    #     cur_seq = 4
    # elif iteration > 1200:
    #     cur_seq = 5

    :param iteration:
    :param max_seq:
    :return:
    '''

    min_seq = 2
    lowest_iter_bound = 5
    cur_seq = tf.math.minimum(iteration, lowest_iter_bound*(max_seq-min_seq)) // lowest_iter_bound + min_seq
    # cur_seq = min(iteration, lowest_iter_bound*(max_seq-min_seq)) // lowest_iter_bound + min_seq

    return cur_seq

def get_first_occurrence_indices(sequence, eos_idx):
    '''
    https://stackoverflow.com/questions/42184663/how-to-find-an-index-of-the-first-matching-element-in-tensorflow
    args:
      sequence: [batch, length]
      eos_idx: scalar
    return:
     tf.tensor with shape (batch), dtype tf.int32
    '''
    batch_size, maxlen = sequence.get_shape().as_list()
    eos_idx = tf.convert_to_tensor(eos_idx)
    tensor = tf.concat( [sequence, tf.tile(eos_idx[None, None], [batch_size, 1])], axis=-1)
    index_all_occurrences = tf.where(tf.math.less(tensor, eos_idx))
    index_all_occurrences = tf.cast(index_all_occurrences, tf.int32)
    index_first_occurrences = tf.math.segment_min(index_all_occurrences[:, 1],
                                           index_all_occurrences[:, 0])
    index_first_occurrences.set_shape([batch_size])
    index_first_occurrences = tf.minimum(index_first_occurrences, maxlen)

    return index_first_occurrences


def normalize_each_row(a_tensor):
    '''
    :param a_tensor: shape (batch_size, seg_point_num, channels)
    :return: normalized_tensor: shape (batch_size, seg_point_num, channels)
    '''
    epsilon = 0.0000001

    min_tensor = tf.reduce_min(a_tensor, axis=-1)
    min_tensor = tf.expand_dims( min_tensor, axis=-1)  # batch_size, seg_point_num, 1
    max_tensor = tf.reduce_max(a_tensor, axis=-1)
    max_tensor = tf.expand_dims( max_tensor, axis=-1)  # batch_size, seg_point_num, 1
    normalized_tensor = tf.math.divide( tf.subtract( a_tensor, min_tensor), tf.subtract( max_tensor, min_tensor ) + epsilon )

    return normalized_tensor


def fit_id_assignments_to_next_contour(id_assignments, prev_seg_mask, cur_seg_max_index):
    '''
    :param id_assignments: shape (batch_size, seg_point_num, 1)
    :param cur_seg_max_index:  TensorShape(), tf.int32
    :return:
    '''

    cur_seg_max_index = tf.cast(cur_seg_max_index-1, id_assignments.dtype)
    total_num_seg_point = tf.shape(id_assignments)[1]
    if cur_seg_max_index.shape == []:
        cur_seg_max_index = tf.expand_dims(cur_seg_max_index, axis=-1)
    cur_seg_max_index = tf.expand_dims(cur_seg_max_index, axis=-1)
    cur_seg_max_index = tf.expand_dims(cur_seg_max_index, axis=-1)
    cur_seg_max_index = tf.repeat( cur_seg_max_index, repeats=total_num_seg_point, axis=1)  # batch_size, total_num_seg_point, 1

    masked_id_assignments = id_assignments * prev_seg_mask
    max_id = tf.reduce_max(masked_id_assignments, axis=1) # batch_size, 1
    max_id = tf.expand_dims( max_id, axis=1)  # batch_size, 1, 1
    max_id = tf.repeat( max_id, repeats=total_num_seg_point, axis=1)  # batch_size, total_num_seg_point, 1

    fit_id_assignments = (masked_id_assignments / max_id ) * cur_seg_max_index  # batch_size, total_num_seg_point, 1

    fit_id_assignments = fit_id_assignments + (prev_seg_mask - 1)  # so that masked id_assignments are negative

    return fit_id_assignments


def resize_seg_points(orig_height, orig_width, new_height, new_width, tracking_points):
    '''
    Find new coordinate of GT tracking points on the resized image

    :param orig_height:
    :param orig_width:
    :param new_height:
    :param new_width:
    :param tracking_points: are in (x, y) order
    :return:
    '''

    if tf.is_tensor(orig_height):
        orig_height = tf.cast(orig_height, dtype='float32')
    else:
        orig_height = tf.constant(orig_height, dtype='float32')
    if tf.is_tensor(orig_width):
        orig_width = tf.cast(orig_width, dtype='float32')
    else:
        orig_width = tf.constant(orig_width, dtype='float32')

    if not tf.is_tensor(new_height):
        new_height = tf.constant(new_height, dtype='float32')
    if not tf.is_tensor(new_width):
        new_width = tf.constant(new_width, dtype='float32')

    Ry = new_height / orig_height
    Rx = new_width / orig_width
    tracking_points = tf.cast(tracking_points, dtype='float32')
    
    x_points = tracking_points[:, :, 0] * Rx  # x-axis (width) column
    y_points = tracking_points[:, :, 1] * Ry  # y-axis (height) row
    points_tensor = tf.round(tf.stack([x_points, y_points], axis=-1))
    # tf.print(Ry, Rx, orig_height, orig_width, new_height, new_width)

    masking_tensor = tracking_points[:, :, 2:]
    
    return tf.concat([points_tensor, masking_tensor], axis=-1)


def rel2abs(cnt, size):
    cnt_intermediate = cnt / 2 + 0.5
    cnt_x = cnt_intermediate[..., 0] * size[0]
    cnt_y = cnt_intermediate[..., 1] * size[1]
    cnt_out = tf.stack([cnt_x, cnt_y], axis=-1)

    return cnt_out


def abs2rel(cnt, size):
    cnt = tf.cast(cnt, dtype='float32')

    cnt_x = tf.divide(cnt[..., 0], size[0])
    cnt_y = tf.divide(cnt[..., 1], size[1])
    cnt_xy = tf.stack([cnt_x, cnt_y], axis=-1)
    cnt_out = (cnt_xy - 0.5) * 2  # [-1 1]

    return cnt_out


def cnt2mask(cnt, size):
    '''
    Range of the intensity of the mask is 0 ~ 1
    :param cnt:
    :param size: Width, Height
    :return:
    '''
    B, N, _ = cnt.shape

    abs_cnt = tf.math.round(rel2abs(cnt, size))
    masks = []
    for i in range(B):
        mask = np.zeros((size[1], size[0], 3))
        abs_cnt_np = abs_cnt[i].numpy().astype('int32')
        mask = cv2.drawContours(mask, [abs_cnt_np], -1, (1, 1, 1), cv2.FILLED)
        mask = tf.convert_to_tensor(np.mean(mask, axis=-1), dtype='float32')  # H, W
        masks.append(tf.expand_dims(mask, axis=-1))  # H, W, 1
    masks = tf.stack(masks, axis=0)  # B, H, W, 1

    return masks


def transform(img, cnt, theta):
    img_align = transform_img(img, theta)
    cnt_align = transform_cnt(cnt, theta)
    return img_align, cnt_align


def transform_img(img, theta):
    # inv_theta = invert(theta)
    grid = affine_grid_generator(*img.shape[:3], theta)
    img_align = bilinear_sampler_2d(*img.shape[:3], img, grid[:,:,:,0], grid[:,:,:,1])

    return img_align


def transform_cnt(cnt, theta):
    # cnt: B, N, 2
    # theta: B, 2, 3

    B, N, _ = cnt.shape
    cnt_align = tf.concat((cnt, tf.ones_like(cnt)[..., :1]), axis=2)  # B, N, 3
    theta = tf.transpose(theta, perm=[0, 2, 1])  # B, 3, 2
    theta = tf.cast(theta, 'float32')
    cnt_align = tf.linalg.matmul(cnt_align, theta)  # B, N, 2

    return cnt_align


def cnt2poly(cnt):
    x_min = tf.math.reduce_min(cnt[..., 0], axis=1, keepdims=True)
    y_min = tf.math.reduce_min(cnt[..., 1], axis=1, keepdims=True)
    xy_min = tf.stack( (x_min, y_min), axis=-1)
    expanded_xy_min = tf.broadcast_to(xy_min, tf.shape(cnt))  # B, N, C

    poly = cnt - expanded_xy_min
    poly = tf.cast(poly, 'float32')

    return poly


def get_adj_ind(n_adj, n_nodes):
    '''
    :param n_adj: number of neighboring points
    :param n_nodes: total number of tracking points
    :return: (n_nodes, n_adj) indices
    e.g.
    array([[5, 6, 1, 2],
       [6, 0, 2, 3],
       [0, 1, 3, 4],
       [1, 2, 4, 5],
       [2, 3, 5, 6],
       [3, 4, 6, 0],
       [4, 5, 0, 1]], dtype=int32)>
    '''
    ind = tf.convert_to_tensor([i for i in range(-n_adj // 2, n_adj // 2 + 1) if i != 0], dtype='int32')
    ind = (tf.range(n_nodes)[:, None] + ind[None]) % n_nodes

    return ind

def erode(mask, it=10):
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=it)
    return mask


def dilate(mask, it=10):
    if it > 0:
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=it)
    return mask


def sample_cnt(cnt, num_cp):
    # cnt: N_all, 2
    cnt = cnt.clone()
    N_all = tf.shape(cnt)[0]

    select = np.linspace(0, N_all - (N_all // num_cp), num_cp, dtype=np.int64)
    sample = cnt[select]  # N, 2
    return sample


def sample_cnt_with_idx(cnt, idx, num_cp):
    total_num = cnt.shape[0]
    select = np.linspace(0, total_num - 1, num_cp).astype(np.int64)
    sampled_idx = ((idx / total_num) * num_cp).astype(np.int64)
    select[sampled_idx] = idx
    sampled_cnt = cnt[select]  # N, 2
    sampled_cnt = sampled_cnt.reshape(-1, 1, 2)

    return sampled_cnt, sampled_idx


def update_cnt(cnt_out, cnt_smpl, start):
    B = cnt_out.shape[0]
    N = cnt_out.shape[1]
    N_smpl = cnt_smpl.shape[1]

    skip = N // N_smpl

    cnt_out[:, start::skip] = cnt_smpl

    return cnt_out


# def upsample_offset(offset, stride=2):
#     B, N, _, _ = offset.shape
#
#     offset_up = offset.clone()
#     offset_up = offset_up.expand(B, N, stride, 2).contiguous().view(B, N * stride, 1, 2)  # B, N*stride, 1, 2
#     offset_up = torch.cat([offset_up.roll(i, dims=1) for i in range(stride)], dim=2)  # B, N*stride, stride, 2
#     offset_up = torch.mean(offset_up, dim=2, keepdim=True)  # B, N*stride, 1, 2)
#     return offset_up


def invert(theta):
    det = theta[:, 0, 0] * theta[:, 1, 1] - theta[:, 0, 1] * theta[:, 1, 0]
    adj_x = -theta[:, 1, 1] * theta[:, 0, 2] + theta[:, 0, 1] * theta[:, 1, 2]
    adj_y = theta[:, 1, 0] * theta[:, 0, 2] - theta[:, 0, 0] * theta[:, 1, 2]

    inv_theta = tf.convert_to_tensor( [ [theta[:, 1, 1], -theta[:, 0, 1], adj_x], [-theta[:, 1, 0], theta[:, 0, 0], adj_y] ] , dtype=theta.dtype)
    # inv_theta[:, 0, 0] = theta[:, 1, 1]
    # inv_theta[:, 1, 1] = theta[:, 0, 0]
    # inv_theta[:, 0, 1] = -theta[:, 0, 1]
    # inv_theta[:, 1, 0] = -theta[:, 1, 0]
    # inv_theta[:, 0, 2] = adj_x
    # inv_theta[:, 1, 2] = adj_y

    inv_theta = inv_theta / tf.reshape(det, shape=(-1, 1, 1))

    return inv_theta


def generate_pos_emb(num_pos):

    emb = np.arange(0, num_pos, 1, dtype=np.float) * 2 * math.pi / num_pos
    sin_p = np.sin(emb)  # N
    cos_p = np.cos(emb)  # N
    emb = np.stack([sin_p, cos_p], axis=1)  # N, 2
    emb = tf.convert_to_tensor(emb, dtype='float32') # torch.FloatTensor(emb)

    return emb


def crop_and_resize(img, cbox, context, size,
                    margin=0, correct=False, resample=Image.BILINEAR):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    assert (isinstance(img, Image.Image))
    assert (isinstance(cbox, np.ndarray))
    min_sz = 10

    W, H = img.size

    if img.mode != 'P':
        img = img.convert('RGB')

    cbox = cbox.copy()

    cbox[0] = cbox[0] + 0.5 * (cbox[2] - 1)
    cbox[1] = cbox[1] + 0.5 * (cbox[3] - 1)

    # define output format
    out_size = size + margin

    if img.mode != 'P':
        avg_color = ImageStat.Stat(img).mean
        avg_color = tuple(int(round(c)) for c in avg_color)
        patch = Image.new(img.mode, (out_size, out_size), color=avg_color)
    else:
        patch = Image.new(img.mode, (out_size, out_size))

    m = (cbox[2] + cbox[3]) * 0.5
    search_range = math.sqrt((cbox[2] + m) * (cbox[3] + m)) * context

    # crop
    crop_sz = int(round(search_range * out_size / size))
    crop_sz = max(5, crop_sz)
    crop_ctr = cbox[:2]

    dldt = crop_ctr - 0.5 * (crop_sz - 1)
    drdb = dldt + crop_sz
    plpt = np.maximum(0, -dldt)
    dldt = np.maximum(0, dldt)
    drdb = np.minimum((W, H), drdb)

    dltrb = np.concatenate((dldt, drdb))
    dltrb = np.round(dltrb).astype('int')
    cp_img = img.crop(dltrb)

    # resize
    cW, cH = cp_img.size
    tW = max(cW * out_size / crop_sz, min_sz)
    tH = max(cH * out_size / crop_sz, min_sz)
    tW = int(round(tW))
    tH = int(round(tH))
    rz_img = cp_img.resize((tW, tH), resample)

    # calculate padding to paste to patch
    plpt = plpt * out_size / crop_sz
    plpt_ = np.round(plpt).astype('int')
    pltrb = np.concatenate((plpt_, plpt_ + (tW, tH)))

    # paste
    patch.paste(rz_img, pltrb)

    # if flag 'correct' is Ture, return information about turning back.
    if correct:
        scale = crop_sz / out_size
        leftmost = crop_ctr - (crop_sz - 1) / 2
        return patch, leftmost, scale

    return patch


def affine_grid_generator(B, H, W, theta):
    '''
    https://pyimagesearch.com/2022/05/23/spatial-transformer-networks-using-tensorflow/
    
    :param B: 
    :param H: 
    :param W: 
    :param theta: 
    :return: 
    '''
    # create normalized 2D grid
    x = tf.linspace(-1.0, 1.0, W)
    y = tf.linspace(-1.0, 1.0, H)
    (xT, yT) = tf.meshgrid(x, y)
    # flatten the meshgrid
    xTFlat = tf.reshape(xT, [-1])
    yTFlat = tf.reshape(yT, [-1])
    # reshape the meshgrid and concatenate ones to convert it to homogeneous form
    ones = tf.ones_like(xTFlat)
    samplingGrid = tf.stack([xTFlat, yTFlat, ones])
    # repeat grid batch size times
    samplingGrid = tf.broadcast_to(samplingGrid, (B, 3, H * W))
    # cast the affine parameters and sampling grid to float32
    # required for matmul
    theta = tf.cast(theta, "float32")
    samplingGrid = tf.cast(samplingGrid, "float32")
    # transform the sampling grid with the affine parameter
    batchGrids = tf.matmul(theta, samplingGrid)  # 2x3 and 3x####
    # reshape the sampling grid to (B, H, W, 2)
    batchGrids = tf.reshape(batchGrids, [B, 2, H, W])
    transposed_batchGrids = tf.transpose(batchGrids, perm=[0, 2, 3, 1])

    return transposed_batchGrids


def bilinear_sampler_2d(B, H, W, featureMap, x, y):
    '''
    From https://pyimagesearch.com/2022/05/23/spatial-transformer-networks-using-tensorflow/

    :param B:
    :param H:
    :param W:
    :param featureMap:
    :param x:
    :param y:
    :return:
    '''

    def get_pixel_value_2d(B, H, W, featureMap, x, y):

        # create batch indices and reshape it
        batchIdx = tf.range(0, B)
        batchIdx = tf.reshape(batchIdx, (B, 1, 1))
        # create the indices matrix which will be used to sample the feature map
        b = tf.tile(batchIdx, (1, H, W))
        indices = tf.stack([b, y, x], 3)
        # gather the feature map values for the corresponding indices
        gatheredPixelValue = tf.gather_nd(featureMap, indices)
        # return the gather pixel values
        return gatheredPixelValue

    # define the bounds of the image
    maxY = tf.cast(H - 1, "int32")
    maxX = tf.cast(W - 1, "int32")
    zero = tf.zeros([], dtype="int32")
    # rescale x and y to feature spatial dimensions
    x = tf.cast(x, "float32")
    y = tf.cast(y, "float32")
    x = 0.5 * ((x + 1.0) * tf.cast(maxX - 1, "float32"))
    y = 0.5 * ((y + 1.0) * tf.cast(maxY - 1, "float32"))
    # grab 4 nearest corner points for each (x, y)
    x0 = tf.cast(tf.floor(x), "int32")
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), "int32")
    y1 = y0 + 1
    # clip to range to not violate feature map boundaries
    x0 = tf.clip_by_value(x0, zero, maxX)
    x1 = tf.clip_by_value(x1, zero, maxX)
    y0 = tf.clip_by_value(y0, zero, maxY)
    y1 = tf.clip_by_value(y1, zero, maxY)

    # get pixel value at corner coords
    Ia = get_pixel_value_2d(B, H, W, featureMap, x0, y0)
    Ib = get_pixel_value_2d(B, H, W, featureMap, x0, y1)
    Ic = get_pixel_value_2d(B, H, W, featureMap, x1, y0)
    Id = get_pixel_value_2d(B, H, W, featureMap, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, "float32")
    x1 = tf.cast(x1, "float32")
    y0 = tf.cast(y0, "float32")
    y1 = tf.cast(y1, "float32")
    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)
    # compute transformed feature map
    transformedFeatureMap = tf.add_n(
        [wa * Ia, wb * Ib, wc * Ic, wd * Id])
    # transformedFeatureMap = wb * Ib + wd * Id # wa * Ia + wc * Ic

    return transformedFeatureMap


def get_pixel_value(a_feature, x_coord_list, y_coord_list):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a 4D tensor image.
    Input
    -----
    - a_feature: tensor of shape (B, H, W, C)
    - x_list: tensor of shape (B, x_coord of Points)
    - y_list: tensor of shape (B, y_coord of Points)
    Returns
    -------
    - output: tensor of shape (B, num_points, C)
    """
    # assert x_coord_list.shape == y_coord_list.shape
    batch_size = tf.shape(x_coord_list)[0]
    number_of_points = tf.shape(x_coord_list)[1]  # returns the dynamic shape whereas Tensor.shape returns the static shape of the tensor

    x_coord_list = tf.reshape(x_coord_list, shape=[-1])
    y_coord_list = tf.reshape(y_coord_list, shape=[-1])
    batch_index_list = tf.repeat(tf.range(batch_size), repeats=number_of_points)
    indices = tf.stack([batch_index_list, y_coord_list, x_coord_list], axis=1)  # shape is (number of points, 3)

    features_at_points = tf.gather_nd(a_feature, indices)
    number_of_channels = a_feature.shape[3]
    features_at_points = tf.reshape(features_at_points, shape=[batch_size, number_of_points, number_of_channels])

    return features_at_points


def bilinear_sampler_1d(input_feature, x, y, abs_scale=False):
    """
    From https://github.com/kevinzakka/spatial-transformer-network/blob/master/stn/transformer.py#L159
    https://stackoverflow.com/questions/52888146/what-is-the-equivalent-of-torch-nn-functional-grid-sample-in-tensorflow-numpy

    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - input_feature: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.

    """

    input_feature = tf.cast(input_feature, dtype='float32')
    H = tf.shape(input_feature)[1]
    W = tf.shape(input_feature)[2]

    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    if not abs_scale:
        x = tf.cast(x, 'float32')
        y = tf.cast(y, 'float32')
        x = 0.5 * ((x + 1.0) * tf.cast(max_x, 'float32'))
        y = 0.5 * ((y + 1.0) * tf.cast(max_y, 'float32'))
    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate input_feature boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(input_feature, x0, y0)
    Ib = get_pixel_value(input_feature, x0, y1)
    Ic = get_pixel_value(input_feature, x1, y0)
    Id = get_pixel_value(input_feature, x1, y1)
    
    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=2)
    wb = tf.expand_dims(wb, axis=2)
    wc = tf.expand_dims(wc, axis=2)
    wd = tf.expand_dims(wd, axis=2)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out


def exact_point_sampler(input_feature, x, y, abs_scale=False):
    """

    Performs sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    Input
    -----
    - input_feature: batch of images in (B, H, W, C) layout.
    - x, y

    """

    input_feature = tf.cast(input_feature, dtype='float32')
    H = tf.shape(input_feature)[1]
    W = tf.shape(input_feature)[2]

    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    if not abs_scale:
        x = tf.cast(x, 'float32')
        y = tf.cast(y, 'float32')
        x = 0.5 * ((x + 1.0) * tf.cast(max_x, 'float32'))
        y = 0.5 * ((y + 1.0) * tf.cast(max_y, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    y0 = tf.cast(tf.floor(y), 'int32')

    # clip to range [0, H-1/W-1] to not violate input_feature boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)

    return get_pixel_value(input_feature, x0, y0)


def sample_2d_features(input_feature, x, y, max_disp):
    """
    Sample input features at the normalized coordinates and around them with tolearance max_disp
    The sampling is done identically for each channel of the input.

    Mainly for cost_volume_at_contour_points

    Input
    -----
    - input_feature: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.

    Output
    -----
    [b, N, C, (max_disp * 2 + 1)^2]
    """

    input_feature = tf.cast(input_feature, dtype='float32')
    H = tf.shape(input_feature)[1] - max_disp*2
    W = tf.shape(input_feature)[2] - max_disp*2

    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    rescaled_x = 0.5 * ((x + 1.0) * tf.cast(max_x, 'float32'))
    rescaled_y = 0.5 * ((y + 1.0) * tf.cast(max_y, 'float32'))
    # add max_disp to account for the padding
    rescaled_x = rescaled_x + max_disp
    rescaled_y = rescaled_y + max_disp

    # grab closest corner point to x and y
    x0 = tf.cast(tf.math.round(rescaled_x), 'int32')
    y0 = tf.cast(tf.math.round(rescaled_y), 'int32')

    values_within_shift = []  # max_disp * 2 + 1, (B, N, C)
    for i_offset in range(-max_disp, max_disp+1):
      for j_offset in range(-max_disp, max_disp+1):
        # clip to range [0, H-1/W-1] to not violate input_feature boundaries
        shifted_x0 = tf.clip_by_value(x0+i_offset, zero, max_x)
        shifted_y0 = tf.clip_by_value(y0+j_offset, zero, max_y)
        values_within_shift.append( get_pixel_value(input_feature, shifted_x0, shifted_y0) )

    stacked_values_within_shift = tf.stack(values_within_shift, axis=-1)

    return stacked_values_within_shift


def normalize_1d_features(a_feature):
    """Normalizes feature tensor (e.g., before computing the cost volume).

    Args:
      a_feature: tf.tensor with dimensions [b, N, c, d^2]

    Returns:
      normalized_feature: tf.tensor with dimensions [b, N, c, d^2] normalized across the channel (c) axis
    """

    mean_tensor, variance_tensor = tf.nn.moments(x=a_feature, axes=[-2], keepdims=True)

    centered_feature = a_feature - mean_tensor
    std_tensor = tf.sqrt(variance_tensor + 1e-16)
    normalized_feature = centered_feature / std_tensor

    return normalized_feature


def cost_volume_at_contour_points(a_feature1, a_feature2, cnt0, cnt1, max_displacement):
    """
    Compute the cost volume (correlation) between 1D contour's feature and 2D image's feature at contour points only

    Displace features2 up to max_displacement in any direction and compute the
    per pixel cost of features1 and the displaced features2.

    Args:
    features1: tf.tensor of shape [b, h, w, c]  which means batch, height, width, channels
    features2: tf.tensor of shape [b, h, w, c]  which means batch, height, width, channels
    cnt0: tf.tensor of shape [b, N, 2]  which means batch, number of tracking points, x_y position. of the first frame
    cnt1: tf.tensor of shape [b, N, 2]  which means batch, number of tracking points, x_y position. of the second frame
    max_displacement: int, maximum displacement for cost volume computation.

    Returns:
    tf.tensor of shape [b, N, (2 * max_displacement + 1) ** 2] of costs for all displacements.
    """

    sampled_feature1 = bilinear_sampler_1d(a_feature1, cnt0[:, :, 0], cnt0[:, :, 1])  # B x N x c
    normalized_sampled_feature1 = normalize_1d_features(tf.expand_dims(sampled_feature1,axis=-1))

    # Set maximum displacement and compute the number of image shifts.
    _, height, width, _ = a_feature2.shape.as_list()
    if max_displacement <= 0 or max_displacement >= height or max_displacement >= width:
        raise ValueError(f'Max displacement of {max_displacement} is too large.')

    max_disp = max_displacement

    # Pad features2 such that shifts do not go out of bounds.
    features2_padded = tf.pad(
        tensor=a_feature2,
        paddings=[[0, 0], [max_disp, max_disp], [max_disp, max_disp], [0, 0]],
        mode='CONSTANT')
    
    # sample 2D features from features2
    sampled_feature2 = sample_2d_features(features2_padded, cnt1[:,:,0], cnt1[:,:,1], max_disp)  # B x N x c x d^2
    normalized_sampled_feature2 = normalize_1d_features(sampled_feature2)

    # compute the cost volume [b, N, d^2] <-- [b, N, c] * [b, N, c, d^2]
    cost_volume = tf.reduce_mean( normalized_sampled_feature1 * normalized_sampled_feature2, axis=2)  # TODO: use einops for readability

    # for debugging, check the following
    # tf.nn.moments(x=normalized_sampled_feature1, axes=[-2], keepdims=True)
    # tf.nn.moments(x=normalized_sampled_feature2, axes=[-2], keepdims=True)

    return cost_volume

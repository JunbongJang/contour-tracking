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
Simple interface for training and inference.
"""

import math
import sys
import time
import cv2

import gin
import tensorflow as tf

from src import tracking_model
from src import tracking_utils
from src import implicit_utils


@gin.configurable
class ContourFlow(object):
  """Simple interface with infer and train methods."""

  def __init__(
      self,
      checkpoint_dir='',
      summary_dir='',
      optimizer='adam',
      learning_rate=0.0001,  # Origianl 0.0002
      only_forward=False,
      level1_num_layers=3,
      level1_num_filters=32,
      level1_num_1x1=0,
      dropout_rate=.25,
      build_selfsup_transformations=None,
      fb_sigma_teacher=0.003,
      fb_sigma_student=0.03,
      train_with_supervision=False,
      train_with_gt_occlusions=False,
      train_with_segmentations=False,
      train_with_seg_points=False,
      train_with_tracking_points=False,
      smoothness_edge_weighting='gaussian',
      teacher_image_version='original',
      stop_gradient_mask=True,
      selfsup_mask='gaussian',
      normalize_before_cost_volume=True,
      original_layer_sizes=False,
      shared_flow_decoder=False,
      channel_multiplier=1,
      use_cost_volume=True,
      use_feature_warp=True,
      num_levels=5,
      accumulate_flow=True,
      occlusion_estimation='wang',
      occ_weights=None,
      occ_thresholds=None,
      occ_clip_max=None,
      smoothness_at_level=2,
      use_bfloat16=False,
      input_image_size=None
  ):
    """Instantiate a UFlow model.

    Args:
      checkpoint_dir: str, location to checkpoint model
      summary_dir: str, location to write tensorboard summary
      optimizer: str, identifier of which optimizer to use
      learning_rate: float, learning rate to use for training
      only_forward: bool, if True, only infer flow in one direction
      level1_num_layers: int, pwc architecture property
      level1_num_filters: int, pwc architecture property
      level1_num_1x1: int, pwc architecture property
      dropout_rate: float, how much dropout to use with pwc net
      build_selfsup_transformations: list of functions which transform the flow
        predicted from the raw images to be in the frame of images transformed
        by geometric_augmentation_fn
      fb_sigma_teacher: float, controls how much forward-backward flow
        consistency is needed by the teacher model in order to supervise the
        student
      fb_sigma_student: float, controls how much forward-backward consistency is
        needed by the student model in order to not receive supervision from the
        teacher model
      train_with_supervision: bool, Whether to train with ground truth flow,
        currently not supported
      train_with_gt_occlusions: bool, if True, use ground truth occlusions
        instead of predicted occlusions during training. Only works with Sintel
        which has dense ground truth occlusions.
      smoothness_edge_weighting: str, controls how smoothness penalty is
        determined
      teacher_image_version: str, which image to give to teacher model
      stop_gradient_mask: bool, whether to stop the gradient of photometric loss
        through the occlusion mask.
      selfsup_mask: str, type of selfsupervision mask to use
      normalize_before_cost_volume: bool, toggles pwc architecture property
      original_layer_sizes: bool, toggles pwc architecture property
      shared_flow_decoder: bool, toogles pwc architecutre property
      channel_multiplier: int, channel factor to use in pwc
      use_cost_volume: bool, toggles pwc architecture property
      use_feature_warp: bool, toggles pwc architecture property
      num_levels: int, how many pwc pyramid layers to use
      accumulate_flow: bool, toggles pwc architecture property
      occlusion_estimation: which type of occlusion estimation to use
      occ_weights: dict of string -> float indicating how to weight occlusions
      occ_thresholds: dict of str -> float indicating thresholds to apply for
        occlusions
      occ_clip_max: dict of string -> float indicating how to clip occlusion
      smoothness_at_level: int, which level to compute smoothness on
      use_bfloat16: bool, whether to run in bfloat16 mode.

    Returns:
      ContourFlow object instance.
    """
    self._only_forward = only_forward
    self._build_selfsup_transformations = build_selfsup_transformations
    self._fb_sigma_teacher = fb_sigma_teacher
    self._fb_sigma_student = fb_sigma_student
    self._train_with_supervision = train_with_supervision
    self._train_with_gt_occlusions = train_with_gt_occlusions
    self._train_with_segmentations = train_with_segmentations
    self._train_with_seg_points = train_with_seg_points
    self._train_with_tracking_points = train_with_tracking_points
    self._smoothness_edge_weighting = smoothness_edge_weighting
    self._smoothness_at_level = smoothness_at_level
    self._teacher_flow_model = None
    self._teacher_feature_model = None
    self._teacher_image_version = teacher_image_version
    self._stop_gradient_mask = stop_gradient_mask
    self._selfsup_mask = selfsup_mask
    self._num_levels = num_levels


    self._tracking_model = tracking_model.PointSetTracker(num_iter=5)

    self._learning_rate = learning_rate
    self._optimizer_type = optimizer
    self._make_or_reset_optimizer()

    # Set up checkpointing.
    self._make_or_reset_checkpoint()
    self.update_checkpoint_dir(checkpoint_dir)

    # Set up tensorboard log files.
    self.summary_dir = summary_dir
    if self.summary_dir:
      self.writer = tf.compat.v1.summary.create_file_writer(summary_dir)
      self.writer.set_as_default()

    self._occlusion_estimation = occlusion_estimation

    if occ_weights is None:
      occ_weights = {
          'fb_abs': 1.0,
          'forward_collision': 1.0,
          'backward_zero': 10.0
      }
    self._occ_weights = occ_weights

    if occ_thresholds is None:
      occ_thresholds = {
          'fb_abs': 1.5,
          'forward_collision': 0.4,
          'backward_zero': 0.25
      }
    self._occ_thresholds = occ_thresholds

    if occ_clip_max is None:
      occ_clip_max = {'fb_abs': 10.0, 'forward_collision': 5.0}
    self._occ_clip_max = occ_clip_max

  def update_checkpoint_dir(self, checkpoint_dir):
    """Changes the checkpoint directory for saving and restoring."""
    self._manager = tf.train.CheckpointManager(
        self._checkpoint, directory=checkpoint_dir, max_to_keep=1)

  def restore(self, reset_optimizer=False, reset_global_step=False):
    """Restores a saved model from a checkpoint."""
    status = self._checkpoint.restore(self._manager.latest_checkpoint).expect_partial()  # expect_partial to silence warning
    try:
      status.assert_existing_objects_matched()
    except AssertionError as e:
      print('Error while attempting to restore UFlow models:', e)
    if reset_optimizer:
      self._make_or_reset_optimizer()
      self._make_or_reset_checkpoint()
    if reset_global_step:
      tf.compat.v1.train.get_or_create_global_step().assign(0)

  def save(self):
    """Saves a model checkpoint."""
    self._manager.save()

  def _make_or_reset_optimizer(self):
    if self._optimizer_type == 'adam':
      self._optimizer = tf.compat.v1.train.AdamOptimizer(
          self._learning_rate, name='Optimizer')
    elif self._optimizer_type == 'sgd':
      self._optimizer = tf.compat.v1.train.GradientDescentOptimizer(
          self._learning_rate, name='Optimizer')
    else:
      raise ValueError('Optimizer "{}" not yet implemented.'.format(
          self._optimizer_type))

  @property
  def optimizer(self):
    return self._optimizer

  def _make_or_reset_checkpoint(self):
    self._checkpoint = tf.train.Checkpoint(
        optimizer=self._optimizer,
        tracking_model=self._tracking_model,
        optimizer_step=tf.compat.v1.train.get_or_create_global_step())


  def infer_tracking_function(self, images,
                                 ground_truth_segmentations,
                                 seg_point1,
                                 seg_point2,
                                 prev_tracking_points,
                                 tracking_pos_emb,
                                 input_height=None,
                                 input_width=None):

    """
    :param images: tf.tensor of shape [batchsize, 2, height, width, 3].
    :param ground_truth_segmentations:
    :param prev_tracking_points:
    :param frame_index:
    :param input_height: height at which the model should be applied if different from image height.
    :param input_width: width at which the model should be applied if different from image width
    :return: tracking points with shape [batch, Number of points, 2]
    """

    """Infers flow from two images.

        Returns:
          Optical flow for each pixel in image1 pointing to image2.
        """

    batch_size, seq_len, orig_height, orig_width, image_channels = images.shape.as_list()

    if input_height is None:
        input_height = orig_height
    if input_width is None:
        input_width = orig_width

    # predict the location of tracking points in the next frame
    prev_patch = images[:,0,:,:,:]
    cur_patch = images[:,1,:,:,:]

    prev_seg_points = seg_point1
    cur_seg_points = seg_point2
    prev_seg_points = tracking_utils.resize_seg_points(orig_height, orig_width, new_height=input_height, new_width=input_width, tracking_points=prev_seg_points)
    cur_seg_points = tracking_utils.resize_seg_points(orig_height, orig_width, new_height=input_height, new_width=input_width, tracking_points=cur_seg_points)

    # Get position embedding of the segmentation points
    tracking_pos_emb = self._tracking_model(prev_seg_points)

    # debug whether images are loaded correctly for prediction
    # cv2.imwrite(f'debug_prev_patch.png', prev_patch[0].numpy()*255)
    # cv2.imwrite(f'debug_cur_patch.png', cur_patch[0].numpy()*255)
    # import pdb;pdb.set_trace()

    # -------------------- To remove negative padding -------------------
    # find the index in seg_points from where negative values start
    prev_seg_points_limit = tracking_utils.get_first_occurrence_indices(prev_seg_points[:,:,0], -0.1)[0]  # index 0 is fine since batch size is 1
    cur_seg_points_limit = tracking_utils.get_first_occurrence_indices(cur_seg_points[:,:,0], -0.1)[0]

    # ------------------------------------------------------------------
    # for every point in the prev contour, find its matching point on the current contour
    
    sampled_prev_contour_indices = tf.linspace(0, prev_seg_points_limit-1, prev_seg_points_limit)
    sampled_prev_contour_indices = tf.expand_dims(sampled_prev_contour_indices, axis=0)   # (1, prev_contour_points)
    sampled_prev_contour_indices = tf.cast(sampled_prev_contour_indices, tf.int32)

    # find 20 nearby points in the current contour
    nearby_cur_sampled_contour_indices = implicit_utils.sample_nearby_points_for_implicit_cycle_consistency(sampled_prev_contour_indices, prev_seg_points, cur_seg_points, cur_seg_points_limit)

    # predict occupancy between 50 initial points x 20 nearby points
    occ_contour_forward = self._tracking_model(prev_patch, cur_patch, prev_seg_points, cur_seg_points, tracking_pos_emb, sampled_prev_contour_indices, nearby_cur_sampled_contour_indices)

    # find the current contour indices with max occupancy
    predicted_cur_contour_indices_tensor = implicit_utils.find_max_corr_contour_indices(occ_contour_forward, nearby_cur_sampled_contour_indices)
    # -------------------------------------------------------------------

    # This part is to compute the validation loss
    sampled_prev_contour_indices = implicit_utils.sample_initial_points(prev_seg_points, prev_seg_points_limit)
    sampled_prev_contour_indices = tf.cast(sampled_prev_contour_indices, tf.int32)  # Problem: only uses discrete representation of the image
    
    # first contour to second contour
    nearby_cur_sampled_contour_indices = implicit_utils.sample_nearby_points_for_implicit_cycle_consistency(sampled_prev_contour_indices, prev_seg_points, cur_seg_points, cur_seg_points_limit)
    occ_contour_forward = self._tracking_model(prev_patch, cur_patch, prev_seg_points, cur_seg_points, tracking_pos_emb, sampled_prev_contour_indices, nearby_cur_sampled_contour_indices)
    predicted_cur_contour_indices = implicit_utils.find_max_corr_contour_indices(occ_contour_forward, nearby_cur_sampled_contour_indices)

    # second contour to first contour
    nearby_prev_sampled_contour_indices = implicit_utils.sample_nearby_points_for_implicit_cycle_consistency(predicted_cur_contour_indices, cur_seg_points, prev_seg_points, prev_seg_points_limit)
    occ_contour_backward = self._tracking_model(cur_patch, prev_patch, cur_seg_points, prev_seg_points, tracking_pos_emb, predicted_cur_contour_indices, nearby_prev_sampled_contour_indices)
    predicted_prev_contour_indices = implicit_utils.find_max_corr_contour_indices(occ_contour_backward, nearby_prev_sampled_contour_indices)
    
    # first contour to second contour
    nearby_new_cur_sampled_contour_indices = implicit_utils.sample_nearby_points_for_implicit_cycle_consistency(predicted_prev_contour_indices, prev_seg_points, cur_seg_points, cur_seg_points_limit)
    occ_contour_forward = self._tracking_model(prev_patch, cur_patch, prev_seg_points, cur_seg_points, tracking_pos_emb, predicted_prev_contour_indices, nearby_new_cur_sampled_contour_indices)

    prev_gt_occ = implicit_utils.create_GT_occupancy(sampled_prev_contour_indices, prev_seg_points.shape[:2], nearby_prev_sampled_contour_indices)   # create GT occupancy on previous contour
    backward_occ_cycle_consistency_loss = tracking_utils.occ_cycle_consistency_loss(prev_gt_occ, occ_contour_backward)

    cur_gt_occ = implicit_utils.create_GT_occupancy(predicted_cur_contour_indices, cur_seg_points.shape[:2], nearby_new_cur_sampled_contour_indices)   # create GT occupancy on current contour
    forward_occ_cycle_consistency_loss = tracking_utils.occ_cycle_consistency_loss(cur_gt_occ, occ_contour_forward)


    return forward_occ_cycle_consistency_loss, backward_occ_cycle_consistency_loss, predicted_cur_contour_indices_tensor, tracking_pos_emb, input_height, input_width
  

  # @tf.function
  def infer(self,
            image1,
            image2,
            segmentation1,
            segmentation2,
            seg_point1,
            seg_point2,
            tracking_point1=None,
            tracking_pos_emb=None,
            input_height=None,
            input_width=None,
            resize_flow_to_img_res=True,
            infer_occlusion=False):

    return self.infer_tracking_function(tf.stack([image1, image2])[None],
                                        tf.stack([segmentation1, segmentation2])[None],
                                        seg_point1,
                                        seg_point2,
                                        tracking_point1,
                                        tracking_pos_emb,
                                        input_height=input_height,
                                        input_width=input_width)

  # -----------------------------------------------------------------------------------------------

  def train_step_no_tf_function(self,
                                batch,
                                current_epoch,
                                weights=None,
                                plot_dir=None,
                                distance_metrics=None,
                                ground_truth_flow=None,
                                ground_truth_valid=None,
                                ground_truth_occlusions=None,
                                ground_truth_segmentations=None,
                                ground_truth_seg_points=None,
                                ground_truth_tracking_points=None,
                                images_without_photo_aug=None,
                                occ_active=None):
    """Perform single gradient step."""
    if weights is None:
      weights = {
          'smooth2': 2.0,
          'edge_constant': 100.0,
          'census': 1.0,
      }
    else:
      # Support values and callables (e.g. to compute weights from global step).
      weights = {k: v() if callable(v) else v for k, v in weights.items()}

    losses, gradients, variables, saved_offset_dict = self._loss_and_grad(
        batch,
        current_epoch,
        weights,
        plot_dir,
        distance_metrics=distance_metrics,
        ground_truth_flow=ground_truth_flow,
        ground_truth_valid=ground_truth_valid,
        ground_truth_occlusions=ground_truth_occlusions,
        ground_truth_segmentations=ground_truth_segmentations,
        ground_truth_seg_points=ground_truth_seg_points,
        ground_truth_tracking_points=ground_truth_tracking_points,
        images_without_photo_aug=images_without_photo_aug,
        occ_active=occ_active)

    self._optimizer.apply_gradients(
        list(zip(gradients, variables)),
        global_step=tf.compat.v1.train.get_or_create_global_step())

    return losses, saved_offset_dict

  # @tf.function
  def train_step(self,
                 batch,
                 current_epoch,
                 weights=None,
                 distance_metrics=None,
                 ground_truth_flow=None,
                 ground_truth_valid=None,
                 ground_truth_occlusions=None,
                 ground_truth_segmentations=None,
                 ground_truth_seg_points=None,
                 ground_truth_tracking_points=None,
                 images_without_photo_aug=None,
                 occ_active=None):
    """Performs a train step on the batch."""

    return self.train_step_no_tf_function(
        batch,
        current_epoch,
        weights,
        distance_metrics=distance_metrics,
        ground_truth_flow=ground_truth_flow,
        ground_truth_valid=ground_truth_valid,
        ground_truth_occlusions=ground_truth_occlusions,
        ground_truth_segmentations=ground_truth_segmentations,
        ground_truth_seg_points=ground_truth_seg_points,
        ground_truth_tracking_points=ground_truth_tracking_points,
        images_without_photo_aug=images_without_photo_aug,
        occ_active=occ_active)

  def train(self,
            data_it,
            current_epoch,
            num_steps,
            weights=None,
            progress_bar=True,
            plot_dir=None,
            distance_metrics=None,
            occ_active=None):
    """Trains flow from a data iterator for a number of gradient steps.

    Args:
      data_it: tf.data.Iterator that produces tensors of shape [b,3,h,w,3].
      num_steps: int, number of gradient steps to train for.
      weights: dictionary with weight for each loss.
      progress_bar: boolean flag for continuous printing of a progress bar.
      plot_dir: location to plot results or None
      distance_metrics: dictionary of which type of distance metric to use for
        photometric losses
      occ_active: dictionary of which occlusion types are active

    Returns:
      a dict that contains all losses.
    """

    # Log dictionary for storing losses of this epoch.
    log = dict()
    # Support constant lr values and callables (for learning rate schedules).
    if callable(self._learning_rate):
      log['learning-rate'] = self._learning_rate()
    else:
      log['learning-rate'] = self._learning_rate

    start_time_data = time.time()
    for _, batch in zip(range(num_steps), data_it):
      stop_time_data = time.time()

      if progress_bar:
        sys.stdout.write('.')
        sys.stdout.flush()
      # Split batch into images, occlusion masks, and ground truth flow.
      images, labels = batch
      ground_truth_flow = labels.get('flow_uv', None)
      ground_truth_valid = labels.get('flow_valid', None)
      ground_truth_occlusions = labels.get('occlusions', None)
      images_without_photo_aug = labels.get('images_without_photo_aug', None)
      ground_truth_segmentations = labels.get('segmentations', None)
      ground_truth_seg_points = labels.get('segmentation_points', None)
      ground_truth_tracking_points = labels.get('tracking_points', None)

      # thresholding is necessary since segmentation is loaded blurrily
      if ground_truth_segmentations is not None:
          seg_threshold = 0
          ground_truth_segmentations = tf.cast((ground_truth_segmentations > seg_threshold), dtype='float32') * 255

      # use code below if thresholding is not used
      # ground_truth_segmentations = tf.cast(ground_truth_segmentations, dtype='float32')

      # -----------------------------------
      # debug whether images are loaded correctly for prediction
      # cv2.imwrite(f'uflow/debug_generated/debug_prev_patch.png', images[0,0,:,:,:].numpy()*255)
      # cv2.imwrite(f'uflow/debug_generated/debug_cur_patch.png', images[0,1,:,:,:].numpy()*255)
      # import pdb; pdb.set_trace()

      # Debug whether segmentation was loaded correctly for training
      # cv2.imwrite(f'test_segmentation_{seg_threshold}.png', ground_truth_segmentations.numpy()[0,0,:,:,0])
      #
      # print( tf.unique_with_counts(tf.reshape(ground_truth_segmentations, -1)) )
      # import pdb; pdb.set_trace()
      # -----------------------------------------

      start_time_train_step = time.time()
      # Use tf.function unless intermediate results have to be plotted.
      if plot_dir is None:
        # Perform a gradient step (optimized by tf.function).
        losses, saved_offset_dict = self.train_step(
            images,
            current_epoch,
            weights,
            distance_metrics=distance_metrics,
            ground_truth_flow=ground_truth_flow,
            ground_truth_valid=ground_truth_valid,
            ground_truth_occlusions=ground_truth_occlusions,
            ground_truth_segmentations=ground_truth_segmentations,
            ground_truth_seg_points=ground_truth_seg_points,
            ground_truth_tracking_points=ground_truth_tracking_points,
            images_without_photo_aug=images_without_photo_aug,
            occ_active=occ_active)
      else:
        # Perform a gradient step without tf.function to allow plotting.
        losses, saved_offset_dict = self.train_step_no_tf_function(
            images,
            current_epoch,
            weights,
            plot_dir,
            distance_metrics=distance_metrics,
            ground_truth_flow=ground_truth_flow,
            ground_truth_valid=ground_truth_valid,
            ground_truth_occlusions=ground_truth_occlusions,
            ground_truth_seg_points=ground_truth_seg_points,
            ground_truth_tracking_points=ground_truth_tracking_points,
            images_without_photo_aug=images_without_photo_aug,
            occ_active=occ_active)

      stop_time_train_step = time.time()

      log_update = losses
      # Compute time in ms.
      log_update['data-time'] = (stop_time_data - start_time_data) * 1000
      log_update['train-time'] = (stop_time_train_step -
                                  start_time_train_step) * 1000

      # Log losses and times.
      for key in log_update:
        if key in log:
          log[key].append(log_update[key])
        else:
          log[key] = [log_update[key]]
        if self.summary_dir:
          tf.summary.scalar(key, log[key])

      # Set start time for data gathering to measure data pipeline efficiency.
      start_time_data = time.time()

    for key in log:
      log[key] = tf.reduce_mean(input_tensor=log[key])

    if progress_bar:
      sys.stdout.write('\n')
      sys.stdout.flush()

    return log, saved_offset_dict

  def _loss_and_grad(self,
                     batch,
                     current_epoch,
                     weights,
                     plot_dir=None,
                     distance_metrics=None,
                     ground_truth_flow=None,
                     ground_truth_valid=None,
                     ground_truth_occlusions=None,
                     ground_truth_segmentations=None,
                     ground_truth_seg_points=None,
                     ground_truth_tracking_points=None,
                     images_without_photo_aug=None,
                     occ_active=None):
    """Apply the model on the data in batch and compute the loss.

    Args:
      batch: tf.tensor of shape [b, seq, h, w, c] that holds a batch of image
        sequences.
      weights: dictionary with float entries per loss.
      plot_dir: str, directory to plot images
      distance_metrics: dict, which distance metrics to use,
      ground_truth_flow: Tensor, optional ground truth flow for first image
      ground_truth_valid: Tensor, indicates locations where gt flow is valid
      ground_truth_occlusions: Tensor, optional ground truth occlusions for
        computing loss. If None, predicted occlusions will be used.
      images_without_photo_aug: optional images without any photometric
        augmentation applied. Will be used for computing photometric losses if
        provided.
      occ_active: optional dict indicating which occlusion methods are active

    Returns:
      A tuple consisting of a tf.scalar that represents the total loss for the
      current batch, a list of gradients, and a list of the respective
      variables.
    """

    # This is where to add the tracking model
    with tf.GradientTape() as tape:
      losses, saved_offset_dict = self.compute_loss_tracking(batch,
                                                              current_epoch,
                                                              ground_truth_seg_points=ground_truth_seg_points,
                                                              ground_truth_tracking_points=ground_truth_tracking_points)

    # print(self._tracking_model.local_alignment.summary())
    variables = ( self._tracking_model.trainable_variables)
    grads = tape.gradient(losses['total-loss'], variables)

    return losses, grads, variables, saved_offset_dict

  def compute_loss_tracking(self,
                           batch,
                           current_epoch,
                           ground_truth_seg_points=None,
                           ground_truth_tracking_points=None):

      # determine seq length
      max_seq_len = int(batch.shape[1])
      # seq_len = tracking_utils.linear_seq_increase(current_epoch, max_seq_len)

      # Get position embedding of the segmentation points
      prev_seg_points = ground_truth_seg_points[:, 0, :, :]
      cur_seg_points = ground_truth_seg_points[:, 1, :, :]
      pos_emb = self._tracking_model(prev_seg_points)

      gt_prev_id_assignments = ground_truth_tracking_points[:, 0, :, :]
      gt_cur_id_assignments = ground_truth_tracking_points[:, 1, :, :]

      # predict the location of tracking points in the next frame
      prev_patch = batch[:, 0, :, :, :]
      cur_patch = batch[:, 1, :, :, :]

      # -------------------- To remove negative padding -------------------
      prev_seg_points_limit = tracking_utils.get_first_occurrence_indices(prev_seg_points[:, :, 0], -0.1)
      cur_seg_points_limit = tracking_utils.get_first_occurrence_indices(cur_seg_points[:, :, 0], -0.1)

      # ------------------------- Compute Forward and backward occupancy ------------------------------------------
      sampled_prev_contour_indices = implicit_utils.sample_initial_points(prev_seg_points, prev_seg_points_limit)
      sampled_prev_contour_indices = tf.cast(sampled_prev_contour_indices, tf.int32)  # Problem: only uses discrete representation of the image
      
      # first contour to second contour
      nearby_cur_sampled_contour_indices = implicit_utils.sample_nearby_points_for_implicit_cycle_consistency(sampled_prev_contour_indices, prev_seg_points, cur_seg_points, cur_seg_points_limit)
      occ_contour_forward = self._tracking_model(prev_patch, cur_patch, prev_seg_points, cur_seg_points, pos_emb, sampled_prev_contour_indices, nearby_cur_sampled_contour_indices)
      predicted_cur_contour_indices = implicit_utils.find_max_corr_contour_indices(occ_contour_forward, nearby_cur_sampled_contour_indices)
      
      # second contour to first contour
      nearby_prev_sampled_contour_indices = implicit_utils.sample_nearby_points_for_implicit_cycle_consistency(predicted_cur_contour_indices, cur_seg_points, prev_seg_points, prev_seg_points_limit)
      occ_contour_backward = self._tracking_model(cur_patch, prev_patch, cur_seg_points, prev_seg_points, pos_emb, predicted_cur_contour_indices, nearby_prev_sampled_contour_indices)
      predicted_prev_contour_indices = implicit_utils.find_max_corr_contour_indices(occ_contour_backward, nearby_prev_sampled_contour_indices)
      
      # first contour to second contour
      nearby_new_cur_sampled_contour_indices = implicit_utils.sample_nearby_points_for_implicit_cycle_consistency(predicted_prev_contour_indices, prev_seg_points, cur_seg_points, cur_seg_points_limit)
      occ_contour_forward = self._tracking_model(prev_patch, cur_patch, prev_seg_points, cur_seg_points, pos_emb, predicted_prev_contour_indices, nearby_new_cur_sampled_contour_indices)
      
      prev_gt_occ = implicit_utils.create_GT_occupancy(sampled_prev_contour_indices, prev_seg_points.shape[:2], nearby_prev_sampled_contour_indices)   # create GT occupancy on previous contour (8, 50, 20)
      backward_occ_cycle_consistency_loss = tracking_utils.occ_cycle_consistency_loss(prev_gt_occ, occ_contour_backward)

      cur_gt_occ = implicit_utils.create_GT_occupancy(predicted_cur_contour_indices, cur_seg_points.shape[:2], nearby_new_cur_sampled_contour_indices)   # create GT occupancy on current contour (8, 50, 20)
      forward_occ_cycle_consistency_loss = tracking_utils.occ_cycle_consistency_loss(cur_gt_occ, occ_contour_forward)
      # ----------------------------------------------------------------------------------------------------------
      
      saved_offset_dict = {0: 0}  # key for sequence number, only one key 0 if seq_num=2

      total_loss = backward_occ_cycle_consistency_loss + forward_occ_cycle_consistency_loss
      # tf.print("total_loss", total_loss, backward_occ_cycle_consistency_loss, forward_occ_cycle_consistency_loss)

      losses = {'total-loss' : total_loss,
                'backward_occ_cycle_consistency_loss': backward_occ_cycle_consistency_loss,
                'forward_occ_cycle_consistency_loss': forward_occ_cycle_consistency_loss}

      return losses, saved_offset_dict

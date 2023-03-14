# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""UFlow: Unsupervised Optical Flow.

This library provides a simple interface for training and inference.
"""

import math
import sys
import time
import cv2

import gin
import tensorflow as tf

from src import contour_flow_model
from src import tracking_model
from src import uflow_utils
from src import tracking_utils


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
      Uflow object instance.
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

    # self._feature_model = contour_flow_model.PWCFeaturePyramid(
    #     level1_num_layers=level1_num_layers,
    #     level1_num_filters=level1_num_filters,
    #     level1_num_1x1=level1_num_1x1,
    #     original_layer_sizes=original_layer_sizes,
    #     num_levels=num_levels,
    #     channel_multiplier=channel_multiplier,
    #     pyramid_resolution='half',
    #     use_bfloat16=use_bfloat16)
    #
    # self._flow_model = contour_flow_model.ContourFlowModel(
    #     dropout_rate=dropout_rate,
    #     normalize_before_cost_volume=normalize_before_cost_volume,
    #     num_levels=num_levels,
    #     use_feature_warp=use_feature_warp,
    #     use_cost_volume=use_cost_volume,
    #     channel_multiplier=channel_multiplier,
    #     accumulate_flow=accumulate_flow,
    #     use_bfloat16=use_bfloat16,
    #     shared_flow_decoder=shared_flow_decoder)

    self._tracking_model = tracking_model.PointSetTracker(num_iter=5, input_image_size=input_image_size)

    # By default, the teacher flow and feature models are the same as
    # the student flow and feature models.
    # self._teacher_flow_model = self._flow_model
    # self._teacher_feature_model = self._feature_model

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

  def set_teacher_models(self, teacher_feature_model, teacher_flow_model):
    self._teacher_feature_model = teacher_feature_model
    self._teacher_flow_model = teacher_flow_model

  @property
  def feature_model(self):
    return self._feature_model

  @property
  def flow_model(self):
    return self._flow_model

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
        # feature_model=self._feature_model,
        # flow_model=self._flow_model,
        tracking_model=self._tracking_model,
        optimizer_step=tf.compat.v1.train.get_or_create_global_step())

  # Use of tf.function breaks exporting the model, see b/138864493
  def infer_no_tf_function(self,
                           image1,
                           image2,
                           segmentation1,
                           segmentation2,
                           seg_point1=None,
                           seg_point2=None,
                           tracking_point1=None,
                           tracking_point2=None,
                           tracking_pos_emb=None,
                           input_height=None,
                           input_width=None,
                           resize_flow_to_img_res=True,
                           infer_occlusion=False,
                           frame_index=None):
    """Infer flow for two images.

    Args:
      image1: tf.tensor of shape [height, width, 3].
      image2: tf.tensor of shape [height, width, 3].
      input_height: height at which the model should be applied if different
        from image height.
      input_width: width at which the model should be applied if different from
        image width
      resize_flow_to_img_res: bool, if True, return the flow resized to the same
        resolution as (image1, image2). If False, return flow at the whatever
        resolution the model natively predicts it.
      infer_occlusion: bool, if True, return both flow and a soft occlusion
        mask, else return just flow.

    Returns:
      Optical flow for each pixel in image1 pointing to image2.
    """

    results = self.batch_infer_no_tf_function(
        tf.stack([image1, image2])[None],
        tf.stack([segmentation1, segmentation2])[None],
        tf.stack([seg_point1, seg_point2])[None],
        tf.stack([tracking_point1, tracking_point2])[None],
        input_height=input_height,
        input_width=input_width,
        resize_flow_to_img_res=resize_flow_to_img_res,
        infer_occlusion=infer_occlusion)

    # Remove batch dimension from all results.
    if isinstance(results, (tuple, list)):
      return [x[0] for x in results]
    else:
      return results[0]


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
    # prev_seg_points_limit = tracking_utils.get_first_occurrence_indices(prev_seg_points[:,:,0], -0.1)[0]  # index 0 is fine since batch size is 1
    # cur_seg_points_limit = tracking_utils.get_first_occurrence_indices(cur_seg_points[:,:,0], -0.1)[0]

    forward_spatial_offset, backward_spatial_offset, saved_offset = self._tracking_model(prev_patch, cur_patch, prev_seg_points, cur_seg_points, tracking_pos_emb)

    return forward_spatial_offset, backward_spatial_offset, tracking_pos_emb, input_height, input_width


  def batch_infer_no_tf_function(self,
                                 images,
                                 segmentations,
                                 seg_points=None,
                                 tracking_points=None,
                                 input_height=None,
                                 input_width=None,
                                 resize_flow_to_img_res=True,
                                 infer_occlusion=False):
    """Infers flow from two images.

    Args:
      images: tf.tensor of shape [batchsize, 2, height, width, 3].
      input_height: height at which the model should be applied if different
        from image height.
      input_width: width at which the model should be applied if different from
        image width
      resize_flow_to_img_res: bool, if True, return the flow resized to the same
        resolution as (image1, image2). If False, return flow at the whatever
        resolution the model natively predicts it.
      infer_occlusion: bool, if True, return both flow and a soft occlusion
        mask, else return just flow.

    Returns:
      Optical flow for each pixel in image1 pointing to image2.
    """

    batch_size, seq_len, orig_height, orig_width, image_channels = images.shape.as_list(
    )

    if input_height is None:
      input_height = orig_height
    if input_width is None:
      input_width = orig_width

    # Ensure a feasible computation resolution. If specified size is not
    # feasible with the model, change it to a slightly higher resolution.
    divisible_by_num = pow(2.0, self._num_levels)
    if (input_height % divisible_by_num != 0 or
        input_width % divisible_by_num != 0):
      print('Cannot process images at a resolution of ' + str(input_height) +
            'x' + str(input_width) + ', since the height and/or width is not a '
            'multiple of ' + str(divisible_by_num) + '.')
      # compute a feasible resolution
      input_height = int(
          math.ceil(float(input_height) / divisible_by_num) * divisible_by_num)
      input_width = int(
          math.ceil(float(input_width) / divisible_by_num) * divisible_by_num)
      print('Inference will be run at a resolution of ' + str(input_height) +
            'x' + str(input_width) + '.')

    # Resize images to desired input height and width.
    if input_height != orig_height or input_width != orig_width:
      images = uflow_utils.resize(
          images, input_height, input_width, is_flow=False)

    # Flatten images by folding sequence length into the batch dimension, apply
    # the feature network and undo the flattening.
    images_flattened = tf.reshape(
        images,
        [batch_size * seq_len, input_height, input_width, image_channels])
    # noinspection PyCallingNonCallable

    features_flattened = self._feature_model(
        images_flattened, split_features_by_sample=False)
    features = [
        tf.reshape(f, [batch_size, seq_len] + f.shape.as_list()[1:])
        for f in features_flattened
    ]

    features1, features2 = [[f[:, i] for f in features] for i in range(2)]

    # split segmentations
    segmentation1 = segmentations[:, 0, :, :, :]
    segmentation2 = segmentations[:, 1, :, :, :]

    # Compute flow in frame of image1.
    # noinspection PyCallingNonCallable
    flow = self._flow_model(features1, features2, segmentation1, segmentation2, training=False)[0]

    if infer_occlusion:
      # noinspection PyCallingNonCallable
      flow_backward = self._flow_model(features2, features1, segmentation1, segmentation2, training=False)[0]
      warps, valid_warp_masks, range_map, occlusion_mask = self.infer_occlusion(flow, flow_backward)
      # originally, the shape is [1, 160, 160, 1] before the resize

      warps = uflow_utils.resize(
          warps, orig_height, orig_width, is_flow=False)

      valid_warp_masks = uflow_utils.resize(
          valid_warp_masks, orig_height, orig_width, is_flow=False)

      occlusion_mask = uflow_utils.resize(
          occlusion_mask, orig_height, orig_width, is_flow=False)

      range_map = uflow_utils.resize(
          range_map, orig_height, orig_width, is_flow=False)

    # Resize and rescale flow to original resolution. This always needs to be
    # done because flow is generated at a lower resolution.
    if resize_flow_to_img_res:
      flow = uflow_utils.resize(flow, orig_height, orig_width, is_flow=True)

    if infer_occlusion:
      return flow, warps, valid_warp_masks, range_map, occlusion_mask

    return flow

  @tf.function
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

    # return self.infer_no_tf_function(image1, image2, segmentation1, segmentation2, tracking_point1, tracking_point2, tracking_pos_emb,
    #                                 input_height, input_width, resize_flow_to_img_res, infer_occlusion, frame_index)

  # @tf.function
  # def batch_infer(self,
  #                 images,
  #                 segmentations,
  #                 input_height=None,
  #                 input_width=None,
  #                 resize_flow_to_img_res=True,
  #                 infer_occlusion=False):
  #
  #   return self.batch_infer_no_tf_function(images, segmentations, input_height, input_width,
  #                                          resize_flow_to_img_res,
  #                                          infer_occlusion)

  def infer_occlusion(self, flow_forward, flow_backward):
    """Gets a 'soft' occlusion mask from the forward and backward flow."""

    flows = {
        (0, 1, 'inference'): [flow_forward],
        (1, 0, 'inference'): [flow_backward],
    }
    warps, valid_warp_masks, range_maps_low_res, occlusion_masks, _, _ = uflow_utils.compute_warps_and_occlusion(
        flows,
        self._occlusion_estimation,
        self._occ_weights,
        self._occ_thresholds,
        self._occ_clip_max,
        occlusions_are_zeros=False)


    warps = warps[(0, 1, 'inference')][0]
    valid_warp_masks = valid_warp_masks[(0, 1, 'inference')][0]
    occlusion_mask_forward = occlusion_masks[(0, 1, 'inference')][0]
    range_maps_low_res = range_maps_low_res[(0, 1, 'inference')][0]

    return warps, valid_warp_masks, range_maps_low_res, occlusion_mask_forward

  def features_no_tf_function(self, image1, image2):
    """Runs the feature extractor portion of the model on image1 and image2."""
    images = tf.stack([image1, image2])
    # noinspection PyCallingNonCallable
    return self._feature_model(images, split_features_by_sample=True)

  @tf.function
  def features(self, image1, image2):
    """Runs the feature extractor portion of the model on image1 and image2."""
    return self.features_no_tf_function(image1, image2)

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

  @tf.function
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

    # with tf.GradientTape() as tape:
    #   losses = self.compute_loss(
    #       batch,
    #       weights,
    #       plot_dir,
    #       distance_metrics=distance_metrics,
    #       ground_truth_flow=ground_truth_flow,
    #       ground_truth_valid=ground_truth_valid,
    #       ground_truth_occlusions=ground_truth_occlusions,
    #       ground_truth_segmentations=ground_truth_segmentations,
    #       ground_truth_tracking_points=ground_truth_tracking_points,
    #       images_without_photo_aug=images_without_photo_aug,
    #       occ_active=occ_active)
    #
    # variables = (
    #     self._feature_model.trainable_variables +
    #     self._flow_model.trainable_variables)
    # grads = tape.gradient(losses['total-loss'], variables)

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
      # prev_seg_points_limit = tracking_utils.get_first_occurrence_indices(prev_seg_points[:, :, 0], -0.1)
      # cur_seg_points_limit = tracking_utils.get_first_occurrence_indices(cur_seg_points[:, :, 0], -0.1)

      # get the biggest value in the tensor
      # prev_seg_points_limit = tf.math.reduce_max(prev_seg_points_limit)
      # cur_seg_points_limit = tf.math.reduce_max(cur_seg_points_limit)
      # -------------------------------------------------------------------
      # find dense correspondence of all segmentation points, whereas gt id assignments are for a subset of points
      forward_spatial_offset, backward_spatial_offset, saved_offset = self._tracking_model(prev_patch, cur_patch, prev_seg_points, cur_seg_points, pos_emb)

      # forward_corr_2d_loss = tracking_utils.corr_2d_loss( gt_prev_id_assignments, gt_cur_id_assignments, forward_corr_2d_matrix)
      # backward_corr_2d_loss = tracking_utils.corr_2d_loss( gt_cur_id_assignments, gt_prev_id_assignments, backward_corr_2d_matrix)
      # corr_cycle_consistency_loss = tracking_utils.corr_cycle_consistency(forward_corr_2d_matrix, backward_corr_2d_matrix)

      # process forward_id_assign by contour length
      # prev_seg_points_limit = tracking_utils.get_first_occurrence_indices(prev_seg_points[:, :, 0], -0.1)
      # cur_seg_points_limit = tracking_utils.get_first_occurrence_indices(cur_seg_points[:, :, 0], -0.1)
      # forward_id_assign = tracking_utils.fit_id_assignments_to_next_contour(forward_id_assign, tf.expand_dims(prev_seg_points[:,:,-1], axis=-1), cur_seg_points_limit)
      # backward_id_assign = tracking_utils.fit_id_assignments_to_next_contour(backward_id_assign, tf.expand_dims(cur_seg_points[:,:,-1], axis=-1), prev_seg_points_limit)

      # total_forward_matching_points_loss = tracking_utils.matching_contour_points_loss(gt_prev_id_assignments, gt_cur_id_assignments, forward_id_assign)
      # total_backward_matching_points_loss = tracking_utils.matching_contour_points_loss(gt_cur_id_assignments, gt_prev_id_assignments, backward_id_assign)
      # total_cycle_consistency_assign_loss = tracking_utils.cycle_consistency_assign_loss(forward_id_assign, backward_id_assign, tf.expand_dims(prev_seg_points[:,:,-1], axis=-1),  tf.expand_dims(cur_seg_points[:,:,-1], axis=-1))
      # total_points_order_loss = tracking_utils.contour_points_order_loss(forward_id_assign)

      # total_forward_matching_points_spatial_metric = tracking_utils.matching_contour_points_spatial_metric(cur_seg_points[:,:,:2], gt_prev_id_assignments, gt_cur_id_assignments, forward_id_assign)
      pred_cur_tracking_points = prev_seg_points[:,:,:2] + forward_spatial_offset
      forward_spatial_points_loss = tracking_utils.matching_spatial_points_loss(gt_prev_id_assignments, gt_cur_id_assignments,
                                                                                      tf.cast(cur_seg_points[:,:,:2], dtype=pred_cur_tracking_points.dtype), pred_cur_tracking_points)

      cycle_consistency_spatial_loss = tracking_utils.cycle_consistency_spatial_loss(prev_seg_points[:,:,:2], prev_seg_points[:,:,-1], cur_seg_points[:,:,:2], cur_seg_points[:,:,-1], forward_spatial_offset, backward_spatial_offset)

      forward_tracker_photometric_loss = tracking_utils.tracker_unsupervised_photometric_loss(prev_patch, cur_patch, prev_seg_points[:,:,:2], prev_seg_points[:,:,-1], pred_cur_tracking_points)

      backward_tracker_photometric_loss = tracking_utils.tracker_unsupervised_photometric_loss(cur_patch, prev_patch, cur_seg_points[:,:,:2], cur_seg_points[:,:,-1], cur_seg_points[:,:,:2] + backward_spatial_offset)

      # forward_tracker_census_loss = tracking_utils.tracker_unsupervised_census_loss(prev_patch, cur_patch, prev_seg_points[:,:,:2], prev_seg_points[:,:,-1], pred_cur_tracking_points)

      # backward_tracker_census_loss = tracking_utils.tracker_unsupervised_census_loss(cur_patch, prev_patch, cur_seg_points[:,:,:2], cur_seg_points[:,:,-1], cur_seg_points[:,:,:2] + backward_spatial_offset)

      forward_linear_spring_loss, forward_normal_force_loss = tracking_utils.mechanical_loss_from_offsets(prev_seg_points[:,:,:2], prev_seg_points[:,:,-1], forward_spatial_offset)
      backward_linear_spring_loss, backward_normal_force_loss = tracking_utils.mechanical_loss_from_offsets(cur_seg_points[:,:,:2], cur_seg_points[:,:,-1], backward_spatial_offset)

      # snake_tension_loss, snake_stiffness_loss = tracking_utils.snake_loss_from_offsets(prev_seg_points[:,:,:2], prev_seg_points[:,:,-1], forward_spatial_offset)

      saved_offset_dict = {0: saved_offset}  # key for sequence number, only one key 0 if seq_num=2

      total_loss = cycle_consistency_spatial_loss + forward_normal_force_loss

      losses = {'total-loss' : total_loss,
                'forward_spatial_points_loss': forward_spatial_points_loss,
                'forward_tracker_photometric_loss': forward_tracker_photometric_loss, 'backward_tracker_photometric_loss': backward_tracker_photometric_loss,
                # 'forward_tracker_census_loss': forward_tracker_census_loss, 'backward_tracker_census_loss': backward_tracker_census_loss,
                'cycle_consistency_spatial_loss': cycle_consistency_spatial_loss,
                # 'snake_tension_loss': snake_tension_loss, 'snake_stiffness_loss': snake_stiffness_loss,
                'forward_linear_spring_loss': forward_linear_spring_loss, 'forward_normal_force_loss': forward_normal_force_loss,
                'backward_linear_spring_loss': backward_linear_spring_loss, 'backward_normal_force_loss': backward_normal_force_loss}

      return losses, saved_offset_dict

  def compute_loss(self,
                   batch,
                   weights,
                   plot_dir=None,
                   distance_metrics=None,
                   ground_truth_flow=None,
                   ground_truth_valid=None,
                   ground_truth_occlusions=None,
                   ground_truth_segmentations=None,
                   ground_truth_tracking_points=None,
                   images_without_photo_aug=None,
                   occ_active=None):
    """Applies the model and computes losses for a batch of image sequences."""
    # Compute only a supervised loss.
    if self._train_with_supervision:
      if ground_truth_flow is None:
        raise ValueError('Need ground truth flow to compute supervised loss.')
      flows = uflow_utils.compute_flow_for_supervised_loss(
          self._feature_model, self._flow_model, batch=batch, training=True)
      losses = uflow_utils.supervised_loss(weights, ground_truth_flow,
                                           ground_truth_valid, flows)
      losses = {key + '-loss': losses[key] for key in losses}
      return losses

    # Use possibly augmented images if non augmented version is not provided.
    if images_without_photo_aug is None:
      images_without_photo_aug = batch

    flows, selfsup_transform_fns = uflow_utils.compute_features_and_flow(
        self._feature_model,
        self._flow_model,
        batch=batch,
        batch_without_aug=images_without_photo_aug,
        training=True,
        build_selfsup_transformations=self._build_selfsup_transformations,
        teacher_feature_model=self._teacher_feature_model,
        teacher_flow_model=self._teacher_flow_model,
        teacher_image_version=self._teacher_image_version,
        ground_truth_segmentations=ground_truth_segmentations,
        ground_truth_tracking_points=ground_truth_tracking_points
    )

    # Prepare images and contours for unsupervised loss (prefer unaugmented images).
    seq_len = int(batch.shape[1])
    images = {i: images_without_photo_aug[:, i] for i in range(seq_len)}
    contours = {i: ground_truth_segmentations[:, i] for i in range(seq_len)}

    # Warp stuff and compute occlusion.
    warps, valid_warp_masks, _, not_occluded_masks, fb_sq_diff, fb_sum_sq = uflow_utils.compute_warps_and_occlusion(
                                                                            flows,
                                                                            occlusion_estimation=self._occlusion_estimation,
                                                                            occ_weights=self._occ_weights,
                                                                            occ_thresholds=self._occ_thresholds,
                                                                            occ_clip_max=self._occ_clip_max,
                                                                            occlusions_are_zeros=True,
                                                                            occ_active=occ_active)

    # Warp images and features.
    warped_images = uflow_utils.apply_warps_stop_grad(images, warps, level=0)

    # Warp contours
    assert len(images) == len(contours)

    # dilate contours and get image at the contour regions
    img_contours = {}
    for dict_i in range(len(images)):
        # img_contours[dict_i] = images[dict_i] * contours[dict_i]  # contour image without dilation

        a_depth = contours[dict_i].shape[-1]
        dilation_filter = tf.zeros( [1, 1, a_depth], tf.float32)  # why zeros? https://stackoverflow.com/questions/54686895/tensorflow-dilation-behave-differently-than-morphological-dilation
        dilated_contour = tf.nn.dilation2d(input=contours[dict_i], filters=dilation_filter, strides=[1,1,1,1], padding='SAME', data_format='NHWC', dilations=[1, 1, 1, 1])
        img_contours[dict_i] = images[dict_i] * dilated_contour

        # debug whether images are loaded correctly for training
        # cv2.imwrite(f'debug_dilated_contour.png', dilated_contour.numpy()[0, :, :, :])
        # cv2.imwrite(f'debug_img_{dict_i}.png', images[dict_i].numpy()[0, :, :, :] * 255)
        # cv2.imwrite(f'debug_contour_{dict_i}.png', contours[dict_i].numpy()[0,:,:,:])
        # cv2.imwrite(f'debug_contour_img_{dict_i}.png', img_contours[dict_i].numpy()[0, :, :, :])
        # print( tf.unique_with_counts(tf.reshape(dilated_contour, -1)) )

    warped_contours = uflow_utils.apply_warps_stop_grad(img_contours, warps, level=0)

    # cv2.imwrite(f'debug_warped_contour_0.png', warped_contours[(0, 1, 'original-teacher')].numpy()[0, :, :, :])
    # cv2.imwrite(f'debug_warped_contour_1.png', warped_contours[(1, 0, 'original-teacher')].numpy()[0, :, :, :])
    # import pdb; pdb.set_trace()

    # Compute losses.
    losses = uflow_utils.compute_loss(
        weights=weights,
        images=images,
        contours=img_contours,
        flows=flows,
        warps=warps,
        valid_warp_masks=valid_warp_masks,
        not_occluded_masks=not_occluded_masks,
        fb_sq_diff=fb_sq_diff,
        fb_sum_sq=fb_sum_sq,
        warped_images=warped_images,
        warped_contours=warped_contours,
        only_forward=self._only_forward,
        selfsup_transform_fns=selfsup_transform_fns,
        fb_sigma_teacher=self._fb_sigma_teacher,
        fb_sigma_student=self._fb_sigma_student,
        plot_dir=plot_dir,
        distance_metrics=distance_metrics,
        smoothness_edge_weighting=self._smoothness_edge_weighting,
        stop_gradient_mask=self._stop_gradient_mask,
        selfsup_mask=self._selfsup_mask,
        ground_truth_occlusions=ground_truth_occlusions,
        smoothness_at_level=self._smoothness_at_level)
    losses = {key + '-loss': losses[key] for key in losses}

    return losses

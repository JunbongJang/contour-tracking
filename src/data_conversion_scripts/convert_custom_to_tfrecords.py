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
This script converts Custom Data ( images, segmentation mask, tracking points ) into the TFRecords format.
"""

import os
from absl import app
from absl import flags
import imageio
import numpy as np
import tensorflow as tf
from src.data_conversion_scripts import conversion_utils
import cv2

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', None, 'Dataset folder.')
flags.DEFINE_string('output_dir', '', 'Location to export to.')
flags.DEFINE_integer('shard', 0, 'Which shard this is.')
flags.DEFINE_integer('num_shards', 100, 'How many total shards there are.')
flags.DEFINE_integer('seq_len', 2, 'How long is the sequence?')
flags.DEFINE_string('img_format', 'jpg', 'image format')
flags.DEFINE_string('mode', 'sparse', 'sparse or dense video frames')
flags.DEFINE_string('data_split', None, 'training or test video frames')


def get_image_name(image_path, file_format):
    return image_path.split('/')[-1].replace(f".{file_format}", '')


def find_least_tracking_points(tracking_points_folders):
    least_tracking_points = None
    for a_folder_path in tracking_points_folders:
        tracking_points_path_list = tf.io.gfile.glob(a_folder_path + '/*.txt')
        a_tracking_point = np.loadtxt(tracking_points_path_list[0], dtype=np.int32)
        if least_tracking_points is None or a_tracking_point.shape[0] < least_tracking_points:
            least_tracking_points = a_tracking_point.shape[0]

    assert least_tracking_points is not None

    return least_tracking_points


def create_list_of_mask_vector(CUSTOM_MAX_TOTAL_SEG_POINTS_NUM, cur_seg_points, next_seg_points):
    '''
    For masking cross-attention

    :param CUSTOM_MAX_TOTAL_SEG_POINTS_NUM:
    :param cur_seg_points:
    :return:
    '''
    if cur_seg_points.shape[0] > CUSTOM_MAX_TOTAL_SEG_POINTS_NUM:
        mask_vector = np.ones(CUSTOM_MAX_TOTAL_SEG_POINTS_NUM)
    else:
        # Create a mask vector with 0 for zero padded seg_points
        mask_vector = np.ones(cur_seg_points.shape[0])
        mask_vector = np.pad(mask_vector, (0, CUSTOM_MAX_TOTAL_SEG_POINTS_NUM - cur_seg_points.shape[0]), 'constant', constant_values=(0))
    mask_vector = np.expand_dims(mask_vector, axis=-1)
    mask_vector = mask_vector.astype(np.int32)
    list_of_mask_vector = [mask_vector]

    # -------------------------------------------------------------------------
    # total_offset = 40
    # assert cur_seg_points.shape[0] > total_offset
    # list_of_mask_vector = []
    # # set mask to 1 for left 20 pixels and right 20 pixels
    # for point_index in range(CUSTOM_MAX_TOTAL_SEG_POINTS_NUM):
    #     # print(point_index, cur_seg_points.shape, next_seg_points.shape)
    #     if point_index < total_offset // 2:
    #         # print('case 1@@')
    #         one_vector = np.ones(total_offset // 2 + point_index)
    #         mask_vector = np.pad(one_vector, (0, CUSTOM_MAX_TOTAL_SEG_POINTS_NUM - one_vector.shape[0]), 'constant', constant_values=(0))
    #
    #     elif total_offset // 2 <= point_index < cur_seg_points.shape[0] :
    #         # print('case 2@@')
    #         cur_point_index = point_index - total_offset // 2
    #         one_vector = np.ones(total_offset)
    #         mask_vector = np.pad(one_vector, (0, CUSTOM_MAX_TOTAL_SEG_POINTS_NUM - one_vector.shape[0]), 'constant', constant_values=(0))
    #         mask_vector = np.roll(mask_vector, cur_point_index)
    #
    #     elif point_index >= cur_seg_points.shape[0]:
    #         # print('case 3@@')
    #         mask_vector = np.zeros(CUSTOM_MAX_TOTAL_SEG_POINTS_NUM)
    #
    #     mask_vector[next_seg_points.shape[0]:] = 0
    #     mask_vector = np.expand_dims(mask_vector, axis=-1)
    #     mask_vector = mask_vector.astype(np.int32)
    #     list_of_mask_vector.append(mask_vector)

    return list_of_mask_vector


def create_seq_from_list(tracking_points_path_list, seq_len, data_split, mode):
    total_tracking_points_path_num = len(tracking_points_path_list)

    if data_split == 'training' and mode == 'dense':
        interval_len = 5
        tracking_points_seq = [[], []]
        for start_index in range(interval_len):
            temp_path_list = tracking_points_path_list[start_index:total_tracking_points_path_num - interval_len + start_index + 1:interval_len]
            temp_path_num = len(temp_path_list)
            temp_seq = [temp_path_list[seq_index:temp_path_num + seq_index + 1 - seq_len] for seq_index in
                        range(seq_len)]  # e.g) [[1,6,...,191], [6,11,16,...,196]]
            tracking_points_seq[0] = tracking_points_seq[0] + temp_seq[0]
            tracking_points_seq[1] = tracking_points_seq[1] + temp_seq[1]
        # returns [[1,6,...,191, 2,7,12,...,192, 3,8,13,...,193, 4,...], [6,11,16,...,196, 7,12,17,...,197, 8,13,...,198, 9,...]]
    else:
        # tracking_points_seq = zip(tracking_points_path_list[:-1], tracking_points_path_list[1:])
        tracking_points_seq = [tracking_points_path_list[seq_index:total_tracking_points_path_num + seq_index + 1 - seq_len] for seq_index in
                               range(seq_len)]  # i.e) if seq_len == 2, [ [1,2,3,4,...198,199], [2,3,4,5,...,199,200] ]

    return tracking_points_seq


def convert_dataset(seq_len, mode, data_split):
    """
    Convert the data to the TFRecord format.
    dataset is images and tracking points label, but no segmentation label

    """

    def write_records(data_path_list, output_folder, num_least_tracking_points, CUSTOM_MAX_TOTAL_SEG_POINTS_NUM):
        """
        Takes in list: 200 x [((im1_path, ..., imn_path), (seg1_path, ..., seqn_path), (track1_path, ..., trackn_path))]
        and writes records.
        """

        # Reading ppm and flo can fail on network filesystem, so copy to tmpdir first.
        tmpdir = '/tmp/convert_custom_to_tfrecords'
        if not os.path.exists(tmpdir):
            os.mkdir(tmpdir)

        filenames = conversion_utils.generate_sharded_filenames(
            os.path.join(output_folder, 'custom@{}'.format(FLAGS.num_shards)))
        with tf.io.TFRecordWriter(filenames[FLAGS.shard]) as record_writer:
            total = len(data_path_list)
            images_per_shard = total // FLAGS.num_shards
            start = images_per_shard * FLAGS.shard
            end = start + images_per_shard
            # Account for num images not being divisible by num shards.
            if FLAGS.shard == FLAGS.num_shards - 1:
                data_path_list = data_path_list[start:]
            else:
                data_path_list = data_path_list[start:end]

            tf.compat.v1.logging.info('Writing %d images per shard', images_per_shard)
            tf.compat.v1.logging.info('Writing range %d to %d of %d total.', start, end, total)

            # create temp path
            seq_len = len(data_path_list[0][0])
            temp_img_path_list = []
            temp_seg_points_path_list = []
            temp_tracking_point_path_list = []
            for seq_index in range(seq_len):
                temp_img_path_list.append( os.path.join(tmpdir, f'img{seq_index}.{FLAGS.img_format}') )
                temp_seg_points_path_list.append( os.path.join(tmpdir, f'segmentation_points{seq_index}.txt') )
                temp_tracking_point_path_list.append( os.path.join(tmpdir, f'tracking_point{seq_index}.txt') )

            max_total_seg_points_num = 0
            # each element of data_path_list consists of seq_len of images and tracking_points

            for i, (images, seg_points, tracking_points) in enumerate(data_path_list):
                img_data_list = []
                seg_points_data_list = []
                tracking_point_data_list = []

                for seq_index in range(seq_len):
                    # --------------- Images --------------
                    if os.path.exists(temp_img_path_list[seq_index]):
                        os.remove(temp_img_path_list[seq_index])
                    tf.io.gfile.copy(images[seq_index], temp_img_path_list[seq_index])
                    a_image = imageio.imread( temp_img_path_list[seq_index] )
                    if a_image.dtype == np.uint16:
                        # convert to uint8
                        a_image = cv2.convertScaleAbs(a_image, alpha=(255.0/65535.0))

                    img_data_list.append( a_image )

                    if len(img_data_list[seq_index].shape) == 2:
                        img_data_list[seq_index] = np.expand_dims(img_data_list[seq_index] , axis=-1)
                        img_data_list[seq_index] = np.repeat(img_data_list[seq_index], 3, axis=-1)
                    elif img_data_list[seq_index].shape[-1] == 4:
                        img_data_list[seq_index] = img_data_list[seq_index][:, :, 0:3]

                    # --------------- Segmentation points -----------------
                    if os.path.exists(temp_seg_points_path_list[seq_index]):
                        os.remove(temp_seg_points_path_list[seq_index])

                    tf.io.gfile.copy(seg_points[seq_index], temp_seg_points_path_list[seq_index])
                    cur_seg_points = np.loadtxt(temp_seg_points_path_list[seq_index], dtype=np.int32)  # convert to uint8 ~~~~~~~~~

                    # zero padding to make every tensor's length the same
                    if cur_seg_points.shape[0] > CUSTOM_MAX_TOTAL_SEG_POINTS_NUM:
                        print('number of seg points is greater than CUSTOM_MAX_TOTAL_SEG_POINTS_NUM', cur_seg_points.shape[0])
                        padded_seg_points = cur_seg_points[:CUSTOM_MAX_TOTAL_SEG_POINTS_NUM,:]
                    else:
                        padded_seg_points = np.pad(cur_seg_points, ((0, CUSTOM_MAX_TOTAL_SEG_POINTS_NUM - cur_seg_points.shape[0]), (0,0)), 'constant', constant_values=(-100))

                    # create masking
                    assert seq_len == 2
                    if seq_index == 0:
                        next_seq_index = 1

                    elif seq_index == 1:
                        next_seq_index = 0
                    else:
                        raise ValueError('@@@ seg_points')
                    if os.path.exists(temp_seg_points_path_list[next_seq_index]):
                        os.remove(temp_seg_points_path_list[next_seq_index])

                    tf.io.gfile.copy(seg_points[next_seq_index], temp_seg_points_path_list[next_seq_index])
                    next_seg_points = np.loadtxt(temp_seg_points_path_list[next_seq_index], dtype=np.int32)

                    list_of_mask_vector = create_list_of_mask_vector(CUSTOM_MAX_TOTAL_SEG_POINTS_NUM, cur_seg_points, next_seg_points)
                    concat_seg_points = np.concatenate((padded_seg_points, *list_of_mask_vector), axis=1)
                    seg_points_data_list.append( concat_seg_points )

                    assert np.int32 == concat_seg_points.dtype
                    assert concat_seg_points.shape == (CUSTOM_MAX_TOTAL_SEG_POINTS_NUM, 3)
                    
                    # --------------- Tracking points ----------------
                    if os.path.exists(temp_tracking_point_path_list[seq_index]):
                        os.remove(temp_tracking_point_path_list[seq_index])

                    tf.io.gfile.copy(tracking_points[seq_index], temp_tracking_point_path_list[seq_index])
                    a_tracking_point = np.loadtxt(temp_tracking_point_path_list[seq_index], dtype=np.int32)
                    a_tracking_point = np.expand_dims(a_tracking_point, axis=-1)
                    tracking_point_data_list.append( a_tracking_point )

                    assert a_tracking_point.shape[1] == 1
                    
                    # --------------------------------
                    # check data shapes
                    if seq_index > 0:
                        assert height == img_data_list[seq_index].shape[0]
                        assert width == img_data_list[seq_index].shape[1]
                        assert total_tracking_point_num == len(tracking_point_data_list[seq_index])

                    height = img_data_list[seq_index].shape[0]
                    width = img_data_list[seq_index].shape[1]
                    total_tracking_point_num = len(tracking_point_data_list[seq_index])
                    total_seg_points_num = len(seg_points_data_list[seq_index])
                    if max_total_seg_points_num < total_seg_points_num:
                        max_total_seg_points_num = total_seg_points_num

                # ------------------------------
                # partition tracking points such that each partition has num_least_tracking_points
                partitioned_tracking_point_data_list = []
                num_all_tracking_points = tracking_point_data_list[0].shape[0]
                if num_all_tracking_points > num_least_tracking_points:
                    # print('partition!')
                    for index in range(0, num_all_tracking_points, num_least_tracking_points):  # iterate floor(num_all / num_least) + 1 times
                        one_partition_tracking_point_data_list = []
                        for a_tracking_point_data in tracking_point_data_list:  # iterate seq_length time
                            if num_all_tracking_points - index < num_least_tracking_points:
                                indexed_tracking_point_data = a_tracking_point_data[-num_least_tracking_points:]
                            else:
                                indexed_tracking_point_data = a_tracking_point_data[index:index + num_least_tracking_points]

                            assert indexed_tracking_point_data.shape[0] == num_least_tracking_points
                            one_partition_tracking_point_data_list.append(indexed_tracking_point_data)

                        partitioned_tracking_point_data_list.append(one_partition_tracking_point_data_list)

                else:
                    partitioned_tracking_point_data_list = [tracking_point_data_list]

                # ------------------------------
                # Case 1: Use only one partition of tracked points since supervision is not done
                a_partitioned_tracking_point_data_list = partitioned_tracking_point_data_list[0]

                # Case 2: uncomment below and then tab below? if more tracking points need to be used
                # for a_partitioned_tracking_point_data_list in partitioned_tracking_point_data_list:

                # save data
                feature = {
                    'height': conversion_utils.int64_feature(height),
                    'width': conversion_utils.int64_feature(width),
                }

                for seq_index in range(seq_len):
                    feature.update({
                        f'image{seq_index}_path': conversion_utils.bytes_feature(str.encode(images[seq_index])),
                        f'segmentation_points{seq_index}_path': conversion_utils.bytes_feature(str.encode(seg_points[seq_index])),
                        f'tracking_points{seq_index}_path': conversion_utils.bytes_feature(str.encode(tracking_points[seq_index])),
                    })

                example = tf.train.SequenceExample(
                    context=tf.train.Features(feature=feature),
                    feature_lists=tf.train.FeatureLists(
                        feature_list={
                            'images':
                                tf.train.FeatureList(feature=[conversion_utils.bytes_feature(image_data.tobytes()) for image_data in img_data_list ]),
                            'segmentation_points':
                                tf.train.FeatureList(feature=[conversion_utils.bytes_feature(seg_points_data.tobytes()) for seg_points_data in seg_points_data_list ]),
                            'tracking_points':
                                tf.train.FeatureList(feature=[conversion_utils.bytes_feature(tracking_point_data.tobytes()) for tracking_point_data in a_partitioned_tracking_point_data_list ])
                        }))


                # log saving
                if i % 10 == 0:
                    tf.compat.v1.logging.info('Writing %d out of %d total.', i, len(data_path_list))
                record_writer.write(example.SerializeToString())

            print('max_total_seg_points_num', max_total_seg_points_num)

        tf.compat.v1.logging.info('Saved results to %s', FLAGS.output_dir)

        # ------------------------------------------- Write Records End -------------------------------------------

    if not tf.io.gfile.exists(FLAGS.output_dir):
        tf.io.gfile.mkdir(FLAGS.output_dir)

    split_folder = os.path.join(FLAGS.output_dir, data_split)
    if not tf.io.gfile.exists(split_folder):
        tf.io.gfile.mkdir(split_folder)

    output_folder = os.path.join(FLAGS.output_dir, data_split)
    if not tf.io.gfile.exists(output_folder):
        tf.io.gfile.mkdir(output_folder)

    data_folder_path = os.path.join(FLAGS.data_dir, data_split)

    image_folders = sorted(tf.io.gfile.glob( data_folder_path + f'/*/images'))
    seg_points_folders = sorted(tf.io.gfile.glob(data_folder_path + f'/*/contour_points'))
    tracking_points_folders = sorted(tf.io.gfile.glob(data_folder_path + f'/*/tracked_points_in_contour_indices'))

    assert len(image_folders) == len(tracking_points_folders)
    assert len(image_folders) == len(seg_points_folders)

    data_list = []
    for image_folder_path, seg_points_folder_path, tracking_points_folder_path in zip(image_folders, seg_points_folders, tracking_points_folders):
        # get image path
        image_path_list = tf.io.gfile.glob(image_folder_path + f"/*.{FLAGS.img_format}")
        sort_by_frame_index = lambda x: int( os.path.basename(x).split('_')[-1].split('.')[0])  # MARS-Net, Jellyfish
        # sort_by_frame_index = lambda x: int( os.path.basename(x)[-8:].split('.')[0] )  # HACKS
        image_path_list = sorted(image_path_list, key=sort_by_frame_index)

        # get segmentation points path
        seg_points_path_list = tf.io.gfile.glob(seg_points_folder_path + f"/*.txt")
        seg_points_path_list = sorted(seg_points_path_list, key=sort_by_frame_index)

        # get their names
        image_name_list = []
        seg_name_list = []
        for a_image_path, cur_seg_points_path in zip(image_path_list, seg_points_path_list):
            image_name_list.append(get_image_name(a_image_path, FLAGS.img_format))
            seg_name_list.append(get_image_name(cur_seg_points_path, 'txt'))

        # get tracking points path
        tracking_points_path_list = tf.io.gfile.glob(tracking_points_folder_path + '/*.txt')
        tracking_points_path_list = sorted(tracking_points_path_list, key=sort_by_frame_index)

        # --------------------------------------------------------------------------------------
        expanded_tracking_points_path_list = []
        # assert len(tracking_points_path_list) == 41
        if data_split == 'test_dense' and len(tracking_points_path_list) == 41:
            print('collect a consecutive frames for dense prediction')
            for path_index, a_path in enumerate(tracking_points_path_list):
                if path_index == 0:
                    expanded_tracking_points_path_list.append(a_path)
                    expanded_tracking_points_path_list.append(a_path)
                    expanded_tracking_points_path_list.append(a_path)
                    expanded_tracking_points_path_list.append(a_path)
                elif path_index >= len(tracking_points_path_list)-1:
                    expanded_tracking_points_path_list.append(a_path)
                else:
                    expanded_tracking_points_path_list.append(a_path)
                    expanded_tracking_points_path_list.append(a_path)
                    expanded_tracking_points_path_list.append(a_path)
                    expanded_tracking_points_path_list.append(a_path)
                    expanded_tracking_points_path_list.append(a_path)
            tracking_points_path_list = expanded_tracking_points_path_list
            assert len(tracking_points_path_list) == 200

            selected_image_path_list = image_path_list
            selected_seg_points_path_list = seg_points_path_list
        else:
            # to predict on sparse frames given 41 tracking points
            # only select images and segmentations that has the corresponding labeled tracking points
            selected_image_path_list = []
            selected_seg_points_path_list = []
            for a_contour_point_path, a_tracking_point_path in zip(seg_points_path_list, tracking_points_path_list):
                # get filenames of tracking points
                tracking_point_filename = get_image_name(a_tracking_point_path, 'txt')
                # query_image_name = 'img' + tracking_point_filename[-4:]  # HACKS
                
                # get images with the same filenames
                # a_index = image_name_list.index(query_image_name)  # HACKS
                a_index = image_name_list.index(tracking_point_filename)  # MARS-Net
                selected_image_path_list.append(image_path_list[a_index])

                a_index = seg_name_list.index(tracking_point_filename)
                selected_seg_points_path_list.append(seg_points_path_list[a_index])

        # -------------------------------------------------------------------------------
        new_tracking_points_seq = create_seq_from_list(tracking_points_path_list, seq_len, data_split, mode)
        new_image_seq = create_seq_from_list(selected_image_path_list, seq_len, data_split, mode)
        new_seg_seq = create_seq_from_list(selected_seg_points_path_list, seq_len, data_split, mode)
        # for training with shuffled sequence
        # assert 'training' == data_split

        # @@@@@@@@@@ for backward tracking test set, reverse sequence
        # assert seq_len == 2
        # assert ('test' in data_split) == True
        # new_tracking_points_seq[0].reverse()
        # new_tracking_points_seq[1].reverse()
        # new_image_seq[0].reverse()
        # new_image_seq[1].reverse()
        # new_seg_seq[0].reverse()
        # new_seg_seq[1].reverse()

        # -------------------------------------------------------------------------------
        zipped_tracking_points_seq = zip(*new_tracking_points_seq)
        zipped_image_seq = zip(*new_image_seq)
        zipped_seg_seq = zip(*new_seg_seq)

        assert len(tracking_points_path_list) == len(selected_image_path_list)
        assert len(tracking_points_path_list) == len(selected_seg_points_path_list)
        assert len(new_tracking_points_seq) == seq_len
        assert len(new_tracking_points_seq) == len(new_image_seq)
        assert len(new_tracking_points_seq) == len(new_seg_seq)
        assert len(new_tracking_points_seq[0]) == len(new_tracking_points_seq[1])
        assert len(new_tracking_points_seq[0]) == len(new_image_seq[0])
        assert len(new_tracking_points_seq[0]) == len(new_seg_seq[0])
        assert len(new_tracking_points_seq[0]) == len(new_image_seq[1])
        assert len(new_tracking_points_seq[0]) == len(new_seg_seq[1])

        data_list.extend(zip(zipped_image_seq, zipped_seg_seq, zipped_tracking_points_seq))

    num_least_tracking_points = find_least_tracking_points(tracking_points_folders)
    print('num_least_tracking_points', num_least_tracking_points)

    CUSTOM_MAX_TOTAL_SEG_POINTS_NUM = 1640 # 1640  # 1150 # TODO
    write_records(data_list, output_folder, num_least_tracking_points, CUSTOM_MAX_TOTAL_SEG_POINTS_NUM)

# ------------------------------------------------------------------------------------

@tf.function
def resize(img, height, width, is_flow, mask=None):
  """Resize an image or flow field to a new resolution.

  In case a mask (per pixel {0,1} flag) is passed a weighted resizing is
  performed to account for missing flow entries in the sparse flow field. The
  weighting is based on the resized mask, which determines the 'amount of valid
  flow vectors' that contributed to each individual resized flow vector. Hence,
  multiplying by the reciprocal cancels out the effect of considering non valid
  flow vectors.

  Args:
    img: tf.tensor, image or flow field to be resized of shape [b, h, w, c]
    height: int, heigh of new resolution
    width: int, width of new resolution
    is_flow: bool, flag for scaling flow accordingly
    mask: tf.tensor, mask (optional) per pixel {0,1} flag

  Returns:
    Resized and potentially scaled image or flow field (and mask).
  """

  def _resize(img, mask=None):
    # _, orig_height, orig_width, _ = img.shape.as_list()
    orig_height = tf.shape(input=img)[1]
    orig_width = tf.shape(input=img)[2]
    if orig_height == height and orig_width == width:
      # early return if no resizing is required
      if mask is not None:
        return img, mask
      else:
        return img

    if mask is not None:
      # multiply with mask, to ensure non-valid locations are zero
      img = tf.math.multiply(img, mask)
      # resize image
      img_resized = tf.compat.v2.image.resize(
          img, (int(height), int(width)), antialias=True)
      # resize mask (will serve as normalization weights)
      mask_resized = tf.compat.v2.image.resize(
          mask, (int(height), int(width)), antialias=True)
      # normalize sparse flow field and mask
      img_resized = tf.math.multiply(img_resized,
                                     tf.math.reciprocal_no_nan(mask_resized))
      mask_resized = tf.math.multiply(mask_resized,
                                      tf.math.reciprocal_no_nan(mask_resized))
    else:
      # normal resize without anti-alaising
      prev_img_dtype = img.dtype
      img_resized = tf.compat.v2.image.resize(img, (int(height), int(width)))
      # img_resized = tf.cast(img, prev_img_dtype)

    if is_flow:
      # If image is a flow image, scale flow values to be consistent with the
      # new image size.
      scaling = tf.reshape([
          float(height) / tf.cast(orig_height, tf.float32),
          float(width) / tf.cast(orig_width, tf.float32)
      ], [1, 1, 1, 2])
      img_resized *= scaling

    if mask is not None:
      return img_resized, mask_resized
    return img_resized

  # Apply resizing at the right shape.
  shape = img.shape.as_list()
  if len(shape) == 3:
    if mask is not None:
      img_resized, mask_resized = _resize(img[None], mask[None])
      return img_resized[0], mask_resized[0]
    else:
      return _resize(img[None])[0]
  elif len(shape) == 4:
    # Input at the right shape.
    return _resize(img, mask)
  elif len(shape) > 4:
    # Reshape input to [b, h, w, c], resize and reshape back.
    img_flattened = tf.reshape(img, [-1] + shape[-3:])
    if mask is not None:
      mask_flattened = tf.reshape(mask, [-1] + shape[-3:])
      img_resized, mask_resized = _resize(img_flattened, mask_flattened)
    else:
      img_resized = _resize(img_flattened)
    # There appears to be some bug in tf2 tf.function
    # that fails to capture the value of height / width inside the closure,
    # leading the height / width undefined here. Call set_shape to make it
    # defined again.
    img_resized.set_shape(
        (img_resized.shape[0], height, width, img_resized.shape[3]))
    result_img = tf.reshape(img_resized, shape[:-3] + img_resized.shape[-3:])
    if mask is not None:
      mask_resized.set_shape(
          (mask_resized.shape[0], height, width, mask_resized.shape[3]))
      result_mask = tf.reshape(mask_resized,
                               shape[:-3] + mask_resized.shape[-3:])
      return result_img, result_mask
    return result_img
  else:
    raise ValueError('Cannot resize an image of shape', shape)


@tf.function
def resize_uint8(img, height, width):
  """Resize an image or flow field to a new resolution.

  In case a mask (per pixel {0,1} flag) is passed a weighted resizing is
  performed to account for missing flow entries in the sparse flow field. The
  weighting is based on the resized mask, which determines the 'amount of valid
  flow vectors' that contributed to each individual resized flow vector. Hence,
  multiplying by the reciprocal cancels out the effect of considering non valid
  flow vectors.

  Args:
    img: tf.tensor, image or flow field to be resized of shape [b, h, w, c]
    height: int, heigh of new resolution
    width: int, width of new resolution
    is_flow: bool, flag for scaling flow accordingly
    mask: tf.tensor, mask (optional) per pixel {0,1} flag

  Returns:
    Resized and potentially scaled image or flow field (and mask).
  """

  def _resize(img):
      orig_height = tf.shape(input=img)[1]
      orig_width = tf.shape(input=img)[2]
      if orig_height == height and orig_width == width:
          # early return if no resizing is required
          return img

      # normal resize without anti-alaising
      prev_img_dtype = img.dtype
      img_resized = tf.compat.v2.image.resize(img, (int(height), int(width)))
      img_resized = tf.cast(img_resized, prev_img_dtype)

      return img_resized


  # Apply resizing at the right shape.
  shape = img.shape.as_list()
  if len(shape) == 3:
      return _resize(img[None])[0]
  elif len(shape) == 4:
    # Input at the right shape.
    return _resize(img)
  elif len(shape) > 4:
      # Reshape input to [b, h, w, c], resize and reshape back.
      img_flattened = tf.reshape(img, [-1] + shape[-3:])
      img_resized = _resize(img_flattened)
      # There appears to be some bug in tf2 tf.function
      # that fails to capture the value of height / width inside the closure,
      # leading the height / width undefined here. Call set_shape to make it
      # defined again.
      img_resized.set_shape(
          (img_resized.shape[0], height, width, img_resized.shape[3]))
      result_img = tf.reshape(img_resized, shape[:-3] + img_resized.shape[-3:])
      return result_img
  else:
      raise ValueError('Cannot resize an image of shape', shape)


def debug_file():
    raw_dataset = tf.data.TFRecordDataset("uflow/assets/pc_celltrack/tfrecord/training/img/custom@0.tfrecord")
    for raw_record in raw_dataset.take(1):
      record = tf.train.SequenceExample.FromString(raw_record.numpy())
      with open('record.txt', 'w') as f:
          f.write(str(record))


def main(_):
    FLAGS.output_dir = FLAGS.data_dir + "tfrecord/"
    
    convert_dataset(FLAGS.seq_len, FLAGS.mode, FLAGS.data_split)
    # convert_dataset_with_segmentation()

    print('@@@@@@@@@@@')
    print('seq_len', FLAGS.seq_len)
    print('mode', FLAGS.mode)
    print('@@@@@@@@@@@')


if __name__ == '__main__':
  # debug_file()
  app.run(main)

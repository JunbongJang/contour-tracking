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

"""
Modified by Junbong Jang
Date: 7/27/2022

This script converts Custom Data ( images, segmentation mask, tracking points ) into the TFRecords format.
"""

import os
from absl import app
from absl import flags
import imageio
import numpy as np
import tensorflow as tf
from uflow.data_conversion_scripts import conversion_utils


FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '', 'Dataset folder.')
flags.DEFINE_string('output_dir', '', 'Location to export to.')
flags.DEFINE_integer('shard', 0, 'Which shard this is.')
flags.DEFINE_integer('num_shards', 100, 'How many total shards there are.')
flags.DEFINE_string('img_format', 'jpg', 'image format')
flags.DEFINE_string('segmentation_format', None, 'segmentation format')


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

def convert_dataset(seq_len):
    """
    Convert the data to the TFRecord format.
    dataset is images and tracking points label, but no segmentation label

    """

    def write_records(data_path_list, output_folder, num_least_tracking_points):
        """Takes in list: 200 x [((im1_path, ..., imn_path), (track1_path, ..., trackn_path), (seg1_path, ..., seqn_path))] and writes records."""

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
            temp_segmentation_path_list = []
            temp_tracking_point_path_list = []
            for seq_index in range(seq_len):
                temp_img_path_list.append( os.path.join(tmpdir, f'img{seq_index}.{FLAGS.img_format}') )
                temp_segmentation_path_list.append( os.path.join(tmpdir, f'segmentation{seq_index}.{FLAGS.img_format}') )
                temp_tracking_point_path_list.append( os.path.join(tmpdir, f'tracking_point{seq_index}.txt') )

            # each element of data_path_list consists of seq_len of images and tracking_points
            for i, (images, segmentations, tracking_points) in enumerate(data_path_list):
                img_data_list = []
                segmentation_data_list = []
                tracking_point_data_list = []

                for seq_index in range(seq_len):
                    # --------------- Images --------------
                    if os.path.exists(temp_img_path_list[seq_index]):
                        os.remove(temp_img_path_list[seq_index])
                    tf.io.gfile.copy(images[seq_index], temp_img_path_list[seq_index])
                    a_image = imageio.imread( temp_img_path_list[seq_index] ).astype('uint8')  # convert to uint8 ~~~~~~~~~
                    img_data_list.append( a_image )

                    if len(img_data_list[seq_index].shape) == 2:
                        img_data_list[seq_index] = np.expand_dims(img_data_list[seq_index] , axis=-1)
                        img_data_list[seq_index] = np.repeat(img_data_list[seq_index], 3, axis=-1)
                    elif img_data_list[seq_index].shape[-1] == 4:
                        img_data_list[seq_index] = img_data_list[seq_index][:, :, 0:3]

                    # --------------- Segmentations -----------------
                    if os.path.exists(temp_segmentation_path_list[seq_index]):
                        os.remove(temp_segmentation_path_list[seq_index])

                    tf.io.gfile.copy(segmentations[seq_index], temp_segmentation_path_list[seq_index])
                    a_segmentation = imageio.imread( temp_segmentation_path_list[seq_index] ).astype('uint8')  # convert to uint8 ~~~~~~~~~
                    segmentation_data_list.append( a_segmentation )
                    segmentation_data_list[seq_index] = np.expand_dims( segmentation_data_list[seq_index], axis=-1)

                    # --------------- Tracking points ----------------
                    if os.path.exists(temp_tracking_point_path_list[seq_index]):
                        os.remove(temp_tracking_point_path_list[seq_index])

                    tf.io.gfile.copy(tracking_points[seq_index], temp_tracking_point_path_list[seq_index])
                    a_tracking_point = np.loadtxt(temp_tracking_point_path_list[seq_index], dtype=np.int32)
                    tracking_point_data_list.append( a_tracking_point )

                    # --------------------------------
                    # check data shapes
                    if seq_index > 0:
                        assert height == img_data_list[seq_index].shape[0]
                        assert width == img_data_list[seq_index].shape[1]
                        assert height == segmentation_data_list[seq_index].shape[0]
                        assert width == segmentation_data_list[seq_index].shape[1]
                        assert total_tracking_point_num == len(tracking_point_data_list[seq_index])

                    height = img_data_list[seq_index].shape[0]
                    width = img_data_list[seq_index].shape[1]
                    total_tracking_point_num = len(tracking_point_data_list[seq_index])

                # ------------------------------
                # partition tracking points such that each partition has num_least_tracking_points
                partitioned_tracking_point_data_list = []
                num_all_tracking_points = tracking_point_data_list[0].shape[0]
                if num_all_tracking_points > num_least_tracking_points:
                    print('partition!')
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
                    print('no partition')
                    partitioned_tracking_point_data_list = [tracking_point_data_list]

                # ------------------------------
                for a_partitioned_tracking_point_data_list in partitioned_tracking_point_data_list:

                    # save data
                    feature = {
                        'height': conversion_utils.int64_feature(height),
                        'width': conversion_utils.int64_feature(width),
                    }

                    for seq_index in range(seq_len):
                        feature.update({
                            f'image{seq_index}_path': conversion_utils.bytes_feature(str.encode(images[seq_index])),
                            f'segmentation{seq_index}_path': conversion_utils.bytes_feature(str.encode(segmentations[seq_index])),
                            f'tracking_points{seq_index}_path': conversion_utils.bytes_feature(str.encode(tracking_points[seq_index])),
                        })

                    example = tf.train.SequenceExample(
                        context=tf.train.Features(feature=feature),
                        feature_lists=tf.train.FeatureLists(
                            feature_list={
                                'images':
                                    tf.train.FeatureList(feature=[conversion_utils.bytes_feature(image_data.tobytes()) for image_data in img_data_list ]),
                                'segmentations':
                                    tf.train.FeatureList(feature=[conversion_utils.bytes_feature(segmentation_data.tobytes()) for segmentation_data in segmentation_data_list ]),
                                'tracking_points':
                                    tf.train.FeatureList(feature=[conversion_utils.bytes_feature(tracking_point_data.tobytes()) for tracking_point_data in a_partitioned_tracking_point_data_list ])
                            }))

                # log saving
                if i % 10 == 0:
                    tf.compat.v1.logging.info('Writing %d out of %d total.', i, len(data_path_list))
                record_writer.write(example.SerializeToString())

        tf.compat.v1.logging.info('Saved results to %s', FLAGS.output_dir)
        # ------------------------------------------- Write Records End -------------------------------------------

    if not tf.io.gfile.exists(FLAGS.output_dir):
        tf.io.gfile.mkdir(FLAGS.output_dir)

    for data_split in ['training']: # 'training', 'valid'
        split_folder = os.path.join(FLAGS.output_dir, data_split)
        if not tf.io.gfile.exists(split_folder):
            tf.io.gfile.mkdir(split_folder)

        output_folder = os.path.join(FLAGS.output_dir, data_split)
        if not tf.io.gfile.exists(output_folder):
            tf.io.gfile.mkdir(output_folder)

        data_folder_path = os.path.join(FLAGS.data_dir, data_split)

        image_folders = sorted(tf.io.gfile.glob(
            data_folder_path + f'/*/images'))  # ['uflow/assets/mpi_sintel/training/clean/alley_1', 'uflow/assets/mpi_sintel/training/clean/alley_2', ...]
        segmentation_folders = sorted(tf.io.gfile.glob(data_folder_path + f'/*/segmentations'))
        tracking_points_folders = sorted(tf.io.gfile.glob(data_folder_path + f'/*/points'))

        assert len(image_folders) == len(tracking_points_folders)
        if FLAGS.segmentation_format is not None:
            assert len(image_folders) == len(segmentation_folders)

        data_list = []
        if data_split == 'training' or data_split == 'valid':
            for image_folder_path, segmentation_folder_path, tracking_points_folder_path in zip(image_folders, segmentation_folders, tracking_points_folders):
                # get image path
                image_path_list = tf.io.gfile.glob(image_folder_path + f"/*.{FLAGS.img_format}")
                sort_by_frame_index = lambda x: int(
                    os.path.basename(x).split('_')[-1].split('.')[0])
                image_path_list = sorted(image_path_list, key=sort_by_frame_index)

                # get segmentation path
                segmentation_path_list = tf.io.gfile.glob(segmentation_folder_path + f"/*.{FLAGS.segmentation_format}")
                segmentation_path_list = sorted(segmentation_path_list, key=sort_by_frame_index)

                # get their names
                image_name_list = []
                segmentation_name_list = []
                for a_image_path, a_segmentation_path in zip(image_path_list, segmentation_path_list):
                    image_name_list.append(get_image_name(a_image_path, FLAGS.img_format))
                    segmentation_name_list.append(get_image_name(a_segmentation_path, FLAGS.img_format))

                # get tracking points path
                tracking_points_path_list = tf.io.gfile.glob(tracking_points_folder_path + '/*.txt')
                tracking_points_path_list = sorted(tracking_points_path_list, key=sort_by_frame_index)
                total_tracking_points_path_num = len(tracking_points_path_list)

                # tracking_points_seq = zip(tracking_points_path_list[:-1], tracking_points_path_list[1:])
                # tracking_points_seq = zip(tracking_points_path_list[:1-seq_len], tracking_points_path_list[1:-3], tracking_points_path_list[2:-2], tracking_points_path_list[3:-1], tracking_points_path_list[4:0])
                tracking_points_seq = [tracking_points_path_list[seq_index:total_tracking_points_path_num+seq_index+1-seq_len] for seq_index in range(seq_len)]

                # ------ renew tracking points every 10 frames to prevent overlapping points  ----------------
                # new_tracking_points_seq = [[], []]
                # for i in range(0, len(tracking_points_seq[0]), 2):
                #     new_tracking_points_seq[0].append(tracking_points_seq[0][i])
                #     new_tracking_points_seq[1].append(tracking_points_seq[1][i])
                # -------------------------------------------------------------------------------------
                zipped_tracking_points_seq = zip(*tracking_points_seq)

                # only select images and segmentations that has the corresponding labeled tracking points
                selected_image_path_list = []
                selected_segmentation_path_list = []
                for a_tracking_point_path in tracking_points_path_list:
                    # get file names of tracking points
                    tracking_point_filename = get_image_name(a_tracking_point_path, 'txt')

                    # get images with indices
                    a_index = image_name_list.index(tracking_point_filename)
                    selected_image_path_list.append(image_path_list[a_index])

                    a_index = segmentation_name_list.index('refined_'+tracking_point_filename)  # TODO: 'refined_'+tracking_point_filename
                    selected_segmentation_path_list.append(segmentation_path_list[a_index])

                # image_seq = zip(selected_image_path_list[:-1], selected_image_path_list[1:])
                image_seq = [selected_image_path_list[seq_index:total_tracking_points_path_num+seq_index+1-seq_len] for seq_index in range(seq_len)]  # i.e) if seq_len == 2, [ [1,2,3,4], [2,3,4,5] ]
                segmentation_seq = [selected_segmentation_path_list[seq_index:total_tracking_points_path_num + seq_index + 1 - seq_len] for seq_index in range(seq_len)]  # i.e) if seq_len == 2, [ [1,2,3,4], [2,3,4,5] ]

                # ------ renew image every 10 frames to prevent overlapping points  ----------------
                # new_image_seq = [[],[]]
                # for i in range(0, len(image_seq[0]), 2):
                #     new_image_seq[0].append(image_seq[0][i])
                #     new_image_seq[1].append(image_seq[1][i])

                # i.e) if seq_len == 2, [ [5,15,25], [10,20,30] ]
                # -------------------------------------------------------------------------------------
                zipped_image_seq = zip(*image_seq)
                zipped_segmentation_seq = zip(*segmentation_seq)

                assert len(tracking_points_path_list) == len(selected_image_path_list)
                if FLAGS.segmentation_format is not None:
                    assert len(tracking_points_path_list) == len(selected_segmentation_path_list)

                data_list.extend(zip(zipped_image_seq, zipped_segmentation_seq, zipped_tracking_points_seq))

        # else:
        #     for image_folder_path, tracking_points_folder_path in zip(image_folders, tracking_points_folders):
        #         image_path_list = tf.io.gfile.glob(image_folder_path + f"/*.{FLAGS.img_format}")
        #         sort_by_frame_index = lambda x: int(
        #             os.path.basename(x).split('_')[-1].split('.')[0])
        #         image_path_list = sorted(image_path_list, key=sort_by_frame_index)
        #         image_pairs = zip(image_path_list[:-1], image_path_list[1:])
        #
        #         tracking_points_path_list = tf.io.gfile.glob(tracking_points_folder_path + '/*.txt')
        #         tracking_points_path_list = sorted(tracking_points_path_list, key=sort_by_frame_index)
        #
        #         # ----------- For missing tracking points path, populate with previous tracking points path --------------
        #         generated_tracking_points_path_list = []
        #         gen_ratio = len(image_path_list) // len(tracking_points_path_list)
        #         for a_tracking_point_path in tracking_points_path_list:
        #             for i in range(gen_ratio):
        #                 generated_tracking_points_path_list.append(a_tracking_point_path)
        #         a_diff = len(image_path_list) - len(generated_tracking_points_path_list)
        #
        #         last_path = generated_tracking_points_path_list[-1]
        #         for i in range(a_diff):
        #             generated_tracking_points_path_list.append(last_path)
        #         # -------------------------------------
        #
        #         tracking_points_pairs = zip(generated_tracking_points_path_list[:-1], generated_tracking_points_path_list[1:])
        #
        #         assert len(generated_tracking_points_path_list) == len(image_path_list)
        #         data_list.extend(zip(image_pairs, tracking_points_pairs))

        num_least_tracking_points = find_least_tracking_points(tracking_points_folders)

        write_records(data_list, output_folder, num_least_tracking_points)


# def write_records_image_only(data_list, output_folder):
#     # Reading ppm and flo can fail on network filesystem, so copy to tmpdir first.
#     tmpdir = '/tmp/convert_custom_to_tfrecords'
#     if not os.path.exists(tmpdir):
#         os.mkdir(tmpdir)
#
#     filenames = conversion_utils.generate_sharded_filenames(
#         os.path.join(output_folder, 'custom@{}'.format(FLAGS.num_shards)))
#     with tf.io.TFRecordWriter(filenames[FLAGS.shard]) as record_writer:
#         total = len(data_list)
#         images_per_shard = total // FLAGS.num_shards
#         start = images_per_shard * FLAGS.shard
#         end = start + images_per_shard
#         # Account for num images not being divisible by num shards.
#         if FLAGS.shard == FLAGS.num_shards - 1:
#             data_list = data_list[start:]
#         else:
#             data_list = data_list[start:end]
#
#         tf.compat.v1.logging.info('Writing %d images per shard', images_per_shard)
#         tf.compat.v1.logging.info('Writing range %d to %d of %d total.', start, end, total)
#
#         img1_path = os.path.join(tmpdir, f'img1.{FLAGS.img_format}')
#         img2_path = os.path.join(tmpdir, f'img2.{FLAGS.img_format}')
#
#         for i, images in enumerate(data_list):
#             if os.path.exists(img1_path):
#                 os.remove(img1_path)
#             if os.path.exists(img2_path):
#                 os.remove(img2_path)
#
#             tf.io.gfile.copy(images[0], img1_path)
#             tf.io.gfile.copy(images[1], img2_path)
#
#             image1_data = imageio.imread(img1_path)
#             image2_data = imageio.imread(img2_path)
#
#             if len(image1_data.shape) == 2:
#                 image1_data = np.expand_dims(image1_data, axis=-1)
#                 image1_data = np.repeat(image1_data, 3, axis=-1)
#                 image2_data = np.expand_dims(image2_data, axis=-1)
#                 image2_data = np.repeat(image2_data, 3, axis=-1)
#
#             # -------------------------------
#             # check data shapes
#             height = image1_data.shape[0]
#             width = image1_data.shape[1]
#
#             assert height == image2_data.shape[0]
#             assert width == image2_data.shape[1]
#             # ------------------------------
#             feature = {
#                 'height': conversion_utils.int64_feature(height),
#                 'width': conversion_utils.int64_feature(width),
#                 'image1_path': conversion_utils.bytes_feature(str.encode(images[0])),
#                 'image2_path': conversion_utils.bytes_feature(str.encode(images[1])),
#             }
#             example = tf.train.SequenceExample(
#                 context=tf.train.Features(feature=feature),
#                 feature_lists=tf.train.FeatureLists(
#                     feature_list={
#                         'images':
#                             tf.train.FeatureList(feature=[
#                                 conversion_utils.bytes_feature(image1_data.tobytes()),
#                                 conversion_utils.bytes_feature(image2_data.tobytes())
#                             ])
#                     }))
#
#             if i % 10 == 0:
#                 tf.compat.v1.logging.info('Writing %d out of %d total.', i,
#                                           len(data_list))
#             record_writer.write(example.SerializeToString())
#
#     tf.compat.v1.logging.info('Saved results to %s', FLAGS.output_dir)

#
# def convert_dataset_with_segmentation():
#
#     def write_records(data_list, output_folder):
#         """Takes in list: [((im1_path, ..., imn_path), (track1_path, ..., trackn_path), (seg1_path, ..., seqn_path))] and writes records."""
#
#         # Reading ppm and flo can fail on network filesystem, so copy to tmpdir first.
#         tmpdir = '/tmp/convert_custom_to_tfrecords'
#         if not os.path.exists(tmpdir):
#             os.mkdir(tmpdir)
#
#         filenames = conversion_utils.generate_sharded_filenames(
#             os.path.join(output_folder, 'custom@{}'.format(FLAGS.num_shards)))
#         with tf.io.TFRecordWriter(filenames[FLAGS.shard]) as record_writer:
#             total = len(data_list)
#             images_per_shard = total // FLAGS.num_shards
#             start = images_per_shard * FLAGS.shard
#             end = start + images_per_shard
#             # Account for num images not being divisible by num shards.
#             if FLAGS.shard == FLAGS.num_shards - 1:
#                 data_list = data_list[start:]
#             else:
#                 data_list = data_list[start:end]
#
#             tf.compat.v1.logging.info('Writing %d images per shard', images_per_shard)
#             tf.compat.v1.logging.info('Writing range %d to %d of %d total.', start, end, total)
#
#             img1_path = os.path.join(tmpdir, f'img1.{FLAGS.img_format}')
#             img2_path = os.path.join(tmpdir, f'img2.{FLAGS.img_format}')
#             segmentation1_path = os.path.join(tmpdir, f'segmentation1.{FLAGS.segmentation_format}')
#             segmentation2_path = os.path.join(tmpdir, f'segmentation2.{FLAGS.segmentation_format}')
#             tracking_point1_path = os.path.join(tmpdir, 'tracking_point1.txt')
#             tracking_point2_path = os.path.join(tmpdir, 'tracking_point2.txt')
#
#             for i, (images, segmentations, tracking_points) in enumerate(data_list):
#                 if os.path.exists(img1_path):
#                     os.remove(img1_path)
#                 if os.path.exists(img2_path):
#                     os.remove(img2_path)
#                 if os.path.exists(segmentation1_path):
#                     os.remove(segmentation1_path)
#                 if os.path.exists(segmentation2_path):
#                     os.remove(segmentation2_path)
#                 if os.path.exists(tracking_point1_path):
#                     os.remove(tracking_point1_path)
#                 if os.path.exists(tracking_point2_path):
#                     os.remove(tracking_point2_path)
#
#                 tf.io.gfile.copy(images[0], img1_path)
#                 tf.io.gfile.copy(images[1], img2_path)
#
#                 image1_data = imageio.imread(img1_path)
#                 image2_data = imageio.imread(img2_path)
#
#                 if len(image1_data.shape) == 2:
#                     image1_data = np.expand_dims(image1_data, axis=-1)
#                     image1_data = np.repeat(image1_data, 3, axis=-1)
#                     image2_data = np.expand_dims(image2_data, axis=-1)
#                     image2_data = np.repeat(image2_data, 3, axis=-1)
#
#                 # ---------------------------------
#                 if segmentations is not None:
#                     tf.io.gfile.copy(segmentations[0], segmentation1_path)
#                     tf.io.gfile.copy(segmentations[1], segmentation2_path)
#
#                     segmentation1_data = np.expand_dims(
#                         imageio.imread(segmentation1_path), axis=-1)
#                     segmentation2_data = np.expand_dims(
#                         imageio.imread(segmentation2_path), axis=-1)  # // 255
#                 else:
#                     raise ValueError('segmentations does not exist!!!')
#                 # else:  # Test has no segmentation data, spoof segmentation data.
#                 #   segmentation1_data = np.zeros(
#                 #       (image1_data.shape[0], image1_data.shape[1], 1), np.uint8)
#                 #   segmentation2_data = np.zeros(
#                 #       (image2_data.shape[0], image2_data.shape[1], 1), np.uint8)
#
#                 # print(np.unique(np.reshape(segmentation1_data, -1), return_counts=True))
#                 # print(np.unique(np.reshape(segmentation2_data, -1), return_counts=True))
#                 # import pdb; pdb.set_trace()
#
#                 # --------------------------------
#                 if tracking_points is not None:
#                     tf.io.gfile.copy(tracking_points[0], tracking_point1_path)
#                     tf.io.gfile.copy(tracking_points[1], tracking_point2_path)
#
#                     tracking_point1_data = np.loadtxt(tracking_point1_path, dtype=np.int32)
#                     tracking_point2_data = np.loadtxt(tracking_point2_path, dtype=np.int32)
#                 else:
#                     raise ValueError('tracking_points does not exist!!!')
#
#                 # -------------------------------
#                 # check data shapes
#                 height = image1_data.shape[0]
#                 width = image1_data.shape[1]
#
#                 assert height == segmentation1_data.shape[0] == segmentation2_data.shape[0]
#                 assert width == segmentation1_data.shape[1] == segmentation2_data.shape[1]
#                 assert height == image2_data.shape[0]
#                 assert width == image2_data.shape[1]
#                 # ------------------------------
#                 feature = {
#                     'height': conversion_utils.int64_feature(height),
#                     'width': conversion_utils.int64_feature(width),
#                     'image1_path': conversion_utils.bytes_feature(str.encode(images[0])),
#                     'image2_path': conversion_utils.bytes_feature(str.encode(images[1])),
#                 }
#                 feature.update({
#                     'segmentation1_path': conversion_utils.bytes_feature(str.encode(segmentations[0])),
#                     'segmentation2_path': conversion_utils.bytes_feature(str.encode(segmentations[1])),
#                 })
#                 feature.update({
#                     'tracking_points1_path': conversion_utils.bytes_feature(str.encode(tracking_points[0])),
#                     'tracking_points2_path': conversion_utils.bytes_feature(str.encode(tracking_points[1])),
#                 })
#                 example = tf.train.SequenceExample(
#                     context=tf.train.Features(feature=feature),
#                     feature_lists=tf.train.FeatureLists(
#                         feature_list={
#                             'images':
#                                 tf.train.FeatureList(feature=[
#                                     conversion_utils.bytes_feature(image1_data.tobytes()),
#                                     conversion_utils.bytes_feature(image2_data.tobytes())
#                                 ]),
#                             'segmentations':
#                                 tf.train.FeatureList(feature=[
#                                     conversion_utils.bytes_feature(segmentation1_data.tobytes()),
#                                     conversion_utils.bytes_feature(segmentation2_data.tobytes())
#                                 ]),
#                             'tracking_points':
#                                 tf.train.FeatureList(feature=[
#                                     conversion_utils.bytes_feature(tracking_point1_data.tobytes()),
#                                     conversion_utils.bytes_feature(tracking_point2_data.tobytes())
#                                 ])
#                         }))
#                 # import pdb; pdb.set_trace()
#                 # ---------------------------------------
#                 # Verified that parse_data() function is not the reason for the bug in loading blurry segmentation
#
#                 # proto = example.SerializeToString()
#                 #
#                 # # Parse context and image sequence from protobuffer.
#                 # context_features = {
#                 #     'height': tf.io.FixedLenFeature([], tf.int64),
#                 #     'width': tf.io.FixedLenFeature([], tf.int64),
#                 # }
#                 # sequence_features = {
#                 #     'images': tf.io.FixedLenSequenceFeature([], tf.string)
#                 # }
#                 # sequence_features['segmentations'] = tf.io.FixedLenSequenceFeature([], tf.string)
#                 #
#                 # context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
#                 #     proto,
#                 #     context_features=context_features,
#                 #     sequence_features=sequence_features,
#                 # )
#                 #
#                 # def deserialize(s, dtype, dims):
#                 #
#                 #     return tf.reshape(tf.io.decode_raw(s, dtype),
#                 #                       [context_parsed['height'], context_parsed['width'], dims])
#                 #
#                 # images = tf.map_fn(
#                 #     lambda s: deserialize(s, tf.uint8, 3),
#                 #     sequence_parsed['images'],
#                 #     dtype=tf.uint8)
#                 #
#                 # images = tf.image.convert_image_dtype(images, tf.float32)
#                 # if height is not None and width is not None:
#                 #   images = resize(images, height, width, is_flow=False)
#                 #
#                 # segmentations1 = tf.map_fn(
#                 #     lambda s: deserialize(s, tf.uint8, 1),
#                 #     sequence_parsed['segmentations'],
#                 #     dtype=tf.uint8)
#                 #
#                 # if height is not None and width is not None:
#                 #     segmentations1 = resize_uint8(segmentations1, height, width)
#                 #
#                 # print(np.unique(np.reshape(segmentations1, -1) , return_counts=True))
#                 # import pdb; pdb.set_trace()
#                 # ----------------------------------------
#
#                 if i % 10 == 0:
#                     tf.compat.v1.logging.info('Writing %d out of %d total.', i,
#                                               len(data_list))
#                 record_writer.write(example.SerializeToString())
#
#         tf.compat.v1.logging.info('Saved results to %s', FLAGS.output_dir)
#
#     """Convert the data to the TFRecord format."""
#     if not tf.io.gfile.exists(FLAGS.output_dir):
#         tf.io.gfile.mkdir(FLAGS.output_dir)
#
#     for data_split in ['training']: # 'test', 'training'
#         split_folder = os.path.join(FLAGS.output_dir, data_split)
#         if not tf.io.gfile.exists(split_folder):
#             tf.io.gfile.mkdir(split_folder)
#
#         output_folder = os.path.join(FLAGS.output_dir, data_split)
#         if not tf.io.gfile.exists(output_folder):
#             tf.io.gfile.mkdir(output_folder)
#
#         image_folder_path = os.path.join(FLAGS.data_dir, data_split)
#         segmentation_folder_path = os.path.join(FLAGS.data_dir, data_split)
#         tracking_points_folder_path = os.path.join(FLAGS.data_dir, data_split)
#
#         # Directory with images.
#         image_folders = sorted(tf.io.gfile.glob(image_folder_path + f'/*/img'))  # ['uflow/assets/mpi_sintel/training/clean/alley_1', 'uflow/assets/mpi_sintel/training/clean/alley_2', ...]
#         # if data_split == 'training':
#         segmentation_folders = sorted(tf.io.gfile.glob(segmentation_folder_path + f'/*/segmentation'))
#         tracking_points_folders = sorted(tf.io.gfile.glob(tracking_points_folder_path + f'/*/tracking_points'))
#
#         assert len(image_folders) == len(segmentation_folders)
#         assert len(image_folders) == len(tracking_points_folders)
#         # else:  # Test has no ground truth flow.
#         #   segmentation_folders = [
#         #       None for _ in image_folders
#         #   ]
#         data_list = []
#         for image_folder_path, segmentation_folder_path, tracking_points_folder_path in zip(image_folders, segmentation_folders, tracking_points_folders):
#             images = tf.io.gfile.glob(image_folder_path + f"/*.{FLAGS.img_format}")
#             # We may want to eventually look at sequences of frames longer than 2.
#             # pylint:disable=g-long-lambda
#             sort_by_frame_index = lambda x: int(
#                 os.path.basename(x).split('_')[-1].split('.')[0])
#             images = sorted(images, key=sort_by_frame_index)
#
#             # if data_split == 'training':
#             segmentations = tf.io.gfile.glob(segmentation_folder_path + f"/*.{FLAGS.segmentation_format}")
#             segmentations = sorted(segmentations, key=sort_by_frame_index)
#
#             tracking_points = tf.io.gfile.glob(tracking_points_folder_path + '/*.txt')
#             tracking_points = sorted(tracking_points, key=sort_by_frame_index)
#
#             # else:  # Test has no ground truth segmentation
#             #   segmentations = [None for _ in images]
#
#             assert len(segmentations) == len(images)
#             assert len(tracking_points) == len(images)
#
#             image_pairs = zip(images[:-1], images[1:])
#             segmentation_pairs = zip(segmentations[:-1], segmentations[1:])
#             tracking_points_pairs = zip(tracking_points[:-1], tracking_points[1:])
#
#             data_list.extend(zip(image_pairs, segmentation_pairs, tracking_points_pairs))
#
#         write_records(data_list, output_folder)


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
    seq_len = 2
    FLAGS.output_dir = FLAGS.data_dir + "tfrecord/"

    convert_dataset(seq_len)
    # convert_dataset_with_segmentation()

    print('@@@@@@@@@ seq_len', seq_len, ' @@@@@@@@@')


if __name__ == '__main__':
  # debug_file()
  app.run(main)

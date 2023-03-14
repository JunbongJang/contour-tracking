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

"""Tests that we can export and load a frozen graph."""

import os
import tempfile

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from uflow.export import freeze_model
from uflow.contour_flow_net import ContourFlow


tf.compat.v1.disable_eager_execution()

HEIGHT = 480
WIDTH = 384


def _export_frozen_graph():
  # save weights
  tempdir = tempfile.mkdtemp()
  sess = tf.compat.v1.Session()
  with sess.as_default():
    uflow = ContourFlow(checkpoint_dir=tempdir)
    image1 = tf.compat.v1.placeholder(
        tf.float32, [HEIGHT, WIDTH, 3], name='first_image')
    image2 = tf.compat.v1.placeholder(
        tf.float32, [HEIGHT, WIDTH, 3], name='second_image')
    segmentation1 = tf.compat.v1.placeholder(
        tf.uint8, [HEIGHT, WIDTH, 1], name='first_segmentation')
    segmentation2 = tf.compat.v1.placeholder(
        tf.uint8, [HEIGHT, WIDTH, 1], name='second_segmentation')
    flow = tf.identity(uflow.infer_no_tf_function(image1, image2, segmentation1, segmentation2), 'flow')

    sess.run(tf.compat.v1.global_variables_initializer())

    image1_np = np.random.randn(HEIGHT, WIDTH, 3)
    image2_np = np.random.randn(HEIGHT, WIDTH, 3)
    image3_np = np.random.randn(HEIGHT, WIDTH, 3)
    segmentation1_np = np.random.randn(HEIGHT, WIDTH, 1)
    segmentation2_np = np.random.randn(HEIGHT, WIDTH, 1)

    flow_output = sess.run(
        flow, feed_dict={
            image1: image1_np,
            image2: image2_np,
            # segmentation1: segmentation1_np,
            # segmentation2: segmentation2_np
        })

    uflow.save()
  # freeze model
  freeze_model.convert_graph(tempdir, HEIGHT, WIDTH, output_row_col=True)
  return tempdir, image1_np, image2_np, segmentation1_np, segmentation2_np, flow_output


def _load_graph(frozen_graph_filename, image1, image2, segmentation1, segmentation2):
  with tf.io.gfile.GFile(frozen_graph_filename, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
  with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def)

  flow = graph.get_tensor_by_name('import/flow:0')
  first_image = graph.get_tensor_by_name('import/first_image:0')
  second_image = graph.get_tensor_by_name('import/second_image:0')
  # first_segmentation = graph.get_tensor_by_name('import/first_segmentation:0')
  # second_segmentation = graph.get_tensor_by_name('import/second_segmentation:0')

  sess = tf.compat.v1.Session(graph=graph)
  return sess.run(flow, feed_dict={first_image: image1, second_image: image2})  # , first_segmentation: segmentation1, second_segmentation: segmentation2


class FreezeModelTest(absltest.TestCase):

  def test_export_and_load(self):

    checkpoint_dir, image1, image2, segmentation1, segmentation2, prefrozen_output = _export_frozen_graph()

    frozen_output = _load_graph(
        os.path.join(checkpoint_dir, 'uflow.pb'), image1, image2, segmentation1, segmentation2)
    diff = np.sum(np.abs(prefrozen_output - frozen_output))

    # Set tolearnace of 10 because if prefrozen_output and frozen_output are actually different, diff is very big
    diff = diff // 10

    self.assertEqual(diff, 0.)


if __name__ == '__main__':
  absltest.main()

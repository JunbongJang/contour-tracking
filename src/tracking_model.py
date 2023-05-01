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
1. encode image1 and image2 to get 2D features
2. sample features at seg_point1 and seg_point2
3. cross attention between two feature vectors
4. output normalized correspondence id
'''

import numpy as np
import tensorflow as tf
import cv2
from einops import rearrange

from src.tracking_utils import generate_pos_emb, get_adj_ind, abs2rel, bilinear_sampler_1d, cnt2poly, cost_volume_at_contour_points, normalize_1d_features


class PointSetTracker(tf.Module):
    def __init__(self, num_iter=5):
        super(PointSetTracker, self).__init__()
        self.num_iter = num_iter
        self.local_alignment = LocalAlignment(num_iter=num_iter)

    def initialize(self, cnt):
        B, N = cnt.shape[:2]

        pos_emb = generate_pos_emb(N)
        pos_emb = tf.expand_dims(pos_emb, axis=0)
        pos_emb = tf.broadcast_to(pos_emb, [B, N, 2])  # copy the same pos_emb and paste to all batch indices

        return pos_emb

    def propagate(self, x0, x1, orig_seg_point1, orig_seg_point2, pos_emb, source_sampled_contour_index, target_sampled_contour_index):
        '''

        :param x0: First Image
        :param x1: Second Image
        :param pos_emb: positional embedding from the number of contour points
        :return:
        '''

        size = x0.shape[-2:-4:-1]  # Width, Height
        seg_point1 = abs2rel(orig_seg_point1, size)
        seg_point2 = abs2rel(orig_seg_point2, size)

        seg_point_mask1 = orig_seg_point1[:, :, 2:]
        seg_point_mask2 = orig_seg_point2[:, :, 2:]

        occ_out = self.local_alignment(x0, x1, seg_point1, seg_point2, pos_emb, seg_point_mask1, seg_point_mask2, source_sampled_contour_index, target_sampled_contour_index)

        return occ_out

    def __call__(self, *args, **kwargs):
        kwargs.pop('training', None)  # because kwargs has training Keyword
        in_len = len(args) + len(kwargs)
        if in_len == 1:
            return self.initialize(*args, **kwargs)
        elif in_len == 7:
            return self.propagate(*args, **kwargs)
        else:
            raise ValueError(f'input length of {in_len} is not supported')


class LocalAlignment(tf.keras.Model):
    def __init__(self, num_iter=5, adj_num=4):
        super(LocalAlignment, self).__init__()
        self.num_iter = num_iter
        self.adj_num = adj_num

        # ------------------------ Post Implementation ----------------------------
        # self.encoder = PostEncoder(out_dim=128)
        # self.align_module = LAM(feature_dim=128 + 4, state_dim=128, output_num=2)
        # -------------------------------------------------------------------------
        self.encoder = LocalEncoder(out_dim=128)
        
        # self.spatial_offset_module = LAM(feature_dim=128 + 4, state_dim=128, output_num=2)

        # Circular convoltuion
        # self.spatial_offset_module = tf.keras.Sequential([
        #     CircConv(256),
        #     tf.keras.layers.ReLU(),
        #     CircConv(64),
        #     tf.keras.layers.ReLU(),
        #     CircConv(2)
        # ])

        # 1D convolution
        # self.spatial_offset_module = tf.keras.Sequential([
        #     LinearConv(256),
        #     tf.keras.layers.ReLU(),
        #     LinearConv(64),
        #     tf.keras.layers.ReLU(),
        #     LinearConv(2)
        # ])

        # Dense layers
        # self.spatial_offset_module = tf.keras.Sequential([
        #     tf.keras.layers.Dense(256),
        #     tf.keras.layers.ReLU(),
        #     tf.keras.layers.Dense(64),
        #     tf.keras.layers.ReLU(),
        #     tf.keras.layers.Dense(2)
        # ])

        # tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=64, value_dim=64)
        self.cross_attn_layer_forward = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=128, value_dim=128)
        self.cross_attn_layer_backward = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=128, value_dim=128)

        self.pos_embed = tf.keras.layers.Dense(128)
        self.occupancy_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(256),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(64),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(1)
        ])

    def call(self, x0, x1, seg_point1, seg_point2, pos_emb, seg_point_mask1, seg_point_mask2, source_sampled_contour_index, target_sampled_contour_index):
        '''

        :param x0: original first feature (B, H, W, C)
        :param x1: orignal second feature (B, H, W, C)
        :param cnt0: contour of the first image (B, N, C) Points are in Width and Height order, unlike the image
        :param pos_emb: (B, N, 2)

        :param seg_point1_limit and seg_point2_limit
        During inference, they are used to remove all negative values that are padded in the data loading procedure to obtain consistent shape
        During training, they remove most of the negative values that are padded

        x1_h : transformed first image to second image
        cnt1_h: transformed first contour to second contour

        :return:
        '''

        # ----------------------------------------------
        # single level, PoST
        # features = self.encoder(x0, x1)
        # a_feature = features[-1]
        
        # forward_sampled_cost_volume = cost_volume_at_contour_points(a_feature, a_feature, seg_point1, seg_point2, max_displacement=1)
        # backward_sampled_cost_volume = cost_volume_at_contour_points(a_feature, a_feature, seg_point2, seg_point1, max_displacement=1)
        
        # poly1 = cnt2poly(seg_point1)  # It represents the contour points' coordinates on +image space. shape is B, N, 2
        # m_in1 = tf.concat((forward_sampled_cost_volume, pos_emb, poly1), axis=-1)  # m_in shape is (B, N, C+4)
        # poly2 = cnt2poly(seg_point2)  # It represents the contour points' coordinates on +image space. shape is B, N, 2
        # m_in2 = tf.concat((backward_sampled_cost_volume, pos_emb, poly2), axis=-1)  # m_in shape is (B, N, C+4)
        
        # N = seg_point1.shape[1]
        # adj = get_adj_ind(self.adj_num, N)
        # forward_spatial_offset = self.align_module(m_in1) # adj
        # backward_spatial_offset = self.align_module(m_in2)
        
        # saved_offset[0] = forward_spatial_offset

        # ----------------------------------------------
        # single level, OURS

        features1 = self.encoder(x0)  # [8, 256, 256, 128]
        features2 = self.encoder(x1) 
        a_feature1 = features1[-1]  
        a_feature2 = features2[-1]

        # TODO: check if bilinear_sampler_1d is necessary since it's sampling at exact 2d coordinates, not floating point
        sampled_feature1 = bilinear_sampler_1d(a_feature1, seg_point1[:, :, 0], seg_point1[:, :, 1])  # B x N x c
        normalized_sampled_feature1 = normalize_1d_features(sampled_feature1)
        poly1 = cnt2poly(seg_point1)  # It represents the contour points' coordinates on +image space. shape is B, N, 2
        concat_sampled_features1 = tf.concat((normalized_sampled_feature1, pos_emb, poly1),
                                             axis=-1)  # m_in shape is (B, N, C+4)

        sampled_feature2 = bilinear_sampler_1d(a_feature2, seg_point2[:, :, 0], seg_point2[:, :, 1])  # B x N x c
        normalized_sampled_feature2 = normalize_1d_features(sampled_feature2)
        poly2 = cnt2poly(seg_point2)  # It represents the contour points' coordinates on +image space. shape is B, N, 2
        concat_sampled_features2 = tf.concat((normalized_sampled_feature2, pos_emb, poly2),
                                             axis=-1)  # m_in shape is (B, N, C+4)

        # cross_attn_tensor shape is ([batch_size, target's num_points, 132])
        # cross_attn_scores shape is ([batch_size, num_heads, target's num_points, source's num_points])
        # implement forward and backward attention
        # t_seg_point_mask1 = tf.transpose(seg_point_mask1, perm=[0,2,1])  # For each target index i, [0,i,:] gives 40 indices with 1 mask
        # t_seg_point_mask2 = tf.transpose(seg_point_mask2, perm=[0,2,1])

        # forward and backward cross attentions
        forward_cross_attn, forward_cross_attn_scores = self.cross_attn_layer_forward(concat_sampled_features1, concat_sampled_features2, return_attention_scores=True)  # target (query), source(key)
        backward_cross_attn, backward_cross_attn_scores = self.cross_attn_layer_backward(concat_sampled_features2, concat_sampled_features1, return_attention_scores=True)  # target, source
        
        # single cross attention
        # forward_cross_attn, forward_cross_attn_scores = self.cross_attn_layer_forward(concat_sampled_features1, concat_sampled_features2, return_attention_scores=True)  # target (query), source(key)
        # backward_cross_attn, backward_cross_attn_scores = self.cross_attn_layer_forward(concat_sampled_features2, concat_sampled_features1, return_attention_scores=True)  # target (query), source(key)
        
        # no cross attention is used
        # forward_cross_attn = concat_sampled_features1 + concat_sampled_features2
        # backward_cross_attn = forward_cross_attn

        # forward_spatial_offset = self.spatial_offset_module(forward_cross_attn)
        # backward_spatial_offset = self.spatial_offset_module(backward_cross_attn)  # shape=(1, 1150, 2), dtype=float32
        # saved_offset[0] = forward_spatial_offset

        # return forward_spatial_offset, backward_spatial_offset, saved_offset
    
        # ----------------------
        # Classify id of each tracking point
        # ----------------------
        # compute 2d Correlation matrix [B, num_seg_points, num_seg_points] <-- [B, num_seg_points, 132] * [B, num_seg_points, 132]
        corr_2d_matrix = tf.einsum('bic,bjc->bij', forward_cross_attn, backward_cross_attn)
        occ_out = self.MLP_score(corr_2d_matrix, source_sampled_contour_index, target_sampled_contour_index)

        # make values along last dimension sum to one
        # softmax_corr_2d_matrix = tf.nn.softmax(corr_2d_matrix, axis=-1) # shape=(B, 1640, 1640), dtype=float32
        # saved_offset[0] = softmax_corr_2d_matrix[0,50,:]

        return occ_out


    def MLP_score(self, cost, source_sampled_contour_index, target_sampled_contour_index):
        """
        Arguments:
            cost: B H_s H_t
            src_coord: B N C
            trg_coord: B N C
        Returns:
            ... D
        """
        
        X, Y = tf.meshgrid(source_sampled_contour_index, target_sampled_contour_index)
        X = tf.reshape(X, [-1])
        Y = tf.reshape(Y, [-1])

        X = tf.repeat( tf.expand_dims(X, axis=0), repeats=8, axis=0)  # (8, 10000)
        Y = tf.repeat( tf.expand_dims(Y, axis=0), repeats=8, axis=0)  # (8, 10000)

        sampled_cost = bilinear_sampler_1d(tf.expand_dims(cost, axis=-1), X, Y, abs_scale=True)
        concat_sampled_cost = tf.stack([tf.squeeze(sampled_cost), X, Y], axis=-1)

        occ = self.occupancy_mlp(concat_sampled_cost)
        occ = tf.reshape(occ, [occ.shape[0], 100, 100, 1])
        
        return occ


class LAM(tf.keras.layers.Layer):
    def __init__(self, feature_dim, state_dim, output_num):
        super(LAM, self).__init__()
        self.head = BasicBlock(feature_dim, state_dim)

        self.dilation_list = [1, 1, 1]  # [1,1,1] # [1, 1, 1, 2, 2, 4, 4]
        self.block_list = []
        for i in range(len(self.dilation_list)):
            a_basic_block = BasicBlock(state_dim, state_dim, dilation=self.dilation_list[i])
            self.block_list.append(a_basic_block)

        fusion_state_dim = 256
        self.fusion = tf.keras.layers.Conv1D(filters=fusion_state_dim, kernel_size=1)

        # conv1d with kernel=1 is equivalent to linear layer
        self.prediction = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=256, kernel_size=1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv1D(filters=64, kernel_size=1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv1D(filters=output_num, kernel_size=1)
        ])

    def call(self, input_x):
        '''
        Architecture Overview:
            Circular Block: 8 circular conv, bn, relu
            Fusion: 1x1 conv, max pooling
            LSTM
            Prediction: 1x1 conv relu, 1x1 conv relu, 1x1 conv

        :param x: interpolated image feature at points + position embeddings of points (N, # of points, C)
        :param adj:
        :return:
        '''
        states = []
        x = self.head(input_x)  # B, N, C

        states.append(x)
        for i in range(len(self.dilation_list)):
            # x = self.__getattribute__('res' + str(i))(x, adj) + x
            x = self.block_list[i](x) + x
            states.append(x)
        state = tf.concat(states, axis=-1)

        global_state = tf.math.reduce_max(self.fusion(state), axis=1, keepdims=True)
        global_state = tf.broadcast_to(global_state, [global_state.shape[0], tf.shape(state)[1], global_state.shape[2]])  # B, N, C
        state = tf.concat([global_state, state], axis=-1)  # state output: (1, 128, 1024), global_state: (1, 128, 256)

        x = self.prediction(state)  # B, N, 1

        return x

class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, state_dim, out_state_dim, n_adj=4, dilation=1):
        super(BasicBlock, self).__init__()
        self.dense_layer = tf.keras.layers.Dense(out_state_dim)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=state_dim)  # key_dim defines the hiddne_dim/output_dim of dense layer
        self.circ_conv = CircConv(state_dim, out_state_dim, n_adj, dilation)
        self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        x = self.mha(query=x, value=x, key=x)
        x = self.circ_conv(x)
        x = self.relu(x)

        return x


class LinearConv(tf.keras.layers.Layer):
    '''
    Refered to https://www.tensorflow.org/tutorials/customization/custom_layers
    https://www.tensorflow.org/guide/keras/custom_layers_and_models
    https://www.tensorflow.org/guide/intro_to_modules

    convert first channel order in Pytorch to last channel order in tensorflow
    '''

    def __init__(self, state_dim, out_state_dim=None, n_adj=4, dilation=1):
        super(LinearConv, self).__init__()

        self.n_adj = n_adj
        self.dilation = dilation
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = tf.keras.layers.Conv1D(filters=out_state_dim, strides=1, kernel_size=self.n_adj * 2 + 1, dilation_rate=self.dilation)

    def call(self, input):
        # adj is not used
        if self.n_adj != 0:
            pad_zeros = tf.zeros_like(input[:, -self.n_adj * self.dilation:, :])
            pad_zeros_right = tf.zeros_like(input[:, :self.n_adj * self.dilation, :])

            input = tf.concat([pad_zeros, input, pad_zeros_right], axis=1)

        return self.fc(input)



class CircConv(tf.keras.layers.Layer):
    '''
    Refered to https://www.tensorflow.org/tutorials/customization/custom_layers
    https://www.tensorflow.org/guide/keras/custom_layers_and_models
    https://www.tensorflow.org/guide/intro_to_modules

    convert first channel order in Pytorch to last channel order in tensorflow
    '''

    def __init__(self, state_dim, out_state_dim=None, n_adj=4, dilation=1):
        super(CircConv, self).__init__()

        self.n_adj = n_adj
        self.dilation = dilation
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = tf.keras.layers.Conv1D(filters=out_state_dim, strides=1, kernel_size=self.n_adj * 2 + 1, dilation_rate=self.dilation)

    def call(self, input):
        # adj is not used
        if self.n_adj != 0:
            input = tf.concat([input[:, -self.n_adj * self.dilation:, :], input, input[:, :self.n_adj * self.dilation, :]], axis=1)

        return self.fc(input)


class VGG16(tf.keras.layers.Layer):
    def __init__(self):
        super(VGG16, self).__init__()
        a_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=[None, None, 3], pooling=None)

        self.conv1_pad = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv1_pad')

        init_kernel = a_model.get_layer('block1_conv1').weights[0]
        init_bias = a_model.get_layer('block1_conv1').bias
        kernel_initializer = tf.keras.initializers.constant(init_kernel)
        bias_initializer = tf.keras.initializers.constant(init_bias)

        self.conv_layer_x0 = tf.keras.layers.Conv2D(64, 3,
                                                    input_shape=(None, None, 3),
                                                    kernel_initializer=kernel_initializer,
                                                    bias_initializer=bias_initializer)

        self.conv_layer_x1 = tf.keras.layers.Conv2D(64, 3,
                                                    input_shape=(None, None, 3),
                                                    kernel_initializer=kernel_initializer,
                                                    bias_initializer=bias_initializer)
        self.conv_layer_m1_h = tf.keras.layers.Conv2D(64, 3, input_shape=(None, None, 1))

        # Methods to create new models by sequential model or function API from keras.applications
        # refer to https://stackoverflow.com/questions/67176547/how-to-remove-first-n-layers-from-a-keras-model
        # https://stackoverflow.com/questions/49546922/keras-replacing-input-layer

        # remove two intermediate layers, create a new input layer, and set 5 outputs
        model_config = a_model.get_config()

        del model_config['layers'][1]
        model_config['layers'][0] = {
            'name': 'new_input',
            'class_name': 'InputLayer',
            'config': {
                'batch_input_shape': a_model.layers[2].input_shape,
                'dtype': 'float32',
                'sparse': False,
                'name': 'new_input'
            },
            'inbound_nodes': []
        }
        model_config['layers'][1]['inbound_nodes'] = [[['new_input', 0, 0, {}]]]
        model_config['input_layers'] = [['new_input', 0, 0]]
        model_config['output_layers'] = [['block1_conv2', 0, 0],
                                         ['block2_conv2', 0, 0],
                                         ['block3_conv3', 0, 0],
                                         ['block4_conv3', 0, 0],
                                         ['block5_conv3', 0, 0]]

        self.modified_model = a_model.__class__.from_config(model_config, custom_objects={})

        weights = [layer.get_weights() for layer in a_model.layers[2:]]
        for layer, weight in zip(self.modified_model.layers[1:], weights):
            layer.set_weights(weight)

    def call(self, x0):
        # Combine features from the first image, second image, and transformed mask
        x0 = self.conv1_pad(x0)
        conv_x0 = self.conv_layer_x0(x0)

        c1, c2, c3, c4, c5 = self.modified_model(conv_x0)

        # the order of features are from coarse to fine !!!
        return c5, c4, c3, c2, c1


class ResNet50(tf.keras.layers.Layer):
    def __init__(self):
        super(ResNet50, self).__init__()
        resnet = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=[None, None, 3], pooling=None)

        self.conv1_pad = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv1_pad')
        self.conv_layer_x0 = tf.keras.layers.Conv2D(64, 3,
                                                    activation='relu',
                                                    input_shape=(None, None, 3))
        self.conv_layer_x1 = tf.keras.layers.Conv2D(64, 3,
                                                    activation='relu',
                                                    input_shape=(None, None, 3))

        # Methods to create new models by sequential model or function API from keras.applications Model are hard to use because ResNet has layers with multiple input or outputs
        # refer to https://stackoverflow.com/questions/67176547/how-to-remove-first-n-layers-from-a-keras-model
        # https://stackoverflow.com/questions/49546922/keras-replacing-input-layer
        # remove two intermidate layers, create a new input layer, and set 5 outputs
        model_config = resnet.get_config()
        del model_config['layers'][1:3]
        model_config['layers'][0] = {
            'name': 'new_input',
            'class_name': 'InputLayer',
            'config': {
                'batch_input_shape': resnet.layers[3].input_shape,
                'dtype': 'float32',
                'sparse': False,
                'name': 'new_input'
            },
            'inbound_nodes': []
        }
        model_config['layers'][1]['inbound_nodes'] = [[['new_input', 0, 0, {}]]]
        model_config['input_layers'] = [['new_input', 0, 0]]
        model_config['output_layers'] = [['conv1_relu', 0, 0],
                                         ['conv2_block3_out', 0, 0],
                                         ['conv3_block4_out', 0, 0],
                                         ['conv4_block6_out', 0, 0],
                                         ['conv5_block3_out', 0, 0]]
        self.modified_resnet = resnet.__class__.from_config(model_config, custom_objects={})
        weights = [layer.get_weights() for layer in resnet.layers[3:]]
        for layer, weight in zip(self.modified_resnet.layers[1:], weights):
            layer.set_weights(weight)



    def call(self, x0, x1):
        # Combine extracted features from the first image and the second image
        x0 = self.conv1_pad(x0)
        x1 = self.conv1_pad(x1)
        conv_x0 = self.conv_layer_x0(x0)
        conv_x1 = self.conv_layer_x1(x1)

        x = (conv_x0 + conv_x1) / 2

        c1, c2, c3, c4, c5 = self.modified_resnet(x)

        return c5, c4, c3, c2, c1


class FeaturePyramid(tf.keras.layers.Layer):
    """
    Referenced https://keras.io/examples/vision/retinanet/
    Builds the Feature Pyramid with the feature maps from the backbone.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
        Currently supports ResNet50 only.
    """

    def __init__(self, out_dim, **kwargs):
        super(FeaturePyramid, self).__init__(name="FeaturePyramid", **kwargs)
        self.conv_c1_1x1 = tf.keras.layers.Conv2D(out_dim, 1, 1, "same")
        self.conv_c2_1x1 = tf.keras.layers.Conv2D(out_dim, 1, 1, "same")
        self.conv_c3_1x1 = tf.keras.layers.Conv2D(out_dim, 1, 1, "same")
        self.conv_c4_1x1 = tf.keras.layers.Conv2D(out_dim, 1, 1, "same")
        self.conv_c5_1x1 = tf.keras.layers.Conv2D(out_dim, 1, 1, "same")

        self.conv_c1_3x3 = tf.keras.layers.Conv2D(out_dim, 3, 1, "same")
        self.conv_c2_3x3 = tf.keras.layers.Conv2D(out_dim, 3, 1, "same")
        self.conv_c3_3x3 = tf.keras.layers.Conv2D(out_dim, 3, 1, "same")
        self.conv_c4_3x3 = tf.keras.layers.Conv2D(out_dim, 3, 1, "same")
        self.conv_c5_3x3 = tf.keras.layers.Conv2D(out_dim, 3, 1, "same")
        self.upsample_2x = tf.keras.layers.UpSampling2D(2)

    def call(self, c5_output, c4_output, c3_output, c2_output, c1_output):
        # are these 1x1 convolutions helpful, except for reducing the computation?
        # What about keeping the channel # the same?
        p1_output = self.conv_c1_1x1(c1_output)
        p2_output = self.conv_c2_1x1(c2_output)
        p3_output = self.conv_c3_1x1(c3_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p5_output = self.conv_c5_1x1(c5_output)

        p4_output = p4_output + self.upsample_2x(p5_output)
        p3_output = p3_output + self.upsample_2x(p4_output)
        p2_output = p2_output + self.upsample_2x(p3_output)
        p1_output = p1_output + self.upsample_2x(p2_output)

        p1_output = self.conv_c1_3x3(p1_output)
        p2_output = self.conv_c2_3x3(p2_output)
        p3_output = self.conv_c3_3x3(p3_output)
        p4_output = self.conv_c4_3x3(p4_output)
        p5_output = self.conv_c5_3x3(p5_output)

        # the order of features are from coarse to fine
        return p5_output, p4_output, p3_output, p2_output, p1_output


class PostEncoder(tf.keras.layers.Layer):
    '''
    outdim corresponds to the number of tracking points
    '''

    def __init__(self, out_dim):
        super(PostEncoder, self).__init__()
        self.encoder = ResNet50()  # VGG16() # VGG19_dropout('uflow/assets/model_weights/vgg19_dropout.hdf5') # VGG19_dropout('')
        self.fpn = FeaturePyramid(out_dim)  # FPN([2048, 1024, 512, 256, 64], out_dim)

    def call(self, x0, x1):
        # x0: original first image
        # x1: original second image
        # p: transformed first image to second image
        # m: Mask from (transformed first contour to second contour)
        rs = self.encoder(x0, x1)
        outs = self.fpn(*rs)

        return outs


class LocalEncoder(tf.keras.layers.Layer):
    '''
    outdim corresponds to the number of tracking points
    '''

    def __init__(self, out_dim):
        super(LocalEncoder, self).__init__()
        self.encoder = VGG16() # VGG19_dropout('uflow/assets/model_weights/vgg19_dropout.hdf5') # VGG19_dropout('')
        self.fpn = FeaturePyramid(out_dim)  # FPN([2048, 1024, 512, 256, 64], out_dim)

    def call(self, x0):
        rs = self.encoder(x0)
        outs = self.fpn(*rs)

        return outs

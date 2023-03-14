'''
Author: Junbong Jang
Date: 4/12/2022

Converted pytorch code from PoST models/networks.py to tensorflow code
'''
import numpy as np
import tensorflow as tf
import cv2

from uflow.tracking_utils import generate_pos_emb, get_adj_ind, rel2abs, abs2rel, bilinear_sampler_1d, transform, cnt2poly, cost_volume_at_contour_points
from uflow.contour_flow_model import normalize_features


class PointSetTracker(tf.Module):
    def __init__(self, num_iter=5, input_image_size=(None, None, 3)):
        super(PointSetTracker, self).__init__()
        self.num_iter = num_iter
        # self.global_alignment = GlobalAlignment()
        # For training, input_image_size is necessary to set weights of the global alignment dense layer
        # if input_image_size[0] is not None and input_image_size[1] is not None:
        #     self.global_alignment(tf.zeros(shape=[1, *input_image_size]), tf.zeros(shape=[1, *input_image_size]), tf.zeros(shape=[1, 10, 2]))
        #     self.global_alignment.affine.fc.set_weights([np.zeros([512,6]), np.array([1, 0, 0, 0, 1, 0]) ])

        self.local_alignment = LocalAlignment(num_iter=num_iter)

    def initialize(self, cnt):
        B, N = cnt.shape[:2]
        rnn_features = [None for _ in range(self.num_iter)]

        pos_emb = generate_pos_emb(N)
        # pos_emb = pos_emb.unsqueeze(0).expand(B, 2, N)
        pos_emb = tf.expand_dims(pos_emb, axis=0)
        pos_emb = tf.broadcast_to(pos_emb, [B, N, 2])  # copy the same pos_emb and paste to all batch indices

        return rnn_features, pos_emb

    def propagate(self, x0, x1, orig_cnt0, pos_emb, prev_rnn_features):
        '''

        :param x0: First Image
        :param x1: Second Image
        :param cnt0: previous Contour
        :param pos_emb: positional embedding from the number of contour points
        :param prev_rnn_features: hidden state of RNN layer
        :return:
        '''

        size = x0.shape[-2:-4:-1]  # Width, Height
        cnt0 = abs2rel(orig_cnt0, size)
        # theta = self.global_alignment(x0, x1, cnt0)
        # tf.print(theta)

        # for debugging spatial transformer
        # a_angle = 30.0
        # theta = tf.convert_to_tensor( [[[tf.math.cos(a_angle),tf.math.sin(a_angle),0], [-tf.math.sin(a_angle),tf.math.cos(a_angle),0]]], dtype='float32' )

        # x1_h, cnt1_h = transform(x0, cnt0, theta)  # global transformed image and contour
        # ------------------ Visualize for debugging -----------------------
        # orig_img_data = (x0[0].numpy() * 255).astype(np.uint8)
        # orig_img_data = cv2.cvtColor(orig_img_data, cv2.COLOR_BGR2RGB)
        # cv2.imwrite('orig_img.png', orig_img_data)
        # cv2_imwrite_data = (x1_h[0].numpy() * 255).astype(np.uint8)
        # cv2_imwrite_data = cv2.cvtColor(cv2_imwrite_data, cv2.COLOR_BGR2RGB)
        # cv2.imwrite('global_transformed_img.png', cv2_imwrite_data)
        #
        # # draw all points on the image
        # import matplotlib.pyplot as plt
        # import pylab
        # plt.imshow(x0[0])
        # cm = pylab.get_cmap('gist_rainbow')
        #
        # for a_index, a_point in enumerate(orig_cnt0[0]):
        #     x, y = a_point
        #     plt.scatter(x=x, y=y, c=np.array([cm(1. * a_index / orig_cnt0.shape[1])]), s=10)
        #
        # plt.axis('off')
        # plt.savefig(f"debug_orig_points_on_image.png", bbox_inches="tight", pad_inches=0)
        # plt.close()
        #
        # for a_index, a_point in enumerate(cnt0[0]):
        #     x, y = a_point
        #     plt.scatter(x=x, y=y, c=np.array([cm(1. * a_index / cnt0.shape[1])]), s=10)
        #
        # plt.axis('off')
        # plt.savefig(f"debug_resized_points_on_image.png", bbox_inches="tight", pad_inches=0)
        # plt.close()

        # import pdb;pdb.set_trace()

        # ---------------------------------------------------------
        # cnt1, rnn_features, offset = self.local_alignment(x1_h, x1, cnt1_h, pos_emb, prev_rnn_features)
        cnt1, rnn_features, saved_offset = self.local_alignment(x0, x1, cnt0, pos_emb, prev_rnn_features)
        cnt1 = rel2abs(cnt1, size)
        # for cur_level, pred_points in saved_offset.items():
        #     abs_offset = rel2abs(saved_offset[cur_level][:,:,:2], size)
        #     uncertainty = tf.expand_dims(saved_offset[cur_level][:, :, -1], axis=-1)
        #     saved_offset[cur_level] = tf.concat([abs_offset, uncertainty], axis=-1)

        return cnt1, rnn_features, saved_offset #, x1_h, rel2abs(cnt1_h, size)

    def __call__(self, *args, **kwargs):
        kwargs.pop('training', None)  # because kwargs has training Keyword
        in_len = len(args) + len(kwargs)
        if in_len == 1 or in_len == 2:
            return self.initialize(*args, **kwargs)
        elif in_len == 5:
            return self.propagate(*args, **kwargs)
        else:
            raise ValueError(f'input length of {in_len} is not supported')


class GlobalAlignment(tf.keras.Model):
    def __init__(self):
        super(GlobalAlignment, self).__init__()
        self.encoder = GlobalEncoder(out_dim=256)
        self.masking = Masking(dim=256)
        self.affine = Affine(dim=512)

        # init_He(self)

    def call(self, x0, x1, cnt0):
        f0 = self.encoder(x0)
        f1 = self.encoder(x1)
        #
        # m0 = cnt2mask(cnt0, f0.shape[-2:-4:-1])
        # ------------------ Visualize cnt2mask for debugging -----------------------
        # a_mask = (m0.numpy() * 255).astype(np.uint8)
        # cv2.imwrite('debug_cnt2mask.png', a_mask[0,:,:,0])
        # ---------------------------------------------------------

        # f0_m = self.masking(f0, m0)
        # theta = self.affine(f1, f0_m)

        # theta = self.affine(x0, x1)
        theta = self.affine(f0, f1)

        return theta


class GlobalEncoder(tf.keras.layers.Layer):

    def __init__(self, out_dim):
        super(GlobalEncoder, self).__init__()
        self.conv12 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, padding="same", activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation='relu')
        self.conv23 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding="same", activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation='relu')
        self.conv34 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, padding="same", activation='relu')
        self.conv4a = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation='relu')
        self.conv4b = tf.keras.layers.Conv2D(out_dim, kernel_size=3, strides=1, padding="same")

    def call(self, in_features):
        x = self.conv12(in_features)
        x = self.conv2(x)
        x = self.conv23(x)
        x = self.conv3(x)
        x = self.conv34(x)
        x = self.conv4a(x)
        x = self.conv4b(x)

        return x


class Affine(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(Affine, self).__init__()
        self.conv45 = tf.keras.layers.Conv2D(filters=dim, kernel_size=3, strides=2, padding="same", activation='relu')  # 16
        self.conv5a = tf.keras.layers.Conv2D(filters=dim, kernel_size=3, strides=1, padding="same", activation='relu')  # 16
        self.conv5b = tf.keras.layers.Conv2D(filters=dim, kernel_size=3, strides=1, padding="same", activation='relu')  # 16
        self.conv56 = tf.keras.layers.Conv2D(filters=dim, kernel_size=3, strides=2, padding="same", activation='relu')  # 32
        self.conv6a = tf.keras.layers.Conv2D(filters=dim, kernel_size=3, strides=1, padding="same", activation='relu')  # 32
        self.conv6b = tf.keras.layers.Conv2D(filters=dim, kernel_size=3, strides=1, padding="same")  # 32

        self.fc = tf.keras.layers.Dense(6)  # weights = [np.zeros([512,6]), np.array([1, 0, 0, 0, 1, 0]) ]

        self.avg_pool2d = tf.keras.layers.GlobalAveragePooling2D()


    def call(self, feat1, feat2):
        x = tf.concat((feat1, feat2), axis=-1)
        x = self.conv45(x)
        x = self.conv5a(x)
        x = self.conv5b(x)
        x = self.conv56(x)
        x = self.conv6a(x)
        x = self.conv6b(x)

        x = self.avg_pool2d(x)
        # x = tf.reshape(x, shape=[-1, x.shape[-1]])

        theta = self.fc(x)
        theta = tf.reshape(theta, shape=[-1, 2, 3])

        return theta


class Masking(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(Masking, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=dim, kernel_size=1, strides=1, padding="same", activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=dim, kernel_size=1, strides=1, padding="same")

    def call(self, feat, mask):
        x = tf.concat((feat, mask), axis=-1)
        x = self.conv1(x)
        x = self.conv2(x)

        return x + feat


class LocalAlignment(tf.keras.Model):
    def __init__(self, num_iter=5, adj_num=4):
        super(LocalAlignment, self).__init__()
        self.num_iter = num_iter
        self.adj_num = adj_num
        self.encoder = LocalEncoder(out_dim=128)

        self.align_modules = []
        for i in range(num_iter):
            module = LAM(feature_dim=128+4, state_dim=128, uncertainty_bool=False)
            self.align_modules.append(module)

        # self.align_module = LAM(feature_dim=128+4, state_dim=128)
        # init_He(self)

    def call(self, x0, x1, cnt0, pos_emb, prev_rnn_features):
        '''

        :param x0: original first image (B, H, W, C)
        :param x1: orignal second image
        :param cnt0: contour of the first image (B, N, C) Points are in Width and Height order, unlike the image
        :param pos_emb:
        :param prev_rnn_features:

        x1_h : transformed first image to second image
        cnt1_h: transformed first contour to second contour

        :return:
        '''

        size = x1.shape[-2:-4:-1]
        N = cnt0.shape[1]
        
        # m1_h = cnt2mask(cnt0, size)
        # ------------------ Visualize cnt2mask for debugging -----------------------
        # a_mask = (m1_h.numpy() * 255).astype(np.uint8)
        # cv2.imwrite('debug_cnt2mask_m1_h.png', a_mask[0,:,:,0])
        # import pdb;pdb.set_trace()
        # ---------------------------------------------------------------------------
        # features = self.encoder(x0, x1)
        features1 = self.encoder(x0)
        features2 = self.encoder(x1)

        rnn_features = []
        cnt1 = cnt0  # cnt0 shape is (B, N, C)
        cnt1 = tf.cast(cnt1, 'float32')
        #----------------------------------------------
        # multi level
        saved_offset = {}
        for a_index, (a_feature1, a_feature2, prev_rnn_feature) in enumerate(zip(features1, features2, prev_rnn_features)):
            # a_feature shape is B, H, W, C
            sampled_cost_volume = cost_volume_at_contour_points(a_feature1, a_feature2, cnt0, cnt1, max_displacement=4)

            poly = cnt2poly(cnt1)  # It represents the contour points' coordinates on +image space. shape is B, N, 2
            m_in = tf.concat((sampled_cost_volume, pos_emb, poly), axis=-1)  # m_in shape is (B, N, C+4)

            adj = get_adj_ind(self.adj_num, N)
            offset, rnn_feature = self.align_modules[a_index](m_in, prev_rnn_feature, adj)
            rnn_features.append(rnn_feature)

            cnt1 = cnt1 + offset[:,:,:2]  # offset shape is (B, N, 3)
            saved_offset[a_index] = offset
        # ---------------------------------------------
        # backup multi level
        # for a_index, (a_feature, prev_rnn_feature) in enumerate(zip(features, prev_rnn_features)):
        #      # a_feature shape is B, H, W, C
        #
        #     sampled_feature = bilinear_sampler_1d(a_cost_volume, cnt1[:,:,0], cnt1[:,:,1])  # F.grid_sample(f, cnt1)  # s shape is (B, N, C)
        #
        #     poly = cnt2poly(cnt1)  # It represents the contour points' coordinates on +image space. shape is B, N, 2
        #     m_in = tf.concat((sampled_feature, pos_emb, poly), axis=-1)  # m_in shape is (B, N, C+4)
        #
        #     adj = get_adj_ind(self.adj_num, N)
        #     offset, rnn_feature = self.align_module(m_in, prev_rnn_feature, adj)
        #     rnn_features.append(rnn_feature)
        #     cnt1 = cnt1 + offset
        #
        #     saved_offset[a_index] = offset

        # --------------------------------------
        # single level
        # sampled_feature = bilinear_sampler_1d(features[0], cnt1[:,:,0], cnt1[:,:,1])  # sampled_feature shape is (B, N, C)
        # poly = cnt2poly(cnt1)  # It represents the contour points' coordinates on +image space. shape is B, N, 2
        # m_in = tf.concat((sampled_feature, pos_emb, poly), axis=-1)  # m_in shape is (B, N, C+4)
        #
        # adj = get_adj_ind(self.adj_num, N)
        # offset, rnn_feature = self.align_module(m_in, prev_rnn_features[0], adj)
        # rnn_features.append(rnn_feature)
        # cnt1 = cnt1 + offset
        # tf.print(offset, summarize=10)


        return cnt1, rnn_features, saved_offset


class LAM(tf.keras.layers.Layer):
    def __init__(self, feature_dim, state_dim, uncertainty_bool):
        super(LAM, self).__init__()
        self.head = BasicBlock(feature_dim, state_dim)

        self.dilation_list = [1, 1, 1] # [1,1,1] # [1, 1, 1, 2, 2, 4, 4]
        self.block_list = []
        for i in range(len(self.dilation_list)):
            a_basic_block = BasicBlock(state_dim, state_dim, dilation=self.dilation_list[i])
            # self.__setattr__('res' + str(i), conv)
            self.block_list.append(a_basic_block)

        fusion_state_dim = 256
        self.fusion = tf.keras.layers.Conv1D(filters=fusion_state_dim, kernel_size=1)

        rnn_state_dim = 256
        self.rnn_state = tf.keras.layers.Conv1D(filters=rnn_state_dim, kernel_size=1)
        self.rnn = tf.keras.layers.LSTM(units=rnn_state_dim, return_state=True, return_sequences=True)

        if uncertainty_bool:
            self.prediction = tf.keras.Sequential([
                tf.keras.layers.Conv1D(filters=256, kernel_size=1),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv1D(filters=64, kernel_size=1),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv1D(filters=3, kernel_size=1)
            ])
        else:
            self.prediction = tf.keras.Sequential([
                tf.keras.layers.Conv1D(filters=256, kernel_size=1),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv1D(filters=64, kernel_size=1),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv1D(filters=2, kernel_size=1)
            ])


    def call(self, input_x, prev_rnn_feature, adj):
        '''
        Architecture Overview:
            Circular Block: 8 circular conv, bn, relu
            Fusion: 1x1 conv, max pooling
            LSTM
            Prediction: 1x1 conv relu, 1x1 conv relu, 1x1 conv

        :param x: interpolated image feature at points + position embeddings of points (N, # of points, C)
        :param prev_rnn_feature:
        :param adj:
        :return:
        '''
        states = []
        x = self.head(input_x, adj)  # B, N, C
        states.append(x)
        for i in range(len(self.dilation_list)):
            # x = self.__getattribute__('res' + str(i))(x, adj) + x
            x = self.block_list[i](x, adj) + x
            states.append(x)

        state = tf.concat(states, axis=-1)

        global_state = tf.math.reduce_max(self.fusion(state), axis=1, keepdims=True)
        global_state = tf.broadcast_to(global_state, [global_state.shape[0], state.shape[1], global_state.shape[2]])  # B, N, C
        state = tf.concat([global_state, state], axis=-1)  # state output: (1, 128, 1024), global_state: (1, 128, 256)

        rnn_state = self.rnn_state(state)    # rnn_state output:(1, 128, 256)
        # TODO: see if combining batch size and number of points together into one dimension helps
        # rB, rN, rC = rnn_state.shape  # B, N, C
        # rnn_state = tf.reshape(rnn_state, shape=[1, rB * rN, rC])

        # For outputs of LSTM, refer to https://stackoverflow.com/questions/67970519/what-does-tensorflow-lstm-return
        # input shape is (batch, timesteps, feature)
        # whole_seq_output: shape=(1, 128, 256), final_hidden_state: shape=(1, 256), final_carry_state: shape=(1, 256)
        whole_seq_output, final_hidden_state, final_carry_state = self.rnn(rnn_state, initial_state=prev_rnn_feature)

        # o = tf.reshape(o, shape=[rB, rN, rC])
        # state = tf.concat([whole_seq_output, state], axis=-1)  # B, N, C  # TODO uncomment later
        x = self.prediction(state)  # B, N, 2

        return x, (final_hidden_state, final_carry_state)


# class MHA(nn.Module):
#     def __init__(self, dim, n_heads):
#         super(MHA, self).__init__()
#         self.dim = dim
#         self.n_heads = n_heads
#
#         self.layer_norm = nn.LayerNorm(dim, eps=1e-6)
#
#         self.conv_q = nn.Linear(dim, dim // 4 * n_heads, bias=False)
#         self.conv_k = nn.Linear(dim, dim // 4 * n_heads, bias=False)
#         self.conv_v = nn.Linear(dim, dim * n_heads, bias=False)
#         self.conv_w = nn.Linear(dim * n_heads, dim, bias=False)
#         self.conv_a = nn.Linear(dim * 2, dim, bias=False)
#
#     def forward(self, x):
#         B, C, K = x.shape
#         H = self.n_heads
#
#         x = x.transpose(1, 2).contiguous()  # B, K, C
#         residual = x
#
#         x = self.layer_norm(x)
#
#         q = self.conv_q(x)  # B, K, H*C//4
#         k = self.conv_k(x)  # B, K, H*C//4
#         v = self.conv_v(x)  # B, K, H*C
#
#         q = q.view(B, K, H, C // 4)  # B, K, H, C//4
#         k = k.view(B, K, H, C // 4)  # B, K, H, C//4
#         v = v.view(B, K, H, C)  # B, K, H, C
#
#         q = q.transpose(1, 2).contiguous()  # B, H, K, C//4
#         k = k.transpose(1, 2).contiguous()  # B, H, K, C//4
#         v = v.transpose(1, 2).contiguous()  # B, H, K, C
#
#         attn = torch.matmul(q / (C // 4) ** 0.5, k.transpose(-2, -1))  # B, H, K, K
#         attn = F.softmax(attn, dim=-1)  # B, H, K, K
#         r = torch.matmul(attn, v)  # B, H, K, C
#
#         r = r.transpose(1, 2).contiguous()  # B, K, H, C
#         r = r.view(B, K, H * C)
#         r = self.conv_w(r)  # B, K, C
#         r = torch.cat([r, residual], dim=2)
#         r = self.conv_a(r)
#         r = r.transpose(1, 2).contiguous()  # B, C, K
#         return r


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

    def call(self, input, adj):
        # adj is not used
        if self.n_adj != 0:
            input = tf.concat( [input[:, -self.n_adj * self.dilation:, :], input, input[:, :self.n_adj * self.dilation, :]], axis=1)
        return self.fc(input)


class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, state_dim, out_state_dim, n_adj=4, dilation=1):
        super(BasicBlock, self).__init__()
        self.dense_layer = tf.keras.layers.Dense(out_state_dim)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=state_dim)  # key_dim defines the hiddne_dim/output_dim of dense layer
        self.circ_conv = CircConv(state_dim, out_state_dim, n_adj, dilation)
        self.relu = tf.keras.layers.ReLU()

    def call(self, x, adj=None):
        # x = self.dense_layer(x)
        x = self.mha(query=x, value=x, key=x)
        x = self.circ_conv(x, adj)
        x = self.relu(x)
        return x


class VGG19_dropout(tf.keras.layers.Layer):
    def __init__(self, weights_path):
        super(VGG19_dropout, self).__init__()
        self.encoder_network = self.make_VGG19_dropout(weights_path)

    def call(self, x):
        c1, c2, c3, c4, c5 = self.encoder_network(x)

        return c5, c4, c3, c2, c1

    def make_VGG19_dropout(self, weights_path):
        # https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/applications/vgg19.py#L45-L230
        inputs = tf.keras.layers.Input(shape=[None, None, 3])
        # Block 1
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
        block1_conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(block1_conv2)
        x = tf.keras.layers.Dropout(0.25)(x)

        # Block 2
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        block2_conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(block2_conv2)
        x = tf.keras.layers.Dropout(0.5)(x)

        # Block 3
        x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        block3_conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(block3_conv4)
        x = tf.keras.layers.Dropout(0.5)(x)

        # Block 4
        x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        block4_conv4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(block4_conv4)
        x = tf.keras.layers.Dropout(0.5)(x)

        # Block 5
        x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        block5_conv4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)

        model = tf.keras.models.Model(inputs=inputs, outputs=[block1_conv2, block2_conv2, block3_conv4, block4_conv4, block5_conv4])

        # Load weights.
        if weights_path == '':
            WEIGHTS_PATH_NO_TOP = ('https://storage.googleapis.com/tensorflow/'
                                   'keras-applications/vgg19/'
                                   'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

            weights_path = tf.keras.utils.get_file(
                'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='253f8cb515780f3b799900260a226db6')
        model.load_weights(weights_path, by_name=True)

        return model


# class ResNet50(tf.keras.layers.Layer):
#     def __init__(self):
#         super(ResNet50, self).__init__()
#         resnet = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=[None, None, 3], pooling=None)
#
#         c1, c2, c3, c4, c5 = [
#             resnet.get_layer(layer_name).output
#             for layer_name in ["conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
#         ]
#
#         self.resnet = tf.keras.Model( inputs=[resnet.input], outputs=[c1, c2, c3, c4, c5] )
#
#     def call(self, x0, x1):
#         c1, c2, c3, c4, c5 = self.resnet(x1)
#
#         return c5, c4, c3, c2, c1


class ResNet50(tf.keras.layers.Layer):
    def __init__(self):
        super(ResNet50, self).__init__()
        resnet = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=[None, None, 3], pooling=None)
        init_kernel = resnet.get_layer('conv1_conv').weights[0]
        init_bias = resnet.get_layer('conv1_conv').bias
        kernel_initializer = tf.keras.initializers.constant(init_kernel)
        bias_initializer = tf.keras.initializers.constant(init_bias)

        self.conv1_pad = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')
        self.conv_layer_x0 = tf.keras.layers.Conv2D(64, 7,
                                                    activation='relu',
                                                    input_shape=(None, None, 3),
                                                    kernel_initializer=kernel_initializer,
                                                    bias_initializer=bias_initializer)
        self.conv_layer_x1 = tf.keras.layers.Conv2D(64, 7,
                                                    activation='relu',
                                                    input_shape=(None, None, 3),
                                                    kernel_initializer=kernel_initializer,
                                                    bias_initializer=bias_initializer)

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


    def call(self, x0):
        # Combine extracted features from the first image and the second image
        x0 = self.conv1_pad(x0)
        # x1 = self.conv1_pad(x1)
        conv_x0 = self.conv_layer_x0(x0)
        # conv_x1 = self.conv_layer_x1(x1)

        x = conv_x0
        # x = (conv_x0 + conv_x1) / 2

        c1, c2, c3, c4, c5 = self.modified_resnet(x)

        return c5, c4, c3, c2, c1


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

        # add residual connections
        # self.modified_model.layers[2].input = self.modified_model.layers[1].output + self.modified_model.layers[1].input
        # self.modified_model.layers[5].input = self.modified_model.layers[4].output + self.modified_model.layers[3].input
        # self.modified_model.layers[9].input = self.modified_model.layers[8].output + self.modified_model.layers[6].input
        # self.modified_model.layers[13].input = self.modified_model.layers[12].output + self.modified_model.layers[10].input
        # self.modified_model.layers[17].input = self.modified_model.layers[16].output + self.modified_model.layers[14].input
        # import pdb;pdb.set_trace()

        weights = [layer.get_weights() for layer in a_model.layers[2:]]
        for layer, weight in zip(self.modified_model.layers[1:], weights):
            layer.set_weights(weight)


    def call(self, x0):
        # Combine features from the first image, second image, and transformed mask
        x0 = self.conv1_pad(x0)
        # x1 = self.conv1_pad(x1)
        # m1_h = self.conv1_pad(m1_h)
        conv_x0 = self.conv_layer_x0(x0)
        # conv_x1 = self.conv_layer_x1(x1)
        # conv_m1_h = self.conv_layer_m1_h(m1_h)

        x = conv_x0
        # x = (conv_x0 + conv_x1) / 2
        # x = (conv_x0 + conv_x1 + conv_m1_h) / 3

        #---------------------
        # x = tf.concat((conv_x0, conv_x1), axis=-1)  # shape is (B, H, W, conv_x0 + conv_x1)
        # x = self.conv_layer_comb(x)
        # ----------------------------
        # for debugging feature map
        # cv2_imwrite_data = (conv_x0[0,:,:,:3].numpy() * 255).astype(np.uint8)
        # cv2_imwrite_data = cv2.cvtColor(cv2_imwrite_data, cv2.COLOR_BGR2RGB)
        # cv2.imwrite('debug_x0_featuremap.png', cv2_imwrite_data)
        # cv2_imwrite_data = (conv_x1[0,:,:,:3].numpy() * 255).astype(np.uint8)
        # cv2_imwrite_data = cv2.cvtColor(cv2_imwrite_data, cv2.COLOR_BGR2RGB)
        # cv2.imwrite('debug_x1_featuremap.png', cv2_imwrite_data)
        # import pdb;pdb.set_trace()
        # ----------------------------
        c1, c2, c3, c4, c5 = self.modified_model(x)

        # the order of features are from coarse to fine !!!
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


class LocalEncoder(tf.keras.layers.Layer):
    '''
    outdim corresponds to the number of tracking points
    '''
    def __init__(self, out_dim):
        super(LocalEncoder, self).__init__()
        self.encoder = VGG16() # VGG19_dropout('uflow/assets/model_weights/vgg19_dropout.hdf5') # VGG19_dropout('')
        self.fpn = FeaturePyramid(out_dim)  # FPN([2048, 1024, 512, 256, 64], out_dim)

    def call(self, x0):
        # x0: original first image
        # x1: original second image
        # p: transformed first image to second image
        # m: Mask from (transformed first contour to second contour)
        rs = self.encoder(x0)
        outs = self.fpn(*rs)

        return outs

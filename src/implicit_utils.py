'''
Junbong Jang
4/30/2023

Modified Tensorflow implementation of Neural Matching Field (NeMF) from https://github.com/KU-CVLAB/NeMF

'''
import tensorflow as tf
from src import tracking_utils


def sample_initial_points(source_seg_points, source_seg_points_limit, NUM_INITIAL_SAMPLE_POINTS=50):
    batch_size = source_seg_points.shape[0]

    # to get valid contour points, not padded points that are meaningless
    maxval_tensor = tf.expand_dims( tf.cast(source_seg_points_limit, tf.float32), axis=-1)
    maxval_tensor = tf.repeat(maxval_tensor, repeats=NUM_INITIAL_SAMPLE_POINTS, axis=-1)

    # sample points in the previous contour
    source_sampled_contour_index = tf.random.uniform(shape=[batch_size, NUM_INITIAL_SAMPLE_POINTS], maxval=maxval_tensor, dtype=tf.float32, seed=10)

    return source_sampled_contour_index


def sample_nearby_points_for_implicit_cycle_consistency(source_sampled_contour_index, source_seg_points, target_seg_points, NUM_NEIGHB_SAMPLE_POINTS=20):
    '''
    from the source contour points, find the closest NUM_NEIGHB_SAMPLE_POINTS points on the target contour

    source_sampled_contour_index  (batch_size, NUM_SAMPLE_POINTS)
    source_seg_points  (batch_size, 1640, 3)
    target_seg_points  (batch_size, 1640, 3)

    return (batch_size, initial_points, nearby_points)
    '''
    batch_size = source_seg_points.shape[0]
    NUM_SOURCE_SAMPLE_POINTS = source_sampled_contour_index.shape[1]

    source_sampled_contour_index = tf.cast(source_sampled_contour_index, dtype=tf.int32)
    temp_sampled_contour_index = tf.reshape(source_sampled_contour_index, shape=[-1])
    batch_index_list = tf.repeat(tf.range(batch_size), repeats=NUM_SOURCE_SAMPLE_POINTS)
    indices = tf.stack([batch_index_list, temp_sampled_contour_index], axis=1)  # shape is (number of points, 3)

    # find the closest points in the current contour with respect to the sampled points in the previous contour
    sampled_source_seg_points = tf.gather_nd(source_seg_points, indices)
    sampled_source_seg_points = tf.reshape(sampled_source_seg_points[:,:2], shape=[batch_size, NUM_SOURCE_SAMPLE_POINTS, 2])
    closest_target_seg_points_id = tracking_utils.get_closest_contour_id(target_seg_points[:,:,:2], None, sampled_source_seg_points)  # [8, 10, 1]

    # get NUM_NEIGHB_SAMPLE_POINTS nearby points for each closest_target_sampled_seg_points
    nearby_target_contour_index_list = []
    for a_index in range(NUM_NEIGHB_SAMPLE_POINTS):
        nearby_target_seg_points_id = closest_target_seg_points_id + a_index - NUM_NEIGHB_SAMPLE_POINTS//2
        nearby_target_contour_index_list.append(nearby_target_seg_points_id)

    nearby_target_contour_index_tensor = tf.concat(nearby_target_contour_index_list,axis=-1)
    nearby_target_contour_index_tensor = tf.math.abs(nearby_target_contour_index_tensor)  # TODO: clip max value with target_seg_points_limit
    # nearby_target_contour_index_tensor = tf.reshape(nearby_target_contour_index_tensor, shape=[batch_size, -1])  # (8, 100)
    
    return nearby_target_contour_index_tensor


def find_max_corr_contour_indices(occ_contour_forward, closest_cur_sampled_contour_indices):
    '''
    For each first contour's point, find the second contour's point with the maximum correspondence value

    occ_contour_forward (batch_size, NUM_SAMPLE_POINTS, NUM_NEARBY_POINTS)
    closest_cur_sampled_contour_indices (batch_size, NUM_SAMPLE_POINTS, NUM_NEARBY_POINTS)

    return (batch_size, NUM_SAMPLE_POINTS)
    '''

    batch_size = occ_contour_forward.shape[0]
    NUM_SAMPLE_POINTS = occ_contour_forward.shape[1]
    NUM_NEARBY_POINTS = occ_contour_forward.shape[2]

    max_indices_of_closest_cur_sampled_contour_indices = tf.math.argmax(occ_contour_forward, axis=-1)
    max_indices_of_closest_cur_sampled_contour_indices = tf.cast(max_indices_of_closest_cur_sampled_contour_indices, dtype=tf.int32)
    max_indices_of_closest_cur_sampled_contour_indices = tf.reshape(max_indices_of_closest_cur_sampled_contour_indices, shape=[-1])

    offset_tensor = tf.linspace(0, (NUM_SAMPLE_POINTS-1)*NUM_NEARBY_POINTS, NUM_SAMPLE_POINTS)
    offset_tensor = tf.cast(offset_tensor, dtype=tf.int32)
    offset_tensor = tf.repeat(tf.expand_dims(offset_tensor, axis=0), repeats=batch_size, axis=0)  # (batch_size, NUM_SAMPLE_POINTS)
    offset_tensor = tf.reshape(offset_tensor, shape=[-1])  # (batch_size * NUM_SAMPLE_POINTS)

    max_indices_of_closest_cur_sampled_contour_indices = max_indices_of_closest_cur_sampled_contour_indices + offset_tensor

    batch_index_list = tf.repeat(tf.range(batch_size), repeats=NUM_SAMPLE_POINTS)
    gather_indices = tf.stack([batch_index_list, max_indices_of_closest_cur_sampled_contour_indices], axis=1)  # shape is (number of points, 2)

    closest_cur_sampled_contour_indices = tf.reshape(closest_cur_sampled_contour_indices, shape=[batch_size, -1])
    predicted_cur_contour_indices = tf.gather_nd(closest_cur_sampled_contour_indices, gather_indices)
    predicted_cur_contour_indices = tf.reshape(predicted_cur_contour_indices, shape=[batch_size, -1]) 

    return predicted_cur_contour_indices


def create_GT_occupnacy(sampled_contour_indices, seg_points_shape, sampled_nearby_contour_indices):
    '''
        sampled_contour_indices shape: (batch_size, NUM_SAMPLE_POINTS) represent the initial points before cycle consistency
        seg_points_shape = (batch_size, 1640) tell the maximum number of points in a contour
        sampled_nearby_contour_indices shape: (batch_size, NUM_SAMPLE_POINTS, NUM_NEARBY_POINTS)   

        Return ground truth occupancy tensor shape:  (batch_size, NUM_SAMPLE_POINTS, NUM_NEARBY_POINTS)
        with value 1 for indices at sampled_contour_indices and 0 otherwise
    '''
    gt_occ = tf.zeros(seg_points_shape)
    updates = tf.ones(shape=sampled_contour_indices.shape[0]*sampled_contour_indices.shape[1])
    updates = tf.cast(updates, tf.float32)

    batch_size = sampled_contour_indices.shape[0]
    NUM_SAMPLE_POINTS = sampled_contour_indices.shape[1]

    assert sampled_contour_indices.shape[1] == sampled_nearby_contour_indices.shape[1]

    batch_index_list = tf.repeat(tf.range(batch_size), repeats=NUM_SAMPLE_POINTS)
    sampled_contour_indices = tf.reshape(sampled_contour_indices, shape=[-1])
    indices = tf.stack([batch_index_list, sampled_contour_indices], axis=1)  # shape is (number of points, 2)

    # Return occupancy tensor with a shape (batch_size, 1640) with value 1 for indices at sampled_contour_indices and 0 otherwise
    gt_occ = tf.tensor_scatter_nd_update(gt_occ, indices, updates)  # (8, 1640)
    
    # assert tf.math.reduce_sum(gt_occ) <= indices.shape[0]  # because some of sampled_contour_indices can be repeated due to uniform sampling
    
    # get gt_occ at sampled_nearby_contour_indices
    import pdb;pdb.set_trace()
    sampled_nearby_contour_indices = tf.reshape(sampled_nearby_contour_indices, shape=[-1])
    indices = tf.stack([batch_index_list, sampled_nearby_contour_indices], axis=1)  # shape is (number of points, 2)

    gt_occ = tf.gather_nd(gt_occ, indices)
    gt_occ = tf.reshape(gt_occ, shape=[batch_size, -1]) 

    return gt_occ


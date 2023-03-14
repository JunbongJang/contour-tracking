'''
Author: Junbong Jang and from various sources
Date: 3/15/2022

Collection of metrics to evaluate performance of contour tracking models
'''

import numpy as np
import math
import cv2
from skimage.morphology import disk


'''
Point Tracking Metrics implemented in Polygonal Point Set Tracking
These measure the distance between the exact corresponding points as follows

from https://github.com/ghnam-ken/PoST/blob/main/eval/metric.py
'''

def normalize_points(pts, image_width, image_height):
    pts = pts.copy()
    pts[...,0] /= image_width
    pts[...,1] /= image_height
    return pts


def point_wise_spatial_accuracy(ps1, ps2, image_width, image_height, thresh):
    '''
        Args) ps1, ps2 : normalized point sets
        Retern) acc: spatial accuracy
    '''
    assert len(ps1) == len(ps2), \
        f"length of given point sets are differenct: len(ps1)={len(ps1)}, len(ps2)={len(ps2)}"

    ps1 = normalize_points(ps1, image_width, image_height)
    ps2 = normalize_points(ps2, image_width, image_height)

    dists = (ps1 - ps2) ** 2
    dists = np.sum(dists, axis=-1)
    dists = np.sqrt(dists)

    return (dists <= thresh)*1


def spatial_accuracy(ps1, ps2, image_width, image_height, thresh):
    '''
        Args) ps1, ps2 : normalized point sets
        Retern) acc: spatial accuracy
    '''
    assert len(ps1) == len(ps2), \
        f"length of given point sets are differenct: len(ps1)={len(ps1)}, len(ps2)={len(ps2)}"

    ps1 = normalize_points(ps1, image_width, image_height)
    ps2 = normalize_points(ps2, image_width, image_height)

    dists = (ps1 - ps2) ** 2
    dists = np.sum(dists, axis=-1)
    dists = np.sqrt(dists)

    acc = np.mean(dists <= thresh)
    return acc

def relative_spatial_accuracy(ps1, ps2, prev_ps1, prev_ps2, image_width, image_height, thresh):
    '''
        Args) ps1, ps2 : normalized point sets
        Retern) acc: temporal accuracy

        ps1: shape (num_points, 2), dtype: np.float32
        ps2: shape (num_points, 2), dtype: np.float32
    '''
    assert len(ps1) == len(ps2), \
            f"length of given point sets are differenct: len(ps1)={len(ps1)}, len(ps2)={len(ps2)}"
    assert len(prev_ps1) == len(prev_ps2), \
            f"length of given point sets are differenct: len(prev_ps1)={len(prev_ps1)}, len(prev_ps2)={len(prev_ps2)}"
    assert len(ps1) == len(prev_ps1)

    ps1 = normalize_points(ps1, image_width, image_height)
    ps2 = normalize_points(ps2, image_width, image_height)
    prev_ps1 = normalize_points(prev_ps1, image_width, image_height)
    prev_ps2 = normalize_points(prev_ps2, image_width, image_height)

    gt_dists = ps1 - prev_ps1
    pred_dists = ps2 - prev_ps2

    diffs = (gt_dists - pred_dists) ** 2
    diffs = np.sum(diffs, axis=-1)
    diffs = np.sqrt(diffs)

    acc = np.mean(diffs <= thresh, axis=-1)
    acc = np.mean(acc)
    return acc


def temporal_accuracy(ps1, ps2, prev_ps1, prev_ps2, image_width, image_height, thresh):
    '''
        Args) ps1, ps2 : normalized point sets
        Retern) acc: temporal accuracy
    '''
    assert len(ps1) == len(ps2), \
            f"length of given point sets are differenct: len(ps1)={len(ps1)}, len(ps2)={len(ps2)}"
    assert len(prev_ps1) == len(prev_ps2), \
            f"length of given point sets are differenct: len(prev_ps1)={len(prev_ps1)}, len(prev_ps2)={len(prev_ps2)}"
    assert len(ps1) == len(prev_ps1)

    ps1 = normalize_points(ps1, image_width, image_height)
    ps2 = normalize_points(ps2, image_width, image_height)
    prev_ps1 = normalize_points(prev_ps1, image_width, image_height)
    prev_ps2 = normalize_points(prev_ps2, image_width, image_height)

    dists_prev = ps1 - ps2
    dists_next = prev_ps1 - prev_ps2

    diffs = (dists_prev - dists_next) ** 2
    diffs = np.sum(diffs, axis=-1)
    diffs = np.sqrt(diffs)

    acc = np.mean(diffs <= thresh, axis=-1)
    acc = np.mean(acc)
    return acc


def contour_accuracy(contour_indices1, contour_indices2, total_contour_length, thresh):
    '''
        Args) contour_indices1, contour_indices2
        Retern) acc: contour accuracy
    '''
    assert len(contour_indices1) == len(contour_indices2), \
        f"length of given point sets are differenct: len(ps1)={len(contour_indices1)}, len(ps2)={len(contour_indices2)}"

    dists = np.abs(contour_indices1 - contour_indices2) / total_contour_length
    acc = np.mean(dists <= thresh )

    return acc

'''
Region similarity J For the evaluation on video object segmentation datasets (CPC and DAVIS2016)

from https://github.com/davisvideochallenge/davis2017-evaluation/blob/master/davis2017/metrics.py
'''

def db_eval_iou(annotation, segmentation, void_pixels=None):
    """  Compute region similarity as the Jaccard Index (J).

    Jaccard Index = (the number in both sets) / (the number in either set) * 100

    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
        void_pixels  (ndarray): optional mask with void pixels
    Return:
        jaccard (float): region similarity
    """
    assert annotation.shape == segmentation.shape, \
        f'Annotation({annotation.shape}) and segmentation:{segmentation.shape} dimensions do not match.'
    annotation = annotation.astype(np.bool)
    segmentation = segmentation.astype(np.bool)

    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape, \
            f'Annotation({annotation.shape}) and void pixels:{void_pixels.shape} dimensions do not match.'
        void_pixels = void_pixels.astype(np.bool)
    else:
        void_pixels = np.zeros_like(segmentation)

    # Intersection between all sets
    inters = np.sum((segmentation & annotation) & np.logical_not(void_pixels), axis=(-2, -1))
    union = np.sum((segmentation | annotation) & np.logical_not(void_pixels), axis=(-2, -1))

    j = inters / union
    if j.ndim == 0:
        j = 1 if np.isclose(union, 0) else j
    else:
        j[np.isclose(union, 0)] = 1
    return j


def db_eval_boundary(annotation, segmentation, void_pixels=None, bound_th=0.008):
    assert annotation.shape == segmentation.shape
    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape
    if annotation.ndim != 2:
        raise ValueError(f'db_eval_boundary does not support tensors with {annotation.ndim} dimensions')

    f_res = f_measure(segmentation, annotation, void_pixels, bound_th=bound_th)

    return f_res

'''
Boundary Accuracy F for the evaluation on video object segmentation datasets (CPC and DAVIS2016)

from https://github.com/davisvideochallenge/davis2017-evaluation/blob/master/davis2017/metrics.py
'''

def f_measure(fg_boundary, gt_boundary, void_pixels=None, bound_pix=1):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.
    Arguments:
        fg_boundary (ndarray): binary predicted contour image.
        gt_boundary (ndarray): binary annotated contour image.
        void_pixels (ndarray): optional mask with void pixels
    Returns:
        F (float): boundaries F-measure
    """

    # TODO Fix Precision value too big problem due to fg_match being too large
    assert np.atleast_3d(fg_boundary).shape[2] == 1
    if void_pixels is not None:
        void_pixels = void_pixels.astype(np.bool)
    else:
        void_pixels = np.zeros_like(fg_boundary).astype(np.bool)

    # bound_th = 0.008
    # bound_pix = bound_th if bound_th >= 1 else \
    #     np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))  # frobenius norm
    intensity_threshold = 0  # slightly better than 127
    # import pdb;pdb.set_trace()
    fg_boundary = (fg_boundary > intensity_threshold).astype(np.uint8)
    gt_boundary = (gt_boundary > intensity_threshold).astype(np.uint8)

    # Get the pixel boundaries of both masks
    fg_boundary = fg_boundary * np.logical_not(void_pixels)
    gt_boundary = gt_boundary * np.logical_not(void_pixels)

    a_disk = disk(bound_pix).astype(np.uint8)
    fg_dil = cv2.dilate(fg_boundary, a_disk )
    gt_dil = cv2.dilate(gt_boundary, a_disk )

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # ---------------- save intermediate outputs for debugging -----------------
    # cv2.imwrite('fg_dil.png', fg_dil*255)
    # cv2.imwrite('gt_dil.png', gt_dil*255)
    # cv2.imwrite('gt_match.png', gt_match*255)
    # cv2.imwrite('fg_match.png', fg_match*255)
    # ---------------------------------------------------------

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F, precision, recall


def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(np.bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
            width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap


def get_warp_error(a_warped_contour, a_contour):
    def l1(x):
        return tf.abs(x + 1e-6)

    def robust_l1(x):
        """Robust L1 metric."""
        return (x ** 2 + 0.001 ** 2) ** 0.5

    error = a_warped_contour.astype(np.int32) - a_contour.astype(np.int32)  # e.g. 127 - 255  or 127 - 0

    cliped_error = np.clip(error, 0, 255)  # clipping to ignore negative values

    final_error = robust_l1(cliped_error)

    return final_error

'''
Spatial Accuracy and Temporal Accuracy implemented in Coherent Parametric Contours for Interactive Video Segmentation by Yao Lu
The distance was computed from the ground truth points to their closest points in the predicted contour

from http://yao.lu/cpc/
'''

# # Compute spatial accuracy
# float acc(vector<Point> gt, vector<Point> res, float th)
# {
# 	int hit = 0;
# 	for (int i = 0; i < gt.size(); i++)
# 	{
# 		if(abs(pointPolygonTest(res, gt[i], true)) <= th)
# 			hit++;
# 	}
# 	return (float)hit / gt.size();
# }
#
# # Compute temporal consistency
# float cons(vector<Point> gt0, vector<Point> gt1, vector<Point> res0, vector<Point> res1, float th)
# {
# 	assert(gt0.size() == gt1.size());
# 	vector<float> dist0, dist1;
# 	int hit = 0;
# 	for (int i = 0; i < gt0.size(); i++)
# 		dist0.push_back(pointPolygonTest(res0, gt0[i], true));
# 	for (int i = 0; i < gt1.size(); i++)
# 		dist1.push_back(pointPolygonTest(res1, gt1[i], true));
# 	for (int i = 0; i < gt0.size(); i++)
# 	{
# 		if (abs(dist0[i] - dist1[i]) <= th)
# 			hit++;
# 	}
# 	return (float)hit / gt0.size();
# }
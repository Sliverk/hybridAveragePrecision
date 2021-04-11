#####################
# Based on https://github.com/hongzhenwang/RRPN-revise
# Licensed under The MIT License
# Author: yanyan, scrin@foxmail.com
#####################
import math
import numba
import numpy as np
from numba import cuda


@numba.jit(nopython=True)
def div_up(m, n):
    return m // n + (m % n > 0)


@cuda.jit('(float32[:], float32[:], float32[:])', device=True, inline=True)
def trangle_area(a, b, c):
    return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) *
            (b[0] - c[0])) / 2.0


@cuda.jit('(float32[:], int32)', device=True, inline=True)
def area(int_pts, num_of_inter):
    area_val = 0.0
    for i in range(num_of_inter - 2):
        area_val += abs(
            trangle_area(int_pts[:2], int_pts[2 * i + 2:2 * i + 4],
                         int_pts[2 * i + 4:2 * i + 6]))
    return area_val


@cuda.jit('(float32[:], int32)', device=True, inline=True)
def sort_vertex_in_convex_polygon(int_pts, num_of_inter):
    if num_of_inter > 0:
        center = cuda.local.array((2, ), dtype=numba.float32)
        center[:] = 0.0
        for i in range(num_of_inter):
            center[0] += int_pts[2 * i]
            center[1] += int_pts[2 * i + 1]
        center[0] /= num_of_inter
        center[1] /= num_of_inter
        v = cuda.local.array((2, ), dtype=numba.float32)
        vs = cuda.local.array((16, ), dtype=numba.float32)
        for i in range(num_of_inter):
            v[0] = int_pts[2 * i] - center[0]
            v[1] = int_pts[2 * i + 1] - center[1]
            d = math.sqrt(v[0] * v[0] + v[1] * v[1])
            v[0] = v[0] / d
            v[1] = v[1] / d
            if v[1] < 0:
                v[0] = -2 - v[0]
            vs[i] = v[0]
        j = 0
        temp = 0
        for i in range(1, num_of_inter):
            if vs[i - 1] > vs[i]:
                temp = vs[i]
                tx = int_pts[2 * i]
                ty = int_pts[2 * i + 1]
                j = i
                while j > 0 and vs[j - 1] > temp:
                    vs[j] = vs[j - 1]
                    int_pts[j * 2] = int_pts[j * 2 - 2]
                    int_pts[j * 2 + 1] = int_pts[j * 2 - 1]
                    j -= 1

                vs[j] = temp
                int_pts[j * 2] = tx
                int_pts[j * 2 + 1] = ty


@cuda.jit(
    '(float32[:], float32[:], int32, int32, float32[:])',
    device=True,
    inline=True)
def line_segment_intersection(pts1, pts2, i, j, temp_pts):
    A = cuda.local.array((2, ), dtype=numba.float32)
    B = cuda.local.array((2, ), dtype=numba.float32)
    C = cuda.local.array((2, ), dtype=numba.float32)
    D = cuda.local.array((2, ), dtype=numba.float32)

    A[0] = pts1[2 * i]
    A[1] = pts1[2 * i + 1]

    B[0] = pts1[2 * ((i + 1) % 4)]
    B[1] = pts1[2 * ((i + 1) % 4) + 1]

    C[0] = pts2[2 * j]
    C[1] = pts2[2 * j + 1]

    D[0] = pts2[2 * ((j + 1) % 4)]
    D[1] = pts2[2 * ((j + 1) % 4) + 1]
    BA0 = B[0] - A[0]
    BA1 = B[1] - A[1]
    DA0 = D[0] - A[0]
    CA0 = C[0] - A[0]
    DA1 = D[1] - A[1]
    CA1 = C[1] - A[1]
    acd = DA1 * CA0 > CA1 * DA0
    bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
    if acd != bcd:
        abc = CA1 * BA0 > BA1 * CA0
        abd = DA1 * BA0 > BA1 * DA0
        if abc != abd:
            DC0 = D[0] - C[0]
            DC1 = D[1] - C[1]
            ABBA = A[0] * B[1] - B[0] * A[1]
            CDDC = C[0] * D[1] - D[0] * C[1]
            DH = BA1 * DC0 - BA0 * DC1
            Dx = ABBA * DC0 - BA0 * CDDC
            Dy = ABBA * DC1 - BA1 * CDDC
            temp_pts[0] = Dx / DH
            temp_pts[1] = Dy / DH
            return True
    return False


@cuda.jit(
    '(float32[:], float32[:], int32, int32, float32[:])',
    device=True,
    inline=True)
def line_segment_intersection_v1(pts1, pts2, i, j, temp_pts):
    a = cuda.local.array((2, ), dtype=numba.float32)
    b = cuda.local.array((2, ), dtype=numba.float32)
    c = cuda.local.array((2, ), dtype=numba.float32)
    d = cuda.local.array((2, ), dtype=numba.float32)

    a[0] = pts1[2 * i]
    a[1] = pts1[2 * i + 1]

    b[0] = pts1[2 * ((i + 1) % 4)]
    b[1] = pts1[2 * ((i + 1) % 4) + 1]

    c[0] = pts2[2 * j]
    c[1] = pts2[2 * j + 1]

    d[0] = pts2[2 * ((j + 1) % 4)]
    d[1] = pts2[2 * ((j + 1) % 4) + 1]

    area_abc = trangle_area(a, b, c)
    area_abd = trangle_area(a, b, d)

    if area_abc * area_abd >= 0:
        return False

    area_cda = trangle_area(c, d, a)
    area_cdb = area_cda + area_abc - area_abd

    if area_cda * area_cdb >= 0:
        return False
    t = area_cda / (area_abd - area_abc)

    dx = t * (b[0] - a[0])
    dy = t * (b[1] - a[1])
    temp_pts[0] = a[0] + dx
    temp_pts[1] = a[1] + dy
    return True


@cuda.jit('(float32, float32, float32[:])', device=True, inline=True)
def point_in_quadrilateral(pt_x, pt_y, corners):
    ab0 = corners[2] - corners[0]
    ab1 = corners[3] - corners[1]

    ad0 = corners[6] - corners[0]
    ad1 = corners[7] - corners[1]

    ap0 = pt_x - corners[0]
    ap1 = pt_y - corners[1]

    abab = ab0 * ab0 + ab1 * ab1
    abap = ab0 * ap0 + ab1 * ap1
    adad = ad0 * ad0 + ad1 * ad1
    adap = ad0 * ap0 + ad1 * ap1

    return abab >= abap and abap >= 0 and adad >= adap and adap >= 0


@cuda.jit('(float32[:], float32[:], float32[:])', device=True, inline=True)
def quadrilateral_intersection(pts1, pts2, int_pts):
    num_of_inter = 0
    for i in range(4):
        if point_in_quadrilateral(pts1[2 * i], pts1[2 * i + 1], pts2):
            int_pts[num_of_inter * 2] = pts1[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1]
            num_of_inter += 1
        if point_in_quadrilateral(pts2[2 * i], pts2[2 * i + 1], pts1):
            int_pts[num_of_inter * 2] = pts2[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1]
            num_of_inter += 1
    temp_pts = cuda.local.array((2, ), dtype=numba.float32)
    for i in range(4):
        for j in range(4):
            has_pts = line_segment_intersection(pts1, pts2, i, j, temp_pts)
            if has_pts:
                int_pts[num_of_inter * 2] = temp_pts[0]
                int_pts[num_of_inter * 2 + 1] = temp_pts[1]
                num_of_inter += 1

    return num_of_inter


@cuda.jit('(float32[:], float32[:])', device=True, inline=True)
def rbbox_to_corners(corners, rbbox):
    # generate clockwise corners and rotate it clockwise
    angle = rbbox[4]
    a_cos = math.cos(angle)
    a_sin = math.sin(angle)
    center_x = rbbox[0]
    center_y = rbbox[1]
    x_d = rbbox[2]
    y_d = rbbox[3]
    corners_x = cuda.local.array((4, ), dtype=numba.float32)
    corners_y = cuda.local.array((4, ), dtype=numba.float32)
    corners_x[0] = -x_d / 2
    corners_x[1] = -x_d / 2
    corners_x[2] = x_d / 2
    corners_x[3] = x_d / 2
    corners_y[0] = -y_d / 2
    corners_y[1] = y_d / 2
    corners_y[2] = y_d / 2
    corners_y[3] = -y_d / 2
    for i in range(4):
        corners[2 * i] = a_cos * corners_x[i] + a_sin * corners_y[i] + center_x
        corners[2 * i +
                1] = -a_sin * corners_x[i] + a_cos * corners_y[i] + center_y


@cuda.jit('(float32[:], float32[:])', device=True, inline=True)
def inter(rbbox1, rbbox2):
    """Compute intersection of two rotated boxes.

    Args:
        rbox1 (np.ndarray, shape=[5]): Rotated 2d box.
        rbox2 (np.ndarray, shape=[5]): Rotated 2d box.

    Returns:
        float: Intersection of two rotated boxes.
    """
    corners1 = cuda.local.array((8, ), dtype=numba.float32)
    corners2 = cuda.local.array((8, ), dtype=numba.float32)
    intersection_corners = cuda.local.array((16, ), dtype=numba.float32)

    rbbox_to_corners(corners1, rbbox1)
    rbbox_to_corners(corners2, rbbox2)

    num_intersection = quadrilateral_intersection(corners1, corners2,
                                                  intersection_corners)
    sort_vertex_in_convex_polygon(intersection_corners, num_intersection)
    # print(intersection_corners.reshape([-1, 2])[:num_intersection])

    return area(intersection_corners, num_intersection)


@cuda.jit('(float32[:], float32[:], int32)', device=True, inline=True)
def devRotateIoUEval(rbox1, rbox2, criterion=-1):
    """Compute rotated iou on device.

    Args:
        rbox1 (np.ndarray, shape=[5]): Rotated 2d box.
        rbox2 (np.ndarray, shape=[5]): Rotated 2d box.
        criterion (int, optional): Indicate different type of iou.
            -1 indicate `area_inter / (area1 + area2 - area_inter)`,
            0 indicate `area_inter / area1`,
            1 indicate `area_inter / area2`.

    Returns:
        float: iou between two input boxes.
    """
    area1 = rbox1[2] * rbox1[3]
    area2 = rbox2[2] * rbox2[3]
    area_inter = inter(rbox1, rbox2)
    if criterion == -1:
        return area_inter / (area1 + area2 - area_inter)
    elif criterion == 0:
        return area_inter / area1
    elif criterion == 1:
        return area_inter / area2
    else:
        return area_inter


@cuda.jit(
    '(int64, int64, float32[:], float32[:], float32[:], int32)',
    fastmath=False)
def rotate_iou_kernel_eval(N,
                           K,
                           dev_boxes,
                           dev_query_boxes,
                           dev_iou,
                           criterion=-1):
    """Kernel of computing rotated iou.

    Args:
        N (int): The number of boxes.
        K (int): The number of query boxes.
        dev_boxes (np.ndarray): Boxes on device.
        dev_query_boxes (np.ndarray): Query boxes on device.
        dev_iou (np.ndarray): Computed iou to return.
        criterion (int, optional): Indicate different type of iou.
            -1 indicate `area_inter / (area1 + area2 - area_inter)`,
            0 indicate `area_inter / area1`,
            1 indicate `area_inter / area2`.
    """
    threadsPerBlock = 8 * 8
    row_start = cuda.blockIdx.x
    col_start = cuda.blockIdx.y
    tx = cuda.threadIdx.x
    row_size = min(N - row_start * threadsPerBlock, threadsPerBlock)
    col_size = min(K - col_start * threadsPerBlock, threadsPerBlock)
    block_boxes = cuda.shared.array(shape=(64 * 5, ), dtype=numba.float32)
    block_qboxes = cuda.shared.array(shape=(64 * 5, ), dtype=numba.float32)

    dev_query_box_idx = threadsPerBlock * col_start + tx
    dev_box_idx = threadsPerBlock * row_start + tx
    if (tx < col_size):
        block_qboxes[tx * 5 + 0] = dev_query_boxes[dev_query_box_idx * 5 + 0]
        block_qboxes[tx * 5 + 1] = dev_query_boxes[dev_query_box_idx * 5 + 1]
        block_qboxes[tx * 5 + 2] = dev_query_boxes[dev_query_box_idx * 5 + 2]
        block_qboxes[tx * 5 + 3] = dev_query_boxes[dev_query_box_idx * 5 + 3]
        block_qboxes[tx * 5 + 4] = dev_query_boxes[dev_query_box_idx * 5 + 4]
    if (tx < row_size):
        block_boxes[tx * 5 + 0] = dev_boxes[dev_box_idx * 5 + 0]
        block_boxes[tx * 5 + 1] = dev_boxes[dev_box_idx * 5 + 1]
        block_boxes[tx * 5 + 2] = dev_boxes[dev_box_idx * 5 + 2]
        block_boxes[tx * 5 + 3] = dev_boxes[dev_box_idx * 5 + 3]
        block_boxes[tx * 5 + 4] = dev_boxes[dev_box_idx * 5 + 4]
    cuda.syncthreads()
    if tx < row_size:
        for i in range(col_size):
            offset = (
                row_start * threadsPerBlock * K + col_start * threadsPerBlock +
                tx * K + i)
            dev_iou[offset] = devRotateIoUEval(block_qboxes[i * 5:i * 5 + 5],
                                               block_boxes[tx * 5:tx * 5 + 5],
                                               criterion)


def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    """Rotated box iou running in gpu. 500x faster than cpu version (take 5ms
    in one example with numba.cuda code). convert from [this project](
    https://github.com/hongzhenwang/RRPN-revise/tree/master/lib/rotation).

    Args:
        boxes (torch.Tensor): rbboxes. format: centers, dims,
            angles(clockwise when positive) with the shape of [N, 5].
        query_boxes (float tensor: [K, 5]): rbboxes to compute iou with boxes.
        device_id (int, optional): Defaults to 0. Device to use.
        criterion (int, optional): Indicate different type of iou.
            -1 indicate `area_inter / (area1 + area2 - area_inter)`,
            0 indicate `area_inter / area1`,
            1 indicate `area_inter / area2`.

    Returns:
        np.ndarray: IoU results.
    """
    boxes = boxes.astype(np.float32)
    query_boxes = query_boxes.astype(np.float32)
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    iou = np.zeros((N, K), dtype=np.float32)
    if N == 0 or K == 0:
        return iou
    threadsPerBlock = 8 * 8
    cuda.select_device(device_id)
    blockspergrid = (div_up(N, threadsPerBlock), div_up(K, threadsPerBlock))

    stream = cuda.stream()
    with stream.auto_synchronize():
        boxes_dev = cuda.to_device(boxes.reshape([-1]), stream)
        query_boxes_dev = cuda.to_device(query_boxes.reshape([-1]), stream)
        iou_dev = cuda.to_device(iou.reshape([-1]), stream)
        rotate_iou_kernel_eval[blockspergrid, threadsPerBlock,
                               stream](N, K, boxes_dev, query_boxes_dev,
                                       iou_dev, criterion)
        iou_dev.copy_to_host(iou.reshape([-1]), stream=stream)
    return iou.astype(boxes.dtype)


@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lidar.
    # TODO: change to use prange for parallel mode, should check the difference
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in numba.prange(N):
        for j in numba.prange(K):
            if rinc[i, j] > 0:
                # iw = (min(boxes[i, 1] + boxes[i, 4], qboxes[j, 1] +
                #         qboxes[j, 4]) - max(boxes[i, 1], qboxes[j, 1]))
                iw = (
                    min(boxes[i, 1], qboxes[j, 1]) -
                    max(boxes[i, 1] - boxes[i, 4],
                        qboxes[j, 1] - qboxes[j, 4]))

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def d3_box_overlap(boxes, qboxes, criterion=-1):
    # from .rotate_iou import rotate_iou_gpu_eval
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]],
                               qboxes[:, [0, 2, 3, 5, 6]], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


def bev_box_overlap(boxes, qboxes, criterion=-1):
    # from .rotate_iou import rotate_iou_gpu_eval
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou


@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]) + qbox_area -
                              iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps

def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]

def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=200):
    """Fast iou algorithm. this function can be used independently to do result
    analysis. Must be used in CAMERA coordinate system.

    Args:
        gt_annos (dict): Must from get_label_annos() in kitti_common.py.
        dt_annos (dict): Must from get_label_annos() in kitti_common.py.
        metric (int): Eval type. 0: bbox, 1: bev, 2: 3d.
        num_parts (int): A parameter for fast calculate algorithm.
    """
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a['name']) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a['name']) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        if metric == 0:
            gt_boxes = np.concatenate([a['bbox'] for a in gt_annos_part], 0)
            dt_boxes = np.concatenate([a['bbox'] for a in dt_annos_part], 0)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        elif metric == 1:
            loc = np.concatenate(
                [a['location'][:, [0, 2]] for a in gt_annos_part], 0)
            dims = np.concatenate(
                [a['dimensions'][:, [0, 2]] for a in gt_annos_part], 0)
            rots = np.concatenate([a['rotation_y'] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1)
            loc = np.concatenate(
                [a['location'][:, [0, 2]] for a in dt_annos_part], 0)
            dims = np.concatenate(
                [a['dimensions'][:, [0, 2]] for a in dt_annos_part], 0)
            rots = np.concatenate([a['rotation_y'] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1)
            overlap_part = bev_box_overlap(gt_boxes,
                                           dt_boxes).astype(np.float64)
        elif metric == 2:
            loc = np.concatenate([a['location'] for a in gt_annos_part], 0)
            dims = np.concatenate([a['dimensions'] for a in gt_annos_part], 0)
            rots = np.concatenate([a['rotation_y'] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1)
            loc = np.concatenate([a['location'] for a in dt_annos_part], 0)
            dims = np.concatenate([a['dimensions'] for a in dt_annos_part], 0)
            rots = np.concatenate([a['rotation_y'] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1)
            overlap_part = d3_box_overlap(gt_boxes,
                                          dt_boxes).astype(np.float64)
        else:
            raise ValueError('unknown metric')
        parted_overlaps.append(overlap_part)
        example_idx += num_part
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                   dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, total_gt_num, total_dt_num



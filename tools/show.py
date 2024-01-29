# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 17:31:42 2022

@author: why
"""

import open3d
import torch
import matplotlib
import numpy as np

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis

def main():
    points = np.fromfile("/home/macs/gzp/openPCDet/data/KITTI/velodyne_reduced/000000.bin", dtype=np.float32).reshape(-1, 4)
    ref_boxes=np.array([[ 14.7530,  -1.0668,  -0.7949,   3.7316,   1.5734,   1.5017,   5.9684],
        [  8.1338,   1.2251,  -0.8056,   3.7107,   1.5718,   1.6053,   2.8312],
        [  6.4539,  -3.8711,  -1.0125,   2.9638,   1.5000,   1.4587,   6.0006],
        [  4.0341,   2.6423,  -0.8589,   3.5184,   1.6252,   1.6226,   6.0168],
        [ 33.5379,  -7.0922,  -0.5771,   4.1590,   1.6902,   1.6740,   2.8695],
        [ 20.2419,  -8.4190,  -0.8768,   2.2857,   1.5067,   1.5971,   5.9195],
        [ 24.9979, -10.1047,  -0.9413,   3.7697,   1.6151,   1.4467,   5.9671],
        [ 55.4206, -20.1693,  -0.5863,   4.1936,   1.6783,   1.5897,   2.8009],
        [ 40.9520,  -9.8718,  -0.5903,   3.7940,   1.5752,   1.5658,   5.9509],
        [ 28.7372,  -1.6067,  -0.3582,   3.7827,   1.5546,   1.5801,   1.2517],
        [ 29.8940, -14.0270,  -0.7138,   0.7105,   0.5286,   1.8181,   1.8177],
        [ 10.5068,   5.3847,  -0.6656,   0.8203,   0.6050,   1.7170,   4.5543],
        [ 14.7198, -13.9145,  -0.7675,   0.6548,   0.6050,   1.8767,   6.3586],
        [ 40.5776,  -7.1297,  -0.4536,   0.7717,   0.6421,   1.8219,   6.3071],
        [ 18.6621,   0.2868,  -0.7225,   0.6963,   0.5903,   1.6171,   3.4939],
        [ 33.5909, -15.3372,  -0.6708,   1.5792,   0.4420,   1.6632,   5.8578],
        [ 53.6673, -16.1789,  -0.2170,   0.9555,   0.5120,   1.9663,   4.0730],
        [ 30.4546,  -3.7337,  -0.3892,   1.6604,   0.5506,   1.7268,   2.8738],
        [ 37.2168,  -6.0348,  -0.4855,   0.8860,   0.5873,   1.7859,   6.3918],
        [ 34.0845,  -4.9617,  -0.4192,   0.8911,   0.6893,   1.8796,   6.0675],
        [ 13.2934,   4.3788,  -0.5723,   1.7745,   0.5844,   1.7321,   5.5894],
        [  1.5887,   8.8918,  -0.5623,   1.7521,   0.3996,   1.6873,   6.9082],
        [  1.6363,  10.6976,  -0.4213,   0.5559,   0.5656,   1.6537,   1.1167],
        [ 10.1203,  -7.5959,  -0.8065,   1.6906,   0.5269,   1.8206,   6.0078],
        [  1.3104,  -5.3168,  -0.9996,   3.8217,   1.5819,   1.5247,   5.7200],
        [  1.9891,   6.9479,  -0.6237,   0.7172,   0.6449,   1.8667,   5.1038],
        [ 37.0710, -16.5266,  -0.6848,   1.4592,   0.5439,   1.6777,   2.5990],
        [ 18.6999,   1.1810,  -0.4766,   0.7327,   0.6436,   1.8375,   5.8503],
        [  2.6479,  17.1586,  -0.1585,   0.5904,   0.6348,   1.8937,   3.6890],
        [  0.9431,  10.5031,  -0.3420,   0.5309,   0.5733,   1.7027,   2.1916],
        [  5.7515, -12.5565,  -0.7717,   0.5685,   0.5493,   1.6204,   2.1157],
        [ 45.0186,  -7.5816,  -0.0797,   3.7895,   1.6455,   1.7168,   4.4490]])
    ref_scores=np.array([0.9654, 0.9511, 0.9037, 0.8834, 0.8346, 0.6788, 0.6594, 0.5516, 0.5041,
        0.4658, 0.3139, 0.3063, 0.2938, 0.2692, 0.2396, 0.2348, 0.2258, 0.2208,
        0.2194, 0.1883, 0.1608, 0.1559, 0.1516, 0.1449, 0.1427, 0.1358, 0.1239,
        0.1227, 0.1212, 0.1207, 0.1192, 0.1003])
    ref_labels=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 2, 3, 2, 2, 3, 3, 2, 3,
        1, 2, 3, 2, 2, 2, 2, 1])
    print(ref_scores)
    draw_scenes(points=points,ref_boxes=ref_boxes,ref_scores=ref_scores, ref_labels=ref_labels)

 
if __name__ == '__main__':
    main()

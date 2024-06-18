import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from typing import List, Optional, Tuple
from matplotlib import pyplot as plt
import os
from tqdm import tqdm

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def draw_2d_bbox(x, y, w, h, theta_rad, thickness,color='r'):
    # 计算旋转矩阵
    # theta_rad = np.radians(angel)
    R = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                   [np.sin(theta_rad), np.cos(theta_rad)]])
    
    # 未旋转时的角点坐标
    corners = np.array([[-w/2, -h/2],
                        [w/2, -h/2],
                        [w/2, h/2],
                        [-w/2, h/2]])
    
    # 应用旋转
    rotated_corners = np.dot(corners, R.T)  # 注意转置，因为这里是点乘矩阵
    rotated_corners += np.array([x, y])  # 平移到中心点位置

    # 绘制边框
    plt.plot(rotated_corners[:, 0], rotated_corners[:, 1], color=color,linewidth=thickness)
    # 连接首尾形成闭合
    plt.plot([rotated_corners[0, 0], rotated_corners[-1, 0]],
             [rotated_corners[0, 1], rotated_corners[-1, 1]], color=color,linewidth=thickness)

def visualize_lidar(
    fpath= str,
    lidar= None,
    bboxes = None,
    labels  = None,
    classes  = None,
    xlim: Tuple[float, float] = (-50, 50),
    ylim: Tuple[float, float] = (-50, 50),
    color: Optional[Tuple[int, int, int]] = None,
    radius: float = 15,
    thickness: float = 25,
) -> None:
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))
    colors = [[0,1,0],[1,0,1],[0,0,1],[1,1,0],[0,1,1]]
    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    if lidar is not None:
        plt.scatter(
            lidar[:, 0],
            lidar[:, 1],
            s=radius,
            c="white",
        )
    # print("labels:  ",labels)
    if bboxes is not None and len(bboxes) > 0:
        # coords = bboxes.corners[:, [0, 3, 7, 4, 0], :2]
        coords = bboxes
        # print("bboxes:   ",bboxes.shape)
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            # print("name:   ",name)
            color = colors[labels[index] % len(colors)]
            draw_2d_bbox(coords[index,  0],coords[index,  1],coords[index,  3],
                            coords[index,  4],coords[index,  6],thickness,
                            color)

    # print("fpath:    ",fpath)
    # mmcv.mkdir_or_exist(os.path.dirname(fpath))
    fig.savefig(
        fpath,
        dpi=10,
        facecolor="black",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def main():
    point_cloud_range = [-69.12, -39.68, 0, 69.12, 39.68, 4]   
    object_classes = [
        'car', 'truck','lockbox','ped','lock'
    ]    
    out_dir_path  = "/home/OpenPCdet_alg/perception-alg/data/lidar_images"
    if os.path.exists(out_dir_path):
        pass
    else:
        os.makedirs(out_dir_path)    
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in tqdm(enumerate(demo_dataset)):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            # print("lidar:   ",data_dict['points'][:, 1:].shape)
            # print("lidar_path:   ",os.path.join(out_dir_path, "lidar", f"{idx}.png"))
            visualize_lidar(
                os.path.join(out_dir_path, f"{idx}.png"),
                data_dict['points'][:, 1:].cpu().numpy(),
                bboxes=pred_dicts[0]['pred_boxes'].cpu(),
                labels=pred_dicts[0]['pred_labels'].cpu(),
                xlim=[point_cloud_range[d] for d in [0, 3]],
                ylim=[point_cloud_range[d] for d in [1, 4]],
                classes=object_classes,
            )

            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            # )

            # if not OPEN3D_FLAG:
            #     mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()

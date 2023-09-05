import argparse
import glob
from pathlib import Path
import os
import json

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
            print("self.sample_file_list[index]:   ",self.sample_file_list[index])
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError
        (filename,extension) = os.path.splitext(os.path.split(self.sample_file_list[index])[-1])
        input_dict = {
            'points': points,
            'frame_id': index,
            'lidar_id': int(filename),
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pointpillar.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='/home/why/mnt/github_demo/OpenPCDet/data/kitti/trainning/velodyne/000000.bin',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default="../checkpoints/pointpillar_7728.pth", help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    labels_path = "SUSTechPOINTS/data/merge_pcd_label/label2"
    map_class = {
        "1" : "Car",
        "2" : "Truck"
    }    
    if os.path.exists(labels_path):
        pass
    else:
        os.makedirs(labels_path)     
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

        for idx, data_dict in enumerate(demo_dataset):
            label_file_new = os.path.join(labels_path, str(data_dict['lidar_id'])) + '.json'

            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            json_data = {}
            json_datas = []
            for i in range(len(pred_dicts[0]['pred_labels'])):
                json_data = {
                    "obj_id": "1",
                    "obj_type": map_class[str(int(pred_dicts[0]['pred_labels'][i]))],
                    "psr": {
                    "position": {
                        "x": float(pred_dicts[0]['pred_boxes'][i][0]),
                        "y": float(pred_dicts[0]['pred_boxes'][i][1]),
                        "z": float(pred_dicts[0]['pred_boxes'][i][2])
                    },
                    "rotation": {
                        "x": 0,
                        "y": 0,
                        "z":  float(pred_dicts[0]['pred_boxes'][i][6])      
                    },
                    "scale": {
                        "x": float(pred_dicts[0]['pred_boxes'][i][3]),
                        "y": float(pred_dicts[0]['pred_boxes'][i][4]),
                        "z": float(pred_dicts[0]['pred_boxes'][i][5])
                    }
                    }
                }       
                json_datas.append(json_data)        
            with open(label_file_new, "a", encoding="utf-8") as f:
                json.dump(json_datas, f)                 

            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            # )

            # if not OPEN3D_FLAG:
            #     mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
import os 
import numpy as np
import open3d as o3d
from tqdm import tqdm
def read_pcd(filepath):
    lidar = []
    with open(filepath,'rb') as f:
        line = f.readline().strip()
        while line:
            linestr = line.split(" ")
            if len(linestr) == 4:
                linestr_convert = list(map(float, linestr))
                lidar.append(linestr_convert)
            line = f.readline().strip()
    return np.array(lidar)

def pcd2bin(pcdfolder, binfolder):
    current_path = os.getcwd()
    ori_path = os.path.join(current_path, pcdfolder)
    file_list = os.listdir(ori_path)
    des_path = os.path.join(current_path, binfolder)
    if os.path.exists(des_path):
        pass
    else:
        os.makedirs(des_path)
    for file in file_list: 
        (filename,extension) = os.path.splitext(file)
        velodyne_file = os.path.join(ori_path, filename) + '.pcd'
        pl = read_pcd(velodyne_file)
        pl = pl.reshape(-1, 4).astype(np.float32)
        velodyne_file_new = os.path.join(des_path, filename) + '.bin'
        pl.tofile(velodyne_file_new)



# pcd存储形式为二进制
def read_binary_pcd_to_numpy(pcd_file_path):
    """
    读取二进制PCD文件并转换为NumPy数组。
    
    参数:
    pcd_file_path (str): PCD文件的路径。
    
    返回:
    numpy.ndarray: 包含点云数据的NumPy数组，每行代表一个点，列依次为X, Y, Z等坐标信息。
    """
    # 使用open3d读取PCD文件
    pcd = o3d.io.read_point_cloud(pcd_file_path)
    
    # 将点云数据转换为NumPy数组
    points = np.asarray(pcd.points)
    
    return points

def convert_pcd_to_bin(pcdfolder, binfolder):
    """
    将二进制PCD文件转换为BIN文件。
    
    参数:
    pcd_file_path (str): 输入的PCD文件路径。
    bin_file_path (str): 输出的BIN文件路径。
    """
    current_path = os.getcwd()
    ori_path = os.path.join(current_path, pcdfolder)
    file_list = os.listdir(ori_path)
    des_path = os.path.join(current_path, binfolder)
    if os.path.exists(des_path):
        pass
    else:
        os.makedirs(des_path)
    for file in tqdm(file_list): 
        (filename,extension) = os.path.splitext(file)
        velodyne_file = os.path.join(ori_path, filename) + '.pcd'
        pcd = o3d.io.read_point_cloud(velodyne_file)
        # 获取点云数据的NumPy数组
        points = np.asarray(pcd.points)        
        


        intensities = np.asarray(pcd.colors)[:, 0] if pcd.has_colors() else np.zeros(points.shape[0])

        # 合并点云坐标和强度信息到一个数组中，假设i（强度）对应于颜色的R通道
        point_cloud_data = np.column_stack((points, intensities))
        # print("points shape:   ",point_cloud_data.shape)
        point_cloud_data = point_cloud_data.astype(np.float32)
        velodyne_file_new = os.path.join(des_path, filename) + '.bin'
        point_cloud_data.tofile(velodyne_file_new)
        # # 将NumPy数组的数据直接写入BIN文件（假设点云数据为3D点，即每点有3个浮点数）
        # with open(bin_file_path, 'wb') as f:
        #     f.write(points.tobytes())  # tobytes()方法将数组转换为字节流直接写入文件

#输入一个相对于python运行脚本当前目录的pcd目录，一个bin生成存放目录
#pcd为二进制
convert_pcd_to_bin("/home/dell2/OpenPCdet_alg/perception-alg/data/deep_learning_detection/side_lidar", "/home/dell2/OpenPCdet_alg/perception-alg/data/deep_learning_detection/side_lidar_bin")
# pcd为其他格式
# pcd2bin("/home/dell2/OpenPCdet_alg/perception-alg/data/deep_learning_detection/side_lidar", "/home/dell2/OpenPCdet_alg/perception-alg/data/deep_learning_detection/side_lidar_bin")

import open3d as o3d
import numpy as np 

def main():

    raw_point = np.load('OpenPCDet/data/custom/points/25.npy') #读取1.npy数据  N*[x,y,z]
    raw_point = raw_point[:, 0:3].reshape(-1, 3)
    #创建窗口对象
    vis = o3d.visualization.Visualizer()
    #设置窗口标题
    vis.create_window(window_name="kitti")
    #设置点云大小
    vis.get_render_option().point_size = 1
    #设置颜色背景为黑色
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    #创建点云对象
    pcd=o3d.open3d.geometry.PointCloud()
    #将点云数据转换为Open3d可以直接使用的数据类型
    pcd.points= o3d.open3d.utility.Vector3dVector(raw_point)
    #设置点的颜色为白色
    pcd.paint_uniform_color([1,1,1])
    #将点云加入到窗口中
    vis.add_geometry(pcd)

    vis.run()
    vis.destroy_window()

    
if __name__=="__main__":
    main()




import os 
import numpy as np
import json
import shutil 
import math
from PIL import Image  

class_map = {"Truck":"Truck","Car":"Car"}

def mkdir(file_path):
    if os.path.exists(file_path):
        pass
    else:
        os.makedirs(file_path) 

def read_pcd(filepath):
    lidar = []
    with open(filepath,'r') as f:
        line = f.readline().strip()
        while line:
            linestr = line.split(" ")
            if len(linestr) == 4:
                linestr_convert = list(map(float, linestr))
                lidar.append(linestr_convert)
            line = f.readline().strip()
    return np.array(lidar)

def pcd2npy(pcdfolder, npyfolder):
    ori_path = pcdfolder
    file_list = os.listdir(ori_path)
    des_path = npyfolder
    if os.path.exists(des_path):
        pass
    else:
        os.makedirs(des_path)
    for file in file_list: 
        (filename,extension) = os.path.splitext(file)
        velodyne_file = os.path.join(ori_path, filename) + '.pcd'
        pl = read_pcd(velodyne_file)
        pl = pl.reshape(-1, 4).astype(np.float32)
        npy_file_new = os.path.join(des_path, filename) + '.npy'
        np.save(npy_file_new, pl)

#输入一个相对于python运行脚本当前目录的pcd目录，一个bin生成存放目录
def pcd2bin(pcdfolder, binfolder):
    ori_path = pcdfolder
    file_list = os.listdir(ori_path)
    des_path = binfolder
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


def json2txt(json_paths,txt_paths):
    json_path_list = os.listdir(json_paths)
    if os.path.exists(txt_paths):
        pass
    else:
        os.makedirs(txt_paths)    
    for json_path in json_path_list:
        # print(json_path)
        (filename,extension) = os.path.splitext(json_path)
        json_path_new = json_paths + "/" +json_path
        txt_file_new = os.path.join(txt_paths, filename) + '.txt'
        with open(json_path_new,'r',encoding='utf8')as fp:
            json_data = json.load(fp)
            for i in range(len(json_data)):
                obj_id = json_data[i]["obj_id"]
                obj_type = json_data[i]["obj_type"]
                position_x = json_data[i]["psr"]["position"]["x"]
                position_y = json_data[i]["psr"]["position"]["y"]
                position_z = json_data[i]["psr"]["position"]["z"]
                scale_x = json_data[i]["psr"]["scale"]["x"]
                scale_y = json_data[i]["psr"]["scale"]["y"]
                scale_z = json_data[i]["psr"]["scale"]["z"]
                rotation_x = json_data[i]["psr"]["rotation"]["x"]
                rotation_y = json_data[i]["psr"]["rotation"]["y"]
                rotation_z = json_data[i]["psr"]["rotation"]["z"]
                alpha = round(rotation_z -  math.atan(position_x / position_y),2)
                line = class_map[obj_type] + " " + "0.00" + " " + "0" + " " +  str(alpha)  + " " + "0.0" + " " + "0.0" + " " + "0.0"+ " " + "0.0" + " "  \
                                    + str(scale_z) + " " + str(scale_y) + " " + str(scale_x)   + " "  + str(position_x)   + " " + str(position_y) + " " + str(position_z) + " " + str(rotation_z) +  "\n"
                with open(txt_file_new,"a") as f:
                    f.write(line)

def calib2calib(json_paths,calib_path):
    json_path_list = os.listdir(json_paths)
    for json_path in json_path_list:
        # print(json_path)
        (filename,extension) = os.path.splitext(json_path)
        json_path_new = json_paths + "/" +json_path
        txt_file_new = os.path.join(calib_path, filename) + '.txt'
        P0 = "P0:" + " " + "1.0" + " " +  "0" + " " + "0" + " " + "0" + " " + "0" + " " + "1.0" + " " + "0.0" + " " + "0.0" + " " + "0.0" + " " + "0.0" + " " + "1.0" + " " + "1.0" +  "\n"
        P1 = "P1:" + " " +  "1.0" + " " + "0" + " " + "0" + " " + "0" + " " + "0.0"  + " " +   "1.0" + " " + "0.0" + " " + "0.0" + " " + "0.0" + " " + "0.0" + " " + "1.0" + " " + "1.0" +  "\n"
        P2 = "P2:" + " " +    "1.0" + " " + "0" + " " + "0" + " " + "0" + " " + "0.0" + " " +   "1.0" + " " + "0.0" + " " + "0.0" + " " + "0.0" + " " + "0.0" + " " + "1.0" + " " + "1.0" +  "\n"
        P3 = "P3:" + " " +    "1.0" + " " + "0" + " " + "0" + " " + "0" + " " + "0.0" + " " +   "1.0" + " " + "0.0" + " " + "0.0" + " " + "0.0" + " " + "0.0" + " " + "1.0" + " " + "1.0" +  "\n"
        R0_rect = "R0_rect:" + " " +"1.0" + " " +  "0" + " " + "0" + " " + "0" + " " + "0" + " " + "1.0" + " " + "0.0" + " " + "0.0" + " " + "1.0"+  "\n"
        Tr_velo_to_cam = "Tr_velo_to_cam:" + " " +"1.0" + " " +  "0" + " " + "0" + " " + "0" + " " + "0" + " " + "1.0" + " " + "0.0" + " " + "0.0" + " " + "0.0" + " " + "0.0" + " " + "1.0" + " " + "1.0" +  "\n"
        Tr_imu_to_velo ="Tr_imu_to_velo:" + " " + "1.0" + " " +  "0" + " " + "0" + " " + "0" + " " + "0" + " " + "1.0" + " " + "0.0" + " " + "0.0" + " " + "0.0" + " " + "0.0" + " " + "1.0" + " " + "1.0" +  "\n"
        with open(txt_file_new,"a") as f:
            f.write(P0)
            f.write(P1)
            f.write(P2)
            f.write(P3)
            f.write(R0_rect)
            f.write(Tr_velo_to_cam)
            f.write(Tr_imu_to_velo)

def img2img(json_paths,image_2_path):
    json_path_list = os.listdir(json_paths)
    for json_path in json_path_list:
        # print(json_path)
        (filename,extension) = os.path.splitext(json_path)
        json_path_new = json_paths + "/" +json_path
        img_file_new = os.path.join(image_2_path, filename) + '.png'
        
        # 创建一个1920x1080的空图片，所有像素设置为白色  
        width = 1920  
        height = 1080  
        image = Image.new('RGB', (width, height), color='white')  
        # 保存为PNG格式  
        image.save(img_file_new)

def txt2json(txt_paths,json_paths):
    txt_path_list = os.listdir(txt_paths)
    if os.path.exists(json_paths):
        pass
    else:
        os.makedirs(json_paths)       
    for txt_path in txt_path_list:        
        (filename,extension) = os.path.splitext(txt_path)
        txt_path_new = txt_paths + "/" +txt_path
        json_file_new = os.path.join(json_paths, filename) + '.json'
        with open(txt_path_new, 'r') as file:
            lines = file.readlines()
            print(f"文件 {filename} 有 {len(lines)} 个框")

            label_file_new = filename.replace(".txt", ".json")
            json_data = {}
            json_datas = []
            map_class = {
                "1" : "Car",
                "2" : "Truck"
            }
            for i, line in enumerate(lines):
                json_data = {
                    "obj_id": "1",
                    "obj_type": map_class[str(int(line.split()[7]))],
                    "psr": {
                    "position": {
                        "x": float(line.split()[0]),
                        "y": float(line.split()[1]),
                        "z": float(line.split()[2])
                    },
                    "rotation": {
                        "x": 0,
                        "y": 0,
                        "z":  float(line.split()[6])      
                    },
                    "scale": {
                        "x": float(line.split()[3]),
                        "y": float(line.split()[4]),
                        "z": float(line.split()[5])
                    }
                    }
                }       
                json_datas.append(json_data)        
        with open(json_file_new, "a", encoding="utf-8") as f:
            json.dump(json_datas, f)          

def split_train_test(data,test_ratio):
    #设置随机数种子，保证每次生成的结果都是一样的
    np.random.seed(42)
    #permutation随机生成0-len(data)随机序列
    shuffled_indices = np.random.permutation(len(data))
    #test_ratio为测试集所占的半分比
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    #iloc选择参数序列中所对应的行
    return data[train_indices],data[test_indices]


def gen_train_test_name(pcdfolder,ImageSets_path,test_ratio):
    file_list = os.listdir(pcdfolder)
    data = []
    for file in file_list: 
        (filename,extension) = os.path.splitext(file)
        data.append(filename)
    data_np = np.array(data)
    train,test = split_train_test(data_np,test_ratio)
    train_path = ImageSets_path + "/" +  "train.txt"
    test_path = ImageSets_path + "/" +  "val.txt"
    for i in train.tolist():
        with open(train_path,"a") as f:
            f.write(str(i) + "\n")    
    for j in test.tolist():
        with open(test_path,"a") as f:
            f.write(str(j) + "\n")                

#custom、ImageSets、points、labels
#输入custom绝对地址
def gen_custom_data(custom_path,pcdfolder,labelsfolder,test_ratio):
    ImageSets_path = custom_path + "/" + "ImageSets"
    if os.path.exists(ImageSets_path):
        pass
    else:
        os.makedirs(ImageSets_path)        
    training_path = custom_path + "/" + "training"    
    if os.path.exists(training_path):
        pass
    else:
        os.makedirs(training_path)       
    gen_train_test_name(pcdfolder,ImageSets_path,test_ratio)
    points_path = training_path + "/" + "velodyne"
    if os.path.exists(points_path):
        pass
    else:
        os.makedirs(points_path)       
    pcd2bin(pcdfolder,points_path)    

    labels_path = training_path + "/" + "label_2"
    if os.path.exists(labels_path):
        pass
    else:
        os.makedirs(labels_path)           
    json2txt(labelsfolder,labels_path)    

    image_2_path =  training_path + "/" + "image_2"
    if os.path.exists(image_2_path):
        pass
    else:
        os.makedirs(image_2_path)     
    img2img(labelsfolder,image_2_path)    

    calib_path =  training_path + "/" + "calib"
    if os.path.exists(calib_path):
        pass
    else:
        os.makedirs(calib_path)     
    calib2calib(labelsfolder,calib_path)        

def merge_pcd_label(pcd_label_folder):
    file_list = os.listdir(pcd_label_folder)
    merge_pcd_label_path = pcd_label_folder + "/merge_pcd_label"
    mkdir(merge_pcd_label_path)
    merge_path_pcd = merge_pcd_label_path + "/lidar"
    merge_path_label = merge_pcd_label_path + "/label"
    mkdir(merge_path_pcd)
    mkdir(merge_path_label)    
    # print("file_list:       ",file_list)
    for file in file_list:
        ori_path = pcd_label_folder + "/" + file
        print("ori_path:  " ,ori_path)
        pcdfolder = ori_path + "/lidar"
        labelsfolder = ori_path + "/label"
        for file_tmp in os.listdir(pcdfolder):
            file_tmp_pcdabspath = pcdfolder + "/" + file_tmp
            shutil.copy(file_tmp_pcdabspath,merge_path_pcd)
        for file_tmp in os.listdir(labelsfolder):
            file_tmp_labelabspath = labelsfolder + "/" + file_tmp
            shutil.copy(file_tmp_labelabspath,merge_path_label)                  
            # shutil.move(file_tmp_pcdabspath,merge_path_pcd)
            # shutil.move(file_tmp_labelabspath,merge_path_label)
    return merge_path_pcd,merge_path_label


def check_pcd_label(pcdfolder ,labelsfolder):
    file_list = os.listdir(pcdfolder)
    pcd = []
    for file in file_list: 
        (filename,extension) = os.path.splitext(file)
        pcd.append(filename)

    file_list = os.listdir(labelsfolder)
    label = []
    for file in file_list: 
        (filename,extension) = os.path.splitext(file)
        label.append(filename)
    pcd = set(pcd)    
    label = set(label)
    print("len(pcd):     ",len(pcd))
    print("len(label):     ",len(label))
    non_intersection_1 = pcd - label    
    non_intersection_2 = label - pcd
    for file in non_intersection_1:
        file_path = pcdfolder + "/" + file + ".pcd"
        if os.path.exists(file_path):  
            try:  
                os.remove(file_path)  
                print("文件已成功删除:  ",file_path)  
            except OSError as e:  
                print(f"删除文件时出错: {e}")        
        else:
            continue        
    for file in non_intersection_2:
        file_path = labelsfolder + "/" + file + ".json"
        if os.path.exists(file_path):  
            try:  
                os.remove(file_path)  
                print("文件已成功删除: ",file_path)  
            except OSError as e:  
                print(f"删除文件时出错: {e}")             
        else:
            continue           
    print("check_pcd_label sucess!")   

if __name__ == '__main__':
    pcd_label_folder = "/home/why/mnt/github_demo/labelcloud/SUSTechPOINTS/SUSTechPOINTS/data"
    # pcd_label_folder = "/home/why/mnt/github_demo/labelcloud/SUSTechPOINTS/SUSTechPOINTS/data_test"
    pcdfolder ,labelsfolder = merge_pcd_label(pcd_label_folder)
    check_pcd_label(pcdfolder ,labelsfolder)
    # custom_path = "/home/why/mnt/github_demo/labelcloud/SUSTechPOINTS/SUSTechPOINTS/data/custom"
    custom_path = "/home/why/mnt/github_demo/labelcloud/SUSTechPOINTS/SUSTechPOINTS/data/kitti_custom"
    test_ratio =   0.1
    # test_ratio =   0
    gen_custom_data(custom_path,pcdfolder,labelsfolder,test_ratio)

     # pcdfolder = "/home/why/mnt/github_demo/labelcloud/SUSTechPOINTS/SUSTechPOINTS/data/module1/lidar"
    # labelsfolder = "/home/why/mnt/github_demo/labelcloud/SUSTechPOINTS/SUSTechPOINTS/data/module1/label"   
# json2txt("/home/why/mnt/github_demo/labelcloud/SUSTechPOINTS/SUSTechPOINTS/data/module1/label",
#          "/home/why/mnt/github_demo/labelcloud/SUSTechPOINTS/SUSTechPOINTS/data/txt_test")




































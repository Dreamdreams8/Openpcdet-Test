# 获取每个类别的评价lwh，以此来设置anchor

import os

if __name__ == "__main__":
    label_path = "/home/why/mnt/github_demo/OpenPCDet/data/custom/labels"
    label_list = os.listdir(label_path)
    # l w h
    P_counts = 0
    Car = [0.0, 0.0, 0.0]
    B_counts = 0
    Truck = [0.0, 0.0, 0.0]

    for label_name in label_list:
        label_file = os.path.join(label_path, label_name)
        with open(label_file, 'r') as f:
            data = f.readlines()
            for line in data:
                temp_list = line.split(" ")
                cls_name = temp_list[-1][:-1]
                if cls_name == "Car":
                    Car[0] += float(temp_list[3])
                    Car[1] += float(temp_list[4])
                    Car[2] += float(temp_list[5])
                    P_counts += 1
                else:
                    Truck[0] += float(temp_list[3])
                    Truck[1] += float(temp_list[4])
                    Truck[2] += float(temp_list[5])
                    B_counts += 1

    print(f"C l{Car[0]/P_counts} w{Car[1]/P_counts} h{Car[2]/P_counts}")
    print(f"T l{Truck[0]/B_counts} w{Truck[1]/B_counts} h{Truck[2]/B_counts}")

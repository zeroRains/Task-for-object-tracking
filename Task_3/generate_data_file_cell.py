import numpy as np
from scipy.io import loadmat
import cv2
import os
from Task_2.utils import normalization
from tqdm import tqdm

path = r"E:\datasets\CRCHistoPhenotypes_2016_04_28\Detection"
save_path = 'cell_path.txt'
dir_path = os.listdir(path)


def generate_gauss_map(img, gt, sigma):
    h, w = img.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    # 获取目标的中心
    center_x = gt[0]
    center_y = gt[1]
    # 计算距离
    dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (sigma * 2)
    # 映射到0~1之中
    response = np.exp(-dist)
    # 归一化
    return normalization(response)


with open(save_path, 'w+') as f:
    for dir_i in tqdm(dir_path):
        if "img" not in dir_i:
            continue
        new_path = os.path.join(path, dir_i)
        img_path = os.path.join(new_path, f"{dir_i}.bmp")
        label_path = os.path.join(new_path, f"{dir_i}_detection.mat")
        gauss_path = os.path.join(new_path, f'{dir_i}_gauss.png')
        idxes = loadmat(label_path)['detection'].astype(np.int)
        img = cv2.imread(img_path, -1)
        target = np.zeros(img.shape[:2])
        for r, c in idxes:
            gauss_gt = generate_gauss_map(target, [r, c], sigma=2)
            target += gauss_gt
            # 这个是细胞核检测论文中标签的制作方法
            # target[c - 1, r - 1] = 1
        target = np.clip(target, 0., 1.)
        target *= 255
        cv2.imwrite(gauss_path, target)
        f.write(f'{img_path} {gauss_path}\n')
print("finished")

import cv2
import os
import pandas as pd
import numpy as np
from Task_2.utils import normalization
from tqdm import tqdm

w, h = (100, 100)  # 滤波器大小
cnt = 0

path = r'E:\datasets\segmentation_WBC'  # 数据路径
path_file = 'data_path.txt'  # 输出文件名称


def get_file_txt():
    data1 = pd.read_csv(os.path.join(path, r'Class Labels of Dataset 1.csv'))
    data2 = pd.read_csv(os.path.join(path, r'Class Labels of Dataset 2.csv'))
    images_id = np.array(data1[data1[r'class label'] == 1][r'image ID'])
    use_data = r'Dataset 1'

    with open(path_file, 'w+') as f:
        for i in images_id:
            path_img = os.path.join(path, use_data, f"{str(i).zfill(3)}.bmp")
            f.write(f'{path_img}\n')


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


def generate_gauss_label(sigma=2):
    with open(path_file, 'r+') as f:
        files = f.readlines()
        for i in tqdm(files):
            now = i.strip()
            img = cv2.imread(now, -1)
            img = cv2.resize(img, (w, h))
            box = cv2.selectROI('demo', img, False, False)
            box = np.array(box).astype(np.int64)
            cv2.destroyAllWindows()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = gray.astype(np.float32)
            gauss_gt = generate_gauss_map(gray, box, sigma)
            gauss_gt *= 255
            cv2.imwrite(now.replace(".bmp", "_gauss.png"), gauss_gt)


if __name__ == '__main__':
    # get_file_txt() # 生成图像文件(data_path.txt)
    generate_gauss_label()  # 标记目标中心点并生成高斯图

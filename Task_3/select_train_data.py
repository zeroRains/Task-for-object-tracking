import cv2
import os
import pandas as pd
import numpy as np
from Task_2.utils import normalization
from tqdm import tqdm




def get_file_txt(path, path_file):
    """

    :param path: 数据路径
    :param path_file: 输出文件名称
    :return:
    """
    data1 = pd.read_csv(os.path.join(path, r'Class Labels of Dataset 1.csv'))
    data2 = pd.read_csv(os.path.join(path, r'Class Labels of Dataset 2.csv'))
    images_id = np.array(data1[data1[r'class label'] == 1][r'image ID'])
    use_data = r'Dataset 1'

    with open(path_file, 'w+') as f:
        for i in images_id:
            path_img = os.path.join(path, use_data, f"{str(i).zfill(3)}.bmp")
            f.write(f'{path_img}\n')


def generate_file_txt(path, path_file):
    files = os.listdir(path)
    with open(path_file, 'w+') as f:
        for i in files:
            path_img = os.path.join(path, i)
            f.write(f'{path_img}\n')


def generate_gauss_map(img, gt, sigma):
    h, w = img.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    # 获取目标的中心
    center_x = gt[0] + 0.5 * gt[2]
    center_y = gt[1] + 0.5 * gt[3]
    # 计算距离
    dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (sigma * 2)
    # 映射到0~1之中
    response = np.exp(-dist)
    # 归一化
    return normalization(response)


def generate_gauss_label(path_file, sigma=2):
    with open(path_file, 'r+') as f:
        files = f.readlines()
        for i in tqdm(files):
            now = i.strip()
            img = cv2.imread(now, -1)
            box = cv2.selectROI('demo', img, False, False)
            box = np.array(box).astype(np.int64)
            cv2.destroyAllWindows()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = gray.astype(np.float32)
            gauss_gt = generate_gauss_map(gray, box, sigma)
            gauss_gt *= 255
            cv2.imwrite(now.replace(".bmp", "_gauss.png"), gauss_gt)


def generate_multi_gauss_label(path_file, sigma=2):
    with open(path_file, 'r+') as f:
        files = f.readlines()
        for i in tqdm(files):
            now = i.strip()
            img = cv2.imread(now, -1)
            boxes = []
            while True:
                box = cv2.selectROI('demo', img, False, False)
                box = np.array(box).astype(np.int64)
                k = cv2.waitKey(0)
                cv2.destroyAllWindows()
                if k == 27:
                    break
                boxes.append(list(box))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = gray.astype(np.float32)
            gauss_map = np.zeros(gray.shape)
            for box in boxes:
                gauss_gt = generate_gauss_map(gray, box, sigma)
                gauss_map += gauss_gt
            gauss_map = np.clip(gauss_map, 0., 1.)
            gauss_map *= 255
            cv2.imwrite(now.replace(".png", "_gauss.png"), gauss_map)


if __name__ == '__main__':
    # get_file_txt(r'E:\datasets\segmentation_WBC','data_path.txt') # 生成图像文件(细胞,data_path.txt)
    # generate_gauss_label('data_path.txt')  # 标记目标中心点并生成高斯图(细胞,单一目标)
    # generate_file_txt("../source/car", "car_path.txt")  # 生成图像文件（车辆，car_path.txt）
    generate_multi_gauss_label("car_path.txt")  # 标记目标中心并生成高斯图(车辆,多个目标)

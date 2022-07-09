from Task_2.utils import pre_process, normalization
import cv2
import numpy as np
from tqdm import tqdm


def pre_process_n(img):
    """
    批量预处理，按照MOSSE的说法，是用来减轻边界效应
    :param img: 批量图像
    :return: 预处理后的图像级
    """
    # 获取这批图像的数量，高度，宽度
    n, h, w = img.shape
    # 首先是log函数，有助于低对比度光照情况
    img = np.log(img + 1)
    # 图像的维度是NxHxW，这里主要是做一个标准化，axis=(1,2)指的是求张量在(1,2)维度的均值，结果应该是Nx1
    # 然后后面的resize把他变成3维的,方便利用广播机制与img进行计算
    img = (img - img.mean(axis=(1, 2)).reshape(-1, 1, 1)) / (img.std(axis=(1, 2)).reshape(-1, 1, 1) + 1e-5)
    # 获得余弦窗
    win_col = np.hanning(w)
    win_row = np.hanning(h)
    col, row = np.meshgrid(win_col, win_row)
    window = col * row
    # 余弦窗与图像相乘
    return img * window


class Detector:
    """
    相关滤波检测器类
    """

    def __init__(self, pretrain_num=128, h=100, w=100, show_in_window=False):
        """
        构造函数
        :param pretrain_num:预训练轮数
        :param h:初始化时，相关滤波器的高
        :param w:初始化时，相关滤波器的宽
        :param show_in_window: 是否使用opencv的方式显示预测结果
        """
        self.pretrain_num = pretrain_num
        self.cell = None  # 相关滤波器
        self.cell_h = h
        self.cell_w = w
        self.show_in_windows = show_in_window

    def train(self, path):
        """
        训练相关滤波器
        :param path: 存储训练图像的文件路径（可以使用select_train_data.py生成）
        """
        imgs = []  # 图像集合
        gts = []  # ground-truth集合
        with open(path, 'r+') as f:
            files = f.readlines()
            for i in files:
                now = i.strip()
                # 读取图像，灰度化，resize成相关滤波器的大小，为方便计算转化成浮点数
                img = cv2.imread(now, -1)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (self.cell_w, self.cell_h))
                gray = gray.astype(np.float32)
                # 读取ground-truth，应该是浮点数的响应值，但是在保存的时候只能保存uin8，所以乘了个255
                # 在这里使用的时候要除以255
                gt = cv2.imread(now.replace('.bmp', '_gauss.png'), -1)
                gt = gt.astype(np.float32)
                gt /= 255.0
                # 数据增强
                imgs.append(gray)
                gts.append(gt)
                # 水平镜像翻转
                imgs.append(cv2.flip(gray, 0))
                gts.append(cv2.flip(gt, 0))
                # 垂直镜像翻转
                imgs.append(cv2.flip(gray, 1))
                gts.append(cv2.flip(gt, 1))
        # 向量化处理
        train_img = np.stack(imgs, axis=0)
        train_gt = np.stack(gts, axis=0)
        # 转化到傅里叶域进行使用
        G = np.fft.fft2(train_gt)
        # 图像预处理
        fi = pre_process_n(train_img)
        # MOSSE滤波器生成公式
        Ai = G * np.conjugate(np.fft.fft2(fi))
        Bi = np.fft.fft2(train_img) * np.conjugate(np.fft.fft2(train_img))
        # 滤波器生成预训练
        for _ in tqdm(range(self.pretrain_num)):
            fi = pre_process_n(train_img)
            Ai = Ai + G * np.conjugate(np.fft.fft2(fi))
            Bi = Bi + np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))
        # 存储生成的滤波器
        self.cell = (Ai / Bi).mean(axis=0)
        print("finished pretrain!")

    def run(self, img_path):
        """
        检测一张图像（目前一张图像只支持一个目标检测）
        :param img_path:图像的路径
        """
        # 读取一张图像，变成灰度图，resize成滤波器大小，浮点数转换
        img = cv2.imread(img_path, -1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.cell_w, self.cell_h))
        gray = gray.astype(np.float32)
        # 图像预处理
        fi = pre_process(gray)
        # 生成相应图
        Gi = np.fft.fft2(fi) * self.cell
        # 相应图映射回空间域
        gi = np.fft.ifft2(Gi)
        # 寻找峰值的坐标
        h, w = np.where(gi == gi.max())
        # 将坐标映射回原图
        h, w = int(1.0 * h * img.shape[0] / gray.shape[0]), int(1.0 * w * img.shape[1] / gray.shape[1])
        # 确定以w，h为中心画的园半径是多少
        r = int(30.0 * img.shape[0] / gray.shape[0])
        # 圈出目标
        cv2.circle(img, (w, h), 1, (0, 0, 255), 15)
        cv2.circle(img, (w, h), r, (255, 0, 0), 5)
        # 显示检测结果
        if self.show_in_windows:
            cv2.imshow('result', img)
            cv2.imshow('correlation map',
                       cv2.resize((normalization(gi.real) * 255.0).astype(np.uint8), (img.shape[1], img.shape[0])))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # 保存检测结果
        cv2.imwrite("result.jpg", img)
        print("get the result!")

    def load(self, path):
        """
        从文件中加载滤波器
        :param path:保存滤波器的文件路径
        """
        self.cell = np.load(path)
        print("correlation filter(H*) load successfully！")

    def save(self, path):
        """
        将滤波器参数保存为文件
        :param path:保存滤波器的文件路径
        """
        np.save(path, self.cell)
        print("correlation filter(H*) model save successfully!")


if __name__ == '__main__':
    detector = Detector(show_in_window=False)  # 实例化图像
    # detector.train('data_path.txt')  # 训练相关滤波器
    # detector.save('./model.npy')  # 保存滤波器成文件
    detector.load('./model.npy')  # 加载滤波器文件
    detector.run('../source/cell.bmp')  # 检测相似物体

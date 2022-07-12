import numpy as np
import cv2


# 归一化
def normalization(data):
    return (data - data.min()) / (data.max() - data.min())


# 生成高斯峰
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


# 预处理，解决边界效应
def pre_process(img):
    # 这个操作对应论文Preprossing部分
    h, w = img.shape
    # 首先是log函数，有助于低对比度光照情况
    img = np.log(img + 1)
    # 标准化
    img = (img - img.mean()) / (img.std() + 1e-5)
    # 获取余弦窗
    win_col = np.hanning(w)
    win_row = np.hanning(h)
    col, row = np.meshgrid(win_col, win_row)
    window = col * row
    # 将余弦窗与图片直接相乘
    return img * window


# 仿射变换，随机翻转（参考代码提供的函数，因为存在一点问题，所以没有使用）
def random_warp(img):
    a = -180 / 16
    b = 180 / 16
    r = a + (b - a) * np.random.uniform()
    # rotate the image...
    # 图像旋转（参数：旋转中心，旋转角度，缩放比例）
    matrix_rot = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), r, 1)
    # 仿射变换，他这里的第一个参数好像是做了一反色的操作（仿射变换的图像，仿射变换矩阵，然后是仿射变换后的尺寸）
    img_rot = cv2.warpAffine(np.uint8(img * 255), matrix_rot, (img.shape[1], img.shape[0]))
    # 将旋转过并且进行仿射变换的图像变成浮点数，转化成0~1之间的数
    img_rot = img_rot.astype(np.float32) / 255
    return img_rot


# 预训练生成滤波器
def pretrain(img, G, pretrain_num, lr=0.125):
    h, w = G.shape
    # 预处理(用来解决目标的不连续问题)
    fi = pre_process(cv2.resize(img, (w, h)))
    # 计算
    Ai = G * np.conjugate(np.fft.fft2(fi))
    Bi = np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))
    for _ in range(pretrain_num):
        fi = pre_process(img)
        Ai = (1 - lr) * Ai + lr * G * np.conjugate(np.fft.fft2(fi))
        Bi = (1 - lr) * Bi + lr * np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))
    return Ai, Bi

# 就算PSR
def cal_psr(g):
    """
    论文中的Peak to Sidelobe Ratio
    g是相关性输出
    :return:
    """
    h, w = g.shape
    # 峰值
    peak = g.max()
    sidelobe = []
    x, y = np.where(g == peak)
    # sidelobe是除开峰值位置周围11x11窗口的像素之外的所有像素（论文原文）
    min_x, min_y = max(0, x[0] - 5), max(0, y[0] - 5)
    max_x, max_y = min(h, x[0] + 5), min(w, y[0] + 5)
    for i in range(h):
        for j in range(w):
            if min_x <= i <= max_x and min_y <= j <= max_y:
                continue
            else:
                sidelobe.append(g[i][j])
    sidelobe = np.array(sidelobe)
    psr = (peak - sidelobe.mean()) / sidelobe.std()
    print(psr.real)
    return psr.real

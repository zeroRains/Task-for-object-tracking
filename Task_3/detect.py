from Task_2.utils import normalization, pre_process
import cv2
import numpy as np
from tqdm import tqdm
import os


def pre_process_n(img):
    n, h, w = img.shape
    img = np.log(img + 1)
    img = (img - img.mean(axis=(1, 2)).reshape(-1, 1, 1)) / (img.std(axis=(1, 2)).reshape(-1, 1, 1) + 1e-5)
    win_col = np.hanning(w)
    win_row = np.hanning(h)
    col, row = np.meshgrid(win_col, win_row)
    window = col * row
    return img * window


class Detector:
    def __init__(self, lr=0.125, sigma=2, pretrain_num=128):
        self.lr = lr
        self.sigma = sigma
        self.pretrain_num = pretrain_num
        self.cell = None

    def train(self, path):
        files = os.listdir(path)
        imgs = []
        for i in files:
            img = cv2.imread(os.path.join(path, i), -1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = gray.astype(np.float32)
            imgs.append(gray)
            imgs.append(cv2.flip(gray, 0))  # 水平镜像翻转
            imgs.append(cv2.flip(gray, 1))  # 垂直镜像翻转
        res = np.stack(imgs, axis=0)  # 将所有图像合成一个向量
        n, h, w = res.shape
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        center_x = w // 2
        center_y = h // 2
        dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (self.sigma ** 2)
        response = normalization(np.exp(-dist))
        gauss_maps = np.expand_dims(response, axis=0).repeat(n, axis=0)  # 生成这个向量的高斯图
        G = np.fft.fft2(gauss_maps)  # 转化到傅里叶域进行使用
        fi = pre_process_n(res)
        Ai = G * np.conjugate(np.fft.fft2(fi))
        Bi = np.fft.fft2(res) * np.conjugate(np.fft.fft2(res))
        for _ in tqdm(range(self.pretrain_num)):
            fi = pre_process_n(res)
            Ai = (1 - self.lr) * Ai + self.lr * G * np.conjugate(np.fft.fft2(fi))
            Bi = (1 - self.lr) * Bi + self.lr * np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))
            if np.isnan((Ai / Bi).mean()):
                print("233")
        self.cell = np.fft.ifft2(np.conjugate(Ai / Bi))
        self.cell = self.cell.real.mean(axis=0)
        print("finished pretrain!")

    def run(self, img_path):
        img = cv2.imread(img_path, -1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ho, wo = gray.shape
        gray = gray.astype(np.float32)
        gray = pre_process(gray)
        h, w = self.cell.shape
        fi = np.fft.fft2(cv2.resize(gray, (w, h)))
        h = np.conjugate(np.fft.fft2(self.cell))
        G = fi * h
        g = np.fft.ifft2(G).real
        g = normalization(g) * 255
        g = cv2.resize(g, (wo, ho)).astype(np.uint8)
        cv2.imshow('relation map', g)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# def run(self):
#     img = cv2.imread(self.path, -1)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = gray.astype(np.float32)
#     # 1. 生成相关滤波器H
#     # y,x,w,h，获取目标物体
#     box = cv2.selectROI('demo', img, False, False)
#     print(box)
#     cv2.destroyAllWindows()
#     # 获取目标框
#     box = np.array(box).astype(np.int64)
#     # 初始化高斯图
#     gauss_map = generate_gauss_map(gray, box, self.sigma)
#     # 取出目标的高斯图和原图
#     gi = gauss_map[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
#     fi = gray[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
#     # 高斯图转入傅里叶域
#     G = np.fft.fft2(gi)
#     # 生成Ai,Bi
#     Ai, Bi = pretrain(fi, G, self.pretrain_num)
#     # 生成滤波器H
#     Ai = self.lr * Ai
#     Bi = self.lr * Bi
#     H = Ai / Bi
#     pos = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]]).astype(np.int64)
#     cv2.rectangle(img, (pos[0], pos[1]), (pos[2], pos[3]), (0, 255, 0), 2)
#
#     # 2. 获取在傅里叶域中经过滤波器H滤波的目标相关图
#     f_target = gray[pos[1]:pos[3], pos[0]:pos[2]]
#     f_target = pre_process(cv2.resize(f_target, (box[2], box[3])))
#     g_target = H * np.fft.fft2(f_target)
#
#     # 3. 使用滑动窗法计算图像中每一个像素的相关图，然后与g_target进行对比
#     fs = np.lib.stride_tricks.sliding_window_view(gray, (box[3], box[2]))
#     x, y, h, w = fs.shape
#     print((x, y))
#     for i in tqdm(range(x)):
#         for j in range(y):
#             if pos[1] <= i <= pos[3] and pos[0] <= j <= pos[2]:
#                 continue
#             fi = pre_process(cv2.resize(fs[i][j], (box[2], box[3])))
#             gi = H * np.fft.fft2(fi)
#             if abs(gi.max().real - g_target.max().real) < 0.00075:
#                 xc, yc = np.where(gi == gi.max())
#                 # 获得左上角的坐标
#                 xl = i + int(xc.mean() - gi.shape[0] / 2)
#                 yl = j + int(yc.mean() - gi.shape[1] / 2)
#                 print("233")
#                 # 获取右下角的坐标
#                 xr = np.clip(xl + box[3], 0, gray.shape[0])
#                 yr = np.clip(yl + box[2], 0, gray.shape[1])
#                 cv2.rectangle(img, (yl, xl), (yr, xr), (0, 225, 0), 2)
#     cv2.imwrite('result.jpg', img)


if __name__ == '__main__':
    detector = Detector()
    detector.train('../source/parking/small')
    detector.run('../source/1.jpg')

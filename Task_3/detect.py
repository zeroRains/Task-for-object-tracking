from Task_2.utils import *
import cv2
import numpy as np
from tqdm import tqdm


class Detector:
    def __init__(self, path, lr=0.125, sigma=100, pretrain_num=128):
        self.path = path
        self.lr = 0.125
        self.sigma = sigma
        self.pretrain_num = pretrain_num

    def run(self):
        img = cv2.imread(self.path, -1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32)
        # 1. 生成相关滤波器H
        # y,x,w,h，获取目标物体
        box = cv2.selectROI('demo', img, False, False)
        print(box)
        cv2.destroyAllWindows()
        # 获取目标框
        box = np.array(box).astype(np.int64)
        # 初始化高斯图
        gauss_map = generate_gauss_map(gray, box, self.sigma)
        # 取出目标的高斯图和原图
        gi = gauss_map[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        fi = gray[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        # 高斯图转入傅里叶域
        G = np.fft.fft2(gi)
        # 生成Ai,Bi
        Ai, Bi = pretrain(fi, G, self.pretrain_num)
        # 生成滤波器H
        Ai = self.lr * Ai
        Bi = self.lr * Bi
        H = Ai / Bi
        pos = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]]).astype(np.int64)
        cv2.rectangle(img, (pos[0], pos[1]), (pos[2], pos[3]), (0, 255, 0), 2)

        # 2. 获取在傅里叶域中经过滤波器H滤波的目标相关图
        f_target = gray[pos[1]:pos[3], pos[0]:pos[2]]
        f_target = pre_process(cv2.resize(f_target, (box[2], box[3])))
        g_target = H * np.fft.fft2(f_target)

        # 3. 使用滑动窗法计算图像中每一个像素的相关图，然后与g_target进行对比
        fs = np.lib.stride_tricks.sliding_window_view(gray, (box[3], box[2]))
        x, y, h, w = fs.shape
        print((x,y))
        for i in tqdm(range(x)):
            for j in range(y):
                if pos[1] <= i <= pos[3] and pos[0] <= j <= pos[2]:
                    continue
                fi = pre_process(cv2.resize(fs[i][j], (box[2], box[3])))
                gi = H * np.fft.fft2(fi)
                if abs(gi.max().real - g_target.max().real) < 0.00075:
                    xc, yc = np.where(gi == gi.max())
                    # 获得左上角的坐标
                    xl = i + int(xc.mean() - gi.shape[0] / 2)
                    yl = j + int(yc.mean() - gi.shape[1] / 2)
                    print("233")
                    # 获取右下角的坐标
                    xr = np.clip(xl + box[3], 0, gray.shape[0])
                    yr = np.clip(yl + box[2], 0, gray.shape[1])
                    cv2.rectangle(img, (yl, xl), (yr, xr), (0, 225, 0), 2)
        cv2.imwrite('result.jpg', img)


if __name__ == '__main__':
    detector = Detector('../source/parking.jpg')
    detector.run()

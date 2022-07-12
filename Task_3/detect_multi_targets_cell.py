from Task_2.utils import pre_process, normalization
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imshow, imread, imsave
from tqdm import tqdm


def tukeywindow(size, alpha=0.5):
    '''The Tukey window, also known as the tapered cosine window, can be regarded as a cosine lobe of width \alpha * N / 2
    that is convolved with a rectangle window of width (1 - \alpha / 2). At \alpha = 1 it becomes rectangular, and
    at \alpha = 0 it becomes a Hann window.
 '''
    # Special cases
    #    if alpha <= 0:
    #        return np.ones(window_length) #rectangular window
    #    elif alpha >= 1:
    #        return np.hanning(window_length)
    w, h = size

    # Normal case
    # Window in x direction (vertical direction)
    X = np.arange(w).reshape(w, 1)
    X = X / (w - 1.0)
    X = X * np.ones((1, h), 'd')
    window_X = np.ones(X.shape)
    # first condition 0 <= x < alpha/2
    first_condition = X < alpha / 2
    window_X[first_condition] = 0.5 * (1 + np.cos(2 * np.pi / alpha * (X[first_condition] - alpha / 2)))
    # second condition already taken care of
    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = X >= (1 - alpha / 2)
    window_X[third_condition] = 0.5 * (1 + np.cos(2 * np.pi / alpha * (X[third_condition] - 1 + alpha / 2)))

    # Window in horizontal directions
    Y = np.arange(h).reshape(1, h)
    Y = Y / (h - 1.0)
    Y = Y * np.ones((w, 1), 'd')
    window_Y = np.ones(Y.shape)
    # first condition 0 <= x < alpha/2
    first_condition = Y < alpha / 2
    window_Y[first_condition] = 0.5 * (1 + np.cos(2 * np.pi / alpha * (Y[first_condition] - alpha / 2)))
    # second condition already taken care of
    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = Y >= (1 - alpha / 2)
    window_Y[third_condition] = 0.5 * (1 + np.cos(2 * np.pi / alpha * (Y[third_condition] - 1 + alpha / 2)))

    # now multiply both wiindows
    window = window_X * window_Y
    return window


def rectTukeywindow(size,
                    alpha=0.5):  # this function will remove all targets outside unit tukey window and then will blur targets with gaussian for fixing targets points according to size of target and for blurring to remove sharp transitions (ringing effect)
    # rectangular tukey window... unity region is defined by alpha..
    [w, h] = size

    # Window in x direction (vertical direction)
    X = np.arange(w).reshape(w, 1)
    X = X / (w - 1.0)
    X = X * np.ones((1, h), 'd')
    window_X = np.ones(X.shape)
    # first condition 0 <= x < alpha/2
    first_condition = X < alpha / 2
    window_X[first_condition] = 0
    # second condition already taken care of
    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = X >= (1 - alpha / 2)
    window_X[third_condition] = 0

    # Window in horizontal directions
    Y = np.arange(h).reshape(1, h)
    Y = Y / (h - 1.0)
    Y = Y * np.ones((w, 1), 'd')
    window_Y = np.ones(Y.shape)
    # first condition 0 <= x < alpha/2
    first_condition = Y < alpha / 2
    window_Y[first_condition] = 0
    # second condition already taken care of
    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = Y >= (1 - alpha / 2)
    window_Y[third_condition] = 0

    # now multiply both wiindows
    window = window_X * window_Y
    return window


class Detector:
    """
    相关滤波检测器类
    """

    def __init__(self, eval_img, sigma=2.0, pretrain_num=4, h=500, w=500, show_in_window=False):
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
        self.eval_img = eval_img
        self.eval_gt = None

        self.sigma = sigma
        self.alpha = 0.25
        self.window = tukeywindow([w, h], self.alpha)
        self.target_window = rectTukeywindow([w, h], self.alpha)

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
                now = i.strip().split()
                if self.eval_img in now[0]:
                    self.eval_img = now[0]
                    self.eval_gt = now[1]
                    print(self.eval_img)
                    continue
                # 读取图像，灰度化，resize成相关滤波器的大小，为方便计算转化成浮点数
                img = cv2.imread(now[0], -1)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # gray = cv2.resize(gray, (self.cell_w, self.cell_h))
                gray = gray.astype(np.float32)
                # 读取ground-truth，应该是浮点数的响应值，但是在保存的时候只能保存uin8，所以乘了个255
                # 在这里使用的时候要除以255
                gt = cv2.imread(now[1], -1)
                gt = gt.astype(np.float32)
                gt /= 255.0
                gt = gt * self.target_window
                g = cv2.GaussianBlur(gt, (3, 3), 0)

                gray = (gray / 255.0) ** 2
                f = (gray - gray.mean()) / (gray.std())
                f = f * self.window

                imgs.append(f)
                gts.append(g)
        # 向量化处理
        train_img = np.stack(imgs, axis=0)
        train_gt = np.stack(gts, axis=0)
        # 转化到傅里叶域进行使用
        G = np.fft.fft2(train_gt)
        F = np.fft.fft2(train_img)
        cF = np.conj(F)
        # MOSSE滤波器生成公式
        Ai = G * cF
        Bi = F * cF
        # # 滤波器生成预训练
        # for _ in tqdm(range(self.pretrain_num)):
        #     fi = pre_process_n(train_img)
        #     Ai = Ai + G * np.conjugate(np.fft.fft2(fi))
        #     Bi = Bi + np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))
        # 存储生成的滤波器
        # self.cell = (Ai / Bi).mean(axis=0) # 这个是ASEF的生成方法
        self.cell = Ai.sum(axis=0) / Bi.sum(axis=0)
        print("finished pretrain!")

    def run(self, img_path=None):
        """
        检测一张图像（目前一张图像只支持一个目标检测）
        :param img_path:图像的路径
        """
        if img_path is None:
            if self.eval_img is not None:
                img_path = self.eval_img
            else:
                print("no image!")
                return
        # 读取一张图像，变成灰度图，resize成滤波器大小，浮点数转换
        img = cv2.imread(img_path, -1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = cv2.resize(gray, (self.cell_w, self.cell_h))
        gray = gray.astype(np.float32)
        gray = (gray / 255.0) ** 2
        f = (gray - gray.mean()) / (gray.std())
        f = f * self.window

        F = np.fft.fft2(f)
        G = self.cell * F
        g = np.fft.ifft2(G)
        cor = np.real(g)
        # cor[cor < 2.1 * 1e-5] = 0
        # 显示检测结果
        if self.show_in_windows:
            plt.figure(1)
            imshow(img)
            plt.title("Test Img")
            plt.show()

            plt.figure(2)
            imshow(cor)
            plt.title("Respond Map")
            plt.show()

            plt.figure(3)
            imshow(imread(self.eval_gt))
            plt.title("Ground Truth")
            plt.show()
            # cv2.imshow('result', img)
            # cv2.imshow('correlation map', cor * 255)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        # 保存检测结果
        # cv2.imwrite("result.png", img)
        imsave("result1.png", img)
        imsave("result2.png", cor)
        imsave('result3.png', imread(self.eval_gt))
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
    index = np.random.randint(1, 100)
    detector = Detector(eval_img=f'img{index}.bmp', show_in_window=True)  # 实例化图像
    detector.train('cell_path.txt')  # 训练相关滤波器
    detector.save('./model_multi_targets_cell.npy')  # 保存滤波器成文件
    # detector.load('./model_multi_targets_cell.npy')  # 加载滤波器文件
    detector.run()  # 检测相似物体

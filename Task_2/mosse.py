import numpy as np
import cv2
import copy


def normalization(data):
    return (data - data.min()) / (data.max() - data.min())


def generate_gauss_map(img, gt, sigma):
    h, w = img.shape
    # 变到0~1之间
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    # 获取目标的中心
    center_x = gt[0] + 0.5 * gt[2]
    center_y = gt[1] + 0.5 * gt[3]
    # 计算距离
    dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * sigma)
    # 映射到0~1之中
    response = np.exp(-dist)
    # 归一化
    return normalization(response)


def pre_process(img):
    h, w = img.shape
    # 转化到0~1之间
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


def pretrain(img, G):
    h, w = G.shape
    fi = pre_process(cv2.resize(img, (w, h)))
    Ai = G * np.conjugate(np.fft.fft2(fi))
    Bi = np.fft.fft2(img) * np.conjugate(np.fft.fft2(img))
    for _ in range(300):
        fi = pre_process(img)
        Ai = Ai + G * np.conjugate(np.fft.fft2(fi))
        Bi = Bi + np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))
    return Ai, Bi


class MOSSE:
    def __init__(self, sigma=100., lr=0.125, fps=40, show_in_window=False):
        """
        构造函数
        :param path: 视频的路径
        :param sigma: 根据GT生成的高斯峰的方差
        :param lr: 学习率
        """
        self.sigma = sigma
        self.lr = lr
        self.fps = fps
        self.first_gt = []
        self.show_in_windows = show_in_window

    def run(self, video_path, out_path):
        cap = cv2.VideoCapture(video_path)  # 输入视频
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 宽高
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 设置输出格式
        out = cv2.VideoWriter(out_path, fourcc, self.fps, (width, height))  # 输出流
        cnt = 0  # 当前帧
        while cap.isOpened():
            cnt += 1
            ret, frame = cap.read()  # 获取帧
            if ret:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_gray = frame_gray.astype(np.float32)
                if cnt == 1:
                    first_frame = frame_gray
                    # 获取初始帧的GT
                    first_gt = cv2.selectROI('getGT', frame, False, False)
                    first_gt = np.array(first_gt).astype(np.int64)
                    self.first_gt = first_gt
                    # 生成高斯峰
                    gauss_map = generate_gauss_map(first_frame, first_gt, self.sigma)
                    # 分别从初始帧和映射图中取出gt中的内容，分别作为fi和gi
                    gi = gauss_map[first_gt[1]:first_gt[1] + first_gt[3], first_gt[0]:first_gt[0] + first_gt[2]]
                    fi = first_frame[first_gt[1]:first_gt[1] + first_gt[3], first_gt[0]:first_gt[0] + first_gt[2]]
                    # 转化到频域
                    G = np.fft.fft2(gi)
                    Ai, Bi = pretrain(fi, G)
                    Ai = self.lr * Ai
                    Bi = self.lr * Bi
                    # h, w = G.shape  # 这里有个坑，w和h的顺序
                    # tmp = pre_process(cv2.resize(fi, (w, h)))
                    # # 根据公式11和12进行计算，因为没有上一帧，所以当做0处理
                    # Ai = G * np.conjugate(np.fft.fft2(tmp)) * self.lr
                    # Bi = np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi)) * self.lr
                    pos = first_gt.copy()
                    # 获取目标物体框的左上右下[minx,miny,maxx,maxy]
                    new_pos = np.array([pos[0], pos[1], pos[0] + pos[2], pos[1] + pos[3]]).astype(np.int64)
                else:
                    # 按照公式10计算滤波器的共轭
                    Hi = Ai / Bi
                    # 获取上一帧得到的框在这一帧的图像
                    fi = frame_gray[new_pos[1]:new_pos[3], new_pos[0]:new_pos[2]]
                    # resize+前处理
                    fi = pre_process(cv2.resize(fi, (self.first_gt[2], self.first_gt[3])))
                    # 按照公式1计算Gi
                    Gi = Hi * np.fft.fft2(fi)
                    # 返回到空间域
                    gi = normalization(np.fft.ifft2(Gi))
                    # 找到这个新生成的gi的峰值
                    max_pos = np.where(gi == gi.max())
                    # 获取新的框左上角坐标的偏移量
                    dy = int(max_pos[0].mean() - gi.shape[0] / 2)
                    dx = int(max_pos[1].mean() - gi.shape[1] / 2)

                    # 更新框的位置
                    pos[0] = pos[0] + dx
                    pos[1] = pos[1] + dy
                    # 更新下一帧要用的框
                    new_pos[0] = np.clip(pos[0], 0, frame.shape[1])
                    new_pos[1] = np.clip(pos[1], 0, frame.shape[0])
                    new_pos[2] = np.clip(pos[0] + pos[2], 0, frame.shape[1])
                    new_pos[3] = np.clip(pos[1] + pos[3], 0, frame.shape[0])
                    new_pos = new_pos.astype(np.int64)

                    # 获取现在的fi
                    fi = frame_gray[new_pos[1]:new_pos[3], new_pos[0]:new_pos[2]]
                    # 预处理
                    fi = pre_process(cv2.resize(fi, (self.first_gt[2], self.first_gt[3])))
                    # 根据公式11和12更新A,B
                    Ai = self.lr * (G * np.conjugate(np.fft.fft2(fi))) + (1 - self.lr) * Ai
                    Bi = self.lr * (np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))) + (1 - self.lr) * Bi
                cv2.rectangle(frame, (new_pos[0], new_pos[1]), (new_pos[2], new_pos[3]), (0, 255, 0), 3)
                out.write(frame)
                if self.show_in_windows:
                    cv2.imshow("result", frame)
                    cv2.waitKey(10)
            else:
                break
        out.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    filter_mosse = MOSSE(show_in_window=True)
    filter_mosse.run('../source/demo2.avi', 'result.avi')

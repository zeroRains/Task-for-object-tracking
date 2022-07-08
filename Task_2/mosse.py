from Task_2.utils import *
import numpy as np
import cv2


class MOSSE:
    def __init__(self, sigma=100., lr=0.125, fps=40, show_in_window=False, pretrain_num=0):
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
        self.pretrain_num = pretrain_num
        self.psr = []

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
                    # 获取初始帧的GT（根据你论文中的Correlation Filter Based Tracking中的第一段中获取第一帧中的物体）
                    first_gt = cv2.selectROI('demo', frame, False, False)
                    first_gt = np.array(first_gt).astype(np.int64)
                    self.first_gt = first_gt
                    # 生成高斯峰（对应MOSSE Filters那一节的第一段）
                    gauss_map = generate_gauss_map(first_frame, first_gt, self.sigma)
                    # 分别从初始帧和映射图中取出gt中的内容，分别作为fi和gi
                    gi = gauss_map[first_gt[1]:first_gt[1] + first_gt[3], first_gt[0]:first_gt[0] + first_gt[2]]
                    fi = first_frame[first_gt[1]:first_gt[1] + first_gt[3], first_gt[0]:first_gt[0] + first_gt[2]]
                    # 转化到频域
                    G = np.fft.fft2(gi)
                    Ai, Bi = pretrain(fi, G, self.pretrain_num)
                    Ai = self.lr * Ai
                    Bi = self.lr * Bi
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
                    self.psr.append(cal_psr(Gi))
                    # 返回到空间域(根据Failure Detection and PSR的说明，gi是用于计算PSR的关键)
                    gi = np.fft.ifft2(Gi)
                    gi = normalization(gi)
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
                cv2.circle(frame, ((new_pos[0] + new_pos[2]) // 2, (new_pos[1] + new_pos[3]) // 2), 10, (0, 255, 0))
                cv2.circle(frame, ((new_pos[0] + new_pos[2]) // 2, (new_pos[1] + new_pos[3]) // 2), 1, (0, 255, 0))
                out.write(frame)
                if self.show_in_windows:
                    cv2.imshow("demo", frame)
                    cv2.waitKey(10)
            else:
                break
        out.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    filter_mosse = MOSSE(show_in_window=True, pretrain_num=128, sigma=100.0)
    filter_mosse.run('../source/demo2.avi', 'result.avi')
    filter_mosse.psr = np.array(filter_mosse.psr)
    print(filter_mosse.psr)
    print(filter_mosse.psr.max())
    print(filter_mosse.psr.min())
    print(filter_mosse.psr.mean())

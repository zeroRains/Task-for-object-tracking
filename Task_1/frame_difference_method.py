import cv2
import copy
import numpy as np


class FrameDifference:
    """
    path：视频的路径
    num: 帧差法中具体差多少帧，默认6，即6帧过完之后才使用跟踪算法
    thread：帧差法的阈值，默认10，即当两帧的灰度计算出来后，大于等于thread的会被保留，低于的会被舍弃
    fps: 追踪后视频的帧率，默认40帧
    contours_len：轮廓长度超过这个值才会保留这个轮廓
    """

    def __init__(self, path, num=6, thread=10, fps=40, contours_len=500, show_in_windows=False):
        self.path = path
        self.num = num
        self.thread = thread
        self.fps = fps
        self.contours_len = contours_len
        self.show_in_windows = show_in_windows

    def run(self, out_path):
        """
        :param out_path: 输出跟踪结果的路径
        :return:
        """
        # 视频输出流
        video = cv2.VideoCapture(self.path)  # 读取原视频
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # 宽高
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 设置输出格式
        out = cv2.VideoWriter(out_path, fourcc, self.fps, (width, height)) # 输出流
        cnt = 0  # 记录当前多少帧
        frames = []
        while video.isOpened():
            ret, frame = video.read()
            frac = copy.deepcopy(frame)
            if ret:
                cnt += 1
                frame = cv2.GaussianBlur(frame, (3, 3), 0)  # 高斯平滑
                print(f"now frame: {cnt}")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if len(frames) < self.num:
                    frames.append(gray)
                    continue
                diff = cv2.absdiff(frames[0], gray)  # 比较距离为num的两帧之间的差距
                frames = frames[1:-1]  # 更新帧
                frames.append(gray)
                kernel = np.ones((3, 3), np.uint8)  # 创建一个全是1的3x3的滤波核
                _, diff = cv2.threshold(diff, self.thread, 255, cv2.THRESH_BINARY)  # 阈值选择
                dilation = cv2.dilate(diff, kernel)  # 膨胀
                diff = cv2.erode(dilation, kernel)  # 腐蚀
                # 获取二值化后的轮廓，第一是轮廓，第二个是轮廓属性
                # 轮廓就是由一个个点构成的，两点之间连线
                contours, hierarchy = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for c in contours:
                    x, y, w, h = cv2.boundingRect(c)  # 获取这些轮廓的框并绘制
                    tmp = cv2.arcLength(c, True)
                    if tmp >= self.contours_len:
                        cv2.rectangle(frac, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if self.show_in_windows:
                    cv2.imshow('result', frac)
                    cv2.waitKey(10)
                out.write(frac)
            else:
                break
        print(f"{out_path} 视频跟踪完成！")
        out.release()
        video.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    path = '../source/demo.mp4'
    method = FrameDifference(path, show_in_windows=True)
    method.run('result.avi')

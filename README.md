# 夏令营考核——基于相关滤波的物体追踪

## Task 1：


- [x] 1. 实现最简单的帧差法运动物体追踪

帧差法：当视频中存在移动物体时，相邻帧之间在灰度上会有所差别，求取两帧图像的灰度差的绝对值，则静止的物体的像素在这个差值结果中的灰度值为0，而移动物体特别是该物体的轮廓处由于存在灰度变化为非0，这样就能大致计算出移动物体的位置，轮廓以及移动路径。

实现代码：[frame_difference](https://github.com/zeroRains/Task-for-object-chacking/blob/master/Task_1/frame_difference_method.py#L1)

实现效果：[result.avi](https://github.com/zeroRains/Task-for-object-chacking/raw/master/Task_1/result.avi)

实现过程：根据参考资料2，学习了帧差法的基础概念，因为他加了很多的图像处理方式，比如高斯平滑，腐蚀，膨胀等，我想试试如果单独只用差分法的结果会怎样，下面将展示部分单独使用差分法，使用高斯滤波，使用腐蚀肿胀，控制轮廓边长输出等等情况。我在实现代码中将帧差法封装成类，设置了四个可以变动的参数。

参数说明：
1. num: 表示帧差法到底差几帧，默认6
2. thread: 表示侦差后的图像，大于等于这个阈值的像素被保留，否则舍弃，默认10
3. fps：输出视频的帧率是多少，默认40
4. contours_len：轮廓的长度超过这个值才会画上检测框，默认500

实现效果截图：
1. 单独使用差分法（不加任何图像处理方式）:

    ![](images/fig1.png)

    > 图像中小框很多，所以这就用上了我们类中的第四个参数，用来去除很多小框

2. 轮廓边长控制

    ![](images/fig2.png)

    > 经过控制轮廓边长后，很多小框都消失了

3. 高斯平滑+轮廓边长控制

    ![](images/fig3.png)
    
    > 好像差别不大

4. 高斯平滑+轮廓边长控制+膨胀+腐蚀

    ![](images/fig4.png)
    
    > 好像好多了

（ps：还有很多调了超参的图就不展示了）

参考资料：
1. [OpenCV 图像处理之膨胀与腐蚀](https://zhuanlan.zhihu.com/p/110330329)
2. [高斯混合模型与差帧法提取前景](https://blog.csdn.net/qq_45087786/article/details/121865855)


## Task 2:


- [ ] 手动复现论文[《Visual Object Tracking using Adaptive Correlation Filters. David S. Bolme, J. Ross Beveridge, Bruce A. Draper, Yui Man Lui. CVPR，2010》](https://ieeexplore.ieee.org/abstract/document/5539960)

论文学习记录：[「论文阅读」Visual Object Tracking using Adaptive Correlation Filters](https://blog.zerorains.top/2022/07/02/%E3%80%8C%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB%E3%80%8DVisual-Object-Tracking-using-Adaptive-Correlation-Filters/)



参考资料：

1. [余弦窗(汉宁窗)的作用——图像预处理](https://blog.csdn.net/dengheCSDN/article/details/78085468)
2. [图像的仿射变换：cv2.warpAffine()](https://zhuanlan.zhihu.com/p/416073892)
3. [TianhongDai/mosse-object-tracking](https://github.com/TianhongDai/mosse-object-tracking)


## Task 3:


- [ ] 将相关滤波检测算法扩展应用到相似形状物体检测领域。(诸如细胞核检测、汽车检测等场景)


# 夏令营考核——基于相关滤波的物体追踪

## Task 1：


- [x] 1. 实现最简单的帧差法运动物体追踪

帧差法：当视频中存在移动物体时，相邻帧之间在灰度上会有所差别，求取两帧图像的灰度差的绝对值，则静止的物体的像素在这个差值结果中的灰度值为0，而移动物体特别是该物体的轮廓处由于存在灰度变化为非0，这样就能大致计算出移动物体的位置，轮廓以及移动路径。

实现代码：

实现效果：

## Task 2:


- [ ] 2. 手动复现论文《Visual Object Tracking using Adaptive Correlation Filters. David S. Bolme, J. Ross Beveridge, Bruce A. Draper, Yui Man Lui. CVPR，2010》

## Task 3:


- [ ] 3. 将相关滤波检测算法扩展应用到相似形状物体检测领域。(诸如细胞核检测、汽车检测等场景)


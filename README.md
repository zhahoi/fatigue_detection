# fatigue_detection
使用libfacedetection和dlib进行面部68个关键点检测，根据面部关键点进行疲劳检测。（c++实现）

****

### 写在前面

本仓库的的疲劳检测算法原理绝大部分来自仓库[fatigue_detecting](https://gitee.com/cungudafa/fatigue_detecting)，该仓库的代码是基于python实现的，本仓库在此基础上，将其改写成了c++的版本，用于加深我对疲劳检测算法原理的理解。在此也十分感谢chatgpt，助我快速实现本仓库代码的实现。



### 基本原理

> 驾驶员在开车时，眼睛是司机行车过程中获得信息很重要的途径，所以这一指标显得尤为重要。研究发现，眼部活动（如眼睑的闭合度和眨眼的次数）、嘴部的动作（如打哈欠的频率）以及头部的动作（如频繁点头），这三种特征都与司机的疲劳状态有极大的关联。
>
> 摘自论文《基于面部特征的疲劳驾驶检测算法研究与实现》_岳田甜



### 实现原理

本仓库实现原理如下：先通过**libfacedetection**面部检测算法，检测到图像或视频中包含人脸的位置，将其面部裁剪出来输入到**Dlib**开源图像数据库中，进行面部68个关键点检测。随后，通过提取的面部关键点进行**眨眼**、**打哈欠**和**瞌睡点头**的次数判定，进而进一步判断检测到的人是否处于疲劳瞌睡状态。

更详细的原理可以看下面的介绍，非常详细，保证你一看就懂：

> **cungudafa博客：Dlib模型之驾驶员疲劳检测系列（眨眼、打哈欠、瞌睡点头、可视化界面）**
>
> [https://blog.csdn.net/cungudafa/article/details/103477960](https://gitee.com/link?target=https%3A%2F%2Fblog.csdn.net%2Fcungudafa%2Farticle%2Fdetails%2F103477960)
>
> [https://blog.csdn.net/cungudafa/article/details/103496881](https://gitee.com/link?target=https%3A%2F%2Fblog.csdn.net%2Fcungudafa%2Farticle%2Fdetails%2F103496881)
>
> [https://blog.csdn.net/cungudafa/article/details/103499230](https://gitee.com/link?target=https%3A%2F%2Fblog.csdn.net%2Fcungudafa%2Farticle%2Fdetails%2F103499230)



### 依赖库(版本仅供参考)

- opencv-3.4.10
- dlib-19.24



### 下载和编译

```sh
$ git clone https://github.com/zhahoi/fatigue_detection.git

$ cd fatigue_detection 
$ mkdir build
$ cd build
$ cmake ..
$ make -j8
```

编译之前，请先修改`CMakeLists.txt`中的以下内容：

```sh
// 改为实际的安装路径
set(OpenCV_DIR "/home/hit/Softwares/opencv-3.4.10/build") 
set(dlib_DIR "/home/hit/Softwares/dlib/build")

/home/xxx/fatigue_detection/facedetection/include/
/home/xxx/fatigue_detection/facedetection/lib/libfacedetection.so
```



### 执行

```sh
$ cd build

$./detect  0   # 调用摄像头推理
$./detect  xxx.mp4  # 调用视频推理
```



### 结果截图

简单的检测结果如下：

![微信截图_20250509113223](C:\Users\HIT-HAYES\Desktop\微信截图_20250509113223.png)



### 注意事项

1. 疲劳检测算法有比较多的阈值参数需要调整，本仓库提供的阈值比较粗略，且仅供参考。实际使用时，请按照自身的情况做调整。
2. 瞌睡点头计算时，需要知道当前摄像头的**外参**和**畸变系数**。本仓库使用的是提供的默认参数，为了获得更好且更准确的算法结果，十分建议先对拍摄用的相机进行标定，随后将外参和畸变系数进行替换。
3. 参考的源工程代码提供了一个可视化QT界面，由于本人的QT水平很差，就先不实现了，如果感兴趣的话，可以自己实现试试看。



### 写在最后

创作不易，如果觉得这个仓库还可以的话，麻烦给一个star，这就是对我最大的鼓励。



### Reference

-[fatigue_detecting](https://gitee.com/cungudafa/fatigue_detecting)

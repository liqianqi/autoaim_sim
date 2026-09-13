# 自瞄仿真器

RoboMaster 步兵自瞄仿真：MuJoCo 场景 + 机载相机共享内存，可接检测 / PnP / 跟踪。在AI和各路开源的帮助下用了一天半时间，我脱离RM有几年了，一直忙自己的事情，前几天才突然有要不要做做RM自瞄仿真想法，恰巧这几天碰了碰mujuco，就整了整这个程序。如果有人看到并且帮助了你们那真是很好了。因为现在RM自瞄水平都非常高，这个程序和RM论坛上的开源肯定相当之简陋，所以想了想还是不放到论坛了。

## 仿真环境

- 红 / 蓝步兵，机载针孔相机（1280×720，无畸变）
- 空格：目标左右巡逻并自旋
- 图像写入共享内存 `/autoaim_onboard`

## 运行

```bash
pip install mujoco numpy opencv-python pyyaml # (具体环境是requirements.txt里面的，缺啥补啥就行)
python scripts/view_infantry.py --team red    
```

另开终端接图：

```bash
python aim/camera_example.py
```

C++ 示例（OpenCV / Eigen / CUDA / TensorRT）：

```bash
cmake -S aim/cpp -B aim/cpp/build && cmake --build aim/cpp/build
./aim/cpp/build/camera_example
```

## 目前效果与说明

- yaw优化用的同济大学2025年开源方案，比原生IPPE稳了很多，但是正对相机视角还是有点问题。
- 我现在的程序是std::abs(yaw) < 15度的时候把观测噪声R对应的部分放大，但是实际上车的时候相机正对装甲板的角度不一定是15度以内(涉及相机到云台系坐标转换)，这个得自己改改。
- 因为观测(ekf.update)允许同时观测两个装甲板，所以理论上ekf的状态估计在识别器可识别范围内陀螺转的越快越稳定。
- 角度用std::remainder做处理，不用害怕过180度的问题。
- 因为仿真模型的车辆容易在急启急停，所以ekf在这一个时刻效果很不好，容易飘，不过RM赛场上很少有车能在几乎同一个时刻实现速度从3到-3的变化(我脱离RM赛场很久了，有可能刻板印象)。
- 我的识别模型就随便训练了一下，关键点回归没那么精确，但即使没那么精确配合yaw优化以及ekf效果还算OK。
- 感谢华南师范大学陈君的rm_vision的unity仿真器和EKF开源，车体装甲板灯条用dae导出的(cursor帮助下)，感谢同济大学开源的yaw优化策略，感谢codex分析和cursor一众模型。

## 许可证

[MIT](LICENSE)

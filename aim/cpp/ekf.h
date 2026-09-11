#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <thread>

#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

struct ObjectInfo;

class EKF {
public:
    EKF()
    {
        init();
    }
    ~EKF() = default;

    void predict();

    void update(Eigen::VectorXd measurement);

    void init();                            // 只设噪声参数, 状态归零
    void init(const Eigen::VectorXd& z);    // 用第一块装甲板观测 (x y z yaw) 初始化车中心

    void setDt(double dt) { dt_ = dt; }
    const Eigen::VectorXd& state() const { return state_; }

    static constexpr double kRadiusLong  = 0.26;  // 侧板
    static constexpr double kRadiusShort = 0.21;  // 前后板
private:
    // x y z yaw vx vy v_yaw r (车中心位置, 当前装甲板 yaw, 中心速度, 角速度, 当前装甲板半径)
    Eigen::VectorXd state_;
    Eigen::VectorXd measurement_; // measurement: x y z yaw, 观测到的装甲板信息
    Eigen::MatrixXd P_;
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd R_;
    double dt_ = 0.01;

    double another_r_ = kRadiusShort;   // 另一组装甲板的半径, 跳板时和 state_[7] 互换
};




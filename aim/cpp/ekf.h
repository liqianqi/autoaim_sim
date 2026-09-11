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

    // 这辆车本帧所有观测 (同一编号的装甲板, 1~2 块, 每块 x y z yaw).
    // 离预测最近的那块当作当前板走原有流程; 若还有一块, 它必然是相邻板 (yaw 差 ±90°), 一起用来约束中心和 yaw.
    // 返回是否有观测被接受.
    bool update(const std::vector<Eigen::Vector4d>& zs);

    void init();                            // 只设噪声参数, 状态归零
    void init(const Eigen::VectorXd& z);    // 用第一块装甲板观测 (x y z yaw) 初始化车中心

    void setDt(double dt) { dt_ = dt; }
    const Eigen::VectorXd& state() const { return state_; }
    double anotherR() const { return another_r_; }

    // 第 i 块装甲板 (i=0 当前跟踪的板, 逆时针依次 1,2,3) 的中心位置和 yaw
    void armor(int i, Eigen::Vector3d& pos, double& yaw) const {
        yaw = std::remainder(state_[3] + i * M_PI / 2, 2 * M_PI);
        const double r = (i % 2 == 0) ? state_[7] : another_r_;
        const double y = (i % 2 == 0) ? state_[1] : state_[1] + another_dy_;
        pos = {state_[0] - r * std::sin(yaw), y, state_[2] - r * std::cos(yaw)};
    }

    static constexpr double kRadiusLong  = 0.26;  // 侧板
    static constexpr double kRadiusShort = 0.21;  // 前后板
    static constexpr double kSideLower   = 0.08;  // 侧板比前后板低 (m), 仿真模型值, 作为 dy 先验
private:
    // 用一块板的观测做一次 EKF 修正. offset: 该板 yaw 相对 state_[3] 的偏角 (当前板 0, 相邻板 ±90°)
    void correct(const Eigen::Vector4d& z, double offset);

    // x y z yaw vx vy v_yaw r (车中心位置, 当前装甲板 yaw, 中心速度, 角速度, 当前装甲板半径)
    Eigen::VectorXd state_;
    Eigen::VectorXd measurement_; // measurement: x y z yaw, 观测到的装甲板信息
    Eigen::MatrixXd P_;
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd R_;
    double dt_ = 0.01;

    double another_r_ = kRadiusShort;   // 另一组装甲板的半径, 跳板时和 state_[7] 互换
    double another_dy_ = 0.0;           // 另一组装甲板相对当前板的高度差 (相机 Y 向下), 跳板时取反
};

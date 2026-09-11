#include "ekf.h"
#include <opencv2/core/utility.hpp>

// x y z yaw vx vy v_yaw r(位置,yaw,速度,角速度,半径. 全部基于车中心的信息(yaw是装甲板信息))
void EKF::predict()
{
    state_[0] = state_[0] + state_[4] * dt_;
    state_[1] = state_[1] + state_[5] * dt_;
    state_[2] = state_[2];
    state_[3] = std::remainder(state_[3] + state_[6] * dt_, 2 * M_PI);
    state_[4] = state_[4];
    state_[5] = state_[5];
    state_[6] = state_[6];
    state_[7] = state_[7];

    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(8, 8);
    F(0,4) = dt_;
    F(1,5) = dt_;
    F(3,6) = dt_;

    // 分段白噪声模型: 假设 dt 内加速度为方差 v 的白噪声
    // 位置-速度块为 [[dt^4/4, dt^3/2], [dt^3/2, dt^2]] * v
    const double d2 = dt_ * dt_, d3 = d2 * dt_, d4 = d3 * dt_;
    const double va = 100.0;  // 平动加速度方差 (m/s^2)^2
    const double vw = 400.0;  // 角加速度方差 (rad/s^2)^2
    Q_.setZero(8, 8);
    for (int i : {0, 1}) {          // (x,vx), (y,vy)
        Q_(i, i)         = d4 / 4 * va;
        Q_(i, i + 4)     = d3 / 2 * va;
        Q_(i + 4, i)     = d3 / 2 * va;
        Q_(i + 4, i + 4) = d2 * va;
    }
    Q_(3, 3) = d4 / 4 * vw;         // (yaw, v_yaw)
    Q_(3, 6) = d3 / 2 * vw;
    Q_(6, 3) = d3 / 2 * vw;
    Q_(6, 6) = d2 * vw;
    Q_(2, 2) = d2 * 1e-2;           // z 缓慢漂移
    Q_(7, 7) = d2 * 1e-4;           // r 几乎不变

    P_ = F * P_ * F.transpose() + Q_;
}

void EKF::update(Eigen::VectorXd measurement)
{
    measurement_ = measurement;

    if(std::abs(std::remainder(measurement_[3] - state_[3], 2 * M_PI)) > M_PI / 3){
        std::swap(state_[7], another_r_);
        state_[3] = measurement_[3];
        state_[7] = std::clamp(state_[7], 0.12, 0.4);
        return ;
    }

    double r = state_[7];
    auto h = [&, r](){
        Eigen::VectorXd h(4);
        h[0] = state_[0] - r * sin(state_[3]);
        h[1] = state_[1];
        h[2] = state_[2] - r * cos(state_[3]);
        h[3] = state_[3];
        return h;
    };

    Eigen::MatrixXd JH = Eigen::MatrixXd::Zero(4, 8);
    const double c = std::cos(state_[3]);
    const double s = std::sin(state_[3]);
    JH.setZero();
    JH(0, 0) = 1;
    JH(0, 3) = -r * c;
    JH(1, 1) = 1;
    JH(2, 2) = 1;
    JH(2, 3) = r * s;
    JH(3, 3) = 1;
    JH(0, 7) = -s;   // ∂h0/∂r
    JH(2, 7) = -c;   // ∂h2/∂r

    Eigen::MatrixXd S = JH * P_ * JH.transpose() + R_;
    Eigen::MatrixXd K = P_ * JH.transpose() * S.inverse();
    Eigen::VectorXd y = measurement_ - h();
    y[3] = std::remainder(y[3], 2 * M_PI);

    state_ = state_ + K * y;

    state_[3] = std::remainder(state_[3], 2 * M_PI);
    state_[7] = std::clamp(state_[7], 0.12, 0.4);

    P_ = (Eigen::MatrixXd::Identity(8, 8) - K * JH) * P_;
}

void EKF::init()
{
    dt_ = 0.01;
    state_ = Eigen::VectorXd::Zero(8);
    measurement_ = Eigen::VectorXd::Zero(4);
    P_ = Eigen::MatrixXd::Identity(8, 8);
    Q_ = Eigen::MatrixXd::Zero(8, 8);   // 每次 predict 按 dt 重算

    R_ = Eigen::MatrixXd::Zero(4, 4);
    R_(0, 0) = 1e-4;   // x: 1 cm
    R_(1, 1) = 1e-4;   // y: 1 cm
    R_(2, 2) = 4e-4;   // z: 2 cm, 深度更差
    R_(3, 3) = 1e-2;   // yaw: ~6 deg
}

// 用第一块装甲板 (x y z yaw) 初始化车中心
void EKF::init(const Eigen::VectorXd& z)
{
    const double yaw = std::remainder(z[3], 2 * M_PI);

    // |yaw| 在 45~135 deg 之间看到的是侧板 (长半径), 否则是前后板 (短半径)
    const double a = std::abs(yaw);
    const bool side = a > M_PI / 4 && a < 3 * M_PI / 4;
    const double r = side ? kRadiusLong : kRadiusShort;
    another_r_     = side ? kRadiusShort : kRadiusLong;

    // 观测模型 h0 = x - r sin(yaw), h2 = z - r cos(yaw) 的逆
    state_ = Eigen::VectorXd::Zero(8);
    state_[0] = z[0] + r * std::sin(yaw);
    state_[1] = z[1];
    state_[2] = z[2] + r * std::cos(yaw);
    state_[3] = yaw;
    state_[7] = r;

    Eigen::VectorXd p0(8);
    //    x    y    z    yaw  vx  vy  v_yaw  r
    p0 << 0.1, 0.1, 0.1, 0.1, 4,  4,  25,    1e-2;
    P_ = p0.asDiagonal();
}
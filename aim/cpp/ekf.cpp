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
    Q_(7, 7) = d2 * 1e-6;           // r 几乎不变

    P_ = F * P_ * F.transpose() + Q_;
}

bool EKF::update(const std::vector<Eigen::Vector4d>& zs)
{
    if (zs.empty()) return false;

    // 当前板 = 离预测板位置最近的观测
    const Eigen::Vector3d pred(state_[0] - state_[7] * std::sin(state_[3]), state_[1],
                               state_[2] - state_[7] * std::cos(state_[3]));
    const Eigen::Vector4d* cur = &zs[0];
    for (const auto& z : zs) {
        if ((z.head<3>() - pred).norm() < (cur->head<3>() - pred).norm()) cur = &z;
    }
    if ((cur->head<3>() - pred).norm() > 0.5) return false;  // 离谱观测, 不吃
    measurement_ = *cur;

    // 跳板: 新的当前板就是刚才的相邻板, 它的几何我们已经估过了, 把状态整体转 90° 即可,
    // 不要用这一帧的原始 PnP yaw 覆盖滤好的 yaw (正对时 PnP yaw 误差十几度, 会直接飘)
    const double d = std::remainder(measurement_[3] - state_[3], 2 * M_PI);
    if(std::abs(d) > M_PI / 3){
        state_[3] = std::remainder(state_[3] + (d > 0 ? M_PI / 2 : -M_PI / 2), 2 * M_PI);
        std::swap(state_[7], another_r_);
        state_[1] += another_dy_;
        another_dy_ = -another_dy_;
    }
    correct(*cur, 0.0);

    // 另一块板 (如果有): 与当前板 yaw 差 ±90°, 用它的 y 直接量高度差, 再一起约束中心和 yaw
    for (const auto& z : zs) {
        if (&z == cur) continue;
        const double d = std::remainder(z[3] - state_[3], 2 * M_PI);
        if (std::abs(std::abs(d) - M_PI / 2) > M_PI / 6) continue;  // 不像相邻板, 跳过

/******************************************************************************************************************/
        // 板转到 ~70° 快出视野时远侧灯条压扁, PnP 的 y 会跳好几 cm; 两块板都在 65° 以内才用它量高度差, 且低通
        if (std::abs(z[3]) < 65 * M_PI / 180 && std::abs((*cur)[3]) < 65 * M_PI / 180) {
            const double dy = z[1] - state_[1];
            another_dy_ += 0.3 * (dy - another_dy_);
            // 更低 (y 更大) 的那块是侧板 = 长半径. 初始化时按 yaw 猜的前/侧板是掷硬币, 这里用高度纠正
            if (std::abs(dy) > 0.04 && (dy > 0) != (another_r_ > state_[7])) std::swap(state_[7], another_r_);
        }
/******************************************************************************************************************/

        correct(z, d > 0 ? M_PI / 2 : -M_PI / 2);
    }
    return true;
}

void EKF::correct(const Eigen::Vector4d& measurement, double offset)
{
    measurement_ = measurement;

    // offset=0 是当前板, 用 state_[7]; 相邻板用 another_r_/another_dy_ (不在状态里, 不被更新)
    const bool current = offset == 0.0;
    const double yaw = std::remainder(state_[3] + offset, 2 * M_PI);
    const double r  = current ? state_[7] : another_r_;
    const double dy = current ? 0.0 : another_dy_;
    auto h = [&, r](){
        Eigen::VectorXd h(4);
        h[0] = state_[0] - r * sin(yaw);
        h[1] = state_[1] + dy;
        h[2] = state_[2] - r * cos(yaw);
        h[3] = yaw;
        return h;
    };

    Eigen::MatrixXd JH = Eigen::MatrixXd::Zero(4, 8);
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);
    JH.setZero();
    JH(0, 0) = 1;
    JH(0, 3) = -r * c;
    JH(1, 1) = 1;
    JH(2, 2) = 1;
    JH(2, 3) = r * s;
    JH(3, 3) = 1;
    if (current) {
        JH(0, 7) = -s;   // ∂h0/∂r
        JH(2, 7) = -c;   // ∂h2/∂r
    }

    // 正对 (|yaw| < 15°) 时 PnP 的 yaw 连符号都不稳, 少信它
    Eigen::MatrixXd R = R_;
    if (std::abs(measurement_[3]) < 15 * M_PI / 180) R(3, 3) *= 10;

    Eigen::MatrixXd S = JH * P_ * JH.transpose() + R;
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
    another_dy_    = side ? -kSideLower : kSideLower;  // 相机 Y 向下: 更高的板 y 更小

    // 观测模型 h0 = x - r sin(yaw), h2 = z - r cos(yaw) 的逆
    state_ = Eigen::VectorXd::Zero(8);
    state_[0] = z[0] + r * std::sin(yaw);
    state_[1] = z[1];
    state_[2] = z[2] + r * std::cos(yaw);
    state_[3] = yaw;
    state_[7] = r;

    Eigen::VectorXd p0(8);
    // r 和中心深度在单块板上几乎共线, r 的先验必须比位置紧, 否则 PnP 的深度偏差会全被 r 吸走
    //    x    y    z    yaw  vx  vy  v_yaw  r
    p0 << 0.1, 0.1, 0.1, 0.1, 4,  4,  25,    1e-4;
    P_ = p0.asDiagonal();
}

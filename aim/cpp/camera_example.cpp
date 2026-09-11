//   python scripts/view_infantry.py --team red
//   cmake -S aim/cpp -B aim/cpp/build && cmake --build aim/cpp/build
//   ./aim/cpp/build/camera_example

#include "ekf.h"
#include "onboard_camera.hpp"
#include "trtengine.h"

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

static constexpr bool kIsBlue = false;
static constexpr double kDeg = 3.14159265358979323846 / 180.0;

static cv::Mat cameraK() {
    return (cv::Mat_<double>(3, 3) << 623.5382907247958, 0, 640, 0, 623.5382907247958, 360, 0, 0,
            1);
}

static cv::Mat cameraD() { return cv::Mat::zeros(5, 1, CV_64F); }

// 物点（米）：X 右、Y 下、Z 朝里，顺序左上、右上、右下、左下。
// 不是灯条外接矩形（137.8x59.6），而是按这个 YOLO 模型在仿真里实际输出的关键点校准的：
// 用真值位姿把关键点反投影到装甲平面；数值和当前 18mm 发光灯条的渲染匹配。
// 网络输出并非严格矩形，所以保留每个角点的独立校准值。
static const std::vector<cv::Point3f> kArmorPts = {
    {-0.06909f, -0.02987f, 0.f},
    {0.06808f, -0.02893f, 0.f},
    {0.06659f, 0.02449f, 0.f},
    {-0.06939f, 0.02391f, 0.f},
};

struct ObjectInfo {
    Eigen::Vector2d pixel_pos_;
    Eigen::Vector3d gimbal_pos_;
    Eigen::Vector3d gimbal_angle_;  // roll, pitch, yaw（弧度）
    int id_ = 0;
    int color_ = 0;
};

struct ArmorPose {
    cv::Vec3d rvec, tvec;
    Eigen::Vector3d rpy;
};

static std::vector<cv::Point2f> orderCorners(const Detection& d) {
    std::vector<cv::Point2f> pts(4);
    for (int i = 0; i < 4; ++i) {
        pts[i] = {d.kpts[i * 2], d.kpts[i * 2 + 1]};
    }
    std::sort(pts.begin(), pts.end(),
              [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });
    const cv::Point2f lt = pts[0].x < pts[1].x ? pts[0] : pts[1];
    const cv::Point2f rt = pts[0].x < pts[1].x ? pts[1] : pts[0];
    const cv::Point2f lb = pts[2].x < pts[3].x ? pts[2] : pts[3];
    const cv::Point2f rb = pts[2].x < pts[3].x ? pts[3] : pts[2];
    return {lt, rt, rb, lb};
}

// yaw=atan2(nx,nz)：法向朝右为正（板左沿更近）。pitch=atan2(-ny,|n_xz|)：法向朝上为正。
// roll：绕法向。正对前装甲的真值约 (0, -15°, 0)，-15° 是装甲仰装 15°。
static Eigen::Vector3d rvecToRPY(const cv::Vec3d& rvec) {
    cv::Mat Rcv;
    cv::Rodrigues(rvec, Rcv);
    Eigen::Matrix3d R;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R(i, j) = Rcv.at<double>(i, j);
        }
    }
    const Eigen::Vector3d n = R.col(2).normalized();
    const Eigen::Vector3d x = R.col(0).normalized();
    Eigen::Vector3d xref = Eigen::Vector3d::UnitX() - n * n.x();
    if (xref.norm() < 1e-6) {
        xref = Eigen::Vector3d::UnitY() - n * n.y();
    }
    xref.normalize();
    return {std::atan2(n.dot(xref.cross(x)), xref.dot(x)),
            std::atan2(-n.y(), std::hypot(n.x(), n.z())), std::atan2(n.x(), n.z())};
}

// 装甲板姿态 = 绕相机 Y 转 yaw，再绕板自身 X 转 -15°（仰装），roll 固定 0。和 rvecToRPY 的约定一致。
static cv::Vec3d yawToRvec(double yaw) {
    const double p = -15 * kDeg;
    const cv::Mat Rx = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, std::cos(p), -std::sin(p), 0, std::sin(p), std::cos(p));
    const cv::Mat Ry = (cv::Mat_<double>(3, 3) << std::cos(yaw), 0, std::sin(yaw), 0, 1, 0, -std::sin(yaw), 0, std::cos(yaw));
    cv::Vec3d rvec;
    cv::Rodrigues(Ry * Rx, rvec);
    return rvec;
}

// 同济 optimize_yaw：位置信 PnP，姿态锁 roll=0、pitch=-15°，只剩 yaw 一个自由度。
// 在“板正对相机”的方位角 ±70° 内 1° 步进搜索，四角点重投影误差最小者胜。
// 单板 PnP 的 yaw 之所以差，是因为它同时在解 6 个自由度，小板上 yaw 和 pitch/roll 互相补偿；锁死后就不会了。
static double optimizeYaw(const std::vector<cv::Point2f>& uv, const cv::Vec3d& tvec) {
    const double az = std::atan2(tvec[0], tvec[2]);  // 板正对相机时法向 yaw 就是它的方位角
    double best_yaw = az, best_err = 1e18;
    for (double yaw = az - 70 * kDeg; yaw <= az + 70 * kDeg; yaw += 1 * kDeg) {
        std::vector<cv::Point2f> proj;
        cv::projectPoints(kArmorPts, yawToRvec(yaw), tvec, cameraK(), cameraD(), proj);
        double err = 0;
        for (int i = 0; i < 4; ++i) {
            err += cv::norm(proj[i] - uv[i]);
        }
        if (err < best_err) {
            best_err = err;
            best_yaw = yaw;
        }
    }
    return best_yaw;
}

// IPPE 两解里取重投影误差最小的（solvePnP 默认行为），只用它的 tvec；姿态由 optimizeYaw 重解。
static bool solveArmor(const Detection& d, ArmorPose& pose) {
    const auto uv = orderCorners(d);
    if (!cv::solvePnP(kArmorPts, uv, cameraK(), cameraD(), pose.rvec, pose.tvec, false,
                      cv::SOLVEPNP_IPPE)) {
        return false;
    }
    if (!std::isfinite(pose.tvec[2]) || pose.tvec[2] < 0.15) {
        return false;
    }
    const double yaw = optimizeYaw(uv, pose.tvec);
    pose.rvec = yawToRvec(yaw);
    pose.rpy = {0, -15 * kDeg, yaw};
    return true;
}

static std::vector<cv::Point> project(const std::vector<cv::Point3f>& obj, const ArmorPose& pose) {
    std::vector<cv::Point2f> uv;
    cv::projectPoints(obj, pose.rvec, pose.tvec, cameraK(), cameraD(), uv);
    std::vector<cv::Point> out;
    for (const auto& p : uv) {
        out.emplace_back(cv::Point2i(p));
    }
    return out;
}

static void drawPose(cv::Mat& img, const ArmorPose& pose) {
    const auto quad = project(kArmorPts, pose);
    for (size_t i = 0; i < quad.size(); ++i) {
        cv::line(img, quad[i], quad[(i + 1) % quad.size()], cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
    }
    constexpr float L = 0.10f;
    const auto ax = project({{0, 0, 0}, {L, 0, 0}, {0, L, 0}, {0, 0, L}}, pose);
    cv::arrowedLine(img, ax[0], ax[1], cv::Scalar(0, 0, 255), 2, cv::LINE_AA, 0, 0.25);
    cv::arrowedLine(img, ax[0], ax[2], cv::Scalar(0, 255, 0), 2, cv::LINE_AA, 0, 0.25);
    cv::arrowedLine(img, ax[0], ax[3], cv::Scalar(255, 0, 0), 2, cv::LINE_AA, 0, 0.25);
    char buf[80];
    std::snprintf(buf, sizeof(buf), "R%+.0f P%+.0f Y%+.0f  %.2fm", pose.rpy.x() / kDeg,
                  pose.rpy.y() / kDeg, pose.rpy.z() / kDeg, pose.tvec[2]);
    cv::putText(img, buf, ax[0] + cv::Point(12, -10), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}

// 把 EKF 估出的整车画出来：车中心 + 四块装甲板的重投影框。
// 当前板用 drawPose 画完整姿态（位置 = 中心 + 半径×yaw 方向，姿态 = yaw 与 -15° 仰装），其余黄色框。
static void drawCar(cv::Mat& img, const EKF& ekf) {
    const auto& s = ekf.state();
    std::vector<cv::Point2f> uv;
    cv::projectPoints(std::vector<cv::Point3f>{{float(s[0]), float(s[1]), float(s[2])}}, cv::Vec3d(),
                      cv::Vec3d(), cameraK(), cameraD(), uv);
    cv::circle(img, uv[0], 5, cv::Scalar(255, 0, 255), -1, cv::LINE_AA);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "v_yaw %+.1f  r %.2f/%.2f", s[6], s[7], ekf.anotherR());
    cv::putText(img, buf, uv[0] + cv::Point2f(10, -8), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 0, 255), 1, cv::LINE_AA);

    for (int i = 0; i < 4; ++i) {
        Eigen::Vector3d p;
        double yaw;
        ekf.armor(i, p, yaw);
        if (std::abs(yaw) > 100 * kDeg) {
            // continue;  // 背面的板不画
        }
        ArmorPose pose;
        pose.rvec = yawToRvec(yaw);
        pose.tvec = {p.x(), p.y(), p.z()};
        pose.rpy = {0, -15 * kDeg, yaw};
        if (i == 0) {
            drawPose(img, pose);
            continue;
        }
        const auto quad = project(kArmorPts, pose);
        for (size_t k = 0; k < 4; ++k) {
            cv::line(img, quad[k], quad[(k + 1) % 4], cv::Scalar(0, 200, 255), 2, cv::LINE_AA);
        }
    }
}

int main() {
    OnboardShm shm;
    uint32_t last = 0;
    bool have_last = false;
    cv::Mat bgr;

    const std::string enginePath = "/home/ubuntu/yolov5_fourpoints/runs/train/shufflev2-armor-sim/weights/best-fp32.engine";
    auto engineData = loadFile(enginePath);
    auto runtime = nvinfer1::createInferRuntime(gLogger);
    auto engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    auto context = engine->createExecutionContext();
    if (engine->getTensorDataType("images") != engine->getTensorDataType("output0")) {
        std::cerr << "engine in/out dtype mismatch\n";
        return 1;
    }

    const size_t inSize = 3 * INPUT_H * INPUT_W, outSize = size_t(NUM_PRED) * NUM_COLS;
    void *dIn = nullptr, *dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dIn, inSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOut, outSize * sizeof(float)));
    context->setTensorAddress("images", dIn);
    context->setTensorAddress("output0", dOut);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    std::vector<float> blob(inSize), output(outSize);
    for (int i = 0; i < 20; ++i) {
        context->enqueueV3(stream);
        cudaStreamSynchronize(stream);
    }

    using Clock = std::chrono::steady_clock;
    const auto t_start = Clock::now();
    auto t_last = t_start;

    EKF ekf;
    int lost = 100;  // 连续丢失帧数；>=100 视为未跟踪

    while (true) {
        uint32_t seq = 0;
        double stamp = 0.0;
        if (!shm.read(&seq, &stamp, &bgr) || (have_last && seq == last)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
            continue;
        }
        last = seq;
        have_last = true;

        const auto t_now = Clock::now();
        const double t = std::chrono::duration<double>(t_now - t_start).count();
        const double dt = std::chrono::duration<double>(t_now - t_last).count();
        t_last = t_now;

        float gain = 0, padW = 0, padH = 0;
        cv::Mat resized = letterbox(bgr, gain, padW, padH);
        blobFromImage(resized, blob.data());
        CUDA_CHECK(cudaMemcpyAsync(dIn, blob.data(), inSize * sizeof(float), cudaMemcpyHostToDevice,
                                   stream));
        context->enqueueV3(stream);
        CUDA_CHECK(cudaMemcpyAsync(output.data(), dOut, outSize * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        const auto det = postprocess(output.data(), 0.25f, 0.35f, gain, padW, padH, bgr.cols, bgr.rows);
        std::vector<Detection> enemies;
        for (const auto& d : det) {
            if (kIsBlue == (d.cls < 6)) {
                enemies.push_back(d);
            }
        }
        std::vector<ObjectInfo> objects;
        drawDetections(bgr, enemies);
        for (const auto& d : enemies) {
            ArmorPose pose;
            if (!solveArmor(d, pose)) {
                continue;
            }
            const auto uv = orderCorners(d);
            ObjectInfo obj;
            obj.pixel_pos_ = {0.5 * (uv[0].x + uv[2].x), 0.5 * (uv[0].y + uv[2].y)};
            obj.gimbal_pos_ = {pose.tvec[0], pose.tvec[1], pose.tvec[2]};
            obj.gimbal_angle_ = pose.rpy;
            obj.id_ = d.cls;
            obj.color_ = d.cls < 6 ? 0 : 1;
            objects.push_back(obj);
        }

        // 整车 EKF：观测 = 本帧所有敌方板 (x y z yaw)。未跟踪时用第一块初始化；跟踪中预测 + 更新。
        std::vector<Eigen::Vector4d> zs;
        for (const auto& o : objects) {
            zs.emplace_back(o.gimbal_pos_.x(), o.gimbal_pos_.y(), o.gimbal_pos_.z(), o.gimbal_angle_.z());
        }
        ekf.setDt(dt);
        if (lost >= 100) {
            if (!zs.empty()) {
                ekf.init(zs[0]);
                lost = 0;
            }
        } else {
            ekf.predict();
            lost = ekf.update(zs) ? 0 : lost + 1;
            drawCar(bgr, ekf);
        }

        char time_buf[48];
        std::snprintf(time_buf, sizeof(time_buf), "t=%.3fs  dt=%.4fs", t, dt);
        cv::putText(bgr, time_buf, {12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255),
                    1, cv::LINE_AA);
        cv::imshow("aim input (onboard)", bgr);
        if ((cv::waitKey(1) & 0xFF) == 27) {
            break;
        }
    }
    return 0;
}

#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <vector>

inline constexpr int INPUT_W = 640;
inline constexpr int INPUT_H = 640;
inline constexpr int NUM_PRED = 25200;
inline constexpr int NUM_COLS = 21;
inline constexpr int NUM_CLS = 12;
inline constexpr const char* CLASS_NAMES[NUM_CLS] = {"B_G", "B_1", "B_2", "B_3", "B_4", "B_5",
                                                     "R_G", "R_1", "R_2", "R_3", "R_4", "R_5"};

class Logger : public nvinfer1::ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) std::cout << "[TRT] " << msg << std::endl;
    }
};

extern Logger gLogger;

#define CUDA_CHECK(x)                                                                    \
    do {                                                                                 \
        cudaError_t e = (x);                                                             \
        if (e != cudaSuccess) {                                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(e) << " @" << __LINE__     \
                      << std::endl;                                                      \
            std::exit(1);                                                                \
        }                                                                                \
    } while (0)

struct Detection {
    float kpts[8];  // 原图像素坐标 x1,y1,x2,y2,x3,y3,x4,y4
    float conf;
    int cls;
};

std::vector<char> loadFile(const std::string& path);
cv::Mat letterbox(const cv::Mat& img, float& gain, float& padW, float& padH);
void blobFromImage(const cv::Mat& img, float* blob);
std::vector<Detection> postprocess(const float* out, float confThres, float iouThres, float gain,
                                   float padW, float padH, int imgW, int imgH);
void drawDetections(cv::Mat& img, const std::vector<Detection>& dets);

#include "trtengine.h"

#include <algorithm>
#include <cmath>
#include <fstream>

Logger gLogger;

std::vector<char> loadFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cerr << "无法打开: " << path << std::endl;
        std::exit(1);
    }
    size_t size = f.tellg();
    f.seekg(0);
    std::vector<char> buf(size);
    f.read(buf.data(), size);
    return buf;
}

cv::Mat letterbox(const cv::Mat& img, float& gain, float& padW, float& padH) {
    gain = std::min(float(INPUT_W) / img.cols, float(INPUT_H) / img.rows);
    int newW = int(std::round(img.cols * gain)), newH = int(std::round(img.rows * gain));
    padW = (INPUT_W - newW) / 2.0f;
    padH = (INPUT_H - newH) / 2.0f;
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(newW, newH));
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(int(padW), int(padH), newW, newH)));
    return out;
}

void blobFromImage(const cv::Mat& img, float* blob) {
    const int area = INPUT_W * INPUT_H;
    for (int y = 0; y < INPUT_H; ++y) {
        const uchar* row = img.ptr<uchar>(y);
        for (int x = 0; x < INPUT_W; ++x) {
            int idx = y * INPUT_W + x;
            blob[0 * area + idx] = row[x * 3 + 2] / 255.0f;  // R
            blob[1 * area + idx] = row[x * 3 + 1] / 255.0f;  // G
            blob[2 * area + idx] = row[x * 3 + 0] / 255.0f;  // B
        }
    }
}

std::vector<Detection> postprocess(const float* out, float confThres, float iouThres, float gain,
                                   float padW, float padH, int imgW, int imgH) {
    struct Cand {
        float kpts[8], score;
        int cls;
        float box[4];  // xyxy (640 尺度)
    };
    std::vector<Cand> cands;
    for (int i = 0; i < NUM_PRED; ++i) {
        const float* row = out + i * NUM_COLS;
        float obj = row[8];
        if (obj < confThres) continue;
        int best = 0;
        float bestCls = 0;
        for (int c = 0; c < NUM_CLS; ++c)
            if (row[9 + c] > bestCls) bestCls = row[9 + c], best = c;
        float score = obj * bestCls;
        if (score < confThres) continue;
        Cand cd;
        std::copy(row, row + 8, cd.kpts);
        cd.score = score;
        cd.cls = best;
        float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
        for (int k = 0; k < 4; ++k) {
            x0 = std::min(x0, cd.kpts[k * 2]);
            x1 = std::max(x1, cd.kpts[k * 2]);
            y0 = std::min(y0, cd.kpts[k * 2 + 1]);
            y1 = std::max(y1, cd.kpts[k * 2 + 1]);
        }
        cd.box[0] = x0;
        cd.box[1] = y0;
        cd.box[2] = x1;
        cd.box[3] = y1;
        cands.push_back(cd);
    }
    std::sort(cands.begin(), cands.end(),
              [](const Cand& a, const Cand& b) { return a.score > b.score; });

    std::vector<Detection> dets;
    std::vector<bool> removed(cands.size(), false);
    const float iominThres = 0.6f;
    for (size_t i = 0; i < cands.size() && dets.size() < 50; ++i) {
        if (removed[i]) continue;
        const Cand& a = cands[i];
        for (size_t j = i + 1; j < cands.size(); ++j) {
            if (removed[j]) continue;
            const Cand& b = cands[j];
            float ix0 = std::max(a.box[0], b.box[0]), iy0 = std::max(a.box[1], b.box[1]);
            float ix1 = std::min(a.box[2], b.box[2]), iy1 = std::min(a.box[3], b.box[3]);
            float inter = std::max(0.f, ix1 - ix0) * std::max(0.f, iy1 - iy0);
            float areaA = (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]);
            float areaB = (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]);
            float iou = inter / (areaA + areaB - inter + 1e-7f);
            float iomin = inter / (std::min(areaA, areaB) + 1e-7f);
            if (iou > iouThres || iomin > iominThres) removed[j] = true;
        }
        Detection d;
        d.conf = a.score;
        d.cls = a.cls;
        for (int k = 0; k < 4; ++k) {
            d.kpts[k * 2] = std::clamp((a.kpts[k * 2] - padW) / gain, 0.f, float(imgW - 1));
            d.kpts[k * 2 + 1] = std::clamp((a.kpts[k * 2 + 1] - padH) / gain, 0.f, float(imgH - 1));
        }
        dets.push_back(d);
    }
    return dets;
}

void drawDetections(cv::Mat& img, const std::vector<Detection>& dets) {
    for (const auto& d : dets) {
        cv::Scalar color = d.cls < 6 ? cv::Scalar(255, 128, 0) : cv::Scalar(0, 64, 255);
        std::vector<cv::Point> pts(4);
        for (int k = 0; k < 4; ++k)
            pts[k] = cv::Point(int(d.kpts[k * 2]), int(d.kpts[k * 2 + 1]));
        for (int k = 0; k < 4; ++k) cv::line(img, pts[k], pts[(k + 1) % 4], color, 2);
        for (int k = 0; k < 4; ++k) cv::circle(img, pts[k], 3, cv::Scalar(0, 255, 0), -1);
        char label[64];
        snprintf(label, sizeof(label), "%s %.2f", CLASS_NAMES[d.cls], d.conf);
        cv::Point org = pts[0] + cv::Point(0, -6);
        cv::putText(img, label, org, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 3);
        cv::putText(img, label, org, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1);
    }
}

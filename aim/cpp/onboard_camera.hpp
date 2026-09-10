#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <opencv2/core.hpp>

// Must match scripts/onboard_feed.py
inline constexpr const char* kOnboardShmName = "/autoaim_onboard";
inline constexpr uint32_t kOnboardHeaderSize = 128;
inline constexpr char kOnboardMagic[4] = {'A', 'I', 'M', '1'};

#pragma pack(push, 1)
struct OnboardHeader {
    char magic[4];
    uint32_t width;
    uint32_t height;
    uint32_t seq;
    double stamp;
    uint32_t writing;
    uint32_t reserved;
    float xmat[9];
    float xpos[3];
    uint8_t pad[48];
};
#pragma pack(pop)

static_assert(sizeof(OnboardHeader) == kOnboardHeaderSize, "header must be 128 bytes");

class OnboardShm {
public:
    explicit OnboardShm(const char* name = kOnboardShmName) {
        fd_ = shm_open(name, O_RDONLY, 0);
        if (fd_ < 0) {
            throw std::runtime_error(
                "shm_open failed; start the sim first: python scripts/view_infantry.py --team red");
        }
        struct stat st {};
        if (fstat(fd_, &st) != 0) {
            close(fd_);
            throw std::runtime_error("fstat failed on shared memory");
        }
        size_ = static_cast<size_t>(st.st_size);
        map_ = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (map_ == MAP_FAILED) {
            close(fd_);
            throw std::runtime_error("mmap failed");
        }
    }

    ~OnboardShm() {
        if (map_ != MAP_FAILED && map_ != nullptr) {
            munmap(map_, size_);
        }
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    OnboardShm(const OnboardShm&) = delete;
    OnboardShm& operator=(const OnboardShm&) = delete;

    // Returns false if the writer is mid-update or the buffer is not ready.
    bool read(uint32_t* seq, double* stamp, cv::Mat* bgr) const {
        const auto* raw = static_cast<const uint8_t*>(map_);
        OnboardHeader h;
        std::memcpy(&h, raw, sizeof(h));
        if (std::memcmp(h.magic, kOnboardMagic, 4) != 0 || h.writing != 0 || h.width == 0 ||
            h.height == 0) {
            return false;
        }
        const size_t nbytes = static_cast<size_t>(h.width) * h.height * 3;
        if (kOnboardHeaderSize + nbytes > size_) {
            return false;
        }
        cv::Mat view(static_cast<int>(h.height), static_cast<int>(h.width), CV_8UC3,
                     const_cast<uint8_t*>(raw + kOnboardHeaderSize));
        view.copyTo(*bgr);
        OnboardHeader h2;
        std::memcpy(&h2, raw, sizeof(h2));
        if (h2.writing != 0 || h2.seq != h.seq) {
            return false;
        }
        *seq = h.seq;
        *stamp = h.stamp;
        return true;
    }

private:
    int fd_ = -1;
    size_t size_ = 0;
    void* map_ = MAP_FAILED;
};

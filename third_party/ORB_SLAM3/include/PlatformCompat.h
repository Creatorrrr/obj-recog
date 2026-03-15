#ifndef ORB_SLAM3_PLATFORM_COMPAT_H
#define ORB_SLAM3_PLATFORM_COMPAT_H

#ifdef _WIN32
#include <chrono>
#include <thread>

inline void usleep(unsigned int microseconds) {
    std::this_thread::sleep_for(std::chrono::microseconds(microseconds));
}

#ifndef __PRETTY_FUNCTION__
#define __PRETTY_FUNCTION__ __FUNCTION__
#endif
#else
#include <unistd.h>
#endif

#endif

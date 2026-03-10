#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>

#define private public
#include "System.h"
#undef private
#include "Atlas.h"
#include "KeyFrame.h"
#include "MapPoint.h"

namespace {

struct Options {
    std::string vocabulary_path;
    std::string settings_path;
    int width = 0;
    int height = 0;
};

bool ReadExact(std::istream& stream, char* dst, std::size_t count) {
    stream.read(dst, static_cast<std::streamsize>(count));
    return stream.gcount() == static_cast<std::streamsize>(count);
}

std::uint32_t ReadU32BE(const std::array<char, 4>& buffer) {
    return (static_cast<std::uint32_t>(static_cast<unsigned char>(buffer[0])) << 24) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(buffer[1])) << 16) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(buffer[2])) << 8) |
           static_cast<std::uint32_t>(static_cast<unsigned char>(buffer[3]));
}

std::uint64_t ReadU64BE(const std::array<char, 8>& buffer) {
    std::uint64_t value = 0;
    for (char byte : buffer) {
        value = (value << 8) | static_cast<std::uint64_t>(static_cast<unsigned char>(byte));
    }
    return value;
}

double ReadTimestampSeconds(const std::array<char, 8>& buffer) {
    return static_cast<double>(ReadU64BE(buffer)) / 1000000.0;
}

std::string Escape(const std::string& value) {
    std::ostringstream out;
    for (char ch : value) {
        if (ch == '"' || ch == '\\') {
            out << '\\' << ch;
        } else {
            out << ch;
        }
    }
    return out.str();
}

float MedianOf(std::vector<float>& values) {
    if (values.empty()) {
        return -1.0f;
    }
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle), values.end());
    float median = values[middle];
    if ((values.size() % 2U) == 0U) {
        const auto max_it = std::max_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle));
        median = (*max_it + median) * 0.5f;
    }
    return median;
}

std::array<float, 16> ToRowMajor(const Sophus::SE3f& pose) {
    const Eigen::Matrix4f matrix = pose.matrix();
    std::array<float, 16> values{};
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            values[static_cast<std::size_t>(row * 4 + col)] = matrix(row, col);
        }
    }
    return values;
}

std::string TrackingStateName(int state, bool relocalized) {
    if (relocalized) {
        return "RELOCALIZED";
    }
    switch (state) {
        case ORB_SLAM3::Tracking::OK:
        case ORB_SLAM3::Tracking::OK_KLT:
            return "TRACKING";
        case ORB_SLAM3::Tracking::RECENTLY_LOST:
            return "RELOCALIZING";
        case ORB_SLAM3::Tracking::LOST:
            return "LOST";
        case ORB_SLAM3::Tracking::NOT_INITIALIZED:
        case ORB_SLAM3::Tracking::NO_IMAGES_YET:
        default:
            return "INITIALIZING";
    }
}

Options ParseArgs(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--vocabulary" && index + 1 < argc) {
            options.vocabulary_path = argv[++index];
        } else if (arg == "--settings" && index + 1 < argc) {
            options.settings_path = argv[++index];
        } else if (arg == "--width" && index + 1 < argc) {
            options.width = std::stoi(argv[++index]);
        } else if (arg == "--height" && index + 1 < argc) {
            options.height = std::stoi(argv[++index]);
        } else {
            throw std::runtime_error("unexpected argument: " + arg);
        }
    }
    if (options.vocabulary_path.empty() || options.settings_path.empty() || options.width <= 0 || options.height <= 0) {
        throw std::runtime_error("usage: orbslam3_bridge --vocabulary <path> --settings <path> --width <w> --height <h>");
    }
    return options;
}

class ScopedStreamRedirect {
   public:
    explicit ScopedStreamRedirect(std::ostream& destination)
        : stream_(std::cout), old_buffer_(stream_.rdbuf(destination.rdbuf())) {}

    ScopedStreamRedirect(const ScopedStreamRedirect&) = delete;
    ScopedStreamRedirect& operator=(const ScopedStreamRedirect&) = delete;

    ~ScopedStreamRedirect() {
        stream_.rdbuf(old_buffer_);
    }

   private:
    std::ostream& stream_;
    std::streambuf* old_buffer_;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = ParseArgs(argc, argv);
        std::unique_ptr<ORB_SLAM3::System> system;
        {
            ScopedStreamRedirect redirect(std::cerr);
            system = std::make_unique<ORB_SLAM3::System>(
                options.vocabulary_path,
                options.settings_path,
                ORB_SLAM3::System::MONOCULAR,
                false
            );
        }

        std::set<unsigned long> known_keyframe_ids;
        int previous_tracking_state = ORB_SLAM3::Tracking::NO_IMAGES_YET;

        while (true) {
            std::array<char, 4> magic{};
            if (!ReadExact(std::cin, magic.data(), magic.size())) {
                break;
            }
            if (std::string(magic.data(), magic.size()) != "SLAM") {
                throw std::runtime_error("invalid frame packet header");
            }

            std::array<char, 8> timestamp_bytes{};
            std::array<char, 4> width_bytes{};
            std::array<char, 4> height_bytes{};
            if (!ReadExact(std::cin, timestamp_bytes.data(), timestamp_bytes.size()) ||
                !ReadExact(std::cin, width_bytes.data(), width_bytes.size()) ||
                !ReadExact(std::cin, height_bytes.data(), height_bytes.size())) {
                throw std::runtime_error("truncated frame packet");
            }

            const double timestamp = ReadTimestampSeconds(timestamp_bytes);
            const int width = static_cast<int>(ReadU32BE(width_bytes));
            const int height = static_cast<int>(ReadU32BE(height_bytes));
            std::vector<unsigned char> payload(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
            if (!ReadExact(std::cin, reinterpret_cast<char*>(payload.data()), payload.size())) {
                throw std::runtime_error("truncated grayscale payload");
            }

            cv::Mat image(height, width, CV_8UC1, payload.data());
            Sophus::SE3f tracked_pose;
            bool map_changed = false;
            int tracking_state = ORB_SLAM3::Tracking::NO_IMAGES_YET;
            {
                ScopedStreamRedirect redirect(std::cerr);
                tracked_pose = system->TrackMonocular(image.clone(), timestamp);
                map_changed = system->MapChanged();
                tracking_state = system->GetTrackingState();
            }
            const bool relocalized =
                (previous_tracking_state == ORB_SLAM3::Tracking::LOST ||
                 previous_tracking_state == ORB_SLAM3::Tracking::RECENTLY_LOST) &&
                (tracking_state == ORB_SLAM3::Tracking::OK || tracking_state == ORB_SLAM3::Tracking::OK_KLT);
            previous_tracking_state = tracking_state;

            std::vector<ORB_SLAM3::KeyFrame*> keyframes;
            std::vector<ORB_SLAM3::MapPoint*> map_points;
            if (system->mpAtlas != nullptr) {
                keyframes = system->mpAtlas->GetAllKeyFrames();
                map_points = system->mpAtlas->GetAllMapPoints();
            }

            std::sort(keyframes.begin(), keyframes.end(), [](const auto* lhs, const auto* rhs) {
                return lhs->mnId < rhs->mnId;
            });

            bool keyframe_inserted = false;
            long long inserted_keyframe_id = -1;
            std::set<unsigned long> current_keyframe_ids;
            std::map<unsigned long, std::array<float, 16>> optimized_keyframes;
            for (ORB_SLAM3::KeyFrame* keyframe : keyframes) {
                if (keyframe == nullptr || keyframe->isBad()) {
                    continue;
                }
                current_keyframe_ids.insert(keyframe->mnId);
                optimized_keyframes[keyframe->mnId] = ToRowMajor(keyframe->GetPoseInverse());
                if (!known_keyframe_ids.count(keyframe->mnId)) {
                    keyframe_inserted = true;
                    inserted_keyframe_id = std::max(inserted_keyframe_id, static_cast<long long>(keyframe->mnId));
                }
            }
            known_keyframe_ids = current_keyframe_ids;
            const bool map_points_changed = map_changed || keyframe_inserted;

            std::vector<std::array<float, 3>> sparse_points;
            sparse_points.reserve(map_points.size());
            for (ORB_SLAM3::MapPoint* map_point : map_points) {
                if (map_point == nullptr || map_point->isBad()) {
                    continue;
                }
                const Eigen::Vector3f position = map_point->GetWorldPos();
                sparse_points.push_back({position.x(), position.y(), position.z()});
            }
            if (sparse_points.size() > 4000) {
                std::vector<std::array<float, 3>> reduced;
                reduced.reserve(4000);
                const double step = static_cast<double>(sparse_points.size() - 1) / 3999.0;
                for (int index = 0; index < 4000; ++index) {
                    reduced.push_back(sparse_points[static_cast<std::size_t>(index * step)]);
                }
                sparse_points.swap(reduced);
            }

            int tracked_feature_count = 0;
            float median_reprojection_error = -1.0f;
            if (system->mpTracker != nullptr) {
                const ORB_SLAM3::Frame& current_frame = system->mpTracker->mCurrentFrame;
                std::vector<float> reprojection_errors;
                const std::size_t limit = std::min({current_frame.mvpMapPoints.size(), current_frame.mvbOutlier.size(), current_frame.mvKeysUn.size()});
                reprojection_errors.reserve(limit);
                for (std::size_t index = 0; index < limit; ++index) {
                    ORB_SLAM3::MapPoint* map_point = current_frame.mvpMapPoints[index];
                    if (map_point == nullptr || map_point->isBad() || current_frame.mvbOutlier[index]) {
                        continue;
                    }
                    const Eigen::Vector3f world = map_point->GetWorldPos();
                    const Eigen::Vector3f camera = tracked_pose * world;
                    if (camera.z() <= 1e-6f) {
                        continue;
                    }
                    const cv::KeyPoint& keypoint = current_frame.mvKeysUn[index];
                    const float projected_u = current_frame.fx * (camera.x() / camera.z()) + current_frame.cx;
                    const float projected_v = current_frame.fy * (camera.y() / camera.z()) + current_frame.cy;
                    const float du = projected_u - keypoint.pt.x;
                    const float dv = projected_v - keypoint.pt.y;
                    reprojection_errors.push_back(std::sqrt((du * du) + (dv * dv)));
                    ++tracked_feature_count;
                }
                median_reprojection_error = MedianOf(reprojection_errors);
            }

            struct ObservationPayload {
                unsigned long keyframe_id;
                unsigned long point_id;
                float u;
                float v;
                float x;
                float y;
                float z;
            };
            std::vector<ObservationPayload> keyframe_observations;
            if (map_points_changed) {
                constexpr std::size_t kMaxObservationKeyframes = 8;
                constexpr std::size_t kMaxObservationsPerKeyframe = 160;
                std::vector<ORB_SLAM3::KeyFrame*> selected_keyframes = keyframes;
                std::sort(
                    selected_keyframes.begin(),
                    selected_keyframes.end(),
                    [](const ORB_SLAM3::KeyFrame* left, const ORB_SLAM3::KeyFrame* right) {
                        const auto left_id = left == nullptr ? 0UL : left->mnId;
                        const auto right_id = right == nullptr ? 0UL : right->mnId;
                        return left_id < right_id;
                    }
                );
                if (selected_keyframes.size() > kMaxObservationKeyframes) {
                    selected_keyframes.erase(
                        selected_keyframes.begin(),
                        selected_keyframes.end() - static_cast<std::ptrdiff_t>(kMaxObservationKeyframes)
                    );
                }
                for (ORB_SLAM3::KeyFrame* keyframe : selected_keyframes) {
                    if (keyframe == nullptr || keyframe->isBad()) {
                        continue;
                    }
                    const std::vector<ORB_SLAM3::MapPoint*> point_matches = keyframe->GetMapPointMatches();
                    const std::size_t observation_limit = std::min(point_matches.size(), keyframe->mvKeysUn.size());
                    std::vector<ObservationPayload> keyframe_payloads;
                    keyframe_payloads.reserve(std::min(observation_limit, kMaxObservationsPerKeyframe));
                    for (std::size_t index = 0; index < observation_limit; ++index) {
                        ORB_SLAM3::MapPoint* map_point = point_matches[index];
                        if (map_point == nullptr || map_point->isBad()) {
                            continue;
                        }
                        const Eigen::Vector3f position = map_point->GetWorldPos();
                        const cv::KeyPoint& keypoint = keyframe->mvKeysUn[index];
                        keyframe_payloads.push_back(
                            ObservationPayload{
                                keyframe->mnId,
                                map_point->mnId,
                                keypoint.pt.x,
                                keypoint.pt.y,
                                position.x(),
                                position.y(),
                                position.z(),
                            }
                        );
                    }
                    if (keyframe_payloads.size() > kMaxObservationsPerKeyframe) {
                        std::vector<ObservationPayload> reduced;
                        reduced.reserve(kMaxObservationsPerKeyframe);
                        const double step =
                            static_cast<double>(keyframe_payloads.size() - 1) /
                            static_cast<double>(kMaxObservationsPerKeyframe - 1);
                        for (std::size_t index = 0; index < kMaxObservationsPerKeyframe; ++index) {
                            reduced.push_back(
                                keyframe_payloads[static_cast<std::size_t>(std::round(step * index))]
                            );
                        }
                        keyframe_payloads.swap(reduced);
                    }
                    keyframe_observations.insert(
                        keyframe_observations.end(),
                        keyframe_payloads.begin(),
                        keyframe_payloads.end()
                    );
                }
            }

            std::array<float, 16> pose_matrix = ToRowMajor(tracked_pose.inverse());
            if (tracking_state == ORB_SLAM3::Tracking::LOST ||
                tracking_state == ORB_SLAM3::Tracking::NOT_INITIALIZED ||
                tracking_state == ORB_SLAM3::Tracking::NO_IMAGES_YET) {
                pose_matrix = ToRowMajor(Sophus::SE3f());
            }

            std::ostringstream output;
            output << std::fixed << std::setprecision(6);
            output << "{\"tracking_state\":\"" << Escape(TrackingStateName(tracking_state, relocalized)) << "\",";
            output << "\"pose_world\":[";
            for (std::size_t index = 0; index < pose_matrix.size(); ++index) {
                if (index) output << ',';
                output << pose_matrix[index];
            }
            output << "],";
            output << "\"tracked_feature_count\":" << tracked_feature_count << ",";
            if (median_reprojection_error >= 0.0f) {
                output << "\"median_reprojection_error\":" << median_reprojection_error << ",";
            } else {
                output << "\"median_reprojection_error\":null,";
            }
            output << "\"keyframe_inserted\":" << (keyframe_inserted ? "true" : "false") << ",";
            if (keyframe_inserted && inserted_keyframe_id >= 0) {
                output << "\"keyframe_id\":" << inserted_keyframe_id << ",";
            } else {
                output << "\"keyframe_id\":null,";
            }
            output << "\"optimized_keyframe_poses\":{";
            bool first_pose = true;
            for (const auto& [keyframe_id, pose] : optimized_keyframes) {
                if (!first_pose) output << ',';
                first_pose = false;
                output << '"' << keyframe_id << "\":[";
                for (std::size_t index = 0; index < pose.size(); ++index) {
                    if (index) output << ',';
                    output << pose[index];
                }
                output << ']';
            }
            output << "},";
            output << "\"sparse_map_points\":[";
            for (std::size_t index = 0; index < sparse_points.size(); ++index) {
                if (index) output << ',';
                output << '[' << sparse_points[index][0] << ',' << sparse_points[index][1] << ',' << sparse_points[index][2] << ']';
            }
            output << "],";
            output << "\"keyframe_observations\":[";
            for (std::size_t index = 0; index < keyframe_observations.size(); ++index) {
                if (index) output << ',';
                const ObservationPayload& observation = keyframe_observations[index];
                output << "{\"keyframe_id\":" << observation.keyframe_id
                       << ",\"point_id\":" << observation.point_id
                       << ",\"u\":" << observation.u
                       << ",\"v\":" << observation.v
                       << ",\"x\":" << observation.x
                       << ",\"y\":" << observation.y
                       << ",\"z\":" << observation.z << '}';
            }
            output << "],";
            output << "\"map_points_changed\":" << (map_points_changed ? "true" : "false") << ",";
            output << "\"map_changed\":" << (map_changed ? "true" : "false");
            output << "}\n";
            std::cout << output.str() << std::flush;
        }

        system->Shutdown();
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << std::endl;
        return 1;
    }
}

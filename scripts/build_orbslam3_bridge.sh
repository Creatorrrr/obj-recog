#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BRIDGE_DIR="${ROOT_DIR}/native/orbslam3_bridge"
BUILD_DIR="${BRIDGE_DIR}/build"
ORB_SLAM3_ROOT="${OBJ_RECOG_ORB_SLAM3_ROOT:-${ROOT_DIR}/third_party/ORB_SLAM3}"

normalize_cmake_versions() {
  python3 - "$1" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
targets = (
    root / "CMakeLists.txt",
    root / "Thirdparty" / "DBoW2" / "CMakeLists.txt",
    root / "Thirdparty" / "g2o" / "CMakeLists.txt",
    root / "Thirdparty" / "Sophus" / "CMakeLists.txt",
)
for path in targets:
    text = path.read_text(encoding="utf-8")
    updated = re.sub(
        r"(?im)^cmake_minimum_required\s*\(\s*version\s+[0-9.]+\s*\)\s*$",
        "cmake_minimum_required(VERSION 3.5)",
        text,
        count=1,
    )
    path.write_text(updated, encoding="utf-8")
PY
}

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake is required to build orbslam3_bridge" >&2
  exit 1
fi

if [ ! -d "${ORB_SLAM3_ROOT}" ]; then
  echo "ORB_SLAM3 checkout not found at ${ORB_SLAM3_ROOT}" >&2
  echo "Run scripts/fetch_orbslam3.sh first or set OBJ_RECOG_ORB_SLAM3_ROOT." >&2
  exit 1
fi

normalize_cmake_versions "${ORB_SLAM3_ROOT}"

if command -v brew >/dev/null 2>&1; then
  OPENCV_PREFIX="$(brew --prefix opencv 2>/dev/null || true)"
  EIGEN_PREFIX="$(brew --prefix eigen 2>/dev/null || true)"
  BOOST_PREFIX="$(brew --prefix boost 2>/dev/null || true)"
  OPENSSL_PREFIX="$(brew --prefix openssl@3 2>/dev/null || true)"

  CMAKE_PREFIX_ENTRIES=()
  if [ -d "${OPENCV_PREFIX}" ]; then
    export OpenCV_DIR="${OPENCV_PREFIX}/lib/cmake/opencv4"
    CMAKE_PREFIX_ENTRIES+=("${OPENCV_PREFIX}")
  fi
  if [ -d "${EIGEN_PREFIX}" ]; then
    export Eigen3_DIR="${EIGEN_PREFIX}/share/eigen3/cmake"
    CMAKE_PREFIX_ENTRIES+=("${EIGEN_PREFIX}")
    export CPLUS_INCLUDE_PATH="${EIGEN_PREFIX}/include/eigen3${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
    export CPATH="${EIGEN_PREFIX}/include/eigen3${CPATH:+:${CPATH}}"
  fi
  if [ -d "${BOOST_PREFIX}" ]; then
    export BOOST_ROOT="${BOOST_PREFIX}"
    export LDFLAGS="-L${BOOST_PREFIX}/lib ${LDFLAGS:-}"
    export CPPFLAGS="-I${BOOST_PREFIX}/include ${CPPFLAGS:-}"
    CMAKE_PREFIX_ENTRIES+=("${BOOST_PREFIX}")
    export CPLUS_INCLUDE_PATH="${BOOST_PREFIX}/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
    export CPATH="${BOOST_PREFIX}/include${CPATH:+:${CPATH}}"
  fi
  if [ -d "${OPENSSL_PREFIX}" ]; then
    export LDFLAGS="-L${OPENSSL_PREFIX}/lib ${LDFLAGS:-}"
    export CPPFLAGS="-I${OPENSSL_PREFIX}/include ${CPPFLAGS:-}"
    CMAKE_PREFIX_ENTRIES+=("${OPENSSL_PREFIX}")
  fi
  if [ "${#CMAKE_PREFIX_ENTRIES[@]}" -gt 0 ]; then
    CMAKE_PREFIX_JOINED="$(IFS=:; echo "${CMAKE_PREFIX_ENTRIES[*]}")"
    export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_JOINED}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
  fi
fi

VOCAB_TXT="${ORB_SLAM3_ROOT}/Vocabulary/ORBvoc.txt"
VOCAB_ARCHIVE="${ORB_SLAM3_ROOT}/Vocabulary/ORBvoc.txt.tar.gz"
if [ ! -f "${VOCAB_TXT}" ]; then
  if [ ! -f "${VOCAB_ARCHIVE}" ]; then
    echo "ORB-SLAM3 vocabulary archive not found at ${VOCAB_ARCHIVE}" >&2
    exit 1
  fi
  tar -xf "${VOCAB_ARCHIVE}" -C "${ORB_SLAM3_ROOT}/Vocabulary"
fi

if [ ! -f "${ORB_SLAM3_ROOT}/lib/libORB_SLAM3.dylib" ] && [ ! -f "${ORB_SLAM3_ROOT}/lib/libORB_SLAM3.so" ]; then
  cmake -S "${ORB_SLAM3_ROOT}" -B "${ORB_SLAM3_ROOT}/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DORB_SLAM3_HEADLESS_VIEWER=ON \
    -DORB_SLAM3_BUILD_EXAMPLES=OFF
  cmake --build "${ORB_SLAM3_ROOT}/build" --target ORB_SLAM3 --config Release
fi

cmake -S "${BRIDGE_DIR}" -B "${BUILD_DIR}" -DORB_SLAM3_ROOT="${ORB_SLAM3_ROOT}"
cmake --build "${BUILD_DIR}" --config Release

echo "Built ${BUILD_DIR}/orbslam3_bridge"

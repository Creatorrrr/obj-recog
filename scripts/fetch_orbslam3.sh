#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${ROOT_DIR}/third_party/ORB_SLAM3"
PINNED_COMMIT="${OBJ_RECOG_ORB_SLAM3_COMMIT:-4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4}"

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

if ! command -v git >/dev/null 2>&1; then
  echo "git is required to fetch ORB_SLAM3" >&2
  exit 1
fi

mkdir -p "${ROOT_DIR}/third_party"

if [ ! -d "${TARGET_DIR}" ]; then
  git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git "${TARGET_DIR}"
fi

git -C "${TARGET_DIR}" fetch --depth 1 origin "${PINNED_COMMIT}"
git -C "${TARGET_DIR}" checkout "${PINNED_COMMIT}"
normalize_cmake_versions "${TARGET_DIR}"

VOCAB_TXT="${TARGET_DIR}/Vocabulary/ORBvoc.txt"
VOCAB_ARCHIVE="${TARGET_DIR}/Vocabulary/ORBvoc.txt.tar.gz"
if [ ! -f "${VOCAB_TXT}" ] && [ -f "${VOCAB_ARCHIVE}" ]; then
  tar -xf "${VOCAB_ARCHIVE}" -C "${TARGET_DIR}/Vocabulary"
fi

echo "ORB_SLAM3 available at ${TARGET_DIR} (${PINNED_COMMIT})"

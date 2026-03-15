#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
UNITY_EDITOR_PATH="${UNITY_EDITOR_PATH:-/Applications/Unity/Hub/Editor/6000.3.11f1/Unity.app}"
DEFAULT_OUTPUT_PATH="${ROOT_DIR}/build/unity/macos/obj-recog-unity.app"
OUTPUT_PATH="${1:-${DEFAULT_OUTPUT_PATH}}"

if [[ "${OUTPUT_PATH}" != /* ]]; then
  OUTPUT_PATH="${ROOT_DIR}/${OUTPUT_PATH}"
fi

if [[ "${UNITY_EDITOR_PATH}" == *.app ]]; then
  UNITY_EDITOR_BIN="${UNITY_EDITOR_PATH}/Contents/MacOS/Unity"
else
  UNITY_EDITOR_BIN="${UNITY_EDITOR_PATH}"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -x "${UNITY_EDITOR_BIN}" ]]; then
  echo "Unity editor not found: ${UNITY_EDITOR_BIN}" >&2
  echo "Set UNITY_EDITOR_PATH to a Unity.app bundle or Unity binary." >&2
  exit 1
fi

if [[ "${OUTPUT_PATH}" != *.app ]]; then
  echo "macOS Unity player output must end with .app: ${OUTPUT_PATH}" >&2
  exit 1
fi

PYTHONPATH="${ROOT_DIR}/src" "${PYTHON_BIN}" -m obj_recog.unity_vendor_check --unity-project-root "${ROOT_DIR}/unity"

mkdir -p "$(dirname "${OUTPUT_PATH}")"

"${UNITY_EDITOR_BIN}" \
  -batchmode \
  -quit \
  -projectPath "${ROOT_DIR}/unity" \
  -buildTarget StandaloneOSX \
  -executeMethod ObjRecog.UnitySim.Editor.MacOsBuild.BuildMacOsPlayer \
  --obj-recog-build-output="${OUTPUT_PATH}" \
  -logFile -

if [[ ! -d "${OUTPUT_PATH}" ]]; then
  echo "Unity build did not produce app bundle: ${OUTPUT_PATH}" >&2
  exit 1
fi

echo "Built ${OUTPUT_PATH}"

# obj-recog

실시간 객체 인식, 단안 3D 복원, Unity 기반 시뮬레이션을 한 프로젝트 안에서 실험할 수 있도록 만든 연구/데모용 런타임입니다.

이 저장소는 단순히 카메라 영상에서 물체를 검출하는 수준이 아니라, 다음 흐름을 하나로 묶습니다.

1. 입력 프레임 수집
2. 객체 검출, 깊이 추정, 파노픽 세그멘테이션
3. 카메라 자세 추정과 SLAM 연동
4. 포인트클라우드 또는 TSDF 메쉬 기반 환경 복원
5. 장면 그래프와 상황 설명 생성
6. Unity 시뮬레이터와의 상호작용 및 에피소드 리포트 저장

## 핵심 기능

- 웹캠 입력 또는 Unity 시뮬레이터 입력을 동일한 파이프라인으로 처리
- YOLO 기반 실시간 객체 검출
- MiDaS 기반 단안 깊이 추정
- Mask2Former 기반 파노픽 세그멘테이션
- ORB 특징 추적 및 ORB-SLAM3 브리지 기반 카메라 포즈 추정
- Open3D 기반 포인트클라우드/메쉬 시각화
- 장면 그래프 기반 공간 관계 메모리 유지
- OpenAI API를 활용한 한국어 상황 설명 및 시뮬레이션 플래너
- Unity Living Room 환경에서 RGB-only 계약으로 에이전트 실험

## 이 프로젝트에서 사용하는 기술

처음 저장소를 볼 때 가장 헷갈리는 부분은 "어떤 기술이 어디에 쓰이는가"입니다. 아래 표만 먼저 보면 전체 그림을 빠르게 잡을 수 있습니다.

| 영역 | 사용 기술 | 역할 |
| --- | --- | --- |
| 런타임 언어 | Python 3.11+ | 전체 파이프라인 오케스트레이션 |
| 수치 연산 | NumPy, SciPy | 좌표 변환, 깊이/포인트클라우드 처리 |
| 비전 입출력 | OpenCV | 카메라 입력, 이미지 전처리, UI 창 렌더링 |
| 딥러닝 실행 | PyTorch | 검출/깊이/세그멘테이션 모델 실행 |
| 객체 검출 | Ultralytics YOLO | 프레임 내 객체 박스와 클래스 추론 |
| 깊이 추정 | MiDaS (`intel-isl/MiDaS`) | 단안 영상에서 의사 깊이 맵 생성 |
| 세그멘테이션 | Hugging Face Transformers, Mask2Former | 물체/구조물 단위 장면 분할 |
| 추적/복원 | ORB 특징 추적, ORB-SLAM3 브리지 | 카메라 이동 추정, 키프레임 기반 맵 구성 |
| 3D 시각화 | Open3D | 포인트클라우드/메쉬 뷰어 |
| 장면 관계 표현 | NetworkX | 객체 간 공간 관계를 그래프로 유지 |
| LLM 연동 | OpenAI API | 한국어 상황 설명, 시뮬레이션 플래닝 |
| 시뮬레이션 | Unity 6 LTS | Living Room 환경, RGB 프레임 공급, 행동 실행 |

실제로는 `src/obj_recog/main.py`가 위 기술들을 한 런타임 안에서 묶어 주는 중심 진입점입니다.

## 구조를 쉽게 이해하는 방법

이 프로젝트는 "입력 -> 인식 -> 위치 추정 -> 맵 구성 -> 설명/행동" 구조로 보면 이해가 쉽습니다.

### 1. 입력 계층

- `frame_source.py`: 웹캠 또는 시뮬레이터 프레임을 `FramePacket` 형태로 통일
- `camera.py`: 실제 카메라 열기, 읽기, 장치 목록 조회
- `unity_rgb.py`: Unity 플레이어와 TCP로 연결해 RGB 프레임을 주고받음

즉, 입력이 라이브 카메라든 Unity든 이후 파이프라인은 거의 같은 방식으로 흘러갑니다.

### 2. 인식 계층

- `detector.py`: YOLO로 객체 박스 검출
- `depth.py`: MiDaS로 단안 깊이 추정
- `segmenter.py`: Mask2Former로 파노픽 세그멘테이션

여기서 만들어진 결과는 단순한 시각화 용도가 아니라, 뒤쪽의 추적과 맵 구성에도 사용됩니다.

### 3. 위치 추정과 복원 계층

- `tracking.py`: ORB 특징점과 PnP 기반 추적
- `slam_bridge.py`: ORB-SLAM3 네이티브 브리지 연동
- `reconstruct.py`: 깊이 맵을 3D 포인트로 역투영
- `mapping.py`: 로컬 포인트맵 또는 TSDF 메쉬 구성

쉽게 말하면 "현재 카메라가 어디에 있는지"와 "지금까지 본 공간을 어떻게 누적할지"를 담당하는 영역입니다.

### 4. 의미 해석 계층

- `scene_graph.py`: 보이는 객체와 구조물의 관계를 그래프로 유지
- `situation_explainer.py`: 현재 프레임과 그래프 정보를 바탕으로 한국어 설명 생성
- `sim_planner.py`: 시뮬레이터에서 다음 행동을 계획

즉, 단순 감지 결과를 넘어서 "무엇이 어디에 있고 지금 어떤 상황인가"를 사람이 읽기 쉬운 형태로 정리합니다.

### 5. 실행 및 시뮬레이션 계층

- `main.py`: 전체 런타임 진입점
- `simulation.py`: Living Room 시나리오 에피소드 실행, 행동 적용, 리포트 저장
- `visualization.py`: OpenCV/Open3D 창 렌더링

라이브 실행에서는 카메라 데모처럼 동작하고, 시뮬레이션 실행에서는 Unity와 상호작용하는 에이전트 런타임처럼 동작합니다.

## 디렉터리 구조

```text
obj-recog/
├── src/obj_recog/          # Python 패키지 본체
├── tests/                  # 단위 테스트 및 통합 테스트
├── unity/                  # Unity Living Room 프로젝트
├── scripts/                # 빌드/검증/보조 스크립트
├── native/orbslam3_bridge/ # ORB-SLAM3 네이티브 브리지
├── third_party/            # 외부 서드파티 소스 체크아웃
├── calibration/            # 카메라 보정 파일
├── models/                 # 런타임에서 참조하는 모델 파일
├── reports/                # 시뮬레이션 결과 및 디버그 산출물
└── docs/plans/             # 설계/작업 문서
```

자주 보게 되는 핵심 하위 모듈은 아래와 같습니다.

- `src/obj_recog/config.py`: CLI 옵션과 런타임 설정 정의
- `src/obj_recog/main.py`: 전체 앱 실행
- `src/obj_recog/simulation.py`: Unity 기반 에피소드 러너
- `src/obj_recog/mapping.py`: 포인트맵/메쉬 맵 구성
- `src/obj_recog/scene_graph.py`: 객체 관계 그래프
- `src/obj_recog/situation_explainer.py`: 한국어 설명 생성

## 사전 요구 사항

- Python 3.11+
- `pip` 또는 다른 Python 패키지 관리자
- 선택 사항: 설명 창에서 LLM을 호출하려면 `OPENAI_API_KEY`
- 시뮬레이션 선택 사항: `unity` 폴더 프로젝트를 열기 위한 Unity 6 LTS `6000.3.11f1`

## 설치

저장소 루트에서 실행합니다.

```bash
python -m pip install -e .[dev]
```

Windows x64 + Python 3.12 환경에서는 프로젝트 메타데이터가 PyTorch/TorchVision을 CUDA 12.8 공식 휠로 고정하므로, 새 환경에서는 GPU 지원이 자동으로 잡히는 구성이 되어 있습니다. 다른 플랫폼이나 Python 버전에서는 기본 PyPI 패키지를 사용합니다.

설치 후 런타임이 어떻게 해석됐는지 확인하려면:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

## Windows CUDA OpenCV

기본 `opencv-python` 휠은 CPU 전용입니다. `--opencv-cuda on` 옵션을 실제로 사용하려면 CUDA 지원 OpenCV를 소스에서 빌드해 현재 `.venv`에 설치해야 합니다.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_opencv_cuda_windows.ps1
```

필요한 도구:

- NVIDIA CUDA Toolkit 12.8
- C++ 워크로드가 포함된 Visual Studio 2022 Build Tools
- CMake
- Ninja

이 스크립트는 공식 `opencv`와 `opencv_contrib` `4.13.0` 소스를 받아 빌드합니다. `opencv_contrib`는 CUDA 이미지 처리 모듈이 `cudev`에 의존하므로 필수입니다.

빌드 후 검증:

```powershell
.\.venv\Scripts\python .\scripts\verify_opencv_cuda.py
```

이후 `python -m pip install -e .[dev]`를 다시 실행하면 CPU 전용 OpenCV가 덮어써질 수 있습니다. 그 경우 위 PowerShell 스크립트를 다시 실행해 `.venv` 안의 CUDA 빌드를 복구하면 됩니다.

## Windows ORB-SLAM3 브리지 빌드

`sim + rgb_only` 조합에서 `3D Reconstruction`을 사용하려면 네이티브 ORB-SLAM3 브리지가 필요합니다. Windows에서는 `obj_recog.main` 실행 전에 먼저 빌드해야 합니다.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_orbslam3_bridge_windows.ps1
```

이 스크립트는 다음을 찾습니다.

- CMake
- C++ 워크로드가 포함된 Visual Studio 2022 Build Tools
- 로컬 `third_party\ORB_SLAM3` 체크아웃
- OpenCV, Eigen3, Boost, OpenSSL의 CMake 패키지 메타데이터

기본 검색 경로 바깥에 라이브러리가 있다면 명시적으로 넘길 수 있습니다.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_orbslam3_bridge_windows.ps1 `
  -OpenCvDir C:\path\to\OpenCVConfig\dir `
  -Eigen3Dir C:\path\to\Eigen3Config\dir `
  -CMakePrefixPath C:\path\to\prefix1;C:\path\to\prefix2
```

예상 산출물은 `native\orbslam3_bridge\build\orbslam3_bridge.exe`입니다.

## CLI 확인

```bash
PYTHONPATH=src python -m obj_recog.main -h
```

## 기본 라이브 실행

기본 카메라 입력으로 실행하는 예시입니다.

```bash
PYTHONPATH=src python -m obj_recog.main \
  --input-source live \
  --width 640 \
  --height 360 \
  --device auto \
  --depth-profile fast \
  --segmentation-mode panoptic \
  --explanation-mode on
```

시작 로그에는 요청한 장치와 실제로 해석된 장치가 함께 출력됩니다.

```text
[obj-recog] runtime accel requested_device=auto resolved_device=cuda precision=fp16 ...
```

이 저장소 루트에서 실행하면 번들된 ORB-SLAM3 vocabulary 파일 `third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt`를 자동으로 찾습니다. 다른 설치 레이아웃에서 실행한다면 `--slam-vocabulary /absolute/path/to/ORBvoc.txt`를 직접 넘기면 됩니다.

유용한 런타임 조작:

- `e` 키 또는 우하단 토글 클릭: 설명 창 on/off
- `Situation Explanation`: 별도 OpenCV 창으로 표시
- `Environment Model`: 시뮬레이션 데이터가 있을 때 별도 Open3D 3인칭 뷰로 표시

## 시뮬레이션 실행

현재 시뮬레이션 경로는 RGB-only Unity 인터페이스를 사용합니다.

- `Unity -> Python`: 카메라 RGB 프레임과 타임스탬프만 전송
- `Python -> Unity`: 이동 및 카메라 팬 동작 명령만 전송
- 온라인 런타임은 라이브 모드와 같은 단안 인식 스택을 사용
- 숨겨진 목표, 충돌 판정, 공식 평가는 온라인 추론 루프 밖에서 관리

Unity 프로젝트는 `unity` 폴더에 있습니다. `Assets/Scenes/LivingRoomMain.unity`는 ApartmentKit `Scene_02`를 기반으로 하고, 그 위에 obj-recog 런타임 연결 계층이 추가되어 있습니다.

`Apartment Kit`은 로컬 vendor 의존성으로 취급되며 Git에는 포함되지 않습니다. 팀원별로 Asset Store 패키지 `Apartment Kit` 버전 `4.2`를 기본 경로로 임포트해 아래 폴더가 존재해야 합니다.

- `Assets/Brick Project Studio/Apartment Kit`
- `Assets/Brick Project Studio/_BPS Basic Assets`

씬을 열거나 sim 모드를 실행하기 전에 로컬 Unity 구성을 먼저 점검합니다.

```bash
PYTHONPATH=src python -m obj_recog.unity_vendor_check --unity-project-root unity
```

그다음 Unity 6 LTS `6000.3.11f1`로 프로젝트를 연 뒤:

- `manual` 모드: Play 모드에서 키보드/마우스로 직접 조작
- `agent` 모드: 스탠드얼론 플레이어를 빌드하고 Python이 실행 및 제어

`Assets/Brick Project Studio` 아래 파일은 직접 수정하지 않는 것이 원칙입니다. 환경 커스터마이징이 필요하면 별도 프로젝트 폴더에 복사본, 프리팹 변형, 머티리얼 변형을 만들어 추적하는 편이 안전합니다.

### manual 모드 조작키

- `W/S`: 전진 / 후진
- `A/D`: 좌우 이동
- `Q`: 몸체 오른쪽 회전
- `E`: 몸체 왼쪽 회전
- 마우스 X/Y: 카메라 팬 및 상하 시선 이동
- `R`: 리셋
- `F1`: HUD 토글
- `Esc`: 커서 해제, 다시 누르면 종료

### macOS에서 Unity 플레이어 빌드

```bash
scripts/build_unity_macos.sh
```

그다음 `.app` 번들을 지정해 Python 런타임에서 실행합니다.

```bash
set -a; source .env; set +a
PYTHONPATH=src python -m obj_recog.main \
  --input-source sim \
  --scenario living_room_navigation_v1 \
  --unity-player-path build/unity/macos/obj-recog-unity.app \
  --width 640 \
  --height 360 \
  --device auto \
  --detector-backend torch \
  --depth-profile fast \
  --segmentation-mode panoptic \
  --camera-calibration calibration/calibration.yaml \
  --sim-planner-model gpt-5-mini \
  --sim-planner-timeout-sec 8 \
  --sim-replan-interval-sec 4 \
  --sim-selfcal-max-sec 6 \
  --sim-action-batch-size 6 \
  --explanation-mode off
```

### Windows에서 sim 실행

빌드된 `.exe` 경로를 지정하면 됩니다.

```powershell
$env:PYTHONPATH = "src"
python -m obj_recog.main `
  --input-source sim `
  --scenario living_room_navigation_v1 `
  --unity-player-path C:\path\to\obj-recog-unity.exe `
  --width 640 `
  --height 360 `
  --device auto `
  --detector-backend torch `
  --depth-profile fast `
  --segmentation-mode panoptic `
  --camera-calibration calibration\calibration.yaml `
  --sim-planner-model gpt-5-mini `
  --sim-planner-timeout-sec 8 `
  --sim-replan-interval-sec 4 `
  --sim-selfcal-max-sec 6 `
  --sim-action-batch-size 6 `
  --explanation-mode off
```

기본 sim 설정은 논리 카메라 `24fps`, 세그멘테이션 갱신 `2프레임` 간격입니다.

### 자주 쓰는 sim 옵션

- `--sim-headless`: 데스크톱 창 없이 실행
- `--unity-host` / `--unity-port`: 이미 실행 중인 Unity 플레이어에 접속
- `--sim-planner-model`: 내비게이션 플래너에 사용할 LLM 선택
- `--sim-interface-mode rgb_only`: RGB-only 계약 강제
- `--unity-player-path`: `agent` 모드로 빌드된 Unity 플레이어 실행

시뮬레이션 중 눈에 보이는 창:

- `Object Recognition`
- `3D Reconstruction`

## 실행 결과물

각 sim 실행은 아래 경로 아래에 에피소드 산출물을 기록합니다.

- `reports/sim/living_room_navigation_v1/.../episode_report.json`
- `reports/sim/living_room_navigation_v1/.../planner_turns.jsonl`
- `reports/sim/living_room_navigation_v1/.../self_calibration.json`

참고:

- `episode_report.json`은 온라인 실행 결과를 기록합니다.
- 공식 성공/실패 판정은 오프라인 평가 단계에서 해석해야 합니다.

## 개발 검증 명령

회귀 확인에 자주 쓰는 명령은 다음과 같습니다.

```bash
pytest -q tests/test_config.py tests/test_main_simulation.py tests/test_living_room_runtime.py tests/test_unity_rgb.py tests/test_offline_benchmark.py
.venv/Scripts/python scripts/verify_opencv_cuda.py
python -m compileall src/obj_recog
git diff --check
```

## 관련 참고 폴더

- `unity/README.md`: Unity Living Room 프로젝트 전용 설명
- `third_party/README.md`: 외부 의존성 관련 참고
- `docs/plans/`: 기능 설계와 작업 기록

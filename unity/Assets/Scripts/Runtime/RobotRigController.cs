using UnityEngine;

namespace ObjRecog.UnitySim
{
    [DisallowMultipleComponent]
    public sealed class RobotRigController : MonoBehaviour
    {
        private const float MinimumCameraFarClipM = 120.0f;

        [SerializeField] private CharacterController robotController;
        [SerializeField] private Transform robotRoot;
        [SerializeField] private Transform cameraPanPivot;
        [SerializeField] private Camera robotCamera;
        [SerializeField] private int imageWidth = 640;
        [SerializeField] private int imageHeight = 360;
        [SerializeField] private float cameraHeightM = 1.25f;
        [SerializeField] private float verticalFieldOfViewDeg = 34.782f;
        [SerializeField] private float maxCameraPanDeg = 70.0f;
        [SerializeField] private float maxCameraPitchDeg = 55.0f;
        [SerializeField] private float moveSpeedMps = 1.6f;
        [SerializeField] private float turnSpeedDegPerSec = 100.0f;
        [SerializeField] private float mousePanSensitivity = 2.4f;
        [SerializeField] private float mousePitchSensitivity = 2.1f;
        [SerializeField] private float cameraNearClipM = 0.2f;
        [SerializeField] private float cameraFarClipM = MinimumCameraFarClipM;

        private Vector3 _startPosition;
        private Quaternion _startRotation;
        private float _startPanDeg;
        private float _startPitchDeg;
        private float _currentPanDeg;
        private float _currentPitchDeg;
        private RenderTexture _captureTarget;
        private Texture2D _captureReadback;

        public Transform RobotRoot => robotRoot;

        public float CameraHeightM => cameraHeightM;

        public void Configure(
            CharacterController controller,
            Transform root,
            Transform panPivot,
            Camera camera,
            int captureWidth,
            int captureHeight,
            float cameraHeight,
            float cameraVerticalFovDeg,
            float cameraNearClip,
            float cameraFarClip
        )
        {
            robotController = controller;
            robotRoot = root;
            cameraPanPivot = panPivot;
            robotCamera = camera;
            imageWidth = captureWidth;
            imageHeight = captureHeight;
            cameraHeightM = cameraHeight;
            verticalFieldOfViewDeg = cameraVerticalFovDeg;
            cameraNearClipM = cameraNearClip;
            cameraFarClipM = Mathf.Max(cameraFarClip, MinimumCameraFarClipM);
            if (robotCamera != null)
            {
                robotCamera.fieldOfView = verticalFieldOfViewDeg;
                robotCamera.nearClipPlane = cameraNearClipM;
                robotCamera.farClipPlane = cameraFarClipM;
            }

            CacheStartPose();
            EnsureCaptureBuffers();
        }

        private void Awake()
        {
            CacheStartPose();
            EnsureCaptureBuffers();
        }

        private void OnDestroy()
        {
            if (_captureTarget != null)
            {
                Destroy(_captureTarget);
            }

            if (_captureReadback != null)
            {
                Destroy(_captureReadback);
            }
        }

        public void ResetRig()
        {
            if (robotRoot == null)
            {
                return;
            }

            if (robotController != null)
            {
                robotController.enabled = false;
            }

            robotRoot.position = _startPosition;
            robotRoot.rotation = _startRotation;
            _currentPanDeg = _startPanDeg;
            _currentPitchDeg = _startPitchDeg;
            ApplyViewRotation();

            if (robotController != null)
            {
                robotController.enabled = true;
            }
        }

        public void ApplyCommand(string primitive, float value)
        {
            switch (primitive)
            {
                case "move_forward":
                    MoveLocal(value, 0.0f);
                    break;
                case "move_backward":
                    MoveLocal(-value, 0.0f);
                    break;
                case "strafe_left":
                    MoveLocal(0.0f, -value);
                    break;
                case "strafe_right":
                    MoveLocal(0.0f, value);
                    break;
                case "turn_left":
                    RotateBody(value);
                    break;
                case "turn_right":
                    RotateBody(-value);
                    break;
                case "camera_pan_left":
                    PanCamera(value);
                    break;
                case "camera_pan_right":
                    PanCamera(-value);
                    break;
                case "pause":
                    break;
                default:
                    throw new System.InvalidOperationException("Unsupported primitive: " + primitive);
            }
        }

        public void ApplyManualInput(
            float forwardAxis,
            float strafeAxis,
            float turnAxis,
            float mousePanAxis,
            float mousePitchAxis,
            float deltaTime
        )
        {
            MoveLocal(forwardAxis * moveSpeedMps * deltaTime, strafeAxis * moveSpeedMps * deltaTime);
            RotateBody(turnAxis * turnSpeedDegPerSec * deltaTime);
            PanCamera(mousePanAxis * mousePanSensitivity);
            PitchCamera(-mousePitchAxis * mousePitchSensitivity);
        }

        public void MoveLocal(float forwardMeters, float strafeMeters)
        {
            if (robotRoot == null)
            {
                return;
            }

            Vector3 delta = (robotRoot.forward * forwardMeters) + (robotRoot.right * strafeMeters);
            if (robotController != null)
            {
                robotController.Move(delta);
                return;
            }

            robotRoot.position += delta;
        }

        public void RotateBody(float deltaDegrees)
        {
            if (robotRoot == null)
            {
                return;
            }

            robotRoot.Rotate(0.0f, deltaDegrees, 0.0f, Space.World);
        }

        public void PanCamera(float deltaDegrees)
        {
            if (cameraPanPivot == null)
            {
                return;
            }

            _currentPanDeg = Mathf.Clamp(_currentPanDeg + deltaDegrees, -maxCameraPanDeg, maxCameraPanDeg);
            ApplyViewRotation();
        }

        public void PitchCamera(float deltaDegrees)
        {
            if (robotCamera == null)
            {
                return;
            }

            _currentPitchDeg = Mathf.Clamp(_currentPitchDeg + deltaDegrees, -maxCameraPitchDeg, maxCameraPitchDeg);
            ApplyViewRotation();
        }

        public byte[] CapturePng()
        {
            if (robotCamera == null)
            {
                return System.Array.Empty<byte>();
            }

            EnsureCaptureBuffers();
            RenderTexture previousTarget = robotCamera.targetTexture;
            RenderTexture previousActive = RenderTexture.active;
            robotCamera.targetTexture = _captureTarget;
            robotCamera.Render();
            RenderTexture.active = _captureTarget;

            _captureReadback.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
            _captureReadback.Apply(false, false);
            byte[] pngBytes = _captureReadback.EncodeToPNG();

            robotCamera.targetTexture = previousTarget;
            RenderTexture.active = previousActive;
            return pngBytes;
        }

        private void CacheStartPose()
        {
            if (robotRoot == null || cameraPanPivot == null)
            {
                return;
            }

            _startPosition = robotRoot.position;
            _startRotation = robotRoot.rotation;
            _startPanDeg = NormalizeSignedDegrees(cameraPanPivot.localEulerAngles.y);
            _startPitchDeg = robotCamera == null
                ? 0.0f
                : NormalizeSignedDegrees(robotCamera.transform.localEulerAngles.x);
            _currentPanDeg = _startPanDeg;
            _currentPitchDeg = _startPitchDeg;
            ApplyViewRotation();
        }

        private void ApplyViewRotation()
        {
            if (cameraPanPivot != null)
            {
                cameraPanPivot.localRotation = Quaternion.Euler(0.0f, _currentPanDeg, 0.0f);
            }

            if (robotCamera != null)
            {
                robotCamera.transform.localRotation = Quaternion.Euler(_currentPitchDeg, 0.0f, 0.0f);
            }
        }

        private void EnsureCaptureBuffers()
        {
            if (imageWidth <= 0 || imageHeight <= 0)
            {
                imageWidth = 640;
                imageHeight = 360;
            }

            if (_captureTarget != null && (_captureTarget.width != imageWidth || _captureTarget.height != imageHeight))
            {
                Destroy(_captureTarget);
                _captureTarget = null;
            }

            if (_captureReadback != null && (_captureReadback.width != imageWidth || _captureReadback.height != imageHeight))
            {
                Destroy(_captureReadback);
                _captureReadback = null;
            }

            if (_captureTarget == null)
            {
                _captureTarget = new RenderTexture(imageWidth, imageHeight, 24, RenderTextureFormat.ARGB32);
                _captureTarget.name = "RobotCameraCapture";
            }

            if (_captureReadback == null)
            {
                _captureReadback = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
            }

            if (robotCamera != null)
            {
                robotCamera.fieldOfView = verticalFieldOfViewDeg;
                robotCamera.nearClipPlane = cameraNearClipM;
                robotCamera.farClipPlane = Mathf.Max(cameraFarClipM, MinimumCameraFarClipM);
            }
        }

        private static float NormalizeSignedDegrees(float degrees)
        {
            float normalized = degrees % 360.0f;
            if (normalized > 180.0f)
            {
                normalized -= 360.0f;
            }

            return normalized;
        }
    }
}

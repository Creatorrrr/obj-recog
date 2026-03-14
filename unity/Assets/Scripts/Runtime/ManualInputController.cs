using UnityEngine;

namespace ObjRecog.UnitySim
{
    [DisallowMultipleComponent]
    public sealed class ManualInputController : MonoBehaviour
    {
        [SerializeField] private RobotRigController robotRig;
        [SerializeField] private SessionState sessionState;
        [SerializeField] private HudOverlay hudOverlay;
        [SerializeField] private bool manualModeEnabled = true;

        private bool _cursorCaptured;
        private bool _quitArmed;

        public bool ManualModeEnabled => manualModeEnabled;

        public bool CursorCaptured => _cursorCaptured;

        public bool QuitArmed => _quitArmed;

        public void Configure(RobotRigController rig, SessionState state, HudOverlay overlay)
        {
            robotRig = rig;
            sessionState = state;
            hudOverlay = overlay;
        }

        public void EnableManualMode(bool enabled)
        {
            manualModeEnabled = enabled;
            if (manualModeEnabled)
            {
                CaptureCursor();
            }
            else
            {
                ReleaseCursor(false);
            }
        }

        private void Start()
        {
            if (manualModeEnabled)
            {
                CaptureCursor();
            }
            else
            {
                ReleaseCursor(false);
            }
        }

        private void OnDisable()
        {
            ReleaseCursor(false);
        }

        private void Update()
        {
            if (!Application.isPlaying)
            {
                return;
            }

            if (Input.GetKeyDown(KeyCode.F1) && hudOverlay != null)
            {
                hudOverlay.Toggle();
            }

            if (!manualModeEnabled)
            {
                return;
            }

            if (Input.GetKeyDown(KeyCode.R) && sessionState != null)
            {
                sessionState.ResetEpisode();
            }

            if (Input.GetKeyDown(KeyCode.Escape))
            {
                if (_cursorCaptured)
                {
                    ReleaseCursor(true);
                    return;
                }

                if (_quitArmed)
                {
                    Application.Quit();
                    return;
                }

                _quitArmed = true;
                return;
            }

            if (!_cursorCaptured)
            {
                if (Input.GetMouseButtonDown(0))
                {
                    CaptureCursor();
                }

                return;
            }

            float forward = 0.0f;
            float strafe = 0.0f;
            float turn = 0.0f;
            if (Input.GetKey(KeyCode.W))
            {
                forward += 1.0f;
            }

            if (Input.GetKey(KeyCode.S))
            {
                forward -= 1.0f;
            }

            if (Input.GetKey(KeyCode.D))
            {
                strafe += 1.0f;
            }

            if (Input.GetKey(KeyCode.A))
            {
                strafe -= 1.0f;
            }

            if (Input.GetKey(KeyCode.Q))
            {
                turn += 1.0f;
            }

            if (Input.GetKey(KeyCode.E))
            {
                turn -= 1.0f;
            }

            float mousePan = Input.GetAxis("Mouse X");
            if (robotRig != null)
            {
                robotRig.ApplyManualInput(
                    forwardAxis: Mathf.Clamp(forward, -1.0f, 1.0f),
                    strafeAxis: Mathf.Clamp(strafe, -1.0f, 1.0f),
                    turnAxis: Mathf.Clamp(turn, -1.0f, 1.0f),
                    mousePanAxis: mousePan,
                    deltaTime: Time.deltaTime
                );
            }
        }

        private void CaptureCursor()
        {
            _cursorCaptured = true;
            _quitArmed = false;
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
        }

        private void ReleaseCursor(bool armQuit)
        {
            _cursorCaptured = false;
            _quitArmed = armQuit;
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
        }
    }
}

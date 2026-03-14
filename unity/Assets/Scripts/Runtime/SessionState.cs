using UnityEngine;

namespace ObjRecog.UnitySim
{
    [DisallowMultipleComponent]
    public sealed class SessionState : MonoBehaviour
    {
        [SerializeField] private RobotRigController robotRig;
        [SerializeField] private Transform robotRoot;
        [SerializeField] private string scenarioId = "living_room_navigation_v1";

        private bool _missionSucceeded;
        private string _statusMessage = "Ready";
        private float _statusUntilTime;

        public bool MissionSucceeded => _missionSucceeded;

        public string ScenarioId => scenarioId;

        public string CurrentStatusMessage
        {
            get
            {
                if (!Application.isPlaying)
                {
                    return _statusMessage;
                }

                return Time.time <= _statusUntilTime ? _statusMessage : string.Empty;
            }
        }

        public void Configure(RobotRigController rig, Transform root)
        {
            robotRig = rig;
            robotRoot = root;
        }

        public void ResetEpisode()
        {
            _missionSucceeded = false;
            if (robotRig != null)
            {
                robotRig.ResetRig();
            }

            ShowTransientStatus("Episode reset", 2.0f);
        }

        public void MarkGoalReached()
        {
            if (_missionSucceeded)
            {
                return;
            }

            _missionSucceeded = true;
            ShowTransientStatus("Goal reached", 5.0f);
        }

        public void ShowTransientStatus(string message, float durationSeconds)
        {
            _statusMessage = message ?? string.Empty;
            _statusUntilTime = Application.isPlaying ? Time.time + Mathf.Max(0.0f, durationSeconds) : 0.0f;
        }

        public bool IsRobotCollider(Collider other)
        {
            if (robotRoot == null || other == null)
            {
                return false;
            }

            Transform current = other.transform;
            return current == robotRoot || current.IsChildOf(robotRoot);
        }
    }
}

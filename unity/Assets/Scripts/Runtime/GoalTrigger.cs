using UnityEngine;

namespace ObjRecog.UnitySim
{
    [DisallowMultipleComponent]
    public sealed class GoalTrigger : MonoBehaviour
    {
        [SerializeField] private SessionState sessionState;

        public void Configure(SessionState state)
        {
            sessionState = state;
        }

        private void OnTriggerEnter(Collider other)
        {
            if (sessionState != null && sessionState.IsRobotCollider(other))
            {
                sessionState.MarkGoalReached();
            }
        }
    }
}

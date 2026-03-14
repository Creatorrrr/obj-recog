using UnityEngine;

namespace ObjRecog.UnitySim
{
    [DisallowMultipleComponent]
    public sealed class HudOverlay : MonoBehaviour
    {
        [SerializeField] private SessionState sessionState;
        [SerializeField] private ManualInputController manualInput;
        [SerializeField] private AgentTcpServer agentServer;
        [SerializeField] private SimulatorBootMode mode = SimulatorBootMode.Manual;
        [SerializeField] private bool visible = true;

        public void Configure(
            SessionState state,
            ManualInputController manual,
            AgentTcpServer server,
            SimulatorBootMode bootMode
        )
        {
            sessionState = state;
            manualInput = manual;
            agentServer = server;
            mode = bootMode;
        }

        public void Toggle()
        {
            visible = !visible;
        }

        private void OnGUI()
        {
            if (!visible)
            {
                return;
            }

            GUILayout.BeginArea(new Rect(12, 12, 500, 260), GUI.skin.box);
            GUILayout.Label($"Mode: {mode}");
            if (sessionState != null)
            {
                GUILayout.Label($"Scenario: {sessionState.ScenarioId}");
                if (!string.IsNullOrWhiteSpace(sessionState.CurrentStatusMessage))
                {
                    GUILayout.Label($"Status: {sessionState.CurrentStatusMessage}");
                }

                GUILayout.Label(sessionState.MissionSucceeded ? "Goal: reached" : "Goal: hidden");
            }

            if (mode == SimulatorBootMode.Manual)
            {
                GUILayout.Label("Controls: W/S move, A/D strafe, Q/E turn, mouse pan");
                GUILayout.Label("R reset, F1 HUD toggle, Esc release cursor, Esc again quit");
                if (manualInput != null && !manualInput.CursorCaptured)
                {
                    GUILayout.Label(manualInput.QuitArmed
                        ? "Cursor free: left click to recapture, Esc again to quit"
                        : "Cursor free: left click to recapture");
                }
            }
            else
            {
                if (agentServer != null)
                {
                    GUILayout.Label($"TCP: {(agentServer.ServerRunning ? "listening" : "stopped")}");
                    GUILayout.Label($"Client: {(agentServer.ClientConnected ? "connected" : "waiting")}");
                    GUILayout.Label($"Endpoint: {agentServer.Host}:{agentServer.Port}");
                }

                GUILayout.Label("Agent mode: Python drives the robot from RGB frames only");
            }

            GUILayout.EndArea();
        }
    }
}

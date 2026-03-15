using System;
using UnityEngine;
using UnityEngine.Rendering;
#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
#endif

namespace ObjRecog.UnitySim
{
    [ExecuteAlways]
    [DisallowMultipleComponent]
    public sealed class LivingRoomAppBootstrap : MonoBehaviour
    {
        private const string RuntimeRootName = "ObjRecogRuntime";
        private const string RobotRigGroupName = "RobotRig";
        private const string RobotRootName = "RobotRoot";
        private const string CameraPanPivotName = "CameraPanPivot";
        private const string RobotCameraName = "RobotCamera";
        private const string RobotSpawnName = "RobotSpawn";
        private const string GoalAnchorName = "GoalAnchor";
        private const string GoalTriggerName = "HiddenGoalTrigger";
        private const string LightingRootName = "ObjRecogLighting";
        private const string KeyLightName = "ObjRecogKeyLight";
        private const string FillLightName = "ObjRecogFillLight";

        [SerializeField] private int layoutRevision = 2;
        [SerializeField] private int captureWidth = 640;
        [SerializeField] private int captureHeight = 360;
        [SerializeField] private float cameraHeightM = 1.25f;
        [SerializeField] private float cameraVerticalFieldOfViewDeg = 34.782f;
        [SerializeField] private float cameraNearClipM = 0.2f;
        [SerializeField] private float cameraFarClipM = 120.0f;
        [SerializeField] private float goalStandOffM = 1.25f;
        [SerializeField] private float spawnStandOffM = 5.4f;
        [SerializeField] private float runtimeGroundHeightM = 0.05f;
        [SerializeField] private float characterRadiusM = 0.28f;
        [SerializeField] private float characterHeightM = 1.55f;
        [SerializeField] private bool autoRebuildInEditor = true;
        [SerializeField] private bool disableVendorInteractions = true;
        [SerializeField] private string preferredTelevisionPrefix = "TV_Apt_01";
        [SerializeField] private Vector3 fallbackSpawnPosition = new Vector3(27.084f, 12.1f, 2.35f);
        [SerializeField] private Vector3 fallbackSpawnEulerAngles = new Vector3(0.0f, 180.0f, 0.0f);
        [SerializeField] private Vector3 fallbackGoalPosition = new Vector3(27.084f, 12.1f, -2.40f);
        [SerializeField] private bool brightenSceneLighting = true;
        [SerializeField] private float ambientIntensity = 1.6f;
        [SerializeField] private Color ambientSkyColor = new Color(0.54f, 0.58f, 0.64f, 1.0f);
        [SerializeField] private Color ambientEquatorColor = new Color(0.34f, 0.36f, 0.39f, 1.0f);
        [SerializeField] private Color ambientGroundColor = new Color(0.18f, 0.17f, 0.16f, 1.0f);
        [SerializeField] private Color keyLightColor = new Color(1.0f, 0.96f, 0.9f, 1.0f);
        [SerializeField] private float keyLightIntensity = 1.45f;
        [SerializeField] private Vector3 keyLightEulerAngles = new Vector3(48.0f, -28.0f, 0.0f);
        [SerializeField] private Color fillLightColor = new Color(0.74f, 0.82f, 0.98f, 1.0f);
        [SerializeField] private float fillLightIntensity = 0.55f;
        [SerializeField] private Vector3 fillLightEulerAngles = new Vector3(28.0f, 145.0f, 0.0f);

        private bool _editorEnsureQueued;

        private void Reset()
        {
            QueueOrEnsureScene();
        }

        private void OnEnable()
        {
            QueueOrEnsureScene();
        }

        private void Awake()
        {
            EnsureRuntimeSetup(force: false);
            ApplyBootMode();
        }

        private void Start()
        {
            ApplyBootMode();
        }

#if UNITY_EDITOR
        private void OnValidate()
        {
            if (!autoRebuildInEditor || Application.isPlaying)
            {
                return;
            }

            QueueOrEnsureScene();
        }
#endif

        [ContextMenu("Rebuild Living Room Runtime")]
        public void RebuildLivingRoom()
        {
            EnsureRuntimeSetup(force: true);
        }

        private void QueueOrEnsureScene()
        {
#if UNITY_EDITOR
            if (!Application.isPlaying && autoRebuildInEditor)
            {
                if (_editorEnsureQueued)
                {
                    return;
                }

                _editorEnsureQueued = true;
                EditorApplication.delayCall += EditorEnsureSceneGraph;
                return;
            }
#endif
            EnsureRuntimeSetup(force: false);
        }

#if UNITY_EDITOR
        private void EditorEnsureSceneGraph()
        {
            _editorEnsureQueued = false;
            if (this == null)
            {
                return;
            }

            EnsureRuntimeSetup(force: false);
        }
#endif

        private void EnsureRuntimeSetup(bool force)
        {
            Transform runtimeRoot = EnsureRuntimeRoot(force);
            if (runtimeRoot == null)
            {
                return;
            }

            Transform spawnAnchor = FindRequiredChild(runtimeRoot, RobotSpawnName);
            Transform goalAnchor = FindRequiredChild(runtimeRoot, GoalAnchorName);
            Transform robotRoot = FindRequiredChild(runtimeRoot, $"{RobotRigGroupName}/{RobotRootName}");
            Transform cameraPanPivot = FindRequiredChild(runtimeRoot, $"{RobotRigGroupName}/{RobotRootName}/{CameraPanPivotName}");
            Transform goalTrigger = FindRequiredChild(runtimeRoot, GoalTriggerName);
            CharacterController controller = robotRoot == null ? null : robotRoot.GetComponent<CharacterController>();
            Camera robotCamera = cameraPanPivot == null ? null : cameraPanPivot.GetComponentInChildren<Camera>(true);

            if (disableVendorInteractions)
            {
                DisableConflictingSceneComponents(robotCamera);
            }

            if (brightenSceneLighting)
            {
                EnsureLightingSetup(runtimeRoot, force);
            }

            AutoPlaceAnchors(spawnAnchor, goalAnchor);
            SyncRuntimeObjects(spawnAnchor, goalAnchor, robotRoot, cameraPanPivot, goalTrigger, controller, robotCamera);
            WireRuntimeComponents(runtimeRoot, robotRoot, cameraPanPivot, goalTrigger, controller, robotCamera);
        }

        private void ApplyBootMode()
        {
            if (!Application.isPlaying)
            {
                return;
            }

            Transform runtimeRoot = transform.Find(RuntimeRootName);
            if (runtimeRoot == null)
            {
                return;
            }

            SessionState sessionState = runtimeRoot.GetComponent<SessionState>();
            ManualInputController manualInput = runtimeRoot.GetComponent<ManualInputController>();
            HudOverlay hudOverlay = runtimeRoot.GetComponent<HudOverlay>();
            AgentTcpServer agentServer = runtimeRoot.GetComponent<AgentTcpServer>();
            RobotRigController robotRig = runtimeRoot.GetComponent<RobotRigController>();
            if (sessionState == null || manualInput == null || hudOverlay == null || agentServer == null || robotRig == null)
            {
                return;
            }

            BootModeConfig bootMode = BootModeResolver.Resolve(Environment.GetCommandLineArgs());
            agentServer.Configure(bootMode.Host, bootMode.Port, robotRig, sessionState);
            agentServer.EnableAgentMode(bootMode.Mode == SimulatorBootMode.Agent);
            hudOverlay.Configure(sessionState, manualInput, agentServer, bootMode.Mode);

            if (bootMode.Mode == SimulatorBootMode.Agent)
            {
                Application.runInBackground = true;
                manualInput.EnableManualMode(false);
                Cursor.lockState = CursorLockMode.Locked;
                Cursor.visible = false;
                agentServer.StartServer();
                sessionState.ResetEpisode();
                sessionState.ShowTransientStatus("Agent mode listening", 5.0f);
                return;
            }

            agentServer.StopServer();
            Application.runInBackground = false;
            manualInput.EnableManualMode(true);
            sessionState.ResetEpisode();
            sessionState.ShowTransientStatus("Manual mode ready", 3.0f);
        }

        private Transform EnsureRuntimeRoot(bool force)
        {
            Transform runtimeRoot = transform.Find(RuntimeRootName);
            if (runtimeRoot == null)
            {
                runtimeRoot = new GameObject(RuntimeRootName).transform;
                runtimeRoot.SetParent(transform, false);
            }

            Transform spawnAnchor = FindOrCreateChild(runtimeRoot, RobotSpawnName);
            Transform goalAnchor = FindOrCreateChild(runtimeRoot, GoalAnchorName);
            spawnAnchor.gameObject.SetActive(true);
            goalAnchor.gameObject.SetActive(true);

            Transform rigGroup = FindOrCreateChild(runtimeRoot, RobotRigGroupName);
            Transform robotRoot = FindOrCreateChild(rigGroup, RobotRootName);
            CharacterController controller = GetOrAddComponent<CharacterController>(robotRoot.gameObject);

            Transform cameraPanPivot = FindOrCreateChild(robotRoot, CameraPanPivotName);
            Transform robotCameraTransform = FindOrCreateChild(cameraPanPivot, RobotCameraName);
            Camera robotCamera = GetOrAddComponent<Camera>(robotCameraTransform.gameObject);
            AudioListener _ = GetOrAddComponent<AudioListener>(robotCameraTransform.gameObject);

            Transform goalTrigger = FindOrCreateChild(runtimeRoot, GoalTriggerName);
            BoxCollider triggerCollider = GetOrAddComponent<BoxCollider>(goalTrigger.gameObject);
            triggerCollider.isTrigger = true;

            if (force || !Application.isPlaying)
            {
                robotRoot.SetParent(rigGroup, false);
                cameraPanPivot.SetParent(robotRoot, false);
                robotCameraTransform.SetParent(cameraPanPivot, false);
                goalTrigger.SetParent(runtimeRoot, false);
            }

            controller.radius = characterRadiusM;
            controller.height = characterHeightM;
            controller.center = new Vector3(0.0f, characterHeightM * 0.5f, 0.0f);
            controller.minMoveDistance = 0.0f;
            controller.stepOffset = 0.3f;
            controller.skinWidth = 0.03f;
            controller.slopeLimit = 45.0f;

            robotCamera.nearClipPlane = cameraNearClipM;
            robotCamera.farClipPlane = Mathf.Max(cameraFarClipM, 120.0f);
            robotCamera.fieldOfView = cameraVerticalFieldOfViewDeg;

#if UNITY_EDITOR
            if (!Application.isPlaying)
            {
                EditorSceneManager.MarkSceneDirty(gameObject.scene);
            }
#endif

            return runtimeRoot;
        }

        private void EnsureLightingSetup(Transform runtimeRoot, bool force)
        {
            if (runtimeRoot == null)
            {
                return;
            }

            Transform lightingRoot = FindOrCreateChild(runtimeRoot, LightingRootName);
            Transform keyLightTransform = FindOrCreateChild(lightingRoot, KeyLightName);
            Transform fillLightTransform = FindOrCreateChild(lightingRoot, FillLightName);

            Light keyLight = GetOrAddComponent<Light>(keyLightTransform.gameObject);
            Light fillLight = GetOrAddComponent<Light>(fillLightTransform.gameObject);

            ConfigureDirectionalLight(
                keyLight,
                KeyLightName,
                keyLightColor,
                keyLightIntensity,
                keyLightEulerAngles,
                shadows: LightShadows.Soft
            );
            ConfigureDirectionalLight(
                fillLight,
                FillLightName,
                fillLightColor,
                fillLightIntensity,
                fillLightEulerAngles,
                shadows: LightShadows.None
            );

            RenderSettings.ambientMode = AmbientMode.Trilight;
            RenderSettings.ambientSkyColor = ambientSkyColor;
            RenderSettings.ambientEquatorColor = ambientEquatorColor;
            RenderSettings.ambientGroundColor = ambientGroundColor;
            RenderSettings.ambientIntensity = ambientIntensity;
            RenderSettings.reflectionIntensity = Mathf.Max(RenderSettings.reflectionIntensity, 1.15f);
            RenderSettings.sun = keyLight;

#if UNITY_EDITOR
            if ((force || !Application.isPlaying) && !Application.isPlaying)
            {
                EditorSceneManager.MarkSceneDirty(gameObject.scene);
            }
#endif
        }

        private void AutoPlaceAnchors(Transform spawnAnchor, Transform goalAnchor)
        {
            if (spawnAnchor == null || goalAnchor == null)
            {
                return;
            }

            if (TryPlaceAnchorsAroundPreferredTelevision(spawnAnchor, goalAnchor))
            {
                return;
            }

            Transform sceneCamera = FindPreferredSceneCamera();
            Vector3 spawnPosition = fallbackSpawnPosition;
            Quaternion spawnRotation = Quaternion.Euler(fallbackSpawnEulerAngles);
            if (sceneCamera != null)
            {
                spawnPosition = sceneCamera.position;
                spawnPosition.y = Mathf.Max(sceneCamera.position.y - cameraHeightM, runtimeGroundHeightM);
                spawnRotation = Quaternion.Euler(0.0f, sceneCamera.rotation.eulerAngles.y, 0.0f);
            }

            spawnAnchor.position = spawnPosition;
            spawnAnchor.rotation = spawnRotation;
            goalAnchor.position = fallbackGoalPosition;
            Vector3 facing = spawnPosition - fallbackGoalPosition;
            facing.y = 0.0f;
            goalAnchor.rotation = facing.sqrMagnitude > 0.001f
                ? Quaternion.LookRotation(facing.normalized, Vector3.up)
                : Quaternion.identity;
        }

        private bool TryPlaceAnchorsAroundPreferredTelevision(Transform spawnAnchor, Transform goalAnchor)
        {
            Transform television = FindPreferredTelevision();
            if (television == null)
            {
                return false;
            }

            Transform livingRoomRoot = FindContainingAncestorNamed(television, "Living Room");
            float groundHeight = ResolveRoomGroundHeight(livingRoomRoot);
            Vector3 televisionPosition = television.position;
            Vector3 roomFacingDirection = ResolveTelevisionRoomFacingDirection(television, livingRoomRoot);
            if (roomFacingDirection.sqrMagnitude < 0.001f)
            {
                return false;
            }

            Vector3 goalPosition = FlattenToGround(televisionPosition + (roomFacingDirection * goalStandOffM), groundHeight);
            Vector3 spawnPosition = ResolveTelevisionSpawnPosition(television, livingRoomRoot, roomFacingDirection, groundHeight);
            Vector3 awayFromTelevision = FlattenDirection(spawnPosition - televisionPosition);
            Vector3 spawnLookTarget = awayFromTelevision.sqrMagnitude > 0.001f
                ? spawnPosition + awayFromTelevision
                : ResolveTelevisionSpawnLookTarget(livingRoomRoot, televisionPosition, spawnPosition);
            spawnAnchor.position = spawnPosition;
            spawnAnchor.rotation = LookToward(spawnPosition, spawnLookTarget);
            goalAnchor.position = goalPosition;
            goalAnchor.rotation = LookToward(goalPosition, televisionPosition);
            return true;
        }

        private void SyncRuntimeObjects(
            Transform spawnAnchor,
            Transform goalAnchor,
            Transform robotRoot,
            Transform cameraPanPivot,
            Transform goalTrigger,
            CharacterController controller,
            Camera robotCamera
        )
        {
            if (spawnAnchor == null || goalAnchor == null || robotRoot == null || cameraPanPivot == null || goalTrigger == null)
            {
                return;
            }

            if (controller != null)
            {
                bool wasEnabled = controller.enabled;
                controller.enabled = false;
                robotRoot.position = spawnAnchor.position;
                robotRoot.rotation = spawnAnchor.rotation;
                controller.enabled = wasEnabled;
            }
            else
            {
                robotRoot.position = spawnAnchor.position;
                robotRoot.rotation = spawnAnchor.rotation;
            }

            cameraPanPivot.localPosition = new Vector3(0.0f, cameraHeightM, 0.0f);
            cameraPanPivot.localRotation = Quaternion.identity;

            if (robotCamera != null)
            {
                robotCamera.transform.localPosition = Vector3.zero;
                robotCamera.fieldOfView = cameraVerticalFieldOfViewDeg;
                robotCamera.nearClipPlane = cameraNearClipM;
                robotCamera.farClipPlane = Mathf.Max(cameraFarClipM, 120.0f);
            }

            goalTrigger.position = goalAnchor.position;
            goalTrigger.rotation = goalAnchor.rotation;

            BoxCollider triggerCollider = goalTrigger.GetComponent<BoxCollider>();
            if (triggerCollider != null)
            {
                triggerCollider.isTrigger = true;
                triggerCollider.center = new Vector3(0.0f, 0.6f, 0.0f);
                triggerCollider.size = new Vector3(0.7f, 1.2f, 0.7f);
            }
        }

        private void WireRuntimeComponents(
            Transform runtimeRoot,
            Transform robotRoot,
            Transform cameraPanPivot,
            Transform goalTrigger,
            CharacterController controller,
            Camera robotCamera
        )
        {
            RobotRigController rigController = GetOrAddComponent<RobotRigController>(runtimeRoot.gameObject);
            rigController.Configure(
                controller,
                robotRoot,
                cameraPanPivot,
                robotCamera,
                captureWidth,
                captureHeight,
                cameraHeightM,
                cameraVerticalFieldOfViewDeg,
                cameraNearClipM,
                cameraFarClipM
            );

            SessionState sessionState = GetOrAddComponent<SessionState>(runtimeRoot.gameObject);
            sessionState.Configure(rigController, robotRoot);

            GoalTrigger trigger = GetOrAddComponent<GoalTrigger>(goalTrigger.gameObject);
            trigger.Configure(sessionState);

            ManualInputController manualInput = GetOrAddComponent<ManualInputController>(runtimeRoot.gameObject);
            HudOverlay hudOverlay = GetOrAddComponent<HudOverlay>(runtimeRoot.gameObject);
            AgentTcpServer agentServer = GetOrAddComponent<AgentTcpServer>(runtimeRoot.gameObject);

            manualInput.Configure(rigController, sessionState, hudOverlay);
            agentServer.Configure("127.0.0.1", 8765, rigController, sessionState);
            hudOverlay.Configure(sessionState, manualInput, agentServer, SimulatorBootMode.Manual);
        }

        private void DisableConflictingSceneComponents(Camera robotCamera)
        {
            GameObject[] roots = gameObject.scene.GetRootGameObjects();
            for (int rootIndex = 0; rootIndex < roots.Length; rootIndex++)
            {
                GameObject root = roots[rootIndex];
                if (root == null)
                {
                    continue;
                }

                MonoBehaviour[] behaviours = root.GetComponentsInChildren<MonoBehaviour>(true);
                for (int index = 0; index < behaviours.Length; index++)
                {
                    MonoBehaviour behaviour = behaviours[index];
                    if (behaviour == null || behaviour == this)
                    {
                        continue;
                    }

                    Type type = behaviour.GetType();
                    string namespaceName = type.Namespace ?? string.Empty;
                    bool keepBehaviour =
                        namespaceName.StartsWith("ObjRecog.UnitySim", StringComparison.Ordinal) ||
                        namespaceName.StartsWith("Unity", StringComparison.Ordinal) ||
                        namespaceName.StartsWith("TMPro", StringComparison.Ordinal);
                    if (!keepBehaviour && behaviour.enabled)
                    {
                        behaviour.enabled = false;
                    }
                }

                Camera[] cameras = root.GetComponentsInChildren<Camera>(true);
                for (int cameraIndex = 0; cameraIndex < cameras.Length; cameraIndex++)
                {
                    Camera camera = cameras[cameraIndex];
                    if (camera == null || camera == robotCamera)
                    {
                        continue;
                    }

                    camera.enabled = false;
                }

                AudioListener[] listeners = root.GetComponentsInChildren<AudioListener>(true);
                for (int listenerIndex = 0; listenerIndex < listeners.Length; listenerIndex++)
                {
                    AudioListener listener = listeners[listenerIndex];
                    if (listener == null)
                    {
                        continue;
                    }

                    bool keepListener = robotCamera != null && listener.gameObject == robotCamera.gameObject;
                    listener.enabled = keepListener;
                }
            }
        }

        private Transform FindPreferredSceneCamera()
        {
            Camera[] cameras = gameObject.scene.GetRootGameObjects().Length == 0
                ? Array.Empty<Camera>()
                : FindSceneComponents<Camera>();

            Transform fallbackCamera = null;
            for (int index = 0; index < cameras.Length; index++)
            {
                Camera camera = cameras[index];
                if (camera == null || camera.transform.IsChildOf(transform))
                {
                    continue;
                }

                if (string.Equals(camera.gameObject.name, RobotCameraName, StringComparison.Ordinal))
                {
                    continue;
                }

                if (string.Equals(camera.gameObject.name, "Camera", StringComparison.OrdinalIgnoreCase))
                {
                    return camera.transform;
                }

                if (fallbackCamera == null)
                {
                    fallbackCamera = camera.transform;
                }
            }

            return fallbackCamera;
        }

        private Transform FindPreferredTelevision()
        {
            Transform[] transforms = FindSceneComponents<Transform>();
            Transform preferredTelevision = null;
            float nearestDistanceSq = float.MaxValue;
            Vector3 sceneCenter = EstimateSceneCenter();
            for (int index = 0; index < transforms.Length; index++)
            {
                Transform candidate = transforms[index];
                if (candidate == null || candidate.IsChildOf(transform))
                {
                    continue;
                }

                if (!candidate.name.StartsWith(preferredTelevisionPrefix, StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                Vector3 delta = candidate.position - sceneCenter;
                delta.y = 0.0f;
                float distanceSq = delta.sqrMagnitude;
                if (distanceSq < nearestDistanceSq)
                {
                    nearestDistanceSq = distanceSq;
                    preferredTelevision = candidate;
                }
            }

            return preferredTelevision;
        }

        private Transform FindNearestNamedTransform(string prefix, Vector3 origin)
        {
            Transform nearest = null;
            float nearestDistanceSq = float.MaxValue;
            Transform[] transforms = FindSceneComponents<Transform>();
            for (int index = 0; index < transforms.Length; index++)
            {
                Transform candidate = transforms[index];
                if (candidate == null || candidate.IsChildOf(transform))
                {
                    continue;
                }

                if (!candidate.name.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                Vector3 delta = candidate.position - origin;
                delta.y = 0.0f;
                float distanceSq = delta.sqrMagnitude;
                if (distanceSq < nearestDistanceSq)
                {
                    nearestDistanceSq = distanceSq;
                    nearest = candidate;
                }
            }

            return nearest;
        }

        private Vector3 ResolveTelevisionRoomFacingDirection(Transform television, Transform livingRoomRoot)
        {
            Transform mediaTable = FindNamedTransformInSubtree(livingRoomRoot, "MediaTable_01");
            if (TryGetDirectionBetween(television.position, mediaTable, out Vector3 mediaTableDirection))
            {
                return mediaTableDirection;
            }

            Transform rug = FindNamedTransformInSubtree(livingRoomRoot, "Rug_Apt_01");
            if (TryGetDirectionBetween(television.position, rug, out Vector3 rugDirection))
            {
                return rugDirection;
            }

            Transform sofa = FindNamedTransformInSubtree(livingRoomRoot, "Sofa_Apt_01");
            if (TryGetDirectionBetween(television.position, sofa, out Vector3 sofaDirection))
            {
                return sofaDirection;
            }

            Vector3 forward = FlattenDirection(television.forward);
            if (forward.sqrMagnitude < 0.001f)
            {
                return Vector3.back;
            }

            return Mathf.Abs(forward.z) >= Mathf.Abs(forward.x)
                ? new Vector3(0.0f, 0.0f, Mathf.Sign(forward.z))
                : Vector3.back;
        }

        private Vector3 ResolveTelevisionSpawnPosition(
            Transform television,
            Transform livingRoomRoot,
            Vector3 roomFacingDirection,
            float groundHeight
        )
        {
            string[] farRoomAnchorPrefixes =
            {
                "Lamp_Floor_01",
                "Table_Side_Apt_01",
            };
            float[] anchorOffsets = { 0.55f, 0.2f, -0.15f };
            Transform bestAnchor = null;
            float bestAnchorDistanceSq = float.NegativeInfinity;

            for (int anchorIndex = 0; anchorIndex < farRoomAnchorPrefixes.Length; anchorIndex++)
            {
                Transform[] anchors = FindNamedTransformsInSubtree(livingRoomRoot, farRoomAnchorPrefixes[anchorIndex]);
                for (int index = 0; index < anchors.Length; index++)
                {
                    Transform anchor = anchors[index];
                    if (anchor == null)
                    {
                        continue;
                    }

                    Vector3 towardTv = FlattenDirection(television.position - anchor.position);
                    if (towardTv.sqrMagnitude < 0.001f)
                    {
                        continue;
                    }

                    Vector3 awayFromTv = -towardTv;
                    float distanceSq = (anchor.position - television.position).sqrMagnitude;
                    if (distanceSq > bestAnchorDistanceSq)
                    {
                        bestAnchorDistanceSq = distanceSq;
                        bestAnchor = anchor;
                    }

                    for (int offsetIndex = 0; offsetIndex < anchorOffsets.Length; offsetIndex++)
                    {
                        float offset = anchorOffsets[offsetIndex];
                        Vector3 candidate = FlattenToGround(anchor.position + (awayFromTv * offset), groundHeight);
                        if (!IsPlacementBlocked(candidate))
                        {
                            return candidate;
                        }
                    }
                }
            }

            if (bestAnchor != null)
            {
                Vector3 awayFromTv = FlattenDirection(bestAnchor.position - television.position);
                if (awayFromTv.sqrMagnitude > 0.001f)
                {
                    Vector3 bestAnchorCandidate = FlattenToGround(bestAnchor.position + (awayFromTv * 0.35f), groundHeight);
                    if (!IsPlacementBlocked(bestAnchorCandidate))
                    {
                        return bestAnchorCandidate;
                    }
                }
            }

            Vector3 sideDirection = new Vector3(-roomFacingDirection.z, 0.0f, roomFacingDirection.x);
            float[] distanceCandidates =
            {
                spawnStandOffM,
                Mathf.Max(goalStandOffM + 3.6f, spawnStandOffM - 0.75f),
                Mathf.Max(goalStandOffM + 2.8f, spawnStandOffM - 1.5f),
                Mathf.Max(goalStandOffM + 2.1f, spawnStandOffM - 2.2f),
            };
            float[] lateralOffsets = { 0.0f, 0.85f, -0.85f, 1.45f, -1.45f };

            for (int distanceIndex = 0; distanceIndex < distanceCandidates.Length; distanceIndex++)
            {
                float distance = distanceCandidates[distanceIndex];
                for (int offsetIndex = 0; offsetIndex < lateralOffsets.Length; offsetIndex++)
                {
                    float lateralOffset = lateralOffsets[offsetIndex];
                    Vector3 candidate = FlattenToGround(
                        television.position + (roomFacingDirection * distance) + (sideDirection * lateralOffset),
                        groundHeight
                    );
                    if (!IsPlacementBlocked(candidate))
                    {
                        return candidate;
                    }
                }
            }

            Transform rug = FindNamedTransformInSubtree(livingRoomRoot, "Rug_Apt_01");
            if (rug != null)
            {
                Vector3 rugPosition = FlattenToGround(rug.position + (sideDirection * 1.2f), groundHeight);
                if (!IsPlacementBlocked(rugPosition))
                {
                    return rugPosition;
                }
            }

            return FlattenToGround(television.position + (roomFacingDirection * spawnStandOffM), groundHeight);
        }

        private static bool TryGetDirectionBetween(Vector3 origin, Transform target, out Vector3 direction)
        {
            if (target == null)
            {
                direction = Vector3.zero;
                return false;
            }

            direction = FlattenDirection(target.position - origin);
            return direction.sqrMagnitude > 0.001f;
        }

        private bool IsPlacementBlocked(Vector3 anchorPosition)
        {
            float probeRadius = Mathf.Max(characterRadiusM * 0.92f, 0.18f);
            Vector3 bottom = anchorPosition + (Vector3.up * (probeRadius + 0.02f));
            Vector3 top = anchorPosition + (Vector3.up * Mathf.Max(characterHeightM - probeRadius, probeRadius + 0.25f));
            Collider[] hits = Physics.OverlapCapsule(
                bottom,
                top,
                probeRadius,
                Physics.DefaultRaycastLayers,
                QueryTriggerInteraction.Ignore
            );
            for (int index = 0; index < hits.Length; index++)
            {
                Collider hit = hits[index];
                if (hit == null || hit.transform.IsChildOf(transform))
                {
                    continue;
                }

                return true;
            }

            return false;
        }

        private Transform FindContainingAncestorNamed(Transform start, string expectedName)
        {
            Transform current = start;
            while (current != null)
            {
                if (string.Equals(current.name, expectedName, StringComparison.OrdinalIgnoreCase))
                {
                    return current;
                }

                current = current.parent;
            }

            return null;
        }

        private Transform FindNamedTransformInSubtree(Transform root, string prefix)
        {
            if (root == null)
            {
                return null;
            }

            Transform[] transforms = root.GetComponentsInChildren<Transform>(true);
            for (int index = 0; index < transforms.Length; index++)
            {
                Transform candidate = transforms[index];
                if (candidate == null)
                {
                    continue;
                }

                if (candidate.name.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                {
                    return candidate;
                }
            }

            return null;
        }

        private Vector3 ResolveTelevisionSpawnLookTarget(Transform livingRoomRoot, Vector3 televisionPosition, Vector3 spawnPosition)
        {
            if (Mathf.Abs(spawnPosition.x - televisionPosition.x) > 1.0f)
            {
                return new Vector3(televisionPosition.x, spawnPosition.y, spawnPosition.z);
            }

            string[] lookTargetPrefixes =
            {
                "Coffee_Table_Apt_01",
                "Rug_Apt_01",
                "Sofa_Apt_01",
            };

            for (int index = 0; index < lookTargetPrefixes.Length; index++)
            {
                Transform target = FindNamedTransformInSubtree(livingRoomRoot, lookTargetPrefixes[index]);
                if (target == null)
                {
                    continue;
                }

                Vector3 horizontalDelta = target.position - spawnPosition;
                horizontalDelta.y = 0.0f;
                if (horizontalDelta.sqrMagnitude > 0.25f)
                {
                    return target.position;
                }
            }

            return televisionPosition;
        }

        private Transform[] FindNamedTransformsInSubtree(Transform root, string prefix)
        {
            if (root == null)
            {
                return Array.Empty<Transform>();
            }

            Transform[] transforms = root.GetComponentsInChildren<Transform>(true);
            int count = 0;
            for (int index = 0; index < transforms.Length; index++)
            {
                Transform candidate = transforms[index];
                if (candidate != null && candidate.name.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                {
                    count += 1;
                }
            }

            Transform[] matches = new Transform[count];
            int offset = 0;
            for (int index = 0; index < transforms.Length; index++)
            {
                Transform candidate = transforms[index];
                if (candidate != null && candidate.name.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                {
                    matches[offset] = candidate;
                    offset += 1;
                }
            }

            return matches;
        }

        private float ResolveRoomGroundHeight(Transform livingRoomRoot)
        {
            string[] preferredFloorReferences =
            {
                "Rug_Apt_01",
                "Coffee_Table_Apt_01",
                "Sofa_Apt_01",
                "MediaTable_01",
            };

            for (int index = 0; index < preferredFloorReferences.Length; index++)
            {
                Transform reference = FindNamedTransformInSubtree(livingRoomRoot, preferredFloorReferences[index]);
                if (reference != null)
                {
                    return reference.position.y;
                }
            }

            Transform sceneCamera = FindPreferredSceneCamera();
            if (sceneCamera != null)
            {
                return Mathf.Max(sceneCamera.position.y - cameraHeightM, runtimeGroundHeightM);
            }

            return runtimeGroundHeightM;
        }

        private Vector3 EstimateSceneCenter()
        {
            Renderer[] renderers = FindSceneComponents<Renderer>();
            Bounds bounds = default;
            bool hasBounds = false;
            for (int index = 0; index < renderers.Length; index++)
            {
                Renderer renderer = renderers[index];
                if (renderer == null || renderer.transform.IsChildOf(transform))
                {
                    continue;
                }

                if (!hasBounds)
                {
                    bounds = renderer.bounds;
                    hasBounds = true;
                    continue;
                }

                bounds.Encapsulate(renderer.bounds);
            }

            return hasBounds ? bounds.center : Vector3.zero;
        }

        private static Vector3 FlattenToGround(Vector3 position, float groundHeight)
        {
            position.y = groundHeight;
            return position;
        }

        private static Vector3 FlattenDirection(Vector3 direction)
        {
            direction.y = 0.0f;
            return direction.sqrMagnitude > 0.001f ? direction.normalized : Vector3.zero;
        }

        private static Quaternion LookToward(Vector3 origin, Vector3 target)
        {
            Vector3 facing = target - origin;
            facing.y = 0.0f;
            return facing.sqrMagnitude > 0.001f
                ? Quaternion.LookRotation(facing.normalized, Vector3.up)
                : Quaternion.identity;
        }

        private T[] FindSceneComponents<T>() where T : Component
        {
            GameObject[] roots = gameObject.scene.GetRootGameObjects();
            int componentCount = 0;
            for (int index = 0; index < roots.Length; index++)
            {
                componentCount += roots[index].GetComponentsInChildren<T>(true).Length;
            }

            T[] allComponents = new T[componentCount];
            int offset = 0;
            for (int index = 0; index < roots.Length; index++)
            {
                T[] components = roots[index].GetComponentsInChildren<T>(true);
                Array.Copy(components, 0, allComponents, offset, components.Length);
                offset += components.Length;
            }

            return allComponents;
        }

        private Transform FindRequiredChild(Transform parent, string path)
        {
            return parent == null ? null : parent.Find(path);
        }

        private static Transform FindOrCreateChild(Transform parent, string childName)
        {
            Transform child = parent.Find(childName);
            if (child != null)
            {
                return child;
            }

            child = new GameObject(childName).transform;
            child.SetParent(parent, false);
            return child;
        }

        private static void ConfigureDirectionalLight(
            Light lightComponent,
            string lightName,
            Color color,
            float intensity,
            Vector3 eulerAngles,
            LightShadows shadows
        )
        {
            if (lightComponent == null)
            {
                return;
            }

            lightComponent.gameObject.name = lightName;
            lightComponent.type = LightType.Directional;
            lightComponent.color = color;
            lightComponent.intensity = intensity;
            lightComponent.shadows = shadows;
            lightComponent.renderMode = LightRenderMode.Auto;
            lightComponent.bounceIntensity = 1.0f;
            lightComponent.transform.localPosition = Vector3.zero;
            lightComponent.transform.localRotation = Quaternion.Euler(eulerAngles);
        }

        private static T GetOrAddComponent<T>(GameObject target) where T : Component
        {
            T existing = target.GetComponent<T>();
            return existing != null ? existing : target.AddComponent<T>();
        }
    }
}

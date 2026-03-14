using System.Collections.Generic;
using UnityEngine;
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
        private const float RoomWidth = 7.2f;
        private const float RoomHeight = 2.5f;
        private const float RoomDepth = 5.4f;
        private const float WallThickness = 0.12f;
        private const string GeneratedRootPrefix = "__GeneratedLivingRoom";

        [SerializeField] private int layoutRevision = 1;
        [SerializeField] private int captureWidth = 640;
        [SerializeField] private int captureHeight = 360;
        [SerializeField] private float cameraHeightM = 1.25f;
        [SerializeField] private float cameraVerticalFieldOfViewDeg = 34.782f;
        [SerializeField] private float cameraNearClipM = 0.2f;
        [SerializeField] private float cameraFarClipM = 8.0f;
        [SerializeField] private bool autoRebuildInEditor = true;

        private readonly Dictionary<string, Material> _materialCache = new Dictionary<string, Material>();
        private bool _editorRebuildQueued;

        private string ExpectedGeneratedRootName => $"{GeneratedRootPrefix}_{layoutRevision}";

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
            EnsureSceneGraph();
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

        [ContextMenu("Rebuild Living Room")]
        public void RebuildLivingRoom()
        {
            RebuildSceneGraph(force: true);
        }

        private void QueueOrEnsureScene()
        {
#if UNITY_EDITOR
            if (!Application.isPlaying && autoRebuildInEditor)
            {
                if (_editorRebuildQueued)
                {
                    return;
                }

                _editorRebuildQueued = true;
                EditorApplication.delayCall += EditorEnsureSceneGraph;
                return;
            }
#endif
            EnsureSceneGraph();
        }

#if UNITY_EDITOR
        private void EditorEnsureSceneGraph()
        {
            _editorRebuildQueued = false;
            if (this == null)
            {
                return;
            }

            EnsureSceneGraph();
        }
#endif

        private void EnsureSceneGraph()
        {
            RebuildSceneGraph(force: false);
            WireRuntimeComponents();
        }

        private void RebuildSceneGraph(bool force)
        {
            Transform generatedRoot = FindGeneratedRoot();
            bool needsBuild = force || generatedRoot == null || generatedRoot.childCount == 0;
            if (!needsBuild)
            {
                return;
            }

            if (generatedRoot != null)
            {
                DestroySafe(generatedRoot.gameObject);
            }

            generatedRoot = new GameObject(ExpectedGeneratedRootName).transform;
            generatedRoot.SetParent(transform, false);

            Transform architectureRoot = CreateGroup(generatedRoot, "Architecture");
            Transform decorRoot = CreateGroup(generatedRoot, "Decor");
            Transform rigRoot = CreateGroup(generatedRoot, "RobotRig");

            BuildArchitecture(architectureRoot);
            BuildDecor(decorRoot);
            BuildRobotRig(rigRoot);
            BuildGoalTrigger(rigRoot);
            BuildLighting(generatedRoot);

#if UNITY_EDITOR
            if (!Application.isPlaying)
            {
                EditorSceneManager.MarkSceneDirty(gameObject.scene);
            }
#endif
        }

        private void WireRuntimeComponents()
        {
            Transform generatedRoot = FindGeneratedRoot();
            if (generatedRoot == null)
            {
                return;
            }

            Transform robotRoot = generatedRoot.Find("RobotRig/RobotRoot");
            Transform cameraPanPivot = generatedRoot.Find("RobotRig/RobotRoot/CameraPanPivot");
            Transform goalTrigger = generatedRoot.Find("RobotRig/HiddenGoalTrigger");
            CharacterController controller = robotRoot == null ? null : robotRoot.GetComponent<CharacterController>();
            Camera robotCamera = cameraPanPivot == null ? null : cameraPanPivot.GetComponentInChildren<Camera>(true);

            RobotRigController rigController = GetOrAddComponent<RobotRigController>(gameObject);
            rigController.Configure(
                controller,
                robotRoot,
                cameraPanPivot,
                robotCamera,
                captureWidth,
                captureHeight,
                cameraHeightM,
                cameraVerticalFieldOfViewDeg
            );

            SessionState sessionState = GetOrAddComponent<SessionState>(gameObject);
            sessionState.Configure(rigController, robotRoot);

            GoalTrigger trigger = goalTrigger == null ? null : goalTrigger.GetComponent<GoalTrigger>();
            if (trigger != null)
            {
                trigger.Configure(sessionState);
            }

            ManualInputController manualInput = GetOrAddComponent<ManualInputController>(gameObject);
            HudOverlay hudOverlay = GetOrAddComponent<HudOverlay>(gameObject);
            AgentTcpServer agentServer = GetOrAddComponent<AgentTcpServer>(gameObject);

            manualInput.Configure(rigController, sessionState, hudOverlay);
            agentServer.Configure("127.0.0.1", 8765, rigController, sessionState);
            hudOverlay.Configure(sessionState, manualInput, agentServer, SimulatorBootMode.Manual);
        }

        private void ApplyBootMode()
        {
            if (!Application.isPlaying)
            {
                return;
            }

            SessionState sessionState = GetComponent<SessionState>();
            ManualInputController manualInput = GetComponent<ManualInputController>();
            HudOverlay hudOverlay = GetComponent<HudOverlay>();
            AgentTcpServer agentServer = GetComponent<AgentTcpServer>();
            if (sessionState == null || manualInput == null || hudOverlay == null || agentServer == null)
            {
                return;
            }

            BootModeConfig bootMode = BootModeResolver.Resolve(System.Environment.GetCommandLineArgs());
            agentServer.Configure(bootMode.Host, bootMode.Port, GetComponent<RobotRigController>(), sessionState);
            agentServer.EnableAgentMode(bootMode.Mode == SimulatorBootMode.Agent);
            hudOverlay.Configure(sessionState, manualInput, agentServer, bootMode.Mode);

            if (bootMode.Mode == SimulatorBootMode.Agent)
            {
                manualInput.EnableManualMode(false);
                Cursor.lockState = CursorLockMode.Locked;
                Cursor.visible = false;
                agentServer.StartServer();
                sessionState.ResetEpisode();
                sessionState.ShowTransientStatus("Agent mode listening", 5.0f);
                return;
            }

            agentServer.StopServer();
            manualInput.EnableManualMode(true);
            sessionState.ResetEpisode();
            sessionState.ShowTransientStatus("Manual mode ready", 3.0f);
        }

        private void BuildArchitecture(Transform parent)
        {
            float halfWidth = RoomWidth * 0.5f;
            float halfDepth = RoomDepth * 0.5f;
            float frontGlassWidth = RoomWidth - 0.30f;

            CreateBox(
                name: "Floor",
                parent: parent,
                localPosition: new Vector3(0.0f, -0.03f, 0.0f),
                localScale: new Vector3(RoomWidth, 0.06f, RoomDepth),
                color: new Color(0.58f, 0.42f, 0.28f)
            );
            CreateBox(
                name: "Ceiling",
                parent: parent,
                localPosition: new Vector3(0.0f, RoomHeight + 0.03f, 0.0f),
                localScale: new Vector3(RoomWidth, 0.06f, RoomDepth),
                color: new Color(0.95f, 0.95f, 0.94f)
            );
            CreateBox(
                name: "WallLeft",
                parent: parent,
                localPosition: new Vector3(-halfWidth + (WallThickness * 0.5f), RoomHeight * 0.5f, 0.0f),
                localScale: new Vector3(WallThickness, RoomHeight, RoomDepth),
                color: new Color(0.93f, 0.93f, 0.91f)
            );
            CreateBox(
                name: "WallRight",
                parent: parent,
                localPosition: new Vector3(halfWidth - (WallThickness * 0.5f), RoomHeight * 0.5f, 0.0f),
                localScale: new Vector3(WallThickness, RoomHeight, RoomDepth),
                color: new Color(0.93f, 0.93f, 0.91f)
            );
            CreateBox(
                name: "WallBack",
                parent: parent,
                localPosition: new Vector3(0.0f, RoomHeight * 0.5f, -halfDepth + (WallThickness * 0.5f)),
                localScale: new Vector3(RoomWidth, RoomHeight, WallThickness),
                color: new Color(0.94f, 0.94f, 0.92f)
            );
            CreateBox(
                name: "FrontGlass",
                parent: parent,
                localPosition: new Vector3(0.0f, 1.30f, halfDepth - 0.02f),
                localScale: new Vector3(frontGlassWidth, 2.15f, 0.04f),
                color: new Color(0.78f, 0.86f, 0.92f, 0.38f),
                colliderEnabled: false,
                transparent: true
            );
            CreateBox(
                name: "FrontFrameTop",
                parent: parent,
                localPosition: new Vector3(0.0f, 2.30f, halfDepth - 0.02f),
                localScale: new Vector3(frontGlassWidth, 0.18f, 0.08f),
                color: new Color(0.95f, 0.95f, 0.94f)
            );
            CreateBox(
                name: "FrontFrameBottom",
                parent: parent,
                localPosition: new Vector3(0.0f, 0.18f, halfDepth - 0.02f),
                localScale: new Vector3(frontGlassWidth, 0.16f, 0.08f),
                color: new Color(0.93f, 0.93f, 0.92f)
            );
            CreateBox(
                name: "FrontFrameLeft",
                parent: parent,
                localPosition: new Vector3(-(frontGlassWidth * 0.5f) + 0.07f, 1.30f, halfDepth - 0.02f),
                localScale: new Vector3(0.08f, 2.15f, 0.08f),
                color: new Color(0.94f, 0.94f, 0.92f)
            );
            CreateBox(
                name: "FrontFrameRight",
                parent: parent,
                localPosition: new Vector3((frontGlassWidth * 0.5f) - 0.07f, 1.30f, halfDepth - 0.02f),
                localScale: new Vector3(0.08f, 2.15f, 0.08f),
                color: new Color(0.94f, 0.94f, 0.92f)
            );
            CreateBox(
                name: "CurtainLeft",
                parent: parent,
                localPosition: new Vector3(-(frontGlassWidth * 0.5f) + 0.30f, 1.45f, halfDepth - 0.09f),
                localScale: new Vector3(0.45f, 2.05f, 0.06f),
                color: new Color(0.82f, 0.78f, 0.72f),
                colliderEnabled: false
            );
            CreateBox(
                name: "CurtainRight",
                parent: parent,
                localPosition: new Vector3((frontGlassWidth * 0.5f) - 0.30f, 1.45f, halfDepth - 0.09f),
                localScale: new Vector3(0.45f, 2.05f, 0.06f),
                color: new Color(0.82f, 0.78f, 0.72f),
                colliderEnabled: false
            );
            CreateBox(
                name: "Rug",
                parent: parent,
                localPosition: new Vector3(-1.35f, 0.01f, 0.50f),
                localScale: new Vector3(2.40f, 0.02f, 1.85f),
                color: new Color(0.84f, 0.78f, 0.68f),
                colliderEnabled: false
            );
        }

        private void BuildDecor(Transform parent)
        {
            BuildSofa(parent);
            BuildCoffeeTable(parent);
            BuildTvConsole(parent);
            BuildDiningTable(parent);
            BuildDiningChair(parent, "DiningChairFront", new Vector3(1.75f, 0.0f, 0.72f), 180.0f);
            BuildDiningChair(parent, "DiningChairBack", new Vector3(1.75f, 0.0f, 2.58f), 0.0f);
            BuildDiningChair(parent, "DiningChairLeft", new Vector3(0.72f, 0.0f, 1.65f), 90.0f);
            BuildDiningChair(parent, "DiningChairRight", new Vector3(2.78f, 0.0f, 1.65f), -90.0f);
            BuildLamp(parent);
        }

        private void BuildSofa(Transform parent)
        {
            Transform group = CreateGroup(parent, "Sofa");
            group.localPosition = new Vector3(-1.85f, 0.0f, -0.15f);
            CreateBox("Base", group, new Vector3(0.0f, 0.22f, 0.0f), new Vector3(2.20f, 0.44f, 0.95f), new Color(0.57f, 0.60f, 0.63f));
            CreateBox("Back", group, new Vector3(0.0f, 0.68f, -0.33f), new Vector3(2.20f, 0.76f, 0.28f), new Color(0.53f, 0.56f, 0.60f));
            CreateBox("ArmLeft", group, new Vector3(-0.98f, 0.52f, 0.0f), new Vector3(0.24f, 0.58f, 0.95f), new Color(0.53f, 0.56f, 0.60f));
            CreateBox("ArmRight", group, new Vector3(0.98f, 0.52f, 0.0f), new Vector3(0.24f, 0.58f, 0.95f), new Color(0.53f, 0.56f, 0.60f));
        }

        private void BuildCoffeeTable(Transform parent)
        {
            Transform group = CreateGroup(parent, "CoffeeTable");
            group.localPosition = new Vector3(-1.45f, 0.0f, 0.95f);
            CreateBox("Top", group, new Vector3(0.0f, 0.40f, 0.0f), new Vector3(1.00f, 0.06f, 0.60f), new Color(0.54f, 0.37f, 0.20f));
            CreateTableLegs(group, 0.42f, 0.18f, 0.42f, 0.30f, 0.05f, new Color(0.33f, 0.23f, 0.13f));
        }

        private void BuildTvConsole(Transform parent)
        {
            Transform group = CreateGroup(parent, "TvConsole");
            group.localPosition = new Vector3(2.15f, 0.0f, -1.80f);
            CreateBox("ConsoleBody", group, new Vector3(0.0f, 0.27f, 0.0f), new Vector3(1.80f, 0.54f, 0.45f), new Color(0.43f, 0.29f, 0.17f));
            CreateBox("TvPanel", group, new Vector3(0.0f, 1.15f, 0.22f), new Vector3(1.40f, 0.80f, 0.06f), new Color(0.08f, 0.08f, 0.09f), colliderEnabled: false);
        }

        private void BuildDiningTable(Transform parent)
        {
            Transform group = CreateGroup(parent, "DiningTable");
            group.localPosition = new Vector3(1.75f, 0.0f, 1.65f);
            CreateBox("Top", group, new Vector3(0.0f, 0.75f, 0.0f), new Vector3(1.60f, 0.07f, 0.90f), new Color(0.66f, 0.49f, 0.28f));
            CreateTableLegs(group, 0.68f, 0.68f, 0.33f, 0.33f, 0.08f, new Color(0.42f, 0.28f, 0.16f));
        }

        private void BuildDiningChair(Transform parent, string name, Vector3 center, float yawDeg)
        {
            Transform group = CreateGroup(parent, name);
            group.localPosition = center;
            group.localRotation = Quaternion.Euler(0.0f, yawDeg, 0.0f);
            CreateBox("Seat", group, new Vector3(0.0f, 0.45f, 0.0f), new Vector3(0.50f, 0.07f, 0.50f), new Color(0.67f, 0.51f, 0.31f));
            CreateBox("Back", group, new Vector3(0.0f, 0.82f, -0.20f), new Vector3(0.50f, 0.62f, 0.08f), new Color(0.59f, 0.44f, 0.26f));
            CreateTableLegs(group, 0.40f, 0.40f, 0.18f, 0.18f, 0.05f, new Color(0.38f, 0.26f, 0.15f));
        }

        private void BuildLamp(Transform parent)
        {
            Transform group = CreateGroup(parent, "FloorLamp");
            group.localPosition = new Vector3(-2.75f, 0.0f, 1.75f);
            CreateCylinder("Base", group, new Vector3(0.0f, 0.03f, 0.0f), new Vector3(0.34f, 0.03f, 0.34f), new Color(0.18f, 0.18f, 0.18f));
            CreateCylinder("Pole", group, new Vector3(0.0f, 0.82f, 0.0f), new Vector3(0.04f, 0.82f, 0.04f), new Color(0.21f, 0.21f, 0.22f));
            CreateCylinder("Shade", group, new Vector3(0.0f, 1.62f, 0.0f), new Vector3(0.26f, 0.18f, 0.26f), new Color(0.90f, 0.84f, 0.72f), colliderEnabled: false);
        }

        private void BuildRobotRig(Transform parent)
        {
            Transform robotRoot = CreateGroup(parent, "RobotRoot");
            robotRoot.localPosition = new Vector3(-2.4f, 0.0f, -1.85f);

            CharacterController controller = GetOrAddComponent<CharacterController>(robotRoot.gameObject);
            controller.height = 1.68f;
            controller.radius = 0.28f;
            controller.center = new Vector3(0.0f, 0.84f, 0.0f);
            controller.slopeLimit = 45.0f;
            controller.stepOffset = 0.25f;

            GameObject body = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            body.name = "RobotBody";
            body.transform.SetParent(robotRoot, false);
            body.transform.localPosition = new Vector3(0.0f, 0.78f, 0.0f);
            body.transform.localScale = new Vector3(0.42f, 0.70f, 0.42f);
            body.GetComponent<Renderer>().sharedMaterial = GetOrCreateMaterial("robot", new Color(0.24f, 0.30f, 0.38f));
            RemoveColliders(body);

            Transform cameraPanPivot = CreateGroup(robotRoot, "CameraPanPivot");
            cameraPanPivot.localPosition = new Vector3(0.0f, cameraHeightM, 0.0f);

            GameObject cameraObject = new GameObject("RobotCamera");
            cameraObject.transform.SetParent(cameraPanPivot, false);
            cameraObject.transform.localPosition = Vector3.zero;
            cameraObject.transform.localRotation = Quaternion.identity;
            Camera robotCamera = cameraObject.AddComponent<Camera>();
            robotCamera.nearClipPlane = cameraNearClipM;
            robotCamera.farClipPlane = cameraFarClipM;
            robotCamera.fieldOfView = cameraVerticalFieldOfViewDeg;
            robotCamera.clearFlags = CameraClearFlags.Skybox;
            robotCamera.allowHDR = true;
            robotCamera.allowMSAA = true;
            cameraObject.tag = "MainCamera";
        }

        private void BuildGoalTrigger(Transform parent)
        {
            Transform triggerRoot = CreateGroup(parent, "HiddenGoalTrigger");
            triggerRoot.localPosition = new Vector3(1.75f, 0.40f, 0.45f);
            BoxCollider collider = GetOrAddComponent<BoxCollider>(triggerRoot.gameObject);
            collider.size = new Vector3(0.85f, 1.40f, 0.70f);
            collider.isTrigger = true;
            GoalTrigger trigger = GetOrAddComponent<GoalTrigger>(triggerRoot.gameObject);
            Renderer renderer = triggerRoot.GetComponent<Renderer>();
            if (renderer != null)
            {
                renderer.enabled = false;
            }
        }

        private void BuildLighting(Transform parent)
        {
            GameObject sun = new GameObject("SunMain");
            sun.transform.SetParent(parent, false);
            sun.transform.position = new Vector3(0.0f, 4.5f, 4.8f);
            sun.transform.rotation = Quaternion.Euler(55.0f, 0.0f, 180.0f);
            Light sunLight = sun.AddComponent<Light>();
            sunLight.type = LightType.Directional;
            sunLight.color = new Color(1.0f, 0.96f, 0.90f);
            sunLight.intensity = 1.15f;
            sunLight.shadows = LightShadows.Soft;

            CreatePointLight(parent, "LivingCeiling", new Vector3(-1.2f, 2.2f, 0.3f), new Color(1.0f, 0.97f, 0.93f), 5.0f, 6.0f);
            CreatePointLight(parent, "DiningCeiling", new Vector3(1.8f, 2.15f, 1.65f), new Color(1.0f, 0.90f, 0.82f), 4.4f, 5.0f);
        }

        private void CreateTableLegs(
            Transform parent,
            float offsetX,
            float offsetZ,
            float legHeight,
            float legCenterY,
            float legWidth,
            Color color
        )
        {
            CreateBox("LegFrontLeft", parent, new Vector3(-offsetX, legCenterY, offsetZ), new Vector3(legWidth, legHeight, legWidth), color);
            CreateBox("LegFrontRight", parent, new Vector3(offsetX, legCenterY, offsetZ), new Vector3(legWidth, legHeight, legWidth), color);
            CreateBox("LegBackLeft", parent, new Vector3(-offsetX, legCenterY, -offsetZ), new Vector3(legWidth, legHeight, legWidth), color);
            CreateBox("LegBackRight", parent, new Vector3(offsetX, legCenterY, -offsetZ), new Vector3(legWidth, legHeight, legWidth), color);
        }

        private void CreatePointLight(Transform parent, string name, Vector3 position, Color color, float intensity, float range)
        {
            GameObject lightObject = new GameObject(name);
            lightObject.transform.SetParent(parent, false);
            lightObject.transform.position = position;
            Light lightComponent = lightObject.AddComponent<Light>();
            lightComponent.type = LightType.Point;
            lightComponent.color = color;
            lightComponent.intensity = intensity;
            lightComponent.range = range;
            lightComponent.shadows = LightShadows.Soft;
        }

        private Transform FindGeneratedRoot()
        {
            for (int index = transform.childCount - 1; index >= 0; index--)
            {
                Transform child = transform.GetChild(index);
                if (child.name == ExpectedGeneratedRootName)
                {
                    return child;
                }

                if (child.name.StartsWith(GeneratedRootPrefix, System.StringComparison.Ordinal))
                {
                    DestroySafe(child.gameObject);
                }
            }

            return null;
        }

        private Transform CreateGroup(Transform parent, string name)
        {
            Transform existing = parent.Find(name);
            if (existing != null)
            {
                return existing;
            }

            GameObject group = new GameObject(name);
            group.transform.SetParent(parent, false);
            return group.transform;
        }

        private GameObject CreateBox(
            string name,
            Transform parent,
            Vector3 localPosition,
            Vector3 localScale,
            Color color,
            bool colliderEnabled = true,
            bool transparent = false
        )
        {
            GameObject box = GameObject.CreatePrimitive(PrimitiveType.Cube);
            box.name = name;
            box.transform.SetParent(parent, false);
            box.transform.localPosition = localPosition;
            box.transform.localRotation = Quaternion.identity;
            box.transform.localScale = localScale;
            box.GetComponent<Renderer>().sharedMaterial = GetOrCreateMaterial(name + "_mat", color, transparent);
            if (!colliderEnabled)
            {
                RemoveColliders(box);
            }

            return box;
        }

        private GameObject CreateCylinder(
            string name,
            Transform parent,
            Vector3 localPosition,
            Vector3 localScale,
            Color color,
            bool colliderEnabled = true
        )
        {
            GameObject cylinder = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            cylinder.name = name;
            cylinder.transform.SetParent(parent, false);
            cylinder.transform.localPosition = localPosition;
            cylinder.transform.localRotation = Quaternion.identity;
            cylinder.transform.localScale = localScale;
            cylinder.GetComponent<Renderer>().sharedMaterial = GetOrCreateMaterial(name + "_mat", color);
            if (!colliderEnabled)
            {
                RemoveColliders(cylinder);
            }

            return cylinder;
        }

        private void RemoveColliders(GameObject gameObject)
        {
            Collider[] colliders = gameObject.GetComponents<Collider>();
            for (int index = 0; index < colliders.Length; index++)
            {
                DestroySafe(colliders[index]);
            }
        }

        private Material GetOrCreateMaterial(string key, Color color, bool transparent = false)
        {
            Material cached;
            if (_materialCache.TryGetValue(key, out cached) && cached != null)
            {
                return cached;
            }

            Shader shader = Shader.Find("Universal Render Pipeline/Lit");
            if (shader == null)
            {
                shader = Shader.Find("Standard");
            }

            if (shader == null)
            {
                shader = Shader.Find("Diffuse");
            }

            Material material = new Material(shader);
            material.name = key;
            ConfigureMaterial(material, color, transparent);
            _materialCache[key] = material;
            return material;
        }

        private void ConfigureMaterial(Material material, Color color, bool transparent)
        {
            material.color = color;
            if (!transparent)
            {
                return;
            }

            if (material.HasProperty("_Surface"))
            {
                material.SetFloat("_Surface", 1.0f);
            }

            if (material.HasProperty("_Mode"))
            {
                material.SetFloat("_Mode", 3.0f);
            }

            material.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            material.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            material.SetInt("_ZWrite", 0);
            material.DisableKeyword("_ALPHATEST_ON");
            material.EnableKeyword("_ALPHABLEND_ON");
            material.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
        }

        private static T GetOrAddComponent<T>(GameObject owner)
            where T : Component
        {
            T existing = owner.GetComponent<T>();
            if (existing != null)
            {
                return existing;
            }

            return owner.AddComponent<T>();
        }

        private static void DestroySafe(Object target)
        {
            if (target == null)
            {
                return;
            }

#if UNITY_EDITOR
            if (!Application.isPlaying)
            {
                DestroyImmediate(target);
                return;
            }
#endif
            Destroy(target);
        }
    }
}

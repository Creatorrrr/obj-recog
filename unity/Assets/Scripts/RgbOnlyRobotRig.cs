using UnityEngine;

public sealed class RgbOnlyRobotRig : MonoBehaviour
{
    [SerializeField] private CharacterController robotController;
    [SerializeField] private Transform robotRoot;
    [SerializeField] private Transform cameraPanPivot;
    [SerializeField] private Camera robotCamera;
    [SerializeField] private int imageWidth = 640;
    [SerializeField] private int imageHeight = 360;

    private Vector3 _startPosition;
    private Quaternion _startRotation;
    private Quaternion _startPanRotation;

    public Camera RobotCamera => robotCamera;

    private void Awake()
    {
        _startPosition = robotRoot.position;
        _startRotation = robotRoot.rotation;
        _startPanRotation = cameraPanPivot.localRotation;
    }

    public void ResetEpisode()
    {
        robotRoot.position = _startPosition;
        robotRoot.rotation = _startRotation;
        cameraPanPivot.localRotation = _startPanRotation;
    }

    public void ApplyCommand(string primitive, float value)
    {
        switch (primitive)
        {
            case "move_forward":
                Move(robotRoot.forward * value);
                break;
            case "move_backward":
                Move(-robotRoot.forward * value);
                break;
            case "strafe_left":
                Move(-robotRoot.right * value);
                break;
            case "strafe_right":
                Move(robotRoot.right * value);
                break;
            case "turn_left":
                robotRoot.Rotate(0.0f, value, 0.0f, Space.World);
                break;
            case "turn_right":
                robotRoot.Rotate(0.0f, -value, 0.0f, Space.World);
                break;
            case "camera_pan_left":
                cameraPanPivot.Rotate(0.0f, value, 0.0f, Space.Self);
                break;
            case "camera_pan_right":
                cameraPanPivot.Rotate(0.0f, -value, 0.0f, Space.Self);
                break;
            case "pause":
                break;
            default:
                throw new System.InvalidOperationException("Unsupported primitive: " + primitive);
        }
    }

    public byte[] CapturePng()
    {
        var renderTexture = new RenderTexture(imageWidth, imageHeight, 24, RenderTextureFormat.ARGB32);
        var previousTarget = robotCamera.targetTexture;
        var previousActive = RenderTexture.active;
        robotCamera.targetTexture = renderTexture;
        robotCamera.Render();
        RenderTexture.active = renderTexture;

        var texture = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        texture.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        texture.Apply(false, false);

        var bytes = texture.EncodeToPNG();
        robotCamera.targetTexture = previousTarget;
        RenderTexture.active = previousActive;
        Destroy(renderTexture);
        Destroy(texture);
        return bytes;
    }

    private void Move(Vector3 delta)
    {
        if (robotController != null)
        {
            robotController.Move(delta);
            return;
        }
        robotRoot.position += delta;
    }
}

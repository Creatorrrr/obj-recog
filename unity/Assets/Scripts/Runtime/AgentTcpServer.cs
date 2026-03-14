using System;
using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace ObjRecog.UnitySim
{
    [DisallowMultipleComponent]
    public sealed class AgentTcpServer : MonoBehaviour
    {
        [Serializable]
        private sealed class RequestEnvelope
        {
            public string kind = string.Empty;
            public string scenario_id = string.Empty;
            public string primitive = string.Empty;
            public float value = 0.0f;
        }

        [Serializable]
        private sealed class FrameEnvelope
        {
            public string kind = "rgb_frame";
            public float timestamp_sec = 0.0f;
            public string image_encoding = "png";
            public string image_bytes_b64 = string.Empty;
        }

        [SerializeField] private string host = "127.0.0.1";
        [SerializeField] private int port = 8765;
        [SerializeField] private RobotRigController robotRig;
        [SerializeField] private SessionState sessionState;
        [SerializeField] private bool agentModeEnabled;

        private readonly ConcurrentQueue<Action> _mainThreadActions = new ConcurrentQueue<Action>();
        private CancellationTokenSource _cancellation;
        private TcpListener _listener;

        public bool ServerRunning { get; private set; }

        public bool ClientConnected { get; private set; }

        public string Host => host;

        public int Port => port;

        public void Configure(string resolvedHost, int resolvedPort, RobotRigController rig, SessionState state)
        {
            host = string.IsNullOrWhiteSpace(resolvedHost) ? "127.0.0.1" : resolvedHost;
            port = resolvedPort > 0 ? resolvedPort : 8765;
            robotRig = rig;
            sessionState = state;
        }

        public void EnableAgentMode(bool enabled)
        {
            agentModeEnabled = enabled;
        }

        private void Start()
        {
            if (Application.isPlaying && agentModeEnabled)
            {
                StartServer();
            }
        }

        private void Update()
        {
            while (_mainThreadActions.TryDequeue(out Action action))
            {
                action();
            }
        }

        private void OnDestroy()
        {
            StopServer();
        }

        public void StartServer()
        {
            if (_listener != null)
            {
                return;
            }

            IPAddress parsedAddress;
            if (!IPAddress.TryParse(host, out parsedAddress))
            {
                parsedAddress = IPAddress.Loopback;
            }

            _cancellation = new CancellationTokenSource();
            _listener = new TcpListener(parsedAddress, port);
            _listener.Start();
            ServerRunning = true;
            Task.Run(() => AcceptLoopAsync(_cancellation.Token));
        }

        public void StopServer()
        {
            if (_cancellation != null)
            {
                _cancellation.Cancel();
                _cancellation.Dispose();
                _cancellation = null;
            }

            if (_listener != null)
            {
                _listener.Stop();
                _listener = null;
            }

            ServerRunning = false;
            ClientConnected = false;
        }

        private async Task AcceptLoopAsync(CancellationToken cancellationToken)
        {
            while (!cancellationToken.IsCancellationRequested && _listener != null)
            {
                TcpClient client;
                try
                {
                    client = await _listener.AcceptTcpClientAsync();
                }
                catch (ObjectDisposedException)
                {
                    return;
                }
                catch (InvalidOperationException)
                {
                    return;
                }

                _ = Task.Run(() => HandleClientAsync(client, cancellationToken), cancellationToken);
            }
        }

        private async Task HandleClientAsync(TcpClient client, CancellationToken cancellationToken)
        {
            ClientConnected = true;
            using (client)
            using (NetworkStream stream = client.GetStream())
            {
                while (!cancellationToken.IsCancellationRequested && client.Connected)
                {
                    byte[] requestBytes = await ReadFrameAsync(stream, cancellationToken);
                    if (requestBytes == null)
                    {
                        ClientConnected = false;
                        return;
                    }

                    RequestEnvelope request = JsonUtility.FromJson<RequestEnvelope>(Encoding.UTF8.GetString(requestBytes));
                    if (request == null)
                    {
                        ClientConnected = false;
                        return;
                    }

                    if (request.kind == "shutdown")
                    {
                        Application.Quit();
                        ClientConnected = false;
                        return;
                    }

                    string response = await ExecuteOnMainThreadAsync(() => HandleRequest(request), cancellationToken);
                    if (string.IsNullOrEmpty(response))
                    {
                        ClientConnected = false;
                        return;
                    }

                    byte[] responseBytes = Encoding.UTF8.GetBytes(response);
                    byte[] header = BitConverter.GetBytes(IPAddress.HostToNetworkOrder(responseBytes.Length));
                    await stream.WriteAsync(header, 0, header.Length, cancellationToken);
                    await stream.WriteAsync(responseBytes, 0, responseBytes.Length, cancellationToken);
                }
            }

            ClientConnected = false;
        }

        private string HandleRequest(RequestEnvelope request)
        {
            if (robotRig == null || sessionState == null)
            {
                throw new InvalidOperationException("AgentTcpServer is not fully configured");
            }

            switch (request.kind)
            {
                case "reset_episode":
                    sessionState.ResetEpisode();
                    return CaptureFrameEnvelope();
                case "action":
                    robotRig.ApplyCommand(request.primitive, request.value);
                    return CaptureFrameEnvelope();
                default:
                    throw new InvalidOperationException("Unsupported request kind: " + request.kind);
            }
        }

        private string CaptureFrameEnvelope()
        {
            var payload = new FrameEnvelope
            {
                timestamp_sec = Time.time,
                image_bytes_b64 = Convert.ToBase64String(robotRig.CapturePng()),
            };
            return JsonUtility.ToJson(payload);
        }

        private Task<string> ExecuteOnMainThreadAsync(Func<string> action, CancellationToken cancellationToken)
        {
            var completion = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
            _mainThreadActions.Enqueue(() =>
            {
                try
                {
                    completion.TrySetResult(action());
                }
                catch (Exception exc)
                {
                    completion.TrySetException(exc);
                }
            });
            cancellationToken.Register(() => completion.TrySetCanceled(cancellationToken));
            return completion.Task;
        }

        private static async Task<byte[]> ReadFrameAsync(NetworkStream stream, CancellationToken cancellationToken)
        {
            var header = new byte[4];
            if (!await ReadExactAsync(stream, header, cancellationToken))
            {
                return null;
            }

            int payloadLength = IPAddress.NetworkToHostOrder(BitConverter.ToInt32(header, 0));
            if (payloadLength <= 0)
            {
                return null;
            }

            var payload = new byte[payloadLength];
            if (!await ReadExactAsync(stream, payload, cancellationToken))
            {
                return null;
            }

            return payload;
        }

        private static async Task<bool> ReadExactAsync(NetworkStream stream, byte[] buffer, CancellationToken cancellationToken)
        {
            int offset = 0;
            while (offset < buffer.Length)
            {
                int read = await stream.ReadAsync(buffer, offset, buffer.Length - offset, cancellationToken);
                if (read <= 0)
                {
                    return false;
                }

                offset += read;
            }

            return true;
        }
    }
}

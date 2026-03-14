using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

public sealed class RgbOnlyTcpServer : MonoBehaviour
{
    [Serializable]
    private sealed class RequestEnvelope
    {
        public string kind = "";
        public string scenario_id = "";
        public string primitive = "";
        public float value = 0.0f;
    }

    [Serializable]
    private sealed class FrameEnvelope
    {
        public string kind = "rgb_frame";
        public float timestamp_sec = 0.0f;
        public string image_encoding = "png";
        public string image_bytes_b64 = "";
    }

    [SerializeField] private string host = "127.0.0.1";
    [SerializeField] private int port = 8765;
    [SerializeField] private RgbOnlyRobotRig robotRig;

    private readonly ConcurrentQueue<Action> _mainThreadActions = new ConcurrentQueue<Action>();
    private CancellationTokenSource _cancellation;
    private TcpListener _listener;

    private void Start()
    {
        _cancellation = new CancellationTokenSource();
        _listener = new TcpListener(IPAddress.Parse(host), port);
        _listener.Start();
        _ = Task.Run(() => AcceptLoopAsync(_cancellation.Token));
    }

    private void Update()
    {
        while (_mainThreadActions.TryDequeue(out var action))
        {
            action();
        }
    }

    private void OnDestroy()
    {
        _cancellation.Cancel();
        _listener.Stop();
    }

    private async Task AcceptLoopAsync(CancellationToken cancellationToken)
    {
        while (!cancellationToken.IsCancellationRequested)
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
            _ = Task.Run(() => HandleClientAsync(client, cancellationToken), cancellationToken);
        }
    }

    private async Task HandleClientAsync(TcpClient client, CancellationToken cancellationToken)
    {
        using (client)
        using (var stream = client.GetStream())
        {
            while (!cancellationToken.IsCancellationRequested && client.Connected)
            {
                var requestBytes = await ReadFrameAsync(stream, cancellationToken);
                if (requestBytes == null)
                {
                    return;
                }

                var request = JsonUtility.FromJson<RequestEnvelope>(Encoding.UTF8.GetString(requestBytes));
                if (request == null)
                {
                    return;
                }

                if (request.kind == "shutdown")
                {
                    Application.Quit();
                    return;
                }

                var response = await ExecuteOnMainThreadAsync(() => HandleRequest(request), cancellationToken);
                var responseBytes = Encoding.UTF8.GetBytes(response);
                var header = BitConverter.GetBytes(IPAddress.HostToNetworkOrder(responseBytes.Length));
                await stream.WriteAsync(header, 0, header.Length, cancellationToken);
                await stream.WriteAsync(responseBytes, 0, responseBytes.Length, cancellationToken);
            }
        }
    }

    private string HandleRequest(RequestEnvelope request)
    {
        switch (request.kind)
        {
            case "reset_episode":
                robotRig.ResetEpisode();
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
        var tcs = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        _mainThreadActions.Enqueue(() =>
        {
            try
            {
                tcs.TrySetResult(action());
            }
            catch (Exception exc)
            {
                tcs.TrySetException(exc);
            }
        });
        cancellationToken.Register(() => tcs.TrySetCanceled(cancellationToken));
        return tcs.Task;
    }

    private static async Task<byte[]> ReadFrameAsync(NetworkStream stream, CancellationToken cancellationToken)
    {
        var header = new byte[4];
        if (!await ReadExactAsync(stream, header, cancellationToken))
        {
            return null;
        }

        var payloadLength = IPAddress.NetworkToHostOrder(BitConverter.ToInt32(header, 0));
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
        var offset = 0;
        while (offset < buffer.Length)
        {
            var read = await stream.ReadAsync(buffer, offset, buffer.Length - offset, cancellationToken);
            if (read <= 0)
            {
                return false;
            }
            offset += read;
        }
        return true;
    }
}

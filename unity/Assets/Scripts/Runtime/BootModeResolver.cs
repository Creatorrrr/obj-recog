using System;
using System.Collections.Generic;

namespace ObjRecog.UnitySim
{
    public enum SimulatorBootMode
    {
        Manual,
        Agent,
    }

    public sealed class BootModeConfig
    {
        public BootModeConfig(SimulatorBootMode mode, string host, int port)
        {
            Mode = mode;
            Host = host;
            Port = port;
        }

        public SimulatorBootMode Mode { get; }

        public string Host { get; }

        public int Port { get; }
    }

    public static class BootModeResolver
    {
        public static BootModeConfig Resolve(string[] args)
        {
            Dictionary<string, string> options = ParseArgs(args);
            string modeToken;
            bool hasMode = options.TryGetValue("--obj-recog-mode", out modeToken);
            string host;
            bool hasHost = options.TryGetValue("--obj-recog-host", out host);
            string portToken;
            bool hasPort = options.TryGetValue("--obj-recog-port", out portToken);
            int parsedPort;
            if (!int.TryParse(portToken, out parsedPort) || parsedPort <= 0)
            {
                parsedPort = 8765;
            }

            SimulatorBootMode resolvedMode = SimulatorBootMode.Manual;
            if (hasMode && string.Equals(modeToken, "agent", StringComparison.OrdinalIgnoreCase))
            {
                resolvedMode = SimulatorBootMode.Agent;
            }
            else if (hasHost || hasPort)
            {
                resolvedMode = SimulatorBootMode.Agent;
            }

            if (string.IsNullOrWhiteSpace(host))
            {
                host = "127.0.0.1";
            }

            return new BootModeConfig(resolvedMode, host, parsedPort);
        }

        private static Dictionary<string, string> ParseArgs(string[] args)
        {
            var parsed = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            for (int index = 0; index < args.Length; index++)
            {
                string token = args[index];
                if (!token.StartsWith("--", StringComparison.Ordinal))
                {
                    continue;
                }

                int equalsIndex = token.IndexOf('=');
                if (equalsIndex > 0)
                {
                    parsed[token.Substring(0, equalsIndex)] = token.Substring(equalsIndex + 1);
                    continue;
                }

                string value = "true";
                if (index + 1 < args.Length && !args[index + 1].StartsWith("--", StringComparison.Ordinal))
                {
                    value = args[index + 1];
                    index += 1;
                }

                parsed[token] = value;
            }

            return parsed;
        }
    }
}

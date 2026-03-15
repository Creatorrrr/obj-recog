using System;
using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace ObjRecog.UnitySim.Editor
{
    public static class MacOsBuild
    {
        private const string ScenePath = "Assets/Scenes/LivingRoomMain.unity";
        private const string OutputArgPrefix = "--obj-recog-build-output=";

        public static void BuildMacOsPlayer()
        {
            string outputPath = ResolveOutputPath();
            string? outputDir = Path.GetDirectoryName(outputPath);
            if (string.IsNullOrWhiteSpace(outputDir))
            {
                throw new InvalidOperationException("macOS build output path must have a parent directory");
            }

            Directory.CreateDirectory(outputDir);

            var buildOptions = new BuildPlayerOptions
            {
                scenes = new[] { ScenePath },
                locationPathName = outputPath,
                target = BuildTarget.StandaloneOSX,
                options = BuildOptions.None,
            };
            BuildReport report = BuildPipeline.BuildPlayer(buildOptions);
            if (report.summary.result != BuildResult.Succeeded)
            {
                throw new InvalidOperationException(
                    $"macOS Unity build failed with result {report.summary.result} at {outputPath}"
                );
            }

            Debug.Log($"Built macOS Unity player at {outputPath}");
        }

        private static string ResolveOutputPath()
        {
            string[] args = Environment.GetCommandLineArgs();
            for (int index = 0; index < args.Length; index++)
            {
                string arg = args[index];
                if (arg.StartsWith(OutputArgPrefix, StringComparison.Ordinal))
                {
                    return Path.GetFullPath(arg.Substring(OutputArgPrefix.Length));
                }

                if (string.Equals(arg, "--obj-recog-build-output", StringComparison.Ordinal) && index + 1 < args.Length)
                {
                    return Path.GetFullPath(args[index + 1]);
                }
            }

            return Path.GetFullPath(Path.Combine(RepoRoot(), "build", "unity", "macos", "obj-recog-unity.app"));
        }

        private static string RepoRoot()
        {
            string projectRoot = Directory.GetParent(Application.dataPath)?.FullName
                ?? throw new InvalidOperationException("Unity project root could not be resolved");
            return Directory.GetParent(projectRoot)?.FullName
                ?? throw new InvalidOperationException("Repository root could not be resolved");
        }
    }
}

param(
    [string]$Configuration = "Release",
    [string]$Generator = "",
    [string]$OrbSlam3Root = "",
    [string]$OpenCvDir = "",
    [string]$Eigen3Dir = "",
    [string]$CMakePrefixPath = "",
    [string]$VcVarsPath = "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step {
    param([string]$Message)
    Write-Host "[orbslam3-bridge] $Message"
}

function Resolve-CommandPath {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [string[]]$Candidates = @()
    )

    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    foreach ($candidate in $Candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }

    throw "Required command not found: $Name"
}

function Import-VcVarsEnvironment {
    param([Parameter(Mandatory = $true)][string]$VcVarsBatchPath)

    Write-Step "Importing Visual Studio build environment from $VcVarsBatchPath"
    $setOutput = & cmd.exe /d /s /c "`"$VcVarsBatchPath`" && set"
    if ($LASTEXITCODE -ne 0) {
        throw "vcvars64.bat failed with exit code $LASTEXITCODE"
    }

    foreach ($line in $setOutput) {
        $parts = $line -split "=", 2
        if ($parts.Length -eq 2) {
            [System.Environment]::SetEnvironmentVariable($parts[0], $parts[1], "Process")
        }
    }
}

function Convert-ToCMakePath {
    param([Parameter(Mandatory = $true)][string]$PathValue)
    return ($PathValue -replace "\\", "/")
}

function Resolve-CMakePackageDir {
    param(
        [string]$ExplicitPath,
        [string[]]$EnvironmentVariables = @(),
        [string]$SearchRoot = "",
        [Parameter(Mandatory = $true)][string]$ConfigFileName
    )

    $candidates = New-Object System.Collections.Generic.List[string]
    if ($ExplicitPath) {
        $candidates.Add($ExplicitPath)
    }
    foreach ($envName in $EnvironmentVariables) {
        $envValue = [System.Environment]::GetEnvironmentVariable($envName, "Process")
        if ($envValue) {
            $candidates.Add($envValue)
        }
    }
    if ($SearchRoot -and (Test-Path -LiteralPath $SearchRoot)) {
        $match = Get-ChildItem -Path $SearchRoot -Recurse -File -Filter $ConfigFileName -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($match) {
            $candidates.Add($match.Directory.FullName)
        }
    }

    foreach ($candidate in $candidates) {
        if (-not $candidate) {
            continue
        }
        if ((Test-Path -LiteralPath $candidate -PathType Leaf) -and (Split-Path -Leaf $candidate) -ieq $ConfigFileName) {
            return (Split-Path -Parent $candidate)
        }
        if (Test-Path -LiteralPath $candidate -PathType Container) {
            $directConfig = Join-Path $candidate $ConfigFileName
            if (Test-Path -LiteralPath $directConfig -PathType Leaf) {
                return $candidate
            }
            $nestedConfig = Get-ChildItem -Path $candidate -Recurse -File -Filter $ConfigFileName -ErrorAction SilentlyContinue |
                Select-Object -First 1
            if ($nestedConfig) {
                return $nestedConfig.Directory.FullName
            }
        }
    }

    return ""
}

function Invoke-CMake {
    param(
        [Parameter(Mandatory = $true)][string]$CMakePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    Write-Step ("cmake " + ($Arguments -join " "))
    & $CMakePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "cmake failed with exit code $LASTEXITCODE"
    }
}

function Add-StringItems {
    param(
        [Parameter(Mandatory = $true)]$List,
        [Parameter(Mandatory = $true)][string[]]$Items
    )

    foreach ($item in $Items) {
        $List.Add($item) | Out-Null
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$resolvedOrbSlam3Root = if ($OrbSlam3Root) {
    (Resolve-Path $OrbSlam3Root).Path
} else {
    (Resolve-Path (Join-Path $repoRoot "third_party\ORB_SLAM3")).Path
}
$bridgeDir = (Resolve-Path (Join-Path $repoRoot "native\orbslam3_bridge")).Path
$bridgeBuildDir = Join-Path $bridgeDir "build"
$orbSlam3BuildDir = Join-Path $resolvedOrbSlam3Root "build"

if (-not (Test-Path -LiteralPath $resolvedOrbSlam3Root -PathType Container)) {
    throw "ORB-SLAM3 checkout not found at $resolvedOrbSlam3Root"
}

$cmakePath = Resolve-CommandPath -Name "cmake" -Candidates @("C:\Program Files\CMake\bin\cmake.exe")
$cmakeDir = Split-Path -Parent $cmakePath
$env:Path = "$cmakeDir;$env:Path"

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    if (-not (Test-Path -LiteralPath $VcVarsPath)) {
        throw "Missing Visual Studio vcvars64.bat: $VcVarsPath"
    }
    Import-VcVarsEnvironment -VcVarsBatchPath $VcVarsPath
}

if (-not $Generator) {
    $ninja = Get-Command ninja -ErrorAction SilentlyContinue
    if ($ninja) {
        $Generator = "Ninja"
    } else {
        $Generator = "Visual Studio 17 2022"
    }
}

$isMultiConfigGenerator = $Generator -like "Visual Studio*" -or $Generator -like "Ninja Multi-Config"
$openCvDirResolved = Resolve-CMakePackageDir `
    -ExplicitPath $OpenCvDir `
    -EnvironmentVariables @("OpenCV_DIR") `
    -SearchRoot (Join-Path $repoRoot "build\opencv-cuda\install") `
    -ConfigFileName "OpenCVConfig.cmake"
$eigen3DirResolved = Resolve-CMakePackageDir `
    -ExplicitPath $Eigen3Dir `
    -EnvironmentVariables @("Eigen3_DIR", "EIGEN3_DIR") `
    -ConfigFileName "Eigen3Config.cmake"

$cmakePrefixEntries = New-Object System.Collections.Generic.List[string]
if ($CMakePrefixPath) {
    foreach ($entry in ($CMakePrefixPath -split ";")) {
        if ($entry) {
            $cmakePrefixEntries.Add((Convert-ToCMakePath $entry))
        }
    }
}

$commonConfigureArgs = New-Object System.Collections.Generic.List[string]
Add-StringItems -List $commonConfigureArgs -Items @("-G", $Generator)
if ($Generator -like "Visual Studio*") {
    Add-StringItems -List $commonConfigureArgs -Items @("-A", "x64")
}
if (-not $isMultiConfigGenerator) {
    $commonConfigureArgs.Add("-DCMAKE_BUILD_TYPE=$Configuration")
}
if ($openCvDirResolved) {
    $commonConfigureArgs.Add("-DOpenCV_DIR=$(Convert-ToCMakePath $openCvDirResolved)")
}
if ($eigen3DirResolved) {
    $commonConfigureArgs.Add("-DEigen3_DIR=$(Convert-ToCMakePath $eigen3DirResolved)")
}
if ($cmakePrefixEntries.Count -gt 0) {
    $commonConfigureArgs.Add("-DCMAKE_PREFIX_PATH=$($cmakePrefixEntries -join ';')")
}

$vocabularyArchive = Join-Path $resolvedOrbSlam3Root "Vocabulary\ORBvoc.txt.tar.gz"
$vocabularyText = Join-Path $resolvedOrbSlam3Root "Vocabulary\ORBvoc.txt"
if (-not (Test-Path -LiteralPath $vocabularyText -PathType Leaf)) {
    if (-not (Test-Path -LiteralPath $vocabularyArchive -PathType Leaf)) {
        throw "ORB-SLAM3 vocabulary archive not found at $vocabularyArchive"
    }
    Write-Step "Extracting ORB vocabulary"
    tar -xf $vocabularyArchive -C (Join-Path $resolvedOrbSlam3Root "Vocabulary")
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to extract ORB vocabulary archive"
    }
}

New-Item -ItemType Directory -Force -Path $orbSlam3BuildDir, $bridgeBuildDir | Out-Null

$orbConfigureArgs = New-Object System.Collections.Generic.List[string]
Add-StringItems -List $orbConfigureArgs -Items @("-S", $resolvedOrbSlam3Root, "-B", $orbSlam3BuildDir)
Add-StringItems -List $orbConfigureArgs -Items $commonConfigureArgs.ToArray()
Add-StringItems -List $orbConfigureArgs -Items @(
    "-DORB_SLAM3_HEADLESS_VIEWER=ON",
    "-DORB_SLAM3_BUILD_EXAMPLES=OFF"
)
Invoke-CMake -CMakePath $cmakePath -Arguments $orbConfigureArgs

$orbBuildArgs = @("--build", $orbSlam3BuildDir, "--target", "ORB_SLAM3", "--config", $Configuration)
Invoke-CMake -CMakePath $cmakePath -Arguments $orbBuildArgs

$bridgeConfigureArgs = New-Object System.Collections.Generic.List[string]
Add-StringItems -List $bridgeConfigureArgs -Items @("-S", $bridgeDir, "-B", $bridgeBuildDir)
Add-StringItems -List $bridgeConfigureArgs -Items $commonConfigureArgs.ToArray()
$bridgeConfigureArgs.Add("-DORB_SLAM3_ROOT=$(Convert-ToCMakePath $resolvedOrbSlam3Root)")
Invoke-CMake -CMakePath $cmakePath -Arguments $bridgeConfigureArgs

$bridgeBuildArgs = @("--build", $bridgeBuildDir, "--target", "orbslam3_bridge", "--config", $Configuration)
Invoke-CMake -CMakePath $cmakePath -Arguments $bridgeBuildArgs

$bridgeBinary = Join-Path $bridgeBuildDir "orbslam3_bridge.exe"
if (-not (Test-Path -LiteralPath $bridgeBinary -PathType Leaf)) {
    throw "Bridge build completed but $bridgeBinary was not found"
}

Write-Step "Built $bridgeBinary"

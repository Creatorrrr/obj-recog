$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step {
    param([string]$Message)
    Write-Host "[opencv-cuda] $Message"
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
    param([Parameter(Mandatory = $true)][string]$VcVarsPath)

    Write-Step "Importing Visual Studio build environment from $VcVarsPath"
    $setOutput = & cmd.exe /d /s /c "`"$VcVarsPath`" && set"
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

function Get-PythonBuildInfo {
    param([Parameter(Mandatory = $true)][string]$PythonExe)

$script = @'
import json
import site
import sys
import sysconfig

import numpy

site_packages = next(path for path in site.getsitepackages() if path.endswith("site-packages"))
data = {
    "version": "{}.{}".format(sys.version_info.major, sys.version_info.minor),
    "executable": sys.executable,
    "base_prefix": sys.base_prefix,
    "include_dir": sysconfig.get_path("include"),
    "platinclude_dir": sysconfig.get_path("platinclude"),
    "library_dir": sysconfig.get_config_var("LIBDIR") or "",
    "library": sysconfig.get_config_var("LIBRARY") or "",
    "site_packages": site_packages,
    "numpy_include": numpy.get_include(),
}
print(json.dumps(data))
'@
    $json = $script | & $PythonExe -
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to query Python build information"
    }
    return $json | ConvertFrom-Json
}

function Get-PythonLibraryPath {
    param(
        [Parameter(Mandatory = $true)]$PythonInfo,
        [Parameter(Mandatory = $true)][string]$RepoRoot
    )

    $candidates = @()
    if ($PythonInfo.library_dir -and $PythonInfo.library) {
        $candidates += (Join-Path $PythonInfo.library_dir $PythonInfo.library)
    }

    $venvRoot = Split-Path -Parent (Split-Path -Parent $PythonInfo.executable)
    if ($PythonInfo.base_prefix) {
        if ($PythonInfo.library) {
            $candidates += (Join-Path $PythonInfo.base_prefix "libs\$($PythonInfo.library)")
        }
        $candidates += (Join-Path $PythonInfo.base_prefix "libs\python312.lib")
    }
    $majorMinor = $PythonInfo.version -replace "\.", ""
    $candidates += @(
        (Join-Path $venvRoot "libs\python$majorMinor.lib"),
        (Join-Path $venvRoot "libs\python312.lib"),
        (Join-Path $RepoRoot ".venv\libs\python$majorMinor.lib"),
        (Join-Path $RepoRoot ".venv\libs\python312.lib")
    )

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return $candidate
        }
    }

    throw "Could not locate python import library"
}

function Download-File {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    if (Test-Path -LiteralPath $Destination) {
        return
    }
    Write-Step "Downloading $Url"
    Invoke-WebRequest -Uri $Url -OutFile $Destination
}

function Convert-ToCMakePath {
    param([Parameter(Mandatory = $true)][string]$PathValue)
    return ($PathValue -replace "\\", "/")
}

function Copy-OpenCvArtifactsToVenv {
    param(
        [Parameter(Mandatory = $true)][string]$BuildDir,
        [Parameter(Mandatory = $true)][string]$InstallDir,
        [Parameter(Mandatory = $true)][string]$SitePackages,
        [Parameter(Mandatory = $true)][string]$VenvScripts,
        [Parameter(Mandatory = $true)][string]$PythonVersion,
        [Parameter(Mandatory = $true)][string]$CudaBinDir
    )

    $cv2Module = Get-ChildItem -Path $BuildDir -Recurse -File -Filter "cv2*.pyd" |
        Sort-Object FullName |
        Select-Object -First 1
    if (-not $cv2Module) {
        throw "Failed to locate built cv2*.pyd"
    }

    $pythonLoaderSource = Join-Path $BuildDir "python_loader\cv2"
    if (-not (Test-Path -LiteralPath $pythonLoaderSource)) {
        throw "Failed to locate python_loader\\cv2 in build output"
    }

    $targetLoaderDir = Join-Path $SitePackages "cv2"
    if (Test-Path -LiteralPath $targetLoaderDir) {
        Remove-Item -LiteralPath $targetLoaderDir -Recurse -Force
    }
    Write-Step "Copying Python loader package -> $targetLoaderDir"
    Copy-Item -LiteralPath $pythonLoaderSource -Destination $targetLoaderDir -Recurse -Force

    $targetExtensionDir = Join-Path $targetLoaderDir "python-$PythonVersion"
    $targetBinDir = Join-Path $targetLoaderDir "bin"
    New-Item -ItemType Directory -Force -Path $targetExtensionDir, $targetBinDir | Out-Null
    Copy-Item -LiteralPath $cv2Module.FullName -Destination (Join-Path $targetExtensionDir $cv2Module.Name) -Force

    $versionConfigPath = Join-Path $targetLoaderDir "config-$PythonVersion.py"
    @"
import os

PYTHON_EXTENSIONS_PATHS = [
    os.path.join(os.path.dirname(__file__), 'python-$PythonVersion')
] + PYTHON_EXTENSIONS_PATHS
"@ | Set-Content -Path $versionConfigPath -Encoding ASCII

    $configPath = Join-Path $targetLoaderDir "config.py"
    $cudaBinCmake = Convert-ToCMakePath $CudaBinDir
    @"
import os

BINARIES_PATHS = [
    os.path.join(os.path.dirname(__file__), 'bin'),
    r'$cudaBinCmake',
] + BINARIES_PATHS
"@ | Set-Content -Path $configPath -Encoding ASCII

    $dllRoots = @($BuildDir, $InstallDir)
    $copiedDlls = New-Object System.Collections.Generic.HashSet[string]
    foreach ($root in $dllRoots) {
        Get-ChildItem -Path $root -Recurse -File -Include "opencv_*.dll","opencv_world*.dll" |
            ForEach-Object {
                if ($copiedDlls.Add($_.Name)) {
                    $targetDll = Join-Path $VenvScripts $_.Name
                    Copy-Item -LiteralPath $_.FullName -Destination $targetDll -Force
                    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $targetBinDir $_.Name) -Force
                }
            }
    }

    $rootCv2Module = Join-Path $SitePackages $cv2Module.Name
    if (Test-Path -LiteralPath $rootCv2Module) {
        Remove-Item -LiteralPath $rootCv2Module -Force
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Missing virtualenv python: $pythonExe"
}

$cmakePath = Resolve-CommandPath -Name "cmake" -Candidates @("C:\Program Files\CMake\bin\cmake.exe")
$ninjaPath = Resolve-CommandPath -Name "ninja" -Candidates @(
    "C:\Users\ckthd\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe\ninja.exe"
)
$nvccPath = Resolve-CommandPath -Name "nvcc" -Candidates @("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe")
$vcvarsPath = "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path -LiteralPath $vcvarsPath)) {
    throw "Missing Visual Studio vcvars64.bat: $vcvarsPath"
}

Import-VcVarsEnvironment -VcVarsPath $vcvarsPath

$env:Path = "{0};{1};{2}" -f (Split-Path -Parent $cmakePath), (Split-Path -Parent $ninjaPath), $env:Path
Write-Step "cmake version"
& $cmakePath --version
Write-Step "ninja version"
& $ninjaPath --version
Write-Step "nvcc version"
& $nvccPath --version
Write-Step "cl version"
& cl.exe

$pythonInfo = Get-PythonBuildInfo -PythonExe $pythonExe
$pythonLibrary = Get-PythonLibraryPath -PythonInfo $pythonInfo -RepoRoot $repoRoot
$sitePackages = $pythonInfo.site_packages
$venvScripts = Join-Path $repoRoot ".venv\Scripts"
$cudaBinDir = Split-Path -Parent $nvccPath
$pythonExecutableCmake = Convert-ToCMakePath $pythonInfo.executable
$pythonIncludeCmake = Convert-ToCMakePath $pythonInfo.include_dir
$pythonLibraryCmake = Convert-ToCMakePath $pythonLibrary
$pythonNumpyIncludeCmake = Convert-ToCMakePath $pythonInfo.numpy_include
$sitePackagesCmake = Convert-ToCMakePath $sitePackages

$opencvRoot = Join-Path $repoRoot "build\opencv-cuda"
$sourceRoot = Join-Path $opencvRoot "src"
$buildDir = Join-Path $opencvRoot "build"
$installDir = Join-Path $opencvRoot "install"
$downloadDir = Join-Path $opencvRoot "downloads"
$archivePath = Join-Path $downloadDir "opencv-4.13.0.zip"
$contribArchivePath = Join-Path $downloadDir "opencv_contrib-4.13.0.zip"
$sourceExtractDir = Join-Path $sourceRoot "opencv-4.13.0"
$contribExtractDir = Join-Path $sourceRoot "opencv_contrib-4.13.0"

New-Item -ItemType Directory -Force -Path $sourceRoot, $buildDir, $installDir, $downloadDir | Out-Null
Download-File -Url "https://github.com/opencv/opencv/archive/refs/tags/4.13.0.zip" -Destination $archivePath
Download-File -Url "https://github.com/opencv/opencv_contrib/archive/refs/tags/4.13.0.zip" -Destination $contribArchivePath

if (-not (Test-Path -LiteralPath $sourceExtractDir)) {
    Write-Step "Extracting OpenCV source"
    Expand-Archive -Path $archivePath -DestinationPath $sourceRoot -Force
}
if (-not (Test-Path -LiteralPath $contribExtractDir)) {
    Write-Step "Extracting OpenCV contrib source"
    Expand-Archive -Path $contribArchivePath -DestinationPath $sourceRoot -Force
}

$buildCache = Join-Path $buildDir "CMakeCache.txt"
$buildFiles = Join-Path $buildDir "CMakeFiles"
if (Test-Path -LiteralPath $buildCache) {
    Remove-Item -LiteralPath $buildCache -Force
}
if (Test-Path -LiteralPath $buildFiles) {
    Remove-Item -LiteralPath $buildFiles -Recurse -Force
}

$cmakeArgs = @(
    "-S", $sourceExtractDir,
    "-B", $buildDir,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_INSTALL_PREFIX=$(Convert-ToCMakePath $installDir)",
    "-DOPENCV_PYTHON3_INSTALL_PATH=$sitePackagesCmake",
    "-DOPENCV_EXTRA_MODULES_PATH=$(Convert-ToCMakePath (Join-Path $contribExtractDir 'modules'))",
    "-DOPENCV_FORCE_PYTHON_LIBS=ON",
    "-DWITH_CUDA=ON",
    "-DWITH_CUBLAS=ON",
    "-DWITH_CUDNN=OFF",
    "-DOPENCV_DNN_CUDA=OFF",
    "-DBUILD_opencv_python3=ON",
    "-DBUILD_TESTS=OFF",
    "-DBUILD_PERF_TESTS=OFF",
    "-DBUILD_EXAMPLES=OFF",
    "-DBUILD_DOCS=OFF",
    "-DBUILD_JAVA=OFF",
    "-DBUILD_LIST=core,imgproc,imgcodecs,highgui,videoio,calib3d,features2d,flann,objdetect,cudev,cudaarithm,cudaimgproc,cudawarping,python3",
    "-DCUDA_ARCH_BIN=8.9",
    "-DPYTHON_EXECUTABLE=$pythonExecutableCmake",
    "-DPYTHON_INCLUDE_DIR=$pythonIncludeCmake",
    "-DPYTHON_INCLUDE_PATH=$pythonIncludeCmake",
    "-DPYTHON_LIBRARY=$pythonLibraryCmake",
    "-DPYTHON_LIBRARIES=$pythonLibraryCmake",
    "-DPYTHON3_EXECUTABLE=$pythonExecutableCmake",
    "-DPYTHON3_INCLUDE_DIR=$pythonIncludeCmake",
    "-DPYTHON3_INCLUDE_PATH=$pythonIncludeCmake",
    "-DPYTHON3_LIBRARY=$pythonLibraryCmake",
    "-DPYTHON3_LIBRARIES=$pythonLibraryCmake",
    "-DPYTHON3_NUMPY_INCLUDE_DIRS=$pythonNumpyIncludeCmake",
    "-DPYTHON3_PACKAGES_PATH=$sitePackagesCmake"
)

Write-Step "Configuring OpenCV"
& $cmakePath @cmakeArgs
if ($LASTEXITCODE -ne 0) {
    throw "cmake configure failed"
}

Write-Step "Building OpenCV"
& $cmakePath --build $buildDir --config Release
if ($LASTEXITCODE -ne 0) {
    throw "cmake build failed"
}

Write-Step "Installing OpenCV"
& $cmakePath --install $buildDir --config Release
if ($LASTEXITCODE -ne 0) {
    throw "cmake install failed"
}

Write-Step "Removing prebuilt OpenCV wheels from .venv"
& $pythonExe -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless

Copy-OpenCvArtifactsToVenv `
    -BuildDir $buildDir `
    -InstallDir $installDir `
    -SitePackages $sitePackages `
    -VenvScripts $venvScripts `
    -PythonVersion $pythonInfo.version `
    -CudaBinDir $cudaBinDir

Write-Step "Verifying CUDA OpenCV"
& $pythonExe (Join-Path $repoRoot "scripts\verify_opencv_cuda.py")
if ($LASTEXITCODE -ne 0) {
    throw "verify_opencv_cuda.py failed"
}

Write-Step "OpenCV CUDA install complete"

@echo off
setlocal

set "ROOT_DIR=%~dp0.."
for %%I in ("%ROOT_DIR%") do set "ROOT_DIR=%%~fI"

if defined PYTHON goto :use_python_env
if defined PYTHON_BIN goto :use_python_bin_env
if exist "%ROOT_DIR%\.venv\Scripts\python.exe" goto :use_venv_python
set "PYTHON_CMD=python"
goto :python_done

:use_python_env
set "PYTHON_CMD=%PYTHON%"
goto :python_done

:use_python_bin_env
set "PYTHON_CMD=%PYTHON_BIN%"
goto :python_done

:use_venv_python
set "PYTHON_CMD=%ROOT_DIR%\.venv\Scripts\python.exe"

:python_done
if defined UNITY_EDITOR_PATH goto :use_unity_env
set "UNITY_EDITOR_BIN=C:\Program Files\Unity\Hub\Editor\6000.3.11f1\Editor\Unity.exe"
goto :unity_done

:use_unity_env
set "UNITY_EDITOR_BIN=%UNITY_EDITOR_PATH%"

:unity_done
if "%~1"=="" goto :use_default_output
set "OUTPUT_ARG=%~1"
if "%OUTPUT_ARG:~1,1%"==":" goto :use_absolute_output
set "OUTPUT_PATH=%ROOT_DIR%\%OUTPUT_ARG%"
goto :output_done

:use_absolute_output
set "OUTPUT_PATH=%OUTPUT_ARG%"
goto :output_done

:use_default_output
set "OUTPUT_PATH=%ROOT_DIR%\build\unity\obj-recog-unity.exe"

:output_done
set "LOG_PATH=%ROOT_DIR%\unity-build.log"
for %%I in ("%OUTPUT_PATH%") do set "OUTPUT_DIR=%%~dpI"

if defined PYTHONPATH goto :prepend_pythonpath
set "PYTHONPATH=%ROOT_DIR%\src"
goto :pythonpath_done

:prepend_pythonpath
set "PYTHONPATH=%ROOT_DIR%\src;%PYTHONPATH%"

:pythonpath_done
where "%PYTHON_CMD%" >nul 2>nul
if errorlevel 1 if not exist "%PYTHON_CMD%" goto :python_not_found

if not exist "%UNITY_EDITOR_BIN%" goto :unity_not_found
if /I not "%OUTPUT_PATH:~-4%"==".exe" goto :bad_output

"%PYTHON_CMD%" -m obj_recog.unity_vendor_check --unity-project-root "%ROOT_DIR%\unity"
if errorlevel 1 exit /b 1

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if errorlevel 1 exit /b 1

"%UNITY_EDITOR_BIN%" ^
  -batchmode ^
  -quit ^
  -projectPath "%ROOT_DIR%\unity" ^
  -buildWindows64Player "%OUTPUT_PATH%" ^
  -logFile "%LOG_PATH%"
if errorlevel 1 exit /b %errorlevel%

if not exist "%OUTPUT_PATH%" goto :missing_output

echo Built %OUTPUT_PATH%
echo Build log: %LOG_PATH%
exit /b 0

:python_not_found
echo python interpreter not found: %PYTHON_CMD% 1>&2
exit /b 1

:unity_not_found
echo Unity editor not found: %UNITY_EDITOR_BIN% 1>&2
echo Set UNITY_EDITOR_PATH to your Unity.exe path. 1>&2
exit /b 1

:bad_output
echo Windows Unity player output must end with .exe: %OUTPUT_PATH% 1>&2
exit /b 1

:missing_output
echo Unity build did not produce player: %OUTPUT_PATH% 1>&2
exit /b 1

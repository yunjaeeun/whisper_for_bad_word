@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

:: Python & FFmpeg 다운로드 링크 (Python 3.13.2 버전 사용)
set PYTHON_URL=https://www.python.org/ftp/python/3.13.2/python-3.13.2-amd64.exe
set FFMPEG_URL=https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip

:: Python 설치 폴더
set PYTHON_INSTALL_DIR=%ProgramFiles%\Python313
set FFMPEG_INSTALL_DIR=%ProgramFiles%\FFmpeg

:: 관리자 권한 확인
fltmc >nul 2>&1
if %errorlevel% neq 0 (
    echo 관리자 권한이 필요합니다. 관리자 모드로 실행해주세요.
    pause
    exit /b
)

:: Python 설치 확인
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo 🔹 Python 3.13.2를 다운로드하고 설치합니다...
    powershell Invoke-WebRequest -Uri %PYTHON_URL% -OutFile python_installer.exe
    start /wait python_installer.exe /quiet InstallAllUsers=1 PrependPath=1
    echo ✅ Python 3.13.2 설치 완료!
) else (
    echo ✅ Python이 이미 설치되어 있습니다.
)

:: FFmpeg 설치 확인
where ffmpeg >nul 2>nul
if %errorlevel% neq 0 (
    echo 🔹 FFmpeg를 다운로드하고 설치합니다...
    powershell Invoke-WebRequest -Uri %FFMPEG_URL% -OutFile ffmpeg.zip
    powershell Expand-Archive -Path ffmpeg.zip -DestinationPath %FFMPEG_INSTALL_DIR%
    echo ✅ FFmpeg 설치 완료!
) else (
    echo ✅ FFmpeg가 이미 설치되어 있습니다.
)

:: 환경 변수 설정
echo 🔹 환경 변수를 설정합니다...
setx PATH "%PYTHON_INSTALL_DIR%;%FFMPEG_INSTALL_DIR%\bin;%PATH%" /M
echo ✅ 환경 변수 설정 완료!

:: 설치 확인
echo 🔹 설치를 확인합니다...
python --version
ffmpeg -version

echo 🎉 Python 3.13.2 및 FFmpeg 설치가 완료되었습니다!
pause

@echo off
REM Build erg_kernel (Windows, MSVC Build Tools + conda env erg_cpp). Output: ..\erg_kernel*.pyd
if "%ERG_CONDA_ENV%"=="" (set ENV=C:\Users\Admin\anaconda3\envs\erg_cpp) else (set ENV=%ERG_CONDA_ENV%)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CONDA_PREFIX=%ENV%
set PATH=%ENV%;%ENV%\Library\bin;%ENV%\Scripts;%PATH%
cd /d %~dp0
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=%ENV%\python.exe -Dpybind11_DIR=%ENV%\Lib\site-packages\pybind11\share\cmake\pybind11 -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=%~dp0.. || exit /b 1
cmake --build build --config Release || exit /b 1

@echo off
REM Build script for CUDA examples

echo Building CUDA Learning Examples...
echo.

if "%~1"=="" goto :help
if "%~1"=="all" goto :build_all
if "%~1"=="01" goto :build_01
if "%~1"=="02" goto :build_02
if "%~1"=="03" goto :build_03
if "%~1"=="clean" goto :clean

echo Unknown target: %~1
goto :help

:build_all
call :build_01
call :build_02
call :build_03
goto :end

:build_01
echo Building 01_vector_add.cu...
nvcc -o 01_vector_add.exe 01_vector_add.cu
if %errorlevel% neq 0 (
    echo Build failed!
    exit /b 1
)
echo Built: 01_vector_add.exe
echo.
goto :end

:build_02
echo Building 02_matrix_mul.cu...
nvcc -o 02_matrix_mul.exe 02_matrix_mul.cu
if %errorlevel% neq 0 (
    echo Build failed!
    exit /b 1
)
echo Built: 02_matrix_mul.exe
echo.
goto :end

:build_03
echo Building 03_neural_network.cu...
nvcc -o 03_neural_network.exe 03_neural_network.cu
if %errorlevel% neq 0 (
    echo Build failed!
    exit /b 1
)
echo Built: 03_neural_network.exe
echo.
goto :end

:clean
echo Cleaning build files...
del /Q *.exe 2>nul
del /Q *.exp 2>nul
del /Q *.lib 2>nul
echo Cleaned.
goto :end

:help
echo Usage: build.bat [target]
echo.
echo Targets:
echo   all    - Build all examples
echo   01     - Build vector addition
echo   02     - Build matrix multiplication  
echo   03     - Build neural network
echo   clean  - Remove built files
echo.

:end

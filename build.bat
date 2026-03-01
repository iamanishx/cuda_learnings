@echo off
setlocal

echo Building CUDA Learning Examples...
echo.

set CCBIN=C:\MinGW\bin

if "%~1"=="" goto :help
if "%~1"=="all" goto :build_all
if "%~1"=="01" goto :build_01
if "%~1"=="02" goto :build_02
if "%~1"=="03" goto :build_03
if "%~1"=="clean" goto :clean

goto :help

:build_all
call :build_01
call :build_02
call :build_03
goto :end

:build_01
echo Building 01_vector_add.cu...
nvcc -ccbin %CCBIN% -o 01_vector_add.exe 01_vector_add.cu
if errorlevel 1 (echo Build failed! & goto :end)
echo Built: 01_vector_add.exe
goto :end

:build_02
echo Building 02_matrix_mul.cu...
nvcc -ccbin %CCBIN% -o 02_matrix_mul.exe 02_matrix_mul.cu
if errorlevel 1 (echo Build failed! & goto :end)
echo Built: 02_matrix_mul.exe
goto :end

:build_03
echo Building 03_neural_network.cu...
nvcc -ccbin %CCBIN% -o 03_neural_network.exe 03_neural_network.cu
if errorlevel 1 (echo Build failed! & goto :end)
echo Built: 03_neural_network.exe
goto :end

:clean
del /Q *.exe *.exp *.lib 2>nul
echo Cleaned.
goto :end

:help
echo Usage: build.bat [all, 01, 02, 03, clean]

:end

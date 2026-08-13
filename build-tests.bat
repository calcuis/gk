@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d C:\Users\calcu\Desktop\ideas\test-gguf\test\ggk\vendor\engine\gk
if not exist build-t (
  cmake -B build-t -G Ninja -DCMAKE_BUILD_TYPE=Release -DGK_CUDA=ON -DGK_BUILD_TESTS=ON -DCMAKE_MAKE_PROGRAM="C:/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe" || exit /b 1
)
cmake --build build-t --target test-cuda -j 10

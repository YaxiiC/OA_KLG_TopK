@echo off
REM Script to install torchradiomics from GitHub on Windows
REM Run this after installing other requirements

echo Installing torchradiomics from GitHub...

pip install git+https://github.com/AIM-Harvard/pytorch-radiomics.git
if errorlevel 1 (
    echo Trying main branch...
    pip install git+https://github.com/AIM-Harvard/pytorch-radiomics.git@main
)
if errorlevel 1 (
    echo Trying master branch...
    pip install git+https://github.com/AIM-Harvard/pytorch-radiomics.git@master
)

echo torchradiomics installation complete!


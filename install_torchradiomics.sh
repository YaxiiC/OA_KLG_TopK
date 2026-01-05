#!/bin/bash
# Script to install torchradiomics from GitHub
# Run this after installing other requirements

echo "Installing torchradiomics from GitHub..."

# Try main branch first
pip install git+https://github.com/AIM-Harvard/pytorch-radiomics.git || \
pip install git+https://github.com/AIM-Harvard/pytorch-radiomics.git@main || \
pip install git+https://github.com/AIM-Harvard/pytorch-radiomics.git@master

echo "torchradiomics installation complete!"


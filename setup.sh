#!/bin/bash

# Update package lists
apt-get update

# Install necessary dependencies for OpenCV and Streamlit
apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1

# Clean up
apt-get clean

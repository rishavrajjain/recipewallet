#!/usr/bin/env bash
# build.sh

# Exit on error
set -o errexit

# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install ffmpeg
# Update package lists and install ffmpeg silently
apt-get update && apt-get install -y ffmpeg
#!/usr/bin/env bash
# build.sh (Corrected version)

# Exit on error
set -o errexit

# 1. Install Python dependencies
pip install -r requirements.txt
pip install --upgrade yt-dlp

# 2. Install ffmpeg using sudo for permissions
echo "Installing ffmpeg..."
sudo apt-get update && sudo apt-get install -y ffmpeg
echo "ffmpeg installation complete."
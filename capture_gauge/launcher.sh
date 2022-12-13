#!/bin/sh
# launcher.sh
# navigate to home directory, then to this directory, then execute python script, then back home

cd /home/pi/dev/capture_gauge
sudo LIBCAMERA_LOG_LEVELS=ERROR python capture_gauge.py
#!/bin/bash

rsync -av --exclude='build' --exclude='bin' \
/home/robo/dart_vision_jia \
dji@192.168.10.22:/home/dji/dart_vision
#!/usr/bin/env bash

PARAMS="--net=host --ipc=host -u $(id -u ${USER}):$(id -g ${USER})"
VOLUMES="-v $PWD/..:/src"
NAME="ubuntu18.04-cuda10.1-cudnn7-devel-py3.6-torch1.3-fixuid"

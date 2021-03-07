#!/usr/bin/env bash

source source.sh
docker run -ti $PARAMS $VOLUMES $NAME $@

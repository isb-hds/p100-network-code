#!/bin/bash

set -x


# These are the paths on the local machine that contain the code and data
# respectively.  They will be mounted to the docker image at the base of
# the directory that jupyter notebooks are stored
# The below extracts the present working directory, assuming you are
# running this script in the directory containing p-100-code repo and
# the data directory.  If that is not they case, just replace with the
# FULL local path to those directories.
LOCAL_CODE=$(pwd -L)/p100-network-code
LOCAL_DATA=$(pwd -L)/data

# This is the directory that notebooks are initially stored in the docker image.
# If you use the jupyter/datascience-notebook, this is properly set
# If you are using a different docker image, see that images
# documentation to figure out where notebooks go.
JUPYTER_BASE=/home/jovyan/work

# This is the port on the host machine where the docker notebook will be
# run
LOCAL_PORT=447


docker run -d -p $LOCAL_PORT:8888 -e USE_HTTPS=yes \
    -e GRANT_SUDO=yes \
    -v $LOCAL_CODE:$JUPYTER_BASE/p100-network-code  \
    -v $LOCAL_DATA:$JUPYTER_BASE/data \
    jupyter/datascience-notebook

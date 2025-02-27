#!/bin/bash

# Script used to organize the CFD mesh. Creates a folder called "meshes" in the top-level 
# directory of the problem, then copies the meshes from ${CAPS_PAR}/${CAPS_NAME} to the meshes
# directory as ${TARGET_NAME}. Then, creates symbolic links in the meshes folder to put to the
# recently copied mesh files, then separate symbolic links in the FUN3D Flow directory to point
# to these mesh files. 
# Written by Brian J. Burke.

BASE_DIR=${PWD}
CAPS_PAR=${BASE_DIR}/geometry/capsFUNtoFEM/Scratch/aflr3
CAPS_NAME="funtofem_CAPS"
TARGET_NAME="funtofem_CAPS"
# FLOW_DIR=${BASE_DIR}/cfd/pullup_turb/Flow
# FLOW_DIR=${BASE_DIR}/cfd/pushdown_turb/Flow
FLOW_DIR=${BASE_DIR}/cfd/cruise_turb/Flow

echo "Base directory: ${BASE_DIR}"

if test -d $BASE_DIR/meshes; then
    echo "meshes directory already exists."
else
    echo "Making meshes directory."
    mkdir meshes
fi

MESH_DIR=${BASE_DIR}/meshes

echo "Copying grid files from CAPS directory to meshes directory..."
cp ${CAPS_PAR}/${CAPS_NAME}.lb8.ugrid ${MESH_DIR}/${CAPS_NAME}.lb8.ugrid
cp ${CAPS_PAR}/${CAPS_NAME}.mapbc ${MESH_DIR}/${CAPS_NAME}.mapbc

cd ${MESH_DIR}
if [ ! -f ${TARGET_NAME}.lb8.ugrid ]; then
    ln -s ${MESH_DIR}/$CAPS_NAME.lb8.ugrid ${TARGET_NAME}.lb8.ugrid
fi
if [ ! -f ${TARGET_NAME}.mapbc ]; then
    ln -s ${MESH_DIR}/$CAPS_NAME.mapbc ${TARGET_NAME}.mapbc
fi

cd ${FLOW_DIR}
if [ ! -f ${TARGET_NAME}.lb8.ugrid ]; then
    ln -s ${MESH_DIR}/${TARGET_NAME}.lb8.ugrid ${TARGET_NAME}.lb8.ugrid
fi
if [ ! -f ${TARGET_NAME}.mapbc ]; then
    ln -s ${MESH_DIR}/${TARGET_NAME}.mapbc ${TARGET_NAME}.mapbc
fi
#!/bin/bash

# Create the CFD directories.

BASE_DIR=${PWD}
CFD_DIR=${BASE_DIR}/cfd

scenarios=("pullup_inviscid" "cruise_inviscid" "pushdown_inviscid")

if test -d $CFD_DIR; then
    echo "CFD directory already exists: ${CFD_DIR}"
else
    echo "Making CFD directory: ${CFD_DIR}"
    mkdir cfd
fi

for scen in "${scenarios[@]}"; do
    if ! test -d $CFD_DIR/"$scen"; then
        echo "Making $scen in cfd"
        mkdir cfd/"$scen"
    fi
    if ! test -d $CFD_DIR/"$scen"/Flow; then
        mkdir cfd/"$scen"/Flow
    fi
    if ! test -d $CFD_DIR/"$scen"/Adjoint; then
        mkdir cfd/"$scen"/Adjoint
    fi
done
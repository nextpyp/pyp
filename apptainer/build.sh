#!/bin/bash

git_cryoluge="/scratch/cryoluge"
git_csp2="/scratch/csp2"
git_mojit="/scratch/mojit"
# TODO: replace these with git URLs

# find the project sources
binds=()

addbind() {
    binds+=(--bind "$1:$2")
}

foundat=
findproject() {
    case "$1" in
        "https://"*)
            foundat="$1"
            ;;
        "http://"*)
            echo "ERROR: always download code with HTTPs, not HTTP  $1"
            exit 1
            ;;
        *)
            if [ -d "$1" ]; then
                addbind "$1" "$2"
                foundat="$2"
            else
                echo "ERROR: no folder found at: $1"
                exit 1
            fi
            ;;
    esac
}

findproject "$git_cryoluge" "/src/cryoluge"
git_cryoluge="$foundat"

findproject "$git_csp2" "/src/csp2"
git_csp2="$foundat"

findproject "$git_mojit" "/src/mojit"
export git_mojit="$foundat"

export APPTAINER_TMPDIR=/scratch/scratch
export TMPDIR=/scratch/tmp

# build the app container
apptainer build \
    --force \
    "${binds[@]}" \
    --build-arg git_cryoluge="$git_cryoluge" \
    --build-arg git_csp2="$git_csp2" \
    --build-arg git_mojit="$git_mojit" \
    -B /scratch:/tmp -B ~/.ssh/:/root \
    pyp.sif \
    apptainer/pyp.def

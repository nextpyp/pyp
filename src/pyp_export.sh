#!/bin/bash
# Wrapper to run pyp in export session mode

# find pyp in the current directory of this script
# or the directory of symlink to this script, if executing through a symlink
pyp=`dirname "$0"`/pyp

# export the corresponding env variable
export_session=export_session $pyp "$@"

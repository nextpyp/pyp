#!/bin/bash
# Wrapper to run pyp in csp mode

# find pyp in the current directory of this script
# or the directory of symlink to this script, if executing through a symlink
pyp=`dirname "$0"`/pyp

# always use csp_no_stacks
clean=clean $pyp "$@"

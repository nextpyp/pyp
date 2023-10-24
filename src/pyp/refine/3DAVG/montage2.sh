#!/bin/bash


# IMOD setup
# module load IMOD/4.7.15
# export IMOD_DIR=${IMOD_DIR:=/home/${USER}/autoem2-alpha/IMOD}
#export IMOD_JAVADIR=${IMOD_JAVADIR:=/usr/local/java}
#if ! echo ${PATH} | /bin/grep -q "$IMOD_DIR/bin" ; then
#    export PATH=$IMOD_DIR/bin:$PATH
#fi
#export IMOD_PLUGIN_DIR=$IMOD_DIR/lib/imodplug
# export LD_LIBRARY_PATH=$IMOD_DIR/lib:$LD_LIBRARY_PATH
# end IMOD setup
# Imagemagick
#

export IMOD_DIR=${PYP_DIR}/external/IMOD_4.10
export PATH=$PATH:$IMOD_DIR/bin

export PATH=${PYP_DIR}/external/TOMO2/asoft/bin:$PATH

datapath=$1
dataset=$2
scratch=/scratch
#rm -fr ${scratch}/${dataset}
#clearscratch
mkdir ${scratch}/${dataset}
cd ${scratch}/${dataset}
cp ${datapath}/${maskfile} .
cp ${datapath}/${dataset}*averages.txt .
cp ${datapath}/${dataset}*.mrc .
cp ${datapath}/filter.xml .
${datapath}/montage_classes_filt.pl ${dataset} filter.xml
cp *.png ${datapath}

rm -rf ${scratch}/${dataset}

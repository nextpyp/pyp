#!/usr/bin/env bash
set -euo pipefail

# install pbzip2 from source
NAME=pbzip2-1.1.13
TARFILE=${NAME}.tar.gz
wget https://launchpad.net/pbzip2/1.1/1.1.13/+download/$TARFILE
tar xvfz $TARFILE
cd $NAME
make -j 4
make install
make clean
cd -
rm -rf $NAME
rm $TARFILE

# make in/out folders we can mount to
mkdir /var/batch
mkdir /var/data
mkdir /var/users
mkdir /var/out
mkdir /var/scratch

# load external packages
cd /opt/pyp/external
git clone https://github.com/nextpyp/spr_pick.git --depth 1
rm -rf spr_pick/.git
# git clone https://github.com/nextpyp/cet_pick.git --depth 1
# rm -rf cet_pick/.git
git clone https://github.com/nextpyp/postprocessing.git --depth 1
rm -rf postprocessing/.git
cd -

apt-get -y install software-properties-common

# install IMOD (legacy version without 16-bit support)
IMOD_FILE=imod_4.11.24_RHEL7-64_CUDA8.0.sh
wget --no-check-certificate https://bio3d.colorado.edu/imod/AMD64-RHEL5/${IMOD_FILE}
sh ${IMOD_FILE} -yes -skip -name IMOD -dir /opt -name IMOD_4.11.24
rm -rf ${IMOD_FILE}

# Newer version of IMOD with 16-bit support
IMOD_FILE=imod_5.1.0_RHEL8-64_CUDA12.0.sh
wget --no-check-certificate https://bio3d.colorado.edu/imod/AMD64-RHEL5/${IMOD_FILE}
sh ${IMOD_FILE} -yes -skip -name IMOD -dir /opt
rm -rf ${IMOD_FILE}

# install CUDA-12.8
CUDA_HOME=/usr/local/cuda-12.8
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
add-apt-repository contrib
apt-get update
apt-get -y install cuda-toolkit-12-8

apt-get -y install cudnn9-cuda-12
apt-get -y install libcu++-dev

# add symlink to cuda library needed by MotionCor3 and AreTomo
ln -s /usr/local/cuda-12.8/targets/x86_64-linux/lib/stubs/libcuda.so /usr/local/cuda-12.8/targets/x86_64-linux/lib/stubs/libcuda.so.1

# build AreTomo2
cd /opt/pyp/external/AreTomo2
rm -rf .git/
make exe -f makefile11 CUDAHOME=${CUDA_HOME} CUDALIB=/usr/local/cuda-12.8/targets/x86_64-linux/lib/stubs -j 4
find . ! -name "AreTomo2" -type f -exec rm -f {} \;
cd -

# build AreTomo3
cd /opt/pyp/external/AreTomo3
rm -rf .git/
make exe -f makefile11 CUDAHOME=${CUDA_HOME} CUDALIB=/usr/local/cuda-12.8/targets/x86_64-linux/lib/stubs -j 4
find . ! -name "AreTomo3" -type f -exec rm -f {} \;
cd -

# build MotionCor3
cd /opt/pyp/external/MotionCor3
rm -rf .git/
make exe -f makefile11 CUDAHOME=${CUDA_HOME} CUDALIB=/usr/local/cuda-12.8/targets/x86_64-linux/lib/stubs -j 4
find . ! -name "MotionCor3" -type f -exec rm -f {} \;
cd -

# build relion-5.0
current=`pwd`
cd /opt
git clone https://github.com/3dem/relion.git
cd relion
git checkout ver5.0
git pull
mkdir build
cd build
cmake -DGUI=OFF ..
make -j 4
make install
rm -rf /opt/relion
cd $current

# Download and unpack MPICH source
wget http://www.mpich.org/static/downloads/3.2.1/mpich-3.2.1.tar.gz
tar -xzf mpich-3.2.1.tar.gz

# Configure, build, and install MPICH
# Choose an installation directory, e.g., /opt/mpich-3.2.1
cd mpich-3.2.1
./configure --prefix=/opt/mpich-3.2.1 --disable-fortran --without-slurm
make
make install

# Clean up installation files to reduce image size
cd ..
rm mpich-3.2.1.tar.gz
rm -rf mpich-3.2.1
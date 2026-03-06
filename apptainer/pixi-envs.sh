#!/usr/bin/env bash
set -euo pipefail

# install milopyp
MILOPYP=/opt/pixi/milopyp
mkdir -p $MILOPYP
cd $MILOPYP
pixi init --format pyproject
pixi add python=3.11.14 cuda=12.8.1
pixi add \
	colored-traceback==0.3.0 \
	overrides==7.7.0 \
	imagesize==1.4.1
pixi add --pypi \
	h5py==3.15.1 \
	hdbscan==0.8.40 \
	torch==2.9.0 \
	torchvision==0.24.0 \
	setuptools==82.0.0
# install dependencies from requirements.txt (ignoring commented out lines)
pixi add --pypi $(awk 'NF && $1 !~ /^#/{print $1}' /opt/pyp/external/cet_pick/requirements.txt | paste -sd' ' -)

# Install topaz from source
cd /opt/pixi
git clone --depth=1 https://github.com/tbepler/topaz.git
cd topaz
git checkout v0.3.18
sed -i 's/"dependencies",\? *//g' pyproject.toml
pixi init --format pyproject
pixi install
pixi add python=3.12.12 cuda=12.8.1
pixi add --pypi \
	h5py==3.15.1 \
	tqdm==4.67.3 \
	torch==2.10.0 \
	torchvision==0.25.0 \
	torchaudio==2.10.0 \
	pandas==3.0.1
rm -rf .git/

# Install pytom-match-pick
PYTOM=/opt/pixi/pytom
mkdir -p $PYTOM
cd $PYTOM
pixi init --format pyproject
pixi add python=3.12.12 cupy=13.5.1 cuda-version=11.8 matplotlib=3.10.8 seaborn=0.13.2
pixi add --pypi pytom-match-pick==0.8.0

# Install tomoDRGN
TOMODRGN=/opt/pyp/external/tomodrgn
cd $TOMODRGN
pixi init --format pyproject
pixi add python=3.10.19
pixi add --pypi \
	torch==2.10.0 \
	pandas==2.3.3 \
	seaborn==0.13.2 \
	scikit-learn==1.7.2 \
	umap-learn==0.5.11 \
	notebook==7.5.4 \
	ipyvolume==0.6.3 \
	pythreejs==2.4.2 \
	ipython_genutils==0.2.0 \
	numpy==1.26.4
pixi install
rm -rf .git/

# Install cryoCARE
CRYOCARE=/opt/pixi/cryocare
mkdir -p $CRYOCARE
cd $CRYOCARE
pixi init
pixi add python=3.8 cudnn=8.0 tensorflow=2.6
pixi add --pypi tifffile==2019.7.26 cryoCARE==0.3.0

# Install cryoDRGN
CRYODRGN=/opt/pixi/cryodrgn
mkdir -p $CRYODRGN
cd $CRYODRGN
pixi init
pixi add python=3.9
pixi add --pypi cryodrgn==3.4.4
sed -i 's/.png/.svgz/g' /opt/pixi/cryodrgn/.pixi/envs/default/lib/python3.9/site-packages/cryodrgn/commands/analyze.py

# Install membrain-seg
MEMBRAIN=/opt/pixi/membrain
mkdir -p $MEMBRAIN
cd $MEMBRAIN
pixi init
pixi add python=3.9
pixi add --pypi membrain-seg==0.0.10

# Install tardis
TARDIS=/opt/pixi/tardis
mkdir -p $TARDIS
cd $TARDIS
pixi init
pixi add python=3.11
pixi add --pypi tardis-em==0.3.19

# Install IsoNet
ISONET=/opt/pixi/IsoNet
cd /opt/pixi
git clone --depth=1 https://github.com/IsoNet-cryoET/IsoNet.git
cd $ISONET
pixi init
pixi add python=3.9.23 cudatoolkit=11.2.2 cudnn=8.1.0.77
cat << EOF >> pixi.toml
[pypi-options]
extra-index-urls = ["https://pypi.nvidia.com/"]
EOF
pixi add --pypi tensorflow==2.10.0 nvidia-tensorrt==8.4.3.1 numpy==1.23.4
pixi add --pypi mrcfile==1.5.4 scipy==1.13.1 fire==0.7.1 tqdm==4.67.3 PyQt5==5.15.11 scikit-image==0.24.0
rm -rf .git

# Install IsoNet2
cd /opt/pixi
git clone --depth=1 https://github.com/IsoNet-cryoET/IsoNet2.git
cd IsoNet2
rm -rf .git

cp isonet2_environment.yml isonet2_environment.pixi.yml
# Remove strict pins that conflict with conda-pinned transitive deps under pixi.
sed -i '/six==/d;/requests==/d;/pytz==/d;/python-dateutil==/d;/pyparsing==/d;/packaging==/d;/markupsafe==/d;/lazy-loader==/d;/kiwisolver==/d;/idna==/d;/fonttools==/d;/contourpy==/d;/charset-normalizer==/d;/certifi==/d' isonet2_environment.pixi.yml
pixi init --import ./isonet2_environment.pixi.yml
pixi add --pypi "IsoNet2 @ file:///opt/pixi/IsoNet2"

# Install WarpToolsG
WARP=/opt/pixi/warp
mkdir -p $WARP
cd $WARP
pixi init -c warpem -c nvidia/label/cuda-11.8.0 -c pytorch -c conda-forge
pixi add python=3.11.9
pixi add warp=2.0.0

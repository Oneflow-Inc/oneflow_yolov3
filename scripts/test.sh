set -e
set -x
bash scripts/clean.sh

export ONEFLOW_DEBUG_MODE=""
mkdir build && cd build
cmake .. && make -j
cd ..
pip3 install -e . --user

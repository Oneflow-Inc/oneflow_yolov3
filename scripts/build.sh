#!/usr/bin/env bash

set -e
set -x
bash scripts/clean.sh

pushd third_party/darknet
sed -i '1s/0/1/' Makefile
make -j`nproc`
popd

mkdir -p build
pushd build
cmake ..
make -j`nproc`
popd
pip3 install -e . --user

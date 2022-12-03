#!/bin/bash

set -e

pushd dependency

echo -e "\n=== Environment Settting for Dynamic AB-Testing with MLOps ==="
echo -e "    WORKDIR: $PWD \n"

python3 -m pip install --upgrade pip
pip3 install -r requirements.txt

echo -e "\n=== Done ==="
popd
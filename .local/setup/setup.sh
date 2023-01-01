#!/bin/bash

pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip3 install git+https://github.com/huggingface/transformers
pip3 install -r .local/setup/requirements_mac.txt

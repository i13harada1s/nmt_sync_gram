#!/bin/bash

APEX_DIR=$HOME/apex

poetry run \
    pip install -v --no-cache-dir \
    --global-option="--cpp_ext" \
    --global-option="--cuda_ext" \
    --global-option="--deprecated_fused_adam" \
    --global-option="--xentropy" \
    --global-option="--fast_multihead_attn" $APEX_DIR

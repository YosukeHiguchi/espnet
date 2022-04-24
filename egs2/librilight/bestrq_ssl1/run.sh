#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_6k"
valid_set="dev"
test_sets=""

ssl_config=conf/tuning/train_bestrq_conformerL_maskprob1e-2_unmaskedonly_l2norm_cb8192_accum4_bs256_amp.yaml


./bestrq.sh \
    --lang en \
    --ngpu 4 \
    --max_wav_duration 30 \
    --ssl_config "${ssl_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --skip_eval true "$@"

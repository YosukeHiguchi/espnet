#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_960"
valid_set="dev"
test_sets="test_clean test_other dev_clean dev_other"

asr_config=conf/tuning/train_bestrq_conformerL_maskprob1e-2_unmaskedonly_l2norm_cb8192_accum4_bs256_amp.yaml
inference_config=conf/decode_asr.yaml


#    --speed_perturb_factors "0.9 1.0 1.1" \
./bestrq.sh \
    --lang en \
    --ngpu 1 \
    --max_wav_duration 30 \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --skip_eval true "$@"

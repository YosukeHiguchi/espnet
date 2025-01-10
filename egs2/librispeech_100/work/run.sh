#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev"
# test_sets="test_clean test_other dev_clean dev_other"
test_sets="test_clean_trim test_other_trim dev_clean_trim dev_other_trim"

asr_config=
inference_config=
inference_asr_model=valid.loss.ave_10best.pth

./asr.sh \
    --lang en \
    --ngpu 1 \
    --nj 16 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model "${inference_asr_model}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"

# $ python local/trim_data.py {dev_clean, dev_other...}
# $ # Manually remove "8131-117029-0017" from test_other_trim
# $ ./run.sh --stage 3 --stop_stage 3 --test_sets "dev_clean_trim dev_other_trim test_clean_trim test_other_trim" --skip_train true
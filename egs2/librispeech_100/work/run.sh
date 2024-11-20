#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Required
# llama2: --hugging_face_model_name_or_path "meta-llama/Llama-2-7b-hf"
# llama3: --hugging_face_model_name_or_path "meta-llama/Llama-3.1-8B"
hugging_face_model_name_or_path=

train_set="train_clean_100"
valid_set="dev"
test_sets="test_clean test_other dev_clean dev_other"

asr_config=conf/tuning/train_asr_conformer_llama2_vocab.yaml
# asr_config=conf/tuning/train_asr+llama2_conformer.yaml
inference_config=conf/tuning/decode_bs10_ctc0.3.yaml

# lower case
for i in `find dump/* -iname "text"`; do
    if [ ! -f ${i}_uc ]; then
        cp -a $i ${i}_uc
        sed 's/[A-Z]/\L&/g' -i $i
    fi
done

./asr.sh \
    --lang en \
    --ngpu 1 \
    --nj 16 \
    --gpu_inference true \
    --inference_nj 1 \
    --token_type hugging_face \
    --hugging_face_model_name_or_path "${hugging_face_model_name_or_path}" \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"

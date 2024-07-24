#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0


stage=1
stop_stage=100000
target_dumpdir="dump_multi_utts"
train_set="train_clean_100_sp"


log "$0 $*"
. utils/parse_options.sh

. ./path.sh


if [ ! -d "${target_dumpdir}" ]; then
    mkdir -p "${target_dumpdir}/raw"
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    for part in dev ${train_set}; do
        target_dir="${target_dumpdir}/raw/${part}"
        echo "Appending utterances: dump/raw/${part} -> ${target_dir}"

        cp -a "dump/raw/${part}" "${target_dir}"
        python local/append_utts.py --file_path "${target_dir}/wav.scp"

        mv "${target_dir}/wav.scp" "${target_dir}/wav.scp.org"
        mv "${target_dir}/wav_multi.scp" "${target_dir}/wav.scp"
        echo "custom" > "${target_dir}/audio_format"
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    for part in dev_clean test_clean dev_other test_other; do
        target_dir="${target_dumpdir}/raw/${part}_prev"
        echo "Appending previous utterance: dump/raw/${part} -> ${target_dir}"

        cp -a "dump/raw/${part}" "${target_dir}"
        python local/append_prev_utt.py --file_path "${target_dir}/wav.scp"

        mv "${target_dir}/wav.scp" "${target_dir}/wav.scp.org"
        mv "${target_dir}/wav_multi.scp" "${target_dir}/wav.scp"
        echo "custom" > "${target_dir}/audio_format"
    done
fi

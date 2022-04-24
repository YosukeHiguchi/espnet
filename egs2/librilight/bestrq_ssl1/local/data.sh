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
stop_stage=3
librispeech_data_url="www.openslr.org/resources/12"
librilight_data_url="https://dl.fbaipublicfiles.com/librilight/data"
librilight_parts="small medium"  # "large" is missing because
train_set="train_6k"  # "train_60k" if large is included
train_dev="dev"

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${LIBRISPEECH}" ] && [ -z "${LIBRILIGHT}" ]; then
    log "Fill the value of 'LIBRISPEECH' and 'LIBRILIGHT' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -e "${LIBRISPEECH}/LibriSpeech/LICENSE.TXT" ]; then
        echo "stage 1a: Data Download to ${LIBRISPEECH} and ${LIBRILIGHT}"
        for part in dev-clean dev-other; do
            local/download_and_untar_librispeech.sh ${LIBRISPEECH} ${librispeech_data_url} ${part}
        done
    else
        log "stage 1a: ${LIBRISPEECH}/LibriSpeech/LICENSE.TXT is already existing. Skip data downloading"
    fi
    if [ ! -d "${LIBRILIGHT}/.complete" ]; then
        echo "Stage 1b: Data Download librilight data to ${LIBRILIGHT}"
        for part in small medium; do
            local/download_and_untar_librilight.sh ${LIBRILIGHT} ${librilight_data_url} ${part}
        done
    else
        log "stage 1b: ${LIBRILIGHT}/.complete is already existing. Skip data downloading"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"
    for part in dev-clean dev-other; do
        # use underscore-separated names in data directories.
        local/data_prep_librispeech.sh ${LIBRISPEECH}/LibriSpeech/${part} data/librispeech_${part//-/_}
    done

    for part in small medium; do
        local/data_prep_librilight.sh ${LIBRILIGHT}/${part}_segmented data/librilight_${part}
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: combine all training and development sets"
    utils/combine_data.sh --extra_files utt2num_frames data/${train_set} data/librilight_small data/librilight_medium
    utils/combine_data.sh --extra_files utt2num_frames data/${train_dev} data/librispeech_dev_clean data/librispeech_dev_other
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

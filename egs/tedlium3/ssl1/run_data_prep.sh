#!/usr/bin/env bash

# after running data preparation in ../../librispeech/mpl
# for MPL
# $ ./run_data_prep.sh --librispeech_dir ../../librispeech/mpl
# for InterMPL
# $ ./run_data_prep.sh --librispeech_dir ../../librispeech/mpl
# $ ./run_data_prep.sh --stage 2 --nbpe 256 --librispeech_dir ../../librispeech/mpl
# $ ./run_data_prep.sh --stage 2 --nbpe 4096 --librispeech_dir ../../librispeech/mpl
# $ . path.sh
# $ python local/concat_outputs.py train_clean_100 unigram1024,unigram1024,unigram1024
# $ python local/concat_outputs.py train_clean_100 unigram256,unigram1024,unigram4096

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
stop_stage=100
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

# bpemode (unigram or bpe)
nbpe=1024
bpemode=unigram

librispeech_dir=

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


# legacy setup
data_type=legacy
train_set=train_trim
train_dev=dev_trim
recog_set="dev test"


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/prepare_data.sh $data_type
    for dset in dev test train; do
        utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/${dset}.orig data/${dset}
    done

    ### pre-process data
    for x in train dev test; do
        data_dir=data/${x}
        echo "processing ${data_dir}"
        mv ${data_dir}/text ${data_dir}/text_org
        paste -d " " <(cat ${data_dir}/text_org | cut -d ' ' -f 1) <(cat ${data_dir}/text_org | cut -d ' ' -f 2- | sed 's/<unk> //g' | sed 's/ <unk>//g' | sed 's/[a-z]/\U&/g') > ${data_dir}/text
    done
fi

feat_tr_dir=${dumpdir}/ted3_${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/ted3_${train_dev}_cmvn_ted3_${train_set}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in test dev train; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 16 --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done

    # remove utt having > 2000 frames or < 10 frames or
    # remove utt having > 400 characters or 0 characters
    remove_longshortdata.sh --maxchars 400 data/train data/train_trim
    remove_longshortdata.sh --maxchars 400 data/dev data/dev_trim

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj 16 --do_delta ${do_delta} \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 16 --do_delta ${do_delta} \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/ted3_${rtask}_cmvn_ted3_${train_set}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 16 --do_delta ${do_delta} \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    [ -z ${librispeech_dir} ] && { echo "empty variable librispeech_dir"; exit 2; };

    dict=${librispeech_dir}/data/lang_char/train_clean_100_${bpemode}${nbpe}_units.txt
    bpemodel=${librispeech_dir}/data/lang_char/train_clean_100_${bpemode}${nbpe}
    [ ! -e ${dict} ] && { echo "couldn't find dictionary ${dict}"; exit 2; };

    data2json.sh --feat ${feat_tr_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}_train_clean_100.json
    for y in dev dev_trim test; do
        feat_recog_dir=${dumpdir}/ted3_${y}_cmvn_ted3_${train_set}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel}.model \
            data/${y} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}_train_clean_100.json
    done
fi

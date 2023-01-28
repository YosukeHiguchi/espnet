#!/usr/bin/env bash

# for MPL
# $ ./run_data_prep.sh
# for InterMPL
# $ ./run_data_prep.sh
# $ ./run_data_prep.sh --stage 2 --nbpe 256
# $ ./run_data_prep.sh --stage 2 --nbpe 4096
# $ . path.sh
# $ python local/concat_outputs.py train_clean_100 unigram1024,unigram1024,unigram1024
# $ python local/concat_outputs.py train_clean_100 unigram256,unigram1024,unigram4096
# (optional) for domain-mismatch experiment using TED3
# go to ../../tedlium3/mpl and follow run_data_prep.sh
# create links to ted3 data under `dump`
# i.e., $ cd dump; ln -s ../../../tedlium3/mpl/dump/* .

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=0        # start from -1 if you need to start from data download
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
nj=10
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

datadir=/groups/gca50130/migrated_from_SFA_GPFS/higuchi/db

# bpemode (unigram or bpe)
nbpe=1024
bpemode=unigram

# exp tag
tag="" # tag for managing experiments.

train_set=train_clean_100

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        local/data_prep.sh ${datadir}/LibriSpeech/${part} data/${part//-/_}
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in dev_clean test_clean dev_other test_other train_clean_100 train_clean_360 train_other_500; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done

    utils/combine_data.sh --extra_files utt2num_frames data/train_860_org data/train_clean_360 data/train_other_500
    mv data/train_clean_100 data/train_clean_100_org
    mv data/train_clean_360 data/train_clean_360_org
    mv data/train_other_500 data/train_other_500_org
    utils/combine_data.sh --extra_files utt2num_frames data/dev_org data/dev_clean data/dev_other

    # remove utt having more than 3000 frames
    for x in train_clean_100 train_clean_360 train_other_500 train_860; do
        remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${x}_org data/${x}
        # compute global CMVN
        compute-cmvn-stats scp:data/${x}/feats.scp data/${x}/cmvn.ark
    done
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/dev_org data/dev

    # dump features for training
    for x in train_clean_100 train_clean_360 train_other_500 train_860; do
        feat_dir=${dumpdir}/${x}/delta${do_delta}; mkdir -p ${feat_dir}

        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
            data/${x}/feats.scp data/${x}/cmvn.ark exp/dump_feats/train/${x} ${feat_dir}

        for rtask in test_clean test_other dev_clean dev_other dev; do
            feat_dir=${dumpdir}/${rtask}_cmvn_${x}/delta${do_delta}; mkdir -p ${feat_dir}
            dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
                data/${rtask}/feats.scp data/${x}/cmvn.ark exp/dump_feats/recog/${rtask}_cmvn_${x} ${feat_dir}
        done
    done
fi

dict=data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_char/${train_set}_${bpemode}${nbpe}
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Dictionary Preparation"
    mkdir -p data/lang_char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    cut -f 2- -d" " data/${train_set}/text > data/lang_char/input.txt
    spm_train --input=data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
    spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Json Data Preparation"
    # make json labels
    for x in train_clean_100 train_clean_360 train_other_500 train_860; do
        feat_dir=${dumpdir}/${x}/delta${do_delta}
        data2json.sh --feat ${feat_dir}/feats.scp --bpecode ${bpemodel}.model \
            data/${x} ${dict} > ${feat_dir}/data_${bpemode}${nbpe}_${train_set}.json

        for y in dev_clean dev_other dev test_clean test_other; do
            feat_dir=${dumpdir}/${y}_cmvn_${x}/delta${do_delta}
            data2json.sh --feat ${feat_dir}/feats.scp --bpecode ${bpemodel}.model \
                data/${y} ${dict} > ${feat_dir}/data_${bpemode}${nbpe}_${train_set}.json
        done
    done
fi

#!/usr/bin/env bash

# ./run_ctc.sh --train_config conf/tuning/train_cfm_ctc.yaml --decode_config conf/tuning/decode_ctc.yaml

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=4        # start from -1 if you need to start from data download
stop_stage=5
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
nj=10
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

preprocess_config=conf/specaug.yaml
train_config=
decode_config=

# decoding parameter
lang_model=

# model average realted (only for transformer)
n_average=10
use_valbest_average=false
use_cerbest_average=true
max_epoch=150

# bpemode (unigram or bpe)
nbpe=1024
bpemode=unigram

# exp tag
tag="" # tag for managing experiments.

train_set=train_clean_100
train_dev=dev_cmvn_train_clean_100
recog_set="\
    dev_clean_cmvn_train_clean_100 \
    dev_other_cmvn_train_clean_100 \
    test_clean_cmvn_train_clean_100 \
    test_other_cmvn_train_clean_100 \
"
#     ted3_dev_cmvn_ted3_train_trim \
#     ted3_test_cmvn_ted3_train_trim \
# "

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}
echo "train: "${feat_tr_dir}
echo "valid: "${feat_dt_dir}


dict=data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_char/${train_set}_${bpemode}${nbpe}
echo "dict: "${dict}


expname=${train_set}_${bpemode}${nbpe}_$(basename ${train_config%.*})
expdir=exp_base/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"
    lab_json=${feat_tr_dir}/data_${bpemode}${nbpe}_${train_set}.json
    dev_json=${feat_dt_dir}/data_${bpemode}${nbpe}_${train_set}.json
    echo "  lab: "${lab_json}
    echo "  dev: "${dev_json}

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${lab_json} \
        --valid-json ${dev_json}

fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    # Average ASR models
    if ${use_valbest_average}; then
        recog_model=model.val${n_average}.avg.best
        opt="--log ${expdir}/results/log"
    elif ${use_cerbest_average}; then
        recog_model=model.cer${n_average}.avg.best
        opt="--log ${expdir}/results/log --metric cer_ctc"
    else
        recog_model=model.last${n_average}.avg.best
        opt="--log"
    fi

    recog_model=${recog_model}.ep${max_epoch}
    echo ${recog_model}
    if [ ! -f ${expdir}/results/${recog_model} ]; then
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average} \
            --max-epoch ${max_epoch}
    fi

    lm_opt=""
    if [ ! -z ${lang_model} ]; then
        # lang_model=rnnlm.model.best
        lm_opt="--rnnlm ${lang_model}"
    fi

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}_${train_set}.json

        #### use CPU for decoding
        ngpu=0

        # set batchsize 0 to disable batch decoding
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            ${lm_opt} \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}_${train_set}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --api v2

        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

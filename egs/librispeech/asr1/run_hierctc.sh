#!/usr/bin/env bash

# bash run.sh --stage 4 --stop_stage 5 --train_config conf/tuning/train_pytorch_hierctc.yaml --decode_config conf/work/decode_hierctc.yaml --bpemode unigram --nbpe 16384 --units unigram256,unigram2048,unigram16384 --use_cerbest_average true --n_average 10 --ctc_index 2
# bash run.sh --stage 4 --stop_stage 5 --train_config conf/tuning/train_pytorch_hierctc.yaml --decode_config conf/work/decode_hierctc.yaml --bpemode unigram --nbpe 16384 --units unigram16384,unigram16384,unigram16384 --use_cerbest_average true --n_average 10 --ctc_index 2
# bash run.sh --stage 4 --stop_stage 5 --train_config conf/tuning/train_pytorch_paractc.yaml --decode_config conf/work/decode_paractc.yaml --bpemode unigram --nbpe 16384 --units unigram256,unigram2048,unigram16384 --use_cerbest_average true --n_average 10 --ctc_index 2

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
nj=32
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

preprocess_config=conf/specaug.yaml
train_config=
lm_config=
decode_config=

# rnnlm related
lm_resume= # specify a snapshot file to resume LM training
lmtag=     # tag for managing LMs

# decoding parameter
recog_model=model.acc.best  # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
lang_model=rnnlm.model.best # set a language model to be used for decoding

# model average realted (only for transformer)
n_average=5                  # the number of ASR models to be averaged
use_valbest_average=false    # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
use_cerbest_average=false
lm_n_average=0               # the number of languge models to be averaged
use_lm_valbest_average=false # if true, the validation `lm_n_average`-best language models will be averaged.
                             # if false, the last `lm_n_average` language models will be averaged.
ctc_index=2

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
datadir=

# base url for downloads.
data_url=www.openslr.org/resources/12

# bpemode (unigram or bpe)
nbpe=
bpemode= # char or unigram
units=

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_clean_100_sp
train_dev=dev
recog_set="test_clean test_other dev_clean dev_other"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    for part in dev-clean test-clean dev-other test-other train-clean-100; do
        # use underscore-separated names in data directories.
        local/data_prep.sh ${datadir}/LibriSpeech/${part} data/${part//-/_}
    done
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in train_clean_100; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
        mv data/${x} data/${x}_org
    done
    for x in dev_clean test_clean dev_other test_other; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done

    utils/perturb_data_dir_speed.sh 0.9  data/train_clean_100_org  data/temp1
    utils/perturb_data_dir_speed.sh 1.0  data/train_clean_100_org  data/temp2
    utils/perturb_data_dir_speed.sh 1.1  data/train_clean_100_org  data/temp3
    utils/combine_data.sh --extra-files utt2uniq data/${train_set}_org data/temp1 data/temp2 data/temp3
    utils/combine_data.sh --extra_files utt2num_frames data/${train_dev}_org data/dev_clean data/dev_other

    # remove utt having more than 3000 frames
    # remove utt having more than 400 characters
    for x in ${train_set} dev; do
        remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${x}_org data/${x}
    done

    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj}  --write_utt2num_frames true \
        data/${train_set} exp/make_fbank/${train_set} ${fbankdir}
    utils/fix_data_dir.sh data/${train_set}

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_char/${train_set}_${bpemode}${nbpe}
nlsyms=data/lang_char/non_lang_syms.txt

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "dictionary: ${dict}"
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_char/

    if [ ${bpemode} = "char" ]; then
        echo "make a non-linguistic symbol list"
        # cut -f 2- data/${train_set}/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
        cat ${nlsyms}

        echo "make a dictionary"
        echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
        text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
        wc -l ${dict}

        # make json labels
        data2json.sh --feat ${feat_tr_dir}/feats.scp --nlsyms ${nlsyms} \
            data/${train_set} ${dict} > ${feat_tr_dir}/data_char.json
        data2json.sh --feat ${feat_dt_dir}/feats.scp --nlsyms ${nlsyms} \
            data/${train_dev} ${dict} > ${feat_dt_dir}/data_char.json
        for rtask in ${recog_set}; do
            feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
            data2json.sh --feat ${feat_recog_dir}/feats.scp \
                --nlsyms ${nlsyms} data/${rtask} ${dict} > ${feat_recog_dir}/data_char.json
        done
    else
        echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
        cut -f 2- -d" " data/${train_set}/text > data/lang_char/input.txt
        spm_train --input=data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
        spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
        wc -l ${dict}

        # make json labels
        data2json.sh --nj ${nj} --feat ${feat_tr_dir}/feats.scp --bpecode ${bpemodel}.model \
            data/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.json
        data2json.sh --nj ${nj} --feat ${feat_dt_dir}/feats.scp --bpecode ${bpemodel}.model \
            data/${train_dev} ${dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.json

        for rtask in ${recog_set}; do
            feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
            data2json.sh --nj ${nj} --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel}.model \
                data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json
        done
    fi
fi

expname=${train_set}_${backend}_$(basename ${train_config%.*})_${units}
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"
    if [ -z ${units} ]; then
        echo "--units is not specified"
        exit 1;
    fi

    unit_list=(${units//,/ })
    dicts=""
    for u in ${unit_list[@]}; do
        dicts+="data/lang_char/${train_set}_${u}_units.txt,"
    done
    echo ${units}
    echo ${dicts}

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dicts} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${units}.json \
        --valid-json ${feat_dt_dir}/data_${units}.json
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    nj=4
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
           [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]] || \
           [[ $(get_yaml.py ${train_config} model-module) = *hierctc* ]] || \
           [[ $(get_yaml.py ${train_config} model-module) = *paractc* ]] || \
           [[ $(get_yaml.py ${train_config} etype) = custom ]] || \
           [[ $(get_yaml.py ${train_config} dtype) = custom ]]; then
        # Average ASR models
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        elif ${use_cerbest_average}; then
            if [[ $(get_yaml.py ${train_config} model-module) = *hierctc* ]] || \
                [[ $(get_yaml.py ${train_config} model-module) = *paractc* ]]; then
                recog_model=model.cer${n_average}ci${ctc_index}.avg.best
                opt="--log ${expdir}/results/log --metric cer_ctc"
            else
                recog_model=model.cer${n_average}.avg.best
                opt="--log ${expdir}/results/log --metric cer_ctc"
            fi
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        echo ${recog_model}
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average} \
            --ctc-index ${ctc_index}
    fi

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_nolm
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        # set batchsize 0 to disable batch decoding
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}

        if [ ${bpemode} = "char" ]; then
            score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}
        else
            score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
        fi

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

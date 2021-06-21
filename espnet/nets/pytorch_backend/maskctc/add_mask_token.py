#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Waseda University (Yosuke Higuchi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Token masking module for Masked LM."""

import numpy
import torch


def mask_uniform(ys_pad, mask_token, eos, ignore_id):
    """Replace random tokens with <mask> label and add <eos> label.

    The number of <mask> is chosen from a uniform distribution
    between one and the target sequence's length.
    :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
    :param int mask_token: index of <mask>
    :param int eos: index of <eos>
    :param int ignore_id: index of padding
    :return: padded tensor (B, Lmax)
    :rtype: torch.Tensor
    :return: padded tensor (B, Lmax)
    :rtype: torch.Tensor
    """
    from espnet.nets.pytorch_backend.nets_utils import pad_list

    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    ys_out = [y.new(y.size()).fill_(ignore_id) for y in ys]
    ys_in = [y.clone() for y in ys]
    for i in range(len(ys)):
        num_samples = numpy.random.randint(1, len(ys[i]) + 1)
        idx = numpy.random.choice(len(ys[i]), num_samples, replace=False)

        ys_in[i][idx] = mask_token
        ys_out[i][idx] = ys[i][idx]

    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)

def mask_uniform_dlp(ys_pad, mask_token, eos, ignore_id):
    from espnet.nets.pytorch_backend.nets_utils import pad_list

    ys = [y[y != ignore_id] for y in ys_pad]
    yslen = len(ys)

    ys_in_del = [y.clone() for y in ys]
    dur_out_del = []
    for i in range(yslen):
        ylen = len(ys[i])
        num_samples = numpy.random.randint(1, ylen + 1)
        idx = numpy.random.choice(ylen, num_samples)

        ys_in_del[i][idx] = mask_token

        unmask_idx = torch.arange(ylen).to(ys[i]).masked_fill((ys_in_del[i] == mask_token), ignore_id)
        _, dur = unmask_idx.unique_consecutive(return_counts=True)

        ys_in_del[i] = ys_in_del[i][dur.cumsum(0) - 1]
        dur_out_del.append(dur.masked_fill((ys_in_del[i] != mask_token), ignore_id))

    ys_in_ins = [torch.stack([y.new(y.size()).fill_(ignore_id), y.clone()]).t().flatten() for y in ys]
    dur_out_ins = [torch.stack([y.new(y.size()).fill_(0), y.new(y.size()).fill_(1)]).t().flatten() for y in ys]
    for i in range(len(ys)):
        ylen = len(ys[i])
        num_samples = numpy.random.randint(1, ylen + 1)
        idx = numpy.random.choice(ylen, num_samples) * 2

        ys_in_ins[i][idx] = mask_token
        tgt_idx = torch.where(ys_in_ins[i] != ignore_id)[0]

        ys_in_ins[i] = ys_in_ins[i][tgt_idx]
        dur_out_ins[i] = dur_out_ins[i][tgt_idx]

    ys_in_del.extend(ys_in_ins)
    dur_out_del.extend(dur_out_ins)

    return pad_list(ys_in_del, eos), pad_list(dur_out_del, ignore_id)
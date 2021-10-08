# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

from argparse import Namespace
import logging
import math

import chainer
from chainer import reporter
import numpy
import torch

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.e2e_asr import Reporter
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.rnn.decoders import CTC_SCORING_RATIO
from espnet.nets.pytorch_backend.transformer.argument import (
    add_arguments_transformer_common,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.utils.cli_utils import strtobool
from espnet.utils.fill_missing_args import fill_missing_args


class Reporter(chainer.Chain):
    """A chainer reporter wrapper."""

    def report(self, loss_ctc, cer_ctcs, loss):

        for i, lc in enumerate(loss_ctc):
            reporter.report({"loss_ctc{}".format(i): lc}, self)

        for i, cer in enumerate(cer_ctcs):
            reporter.report({"cer_ctc{}".format(i): cer}, self)

        reporter.report({"loss": loss}, self)

class E2E(ASRInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        group = parser.add_argument_group("transformer model setting")
        group = add_arguments_transformer_common(group)
        E2E.add_paractc_arguments(parser)

        return parser

    @staticmethod
    def add_paractc_arguments(parser):
        group = parser.add_argument_group("paractc specific setting")
        group.add_argument(
            "--n-adaptive-layers",
            default=0,
            type=int,
        )

        return parser

    @property
    def attention_plot_class(self):
        """Return PlotAttentionReport."""
        return PlotAttentionReport

    def get_total_subsampling_factor(self):
        """Get total subsampling factor."""
        return self.encoder.conv_subsampling_factor * int(numpy.prod(self.subsample))

    def __init__(self, idim, odim, args, ignore_id=-1):
        torch.nn.Module.__init__(self)
        logging.warning('paractc')

        # fill missing arguments for compatibility
        args = fill_missing_args(args, self.add_arguments)

        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate

        self.encoder = Encoder(
            idim=idim,
            selfattention_layer_type=args.transformer_encoder_selfattn_layer_type,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            conv_wshare=args.wshare,
            conv_kernel_length=args.ldconv_encoder_kernel_length,
            conv_usebias=args.ldconv_usebias,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
        )

        # no decoder
        self.decoder = None
        self.criterion = None

        self.blank = 0
        self.odim = odim ## list
        self.ignore_id = ignore_id
        self.subsample = get_subsample(args, mode="asr", arch="transformer")
        self.reporter = Reporter()
        self.reset_parameters(args)
        self.adim = args.adim  # used for CTC (equal to d_model)

        # adaptive layer
        self.n_adaptive_layers = args.n_adaptive_layers
        if self.n_adaptive_layers > 0:
            self.adaptive_layers = torch.nn.ModuleList()
            for i in range(len(odim)):
                lns = [
                    torch.nn.Linear(self.adim, self.adim) for i in range(self.n_adaptive_layers)
                ]
                self.adaptive_layers.append(torch.nn.Sequential(*lns))

        # CTC
        self.ctcs = torch.nn.ModuleList()
        for i in range(len(odim)):
            self.ctcs.append(
                CTC(
                    odim[i], args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True
                )
            )

        if args.report_cer or args.report_wer:
            self.error_calculators = []
            for cl in args.char_list:
                self.error_calculators.append(
                    ErrorCalculator(
                        cl,
                        args.sym_space,
                        args.sym_blank,
                        args.report_cer,
                        args.report_wer,
                    )
                )
        else:
            self.error_calculators = None
        self.rnnlm = None

    def reset_parameters(self, args):
        """Initialize parameters."""
        # initialize parameters
        initialize(self, args.transformer_init)

    def forward(self, xs_pad, ilens, yss_pad):
        # 1. forward encoder
        xs_pad = xs_pad[:, : max(ilens)]  # for data parallel
        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)

        loss_ctcs = []
        cer_ctcs = []
        batch_size = xs_pad.size(0)
        hs_len = hs_mask.view(batch_size, -1).sum(1)
        itr = len(self.odim)
        for i in range(itr):
            if self.n_adaptive_layers > 0:
                hs_pad_adapted = self.adaptive_layers[i](hs_pad)
            else:
                hs_pad_adapted = hs_pad
            loss_ctcs.append(
                self.ctcs[i](hs_pad_adapted.view(batch_size, -1, self.adim), hs_len, yss_pad[i])
            )

            if not self.training and self.error_calculators is not None:
                ys_hat = self.ctcs[i].argmax(hs_pad_adapted.view(batch_size, -1, self.adim)).data
                cer_ctcs.append(
                    self.error_calculators[i](ys_hat.cpu(), yss_pad[i].cpu(), is_ctc=True)
                )

            # for visualization
            if not self.training:
                self.ctcs[i].softmax(hs_pad_adapted)

        self.loss = 0
        loss_ctc_data = []
        for i, lc in enumerate(loss_ctcs):
            self.loss = self.loss + lc / itr
            loss_ctc_data.append(float(lc))
        loss_data = float(self.loss)

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(
                loss_ctc_data, cer_ctcs, loss_data
            )
        else:
            logging.warning("loss (=%f) is not correct", loss_data)
        return self.loss

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def encode(self, x):
        """Encode acoustic features.

        :param ndarray x: source acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        x = torch.as_tensor(x).unsqueeze(0)
        enc_output, _ = self.encoder(x, None)
        return enc_output.squeeze(0)

    def recognize(self, x, recog_args, char_list=None, rnnlm=None, use_jit=False):
        enc_output = self.encode(x).unsqueeze(0)

        from itertools import groupby

        if self.n_adaptive_layers > 0:
            enc_output = self.adaptive_layers[recog_args.ctc_index](enc_output)
        lpz = self.ctcs[recog_args.ctc_index].argmax(enc_output)
        collapsed_indices = [x[0] for x in groupby(lpz[0])]
        hyp = [x for x in filter(lambda x: x != self.blank, collapsed_indices)]
        nbest_hyps = [{"score": 0.0, "yseq": [self.odim[-1] - 1] + hyp}]
        if recog_args.beam_size > 1:
            raise NotImplementedError("Pure CTC beam search is not implemented.")

        return nbest_hyps

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights (B, H, Lmax, Tmax)
        :rtype: float ndarray
        """
        self.eval()
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad)
        ret = dict()
        for name, m in self.named_modules():
            if (
                isinstance(m, MultiHeadedAttention)
                or isinstance(m, RelPositionMultiHeadedAttention)
            ):
                ret[name] = m.attn.cpu().numpy()
        self.train()
        return ret

    def calculate_all_ctc_probs(self, xs_pad, ilens, ys_pad):
        """E2E CTC probability calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: CTC probability (B, Tmax, vocab)
        :rtype: float ndarray
        """
        ret = []
        self.eval()
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad)
        for name, m in self.named_modules():
            if isinstance(m, CTC) and m.probs is not None:
                ret.append(m.probs.cpu().numpy())
        self.train()
        return ret

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
from espnet.nets.pytorch_backend.conformer.encoder import Encoder as CfmEncoder
from espnet.nets.pytorch_backend.conformer.argument import (
    add_arguments_conformer_common,  # noqa: H301
)
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
        group = add_arguments_conformer_common(group)
        E2E.add_hierctc_arguments(parser)

        return parser

    @staticmethod
    def add_hierctc_arguments(parser):
        group = parser.add_argument_group("hierctc specific setting")
        group.add_argument(
            "--enc-each-nlayer",
            default="4,4,4",
            type=str,
        )
        group.add_argument(
            "--use-conformer",
            default=False,
            type=strtobool,
        )
        group.add_argument(
            "--feedback-ctc-result",
            default=True,
            type=strtobool,
        )

        return parser

    @property
    def attention_plot_class(self):
        """Return PlotAttentionReport."""
        return PlotAttentionReport

    def get_total_subsampling_factor(self):
        """Get total subsampling factor."""
        subsampling_factor = 1
        itr = len(self.odim)
        for i in range(itr):
            subsampling_factor * self.encoders[i].conv_subsampling_factor
        return subsampling_factor * int(numpy.prod(self.subsample))

    def __init__(self, idim, odim, args, ignore_id=-1):
        torch.nn.Module.__init__(self)

        logging.warning('hierctc')

        # fill missing arguments for compatibility
        args = fill_missing_args(args, self.add_arguments)

        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate

        self.encoders = torch.nn.ModuleList()
        self.linears = torch.nn.ModuleList()
        enc_each_nlayer = list(map(int, args.enc_each_nlayer.split(",")))
        assert len(odim) == len(enc_each_nlayer)
        for i in range(len(odim)):
            input_dim = idim if i == 0 else args.adim
            input_layer = args.transformer_input_layer if i == 0 else "identity"

            self.use_conformer = args.use_conformer
            if args.use_conformer:
                self.encoders.append(
                    CfmEncoder(
                        idim=input_dim, #
                        attention_dim=args.adim,
                        attention_heads=args.aheads,
                        linear_units=args.eunits,
                        num_blocks=enc_each_nlayer[i], #
                        input_layer=input_layer, #
                        dropout_rate=args.dropout_rate,
                        positional_dropout_rate=args.dropout_rate,
                        attention_dropout_rate=args.transformer_attn_dropout_rate,
                        pos_enc_layer_type=args.transformer_encoder_pos_enc_layer_type,
                        selfattention_layer_type=args.transformer_encoder_selfattn_layer_type,
                        activation_type=args.transformer_encoder_activation_type,
                        macaron_style=args.macaron_style,
                        use_cnn_module=args.use_cnn_module,
                        cnn_module_kernel=args.cnn_module_kernel,
                        return_posemb=True, #
                    )
                )
            else:
                self.encoders.append(
                    Encoder(
                        idim=input_dim, #
                        selfattention_layer_type=args.transformer_encoder_selfattn_layer_type,
                        attention_dim=args.adim,
                        attention_heads=args.aheads,
                        conv_wshare=args.wshare,
                        conv_kernel_length=args.ldconv_encoder_kernel_length,
                        conv_usebias=args.ldconv_usebias,
                        linear_units=args.eunits,
                        num_blocks=enc_each_nlayer[i], #
                        input_layer=input_layer, #
                        dropout_rate=args.dropout_rate,
                        positional_dropout_rate=args.dropout_rate,
                        attention_dropout_rate=args.transformer_attn_dropout_rate,
                    )
                )
            if i < len(odim) - 1: # not for the last layer
                self.linears.append(torch.nn.Linear(odim[i], args.adim))

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

        # CTC
        self.feedback_ctc_result = args.feedback_ctc_result
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

        hs_pad = xs_pad
        hs_mask = src_mask
        loss_ctcs = []
        cer_ctcs = []
        batch_size = xs_pad.size(0)
        itr = len(self.odim)
        for i in range(itr):
            if self.use_conformer:
                tpl = hs_pad if i == 0 else (hs_pad, hs_posemb)
                hs_pad, hs_posemb, hs_mask = self.encoders[i](tpl, hs_mask)
            else:
                hs_pad, hs_mask = self.encoders[i](hs_pad, hs_mask)
            hs_len = hs_mask.view(batch_size, -1).sum(1)
            loss_ctcs.append(
                self.ctcs[i](hs_pad.view(batch_size, -1, self.adim), hs_len, yss_pad[i])
            )

            if not self.training and self.error_calculators is not None:
                ys_hat = self.ctcs[i].argmax(hs_pad.view(batch_size, -1, self.adim)).data
                cer_ctcs.append(
                    self.error_calculators[i](ys_hat.cpu(), yss_pad[i].cpu(), is_ctc=True)
                )
            # for visualization
            if i == itr - 1 and not self.training:
                self.ctcs[i].softmax(hs_pad)

            if i < itr - 1 and self.feedback_ctc_result:
                hs_pad = hs_pad + self.linears[i](self.ctcs[i].softmax(hs_pad))

        self.loss = 0
        loss_ctc_data = []
        for i, lc in enumerate(loss_ctcs):
            self.loss = self.loss + lc / itr
            loss_ctc_data.append(float(lc))

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

    def encode(self, x, ctc_index=100):
        self.eval()
        x = torch.as_tensor(x).unsqueeze(0)

        h = x
        itr = len(self.odim)
        for i in range(itr):
            if self.use_conformer:
                tpl = h if i == 0 else (h, h_posemb)
                h, h_posemb, _ = self.encoders[i](tpl, None)
            else:
                h, _ = self.encoders[i](h, None)

            if i == ctc_index:
                break

            if i < itr - 1 and self.feedback_ctc_result:
                h = h + self.linears[i](self.ctcs[i].softmax(h))

        return h.squeeze(0)

    def recognize(self, x, recog_args, char_list=None, rnnlm=None, use_jit=False):
        enc_output = self.encode(x).unsqueeze(0)

        from itertools import groupby

        lpz = self.ctcs[-1].argmax(enc_output)
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
        self.eval()
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad)
        ret = []
        for name, m in self.named_modules():
            if isinstance(m, CTC) and m.probs is not None:
                ret.append(m.probs.cpu().numpy())
        self.train()
        return ret

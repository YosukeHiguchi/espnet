from contextlib import contextmanager
from dataclasses import dataclass
from distutils.version import LooseVersion
from itertools import groupby
import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import os
import numpy
import random
import torch
from typeguard import check_argument_types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from espnet.nets.beam_search import Hypothesis
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.maskctc.add_mask_token import mask_uniform_for_bert
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,  # noqa: H301
    ScaledPositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
    LegacyRelPositionalEncoding,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.predecoder.abs_predecoder import AbsPreDecoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.transducer.error_calculator import ErrorCalculatorTransducer
from espnet2.asr_transducer.utils import get_transducer_task_io
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling2
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling6
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling8
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield

class BECTRAModel(AbsESPnetModel):
    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        predecoder: Optional[AbsPreDecoder],
        postencoder: Optional[AbsPostEncoder],
        decoder: AbsDecoder,
        ctc: CTC,
        joint_network: Optional[torch.nn.Module] = None,
        asr_weight: float = 0.3,
        interctc_weight: float = 0.0,
        aux_weight: float = 0.5,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        sym_mask: str = "<mask>",
        extract_feats_in_collect_stats: bool = True,
        predec_drop_rate: float = 0.0,
        embed_enc_layer: str = "conv2d",
        rnnt_type: str = "warprnnt",
        concat_network_nblocks: int = 6,
    ):
        assert check_argument_types()

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.blank_id = 0
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.token_list = token_list.copy()

        self.bert_vocab_size = predecoder.vocab_size()
        self.bert_cls_token = predecoder.cls_token_id()
        self.bert_sep_token = predecoder.sep_token_id()
        self.bert_pad_token = predecoder.pad_token_id() # used as blank
        self.bert_mask_token = predecoder.mask_token_id()
        self.bert_token_list = predecoder.token_list()

        self.asr_weight = asr_weight
        self.interctc_weight = interctc_weight
        self.aux_weight = aux_weight
        logging.warning('asr_weight: {}'.format(self.asr_weight))
        logging.warning('interctc_weight: {}'.format(self.interctc_weight))
        logging.warning('aux_weight: {}'.format(self.aux_weight))

        self.predec_drop_rate = predec_drop_rate
        logging.warning('predec_drop_rate: {}'.format(self.predec_drop_rate))

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize

        # network
        ## encoder
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder
        if not hasattr(self.encoder, "interctc_use_conditioning"):
            self.encoder.interctc_use_conditioning = False
        if self.encoder.interctc_use_conditioning:
            self.encoder.conditioning_layer = torch.nn.Linear(
                vocab_size, self.encoder.output_size()
            )

        ## bert
        self.bert = predecoder

        ## self-attention
        self.concat_network = TransformerEncoder(
            input_size=self.encoder.output_size(),
            attention_heads=4,
            linear_units=2048,
            num_blocks=concat_network_nblocks,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            input_layer="identity",
        )

        ## embedding-related
        self.embed_enc_layer = embed_enc_layer
        if embed_enc_layer == "conv2d":
            self.embed_enc = Conv2dSubsampling(
                self.encoder.output_size(),
                self.concat_network.output_size(),
                self.concat_network._dropout_rate,
            )
        elif embed_enc_layer == "conv2d2":
            self.embed_enc = Conv2dSubsampling2(
                self.encoder.output_size(),
                self.concat_network.output_size(),
                self.concat_network._dropout_rate,
            )
        else:
            raise NotImplementedError
        self.embed_bert = torch.nn.Linear(
            self.bert.output_size(),
            self.concat_network.output_size(),
        )

        ## prediction network
        self.prediction_network = decoder

        ## joint network
        self.joint_network = joint_network

        # losses
        ## CTC
        self.error_calculator_enc = None
        if self.asr_weight == 0.0:
            self.ctc_enc = None
        else:
            self.ctc_enc = ctc
            if report_cer or report_wer:
                self.error_calculator_enc = ErrorCalculator(
                    token_list, sym_space, sym_blank, report_cer, report_wer
                )

        self.ctc_dec = CTC(
            odim=self.bert_vocab_size,
            encoder_output_size=self.encoder.output_size(),
            dropout_rate=ctc.dropout_rate,
        )
        if report_cer or report_wer:
            self.error_calculator_dec = ErrorCalculator(
                self.bert_token_list,
                sym_space,
                self.bert_token_list[self.bert_pad_token], # '[PAD]' is used for blank
                report_cer,
                report_wer
            )

        ## RNN-T
        self.rnnt_type = rnnt_type
        logging.warning("rnnt_type: {}".format(self.rnnt_type))
        if rnnt_type == "warprnnt":
            from warprnnt_pytorch import RNNTLoss
            self.transducer_loss = RNNTLoss(
                blank=self.blank_id,
                fastemit_lambda=0.0,
            )
        elif rnnt_type == "torchaudio":
            import torchaudio
            self.transducer_loss = torchaudio.transforms.RNNTLoss(
                blank=self.blank_id,
            )

        self.error_calculator_trans = None
        if report_cer or report_wer:
            self.error_calculator_trans = ErrorCalculatorTransducer(
                decoder,
                joint_network,
                self.bert_token_list,
                sym_space,
                self.bert_token_list[self.bert_pad_token], # '[PAD]' is used for blank
                report_cer=report_cer,
                report_wer=report_wer,
            )

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        text_bert: torch.Tensor,
        text_bert_lengths: torch.Tensor,
        text_bert_org: torch.Tensor,
        text_bert_org_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        batch_size = speech.shape[0]

        # For data-parallel
        text = text[:, : text_lengths.max()] # ASR
        text_bert = text_bert[:, : text_bert_lengths.max()]
        text_bert_org = text_bert_org[:, : text_bert_org_lengths.max()]

        # Define stats to report
        stats = dict()

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        # 2. Encoder CTC
        if self.asr_weight != 0.0:
            loss_ctc_enc, cer_ctc_enc = self._calc_enc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

            stats["loss_ctc_enc"] = (
                loss_ctc_enc.detach() if loss_ctc_enc is not None else None
            )
            stats["cer_ctc_enc"] = cer_ctc_enc

            # 2a. Intermediate CTC (optional)
            loss_interctc = 0.0
            if self.interctc_weight != 0.0 and intermediate_outs is not None:
                for layer_idx, intermediate_out in intermediate_outs:
                    # we assume intermediate_out has the same length & padding
                    # as those of encoder_out
                    loss_ic, cer_ic = self._calc_enc_loss(
                        intermediate_out, encoder_out_lens, text, text_lengths
                    )
                    loss_interctc = loss_interctc + loss_ic

                    # Collect Intermedaite CTC stats
                    stats["loss_interctc_layer{}".format(layer_idx)] = (
                        loss_ic.detach() if loss_ic is not None else None
                    )
                    stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic

                loss_interctc = loss_interctc / len(intermediate_outs)

                # calculate whole encoder loss
                loss_ctc_enc = (
                    1 - self.interctc_weight
                ) * loss_ctc_enc + self.interctc_weight * loss_interctc

        # 3. BERT
        enc_emb, enc_mask = self.embed_enc(
            encoder_out,
            ~make_pad_mask(encoder_out_lens)[:, None, :],
        )
        encoder_out_lens = enc_mask.squeeze(1).sum(1)
        enc_mask = (
            ~make_pad_mask(encoder_out_lens)[:, None, :]
        ).to(encoder_out.device)

        if random.random() < self.predec_drop_rate and self.training:
            embs = enc_emb
            masks = enc_mask
        else:
            # apply masks
            ys_in_pad, _ = mask_uniform_for_bert(
                text_bert,
                self.bert_mask_token,
                self.bert_pad_token,
                self.ignore_id,
            )

            # forward BERT
            bert_out, _ = self.bert(
                ys_in_pad, text_bert_lengths,
            )

            # embed
            bert_emb = self.embed_bert(bert_out)
            bert_mask = (
                ~make_pad_mask(text_bert_lengths)[:, None, :]
            ).to(encoder_out.device)

            embs = torch.cat([enc_emb, bert_emb], dim=1)
            masks = torch.cat([enc_mask, bert_mask], dim=2)

        # 4. Concat network
        concat_out, _, _ = self.concat_network(
            embs,
            encoder_out_lens, # not used
            masks=masks,
        )

        # 4. Concat network CTC
        loss_ctc_dec = self.ctc_dec(
            concat_out[:, :encoder_out_lens.max()],
            encoder_out_lens,
            text_bert_org,
            text_bert_org_lengths,
        )

        cer_ctc_dec = None
        if not self.training and self.error_calculator_dec is not None:
            ys_hat_dec = self.ctc_dec.argmax(
                concat_out[:, :encoder_out_lens.max()]
            ).data
            cer_ctc_dec = self.error_calculator_dec(
                ys_hat_dec.cpu(), text_bert_org.cpu(), is_ctc=True
            )

        stats["loss_ctc_dec"] = (
            loss_ctc_dec.detach() if loss_ctc_dec is not None else None
        )
        stats["cer_ctc_dec"] = cer_ctc_dec

        # 5. Transducer
        pred_in, target, t_len, u_len = get_transducer_task_io(
            text,
            encoder_out_lens,
            ignore_id=self.ignore_id,
            blank_id=self.blank_id,
        )

        self.prediction_network.set_device(encoder_out.device)
        pred_out = self.prediction_network(pred_in)

        joint_out = self.joint_network(
            concat_out[:, :encoder_out_lens.max()].unsqueeze(2),
            pred_out.unsqueeze(1),
        )

        loss_transducer = self.transducer_loss(
            joint_out,
            target,
            t_len,
            u_len,
        )

        cer_transducer, wer_transducer = None, None
        if not self.training and self.error_calculator_trans is not None:
            cer_transducer, wer_transducer = self.error_calculator_trans(
                concat_out[:, :encoder_out_lens.max()], target
            )

        stats["loss_transducer"] = (
            loss_transducer.detach() if loss_transducer is not None else None
        )
        stats["cer_transducer"] = cer_transducer
        stats["wer_transducer"] = wer_transducer

        # 4. loss definition
        if self.asr_weight == 0.0:
            loss = (
                self.aux_weight * loss_ctc_dec + \
                (1 - self.aux_weight) * loss_transducer
            )
        else:
            loss_ctc = (
                self.asr_weight * loss_ctc_enc + \
                (1 - self.asr_weight) * loss_ctc_dec
            )
            loss = (
                self.aux_weight * loss_ctc + \
                (1 - self.aux_weight) * loss_transducer
            )

        # Collect total loss stats
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        text_bert: torch.Tensor,
        text_bert_lengths: torch.Tensor,
        text_bert_org: torch.Tensor,
        text_bert_org_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.extract_feats_in_collect_stats:
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )
            feats, feats_lengths = speech, speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        if self.encoder.interctc_use_conditioning:
            encoder_out, encoder_out_lens, _ = self.encoder(
                feats, feats_lengths, ctc=self.ctc_enc
            )
        else:
            encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens

        return encoder_out, encoder_out_lens

    def _calc_enc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc_enc(
            encoder_out, encoder_out_lens, ys_pad, ys_pad_lens
        )

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator_enc is not None:
            ys_hat = self.ctc_enc.argmax(encoder_out).data
            cer_ctc = self.error_calculator_enc(
                ys_hat.cpu(), ys_pad.cpu(), is_ctc=True
            )
        return loss_ctc, cer_ctc

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths


@dataclass
class Hypo:
    score: float
    yseq: List[int]
    dec_state: Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        List[Optional[torch.Tensor]],
        torch.Tensor,
    ]
    logpseq: List[float]


class BECTRAInference(torch.nn.Module):
    """BECTRA inference"""

    def __init__(
        self,
        asr_model: BECTRAModel,
        n_iterations: int,
        beam_size: int,
        dataset: str = "librispeech",
    ):
        super().__init__()

        self.encoder = asr_model.encoder
        self.ctc_enc = asr_model.ctc_enc
        self.ctc_dec = asr_model.ctc_dec
        self.bert = asr_model.bert
        self.concat_network = asr_model.concat_network
        self.embed_enc = asr_model.embed_enc
        self.embed_bert = asr_model.embed_bert
        self.prediction_network = asr_model.prediction_network
        self.joint_network = asr_model.joint_network

        asr_model.bert.reset_embeddings_position_ids() # IMPORTANT!!!
        self.bert_tokenizer = asr_model.bert.tokenizer
        self.bert_cls_token = asr_model.bert_cls_token
        self.bert_sep_token = asr_model.bert_sep_token
        self.bert_mask_token = asr_model.bert_mask_token

        self.tokenizer = None # defined in bin/asr_inference_maskctc.py:L112
        self.converter = TokenIDConverter(token_list=asr_model.token_list)

        self.n_iterations = n_iterations
        self.beam_size = beam_size

        self.dataset = dataset
        logging.warning("dataset: {}".format(self.dataset))
        if self.dataset == "csj":
            self.han2zen = str.maketrans({chr(0x0021 + i): chr(0xFF01 + i) for i in range(94)})

    def ids2text(self, ids: List[int]):
        text = "".join(self.converter.ids2tokens(ids))
        return text

    def forward(
        self,
        enc_out: torch.Tensor,
    ) -> List[Hypothesis]:
        enc_out = enc_out.unsqueeze(0)
        try:
            enc_emb, _ = self.embed_enc(enc_out, None)
        except:
            logging.warning("enc_out too short: {}".format(enc_out.shape))
            yseq = torch.tensor([0] + [] + [0], device=enc_out.device)

            return Hypothesis(yseq=yseq)

        num_iter = self.n_iterations

        # encoder CTC
        if self.ctc_enc is not None:
            ctc_enc_probs, ctc_enc_ids = torch.exp(
                self.ctc_enc.log_softmax(enc_out)
            ).max(dim=-1)
            y_enc_hat = torch.stack([x[0] for x in groupby(ctc_enc_ids[0])])
            y_enc_nonblank_idx = torch.nonzero(y_enc_hat != 0).squeeze(-1)

            text_enc = self.ids2text(y_enc_hat[y_enc_nonblank_idx].tolist()).replace('▁', ' ')
            logging.info("ctc:\n{}".format(text_enc))

            # convert encoder results to bert tokens
            bert_text_ids = self.bert_tokenizer(text_enc)['input_ids']
            if self.bert_tokenizer.unk_token_id in bert_text_ids:
                logging.warning("Unk token found: {}".format(
                    self.bert_tokenizer.convert_ids_to_tokens(bert_text_ids)
                ))
            y_bert_in = torch.Tensor([bert_text_ids[1:-1]]).long().to(y_enc_hat.device)
            if num_iter > 0:
                y_bert_in[0][:] = self.bert_mask_token
            else:
                final_hyp = y_bert_in
        else:
            y_bert_in = None

        # decoder ASR
        for k in range(1, num_iter + 1):
            # if the bert input is empty
            if len(y_bert_in[0]) == 0:
                final_hyp = y_bert_in
                break

            # prepend/append cls/sep tokens
            if y_bert_in is not None:
                if y_bert_in[0][0] != self.bert_cls_token:
                    y_bert_in = torch.cat(
                        [
                            enc_out.new([self.bert_cls_token]).long(),
                            y_bert_in[0],
                            enc_out.new([self.bert_sep_token]).long(),
                        ]
                    ).unsqueeze(0)

                bert_out, _ = self.bert(y_bert_in, [y_bert_in.size(1)])

                bert_emb = self.embed_bert(bert_out)
                embs = torch.cat([enc_emb, bert_emb], dim=1)
            else:
                embs = enc_emb

            # forward concat network
            concat_out, _, _ = self.concat_network(embs, [embs.size(1)])

            concat_enc_out = concat_out[:, :enc_emb.size(1)]
            ctc_dec_probs, ctc_dec_ids = torch.exp(
                self.ctc_dec.log_softmax(concat_enc_out)
            ).max(dim=-1)
            y_dec_hat = torch.stack([x[0] for x in groupby(ctc_dec_ids[0])])
            y_dec_nonblank_idx = torch.nonzero(y_dec_hat != 0).squeeze(-1)

            logging.info("dec:\n{}".format(
                self.bert_tokenizer.decode(y_dec_hat[y_dec_nonblank_idx])
            ))

            # last iteration
            if k == num_iter or len(y_dec_nonblank_idx) == 0:
                t_len = enc_emb.size(1)
                self.prediction_network.set_device(enc_out.device)

                logging.info("beam_size: {}".format(self.beam_size))
                if self.beam_size == 0:
                    # bert-ctc decoding
                    final_hyp = y_dec_hat[y_dec_nonblank_idx].unsqueeze(0)
                elif self.beam_size == 1:
                    # transducer greedy decoding
                    dec_state = self.prediction_network.init_state(1)
                    hyp = Hypo(score=0.0, yseq=[0], logpseq=[1.0], dec_state=dec_state)
                    cache = {}

                    pred_out, state, _ = self.prediction_network.score(hyp, cache)

                    for t in range(t_len):
                        logp = self.joint_network(
                            concat_out[:, t], pred_out.unsqueeze(0)
                        ).log_softmax(-1)
                        top_logp, pred = torch.max(logp, dim=-1)

                        if pred != 0:
                            hyp.yseq.append(int(pred))
                            hyp.logpseq.append(float(top_logp))
                            hyp.score += float(top_logp)

                            hyp.dec_state = state

                            pred_out, state, _ = self.prediction_network.score(hyp, cache)

                    preds = torch.Tensor(hyp.yseq[1:]).long().to(enc_out.device)
                    probs = torch.Tensor(hyp.logpseq[1:]).float().to(enc_out.device).exp()

                    logging.info("dec:\n{}".format(self.ids2text(preds)))
                    final_hyp = preds.unsqueeze(0)
                else:
                    # transducer beam-search decoding
                    dec_state = self.prediction_network.init_state(1)
                    kept_hyps = [
                        Hypo(score=0.0, yseq=[0], logpseq=[1.0], dec_state=dec_state)
                    ]
                    cache = {}

                    for t in range(t_len):
                        hyps = kept_hyps
                        kept_hyps = []

                        while True:
                            max_hyp = max(hyps, key=lambda x: x.score)
                            hyps.remove(max_hyp)

                            pred_out, state, _ = self.prediction_network.score(max_hyp, cache)

                            logp = self.joint_network(
                                concat_out[:, t], pred_out.unsqueeze(0)
                            ).log_softmax(-1)
                            topk = logp[0, 1:].topk(self.beam_size, dim=-1) # ignore blank

                            kept_hyps.append(
                                Hypo(
                                    score=max_hyp.score + float(logp[0, 0:1]),
                                    yseq=max_hyp.yseq[:],
                                    dec_state=max_hyp.dec_state,
                                    logpseq=max_hyp.logpseq[:],
                                )
                            )

                            for logp, idx in zip(*topk):
                                hyps.append(
                                    Hypo(
                                        score=max_hyp.score + float(logp),
                                        yseq=max_hyp.yseq[:] + [int(idx + 1)],
                                        dec_state=state,
                                        logpseq=max_hyp.logpseq[:] + [float(logp)],
                                    )
                                )

                            hyps_max = float(max(hyps, key=lambda x: x.score).score)
                            kept_most_prob = sorted(
                                [hyp for hyp in kept_hyps if hyp.score > hyps_max],
                                key=lambda x: x.score,
                            )

                            if len(kept_most_prob) >= self.beam_size:
                                kept_hyps = kept_most_prob
                                break

                    best_hyp = max(kept_hyps, key=lambda x: x.score)
                    preds = torch.Tensor(best_hyp.yseq[1:]).long().to(enc_out.device)
                    probs = torch.Tensor(best_hyp.logpseq[1:]).float().to(enc_out.device).exp()

                    logging.info("dec:\n{}".format(self.ids2text(preds)))
                    final_hyp = preds.unsqueeze(0)

                break

            probs_hat = []
            cnt = 0
            for i, y in enumerate(y_dec_hat.tolist()):
                probs_hat.append(-1)
                while cnt < ctc_dec_ids.shape[1] and y == ctc_dec_ids[0][cnt]:
                    if probs_hat[i] < ctc_dec_probs[0][cnt]:
                        probs_hat[i] = ctc_dec_probs[0][cnt].item()
                    cnt += 1
            probs_hat = torch.from_numpy(numpy.array(probs_hat))

            mask_num = (len(y_dec_nonblank_idx) * (num_iter - k)) // num_iter
            mask_idx = torch.topk(
                probs_hat[y_dec_nonblank_idx], mask_num, dim=-1, largest=False
            )[1]

            y_bert_in = y_dec_hat[y_dec_nonblank_idx].unsqueeze(0)
            y_bert_in[0][mask_idx] = self.bert_mask_token

            logging.info("mask:{}".format(mask_num))
            logging.info("dec:\n{}".format(self.bert_tokenizer.decode(y_bert_in[0])))

        if len(final_hyp[0]) > 0:
            if self.beam_size >= 1:
                # bectra
                yseq = final_hyp[0].tolist()
            else:
                # bert-ctc
                if final_hyp[0][0] == self.bert_cls_token:
                    yseq = self.bert_tokenizer.decode(final_hyp[0][1:-1].tolist())
                else:
                    yseq = self.bert_tokenizer.decode(final_hyp[0].tolist())

                if self.dataset == "librispeech":
                    yseq = yseq.upper()
                elif self.dataset == "aishell":
                    yseq = yseq.replace(" ", "")
                elif self.dataset == "csj":
                    yseq = yseq.replace("[UNK]", "ヮ").replace(" ", "")
                    yseq = yseq.translate(self.han2zen)

                logging.info("out:\n{}".format(yseq))
                yseq = self.converter.tokens2ids(
                    self.tokenizer.text2tokens(yseq)
                )
        else:
            yseq = final_hyp[0].tolist()
            logging.info("out:\n{}".format("#No output"))


        yseq = torch.tensor(
            [0] + yseq + [0], device=enc_out.device
        )

        return Hypothesis(yseq=yseq)

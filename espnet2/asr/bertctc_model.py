from contextlib import contextmanager
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
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.predecoder.abs_predecoder import AbsPreDecoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
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


class BERTCTCModel(AbsESPnetModel):
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
        decoder: AbsEncoder, ###
        ctc: CTC,
        joint_network: Optional[torch.nn.Module] = None,
        asr_weight: float = 0.3,
        interctc_weight: float = 0.0,
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
        slu_type: str = None,
        slu_weight: float = 1.0,
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
        logging.warning('asr_weight: {}'.format(self.asr_weight))
        logging.warning('interctc_weight: {}'.format(self.interctc_weight))

        self.predec_drop_rate = predec_drop_rate
        logging.warning('predec_drop_rate: {}'.format(self.predec_drop_rate))

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize

        # encoder-related
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder
        if not hasattr(self.encoder, "interctc_use_conditioning"):
            self.encoder.interctc_use_conditioning = False
        if self.encoder.interctc_use_conditioning:
            self.encoder.conditioning_layer = torch.nn.Linear(
                vocab_size, self.encoder.output_size()
            )

        # decoder-related
        self.decoder = decoder # positional encoding is not defined here
        self.predecoder = predecoder

        # embedding-related
        self.embed_enc_layer = embed_enc_layer
        if embed_enc_layer == "linear":
            pos_enc_class = self.decoder._pos_enc_class
            self.embed_enc = torch.nn.Sequential(
                torch.nn.Linear(
                    self.encoder.output_size(),
                    self.decoder.output_size(),
                ),
                torch.nn.LayerNorm(self.decoder._output_size,),
                torch.nn.Dropout(self.decoder._dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(
                    self.decoder.output_size(),
                    self.decoder._positional_dropout_rate,
                ),
            )
        elif embed_enc_layer == "conv2d":
            self.embed_enc = Conv2dSubsampling(
                self.encoder.output_size(),
                self.decoder.output_size(),
                self.decoder._dropout_rate,
            )
        elif embed_enc_layer == "conv2d2":
            self.embed_enc = Conv2dSubsampling2(
                self.encoder.output_size(),
                self.decoder.output_size(),
                self.decoder._dropout_rate,
            )
        self.embed_bert = torch.nn.Linear(
            self.predecoder.output_size(),
            self.decoder.output_size(),
        )

        # CTC
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

        self.slu_type = slu_type
        self.slu_weight = slu_weight
        if self.slu_type == 'ic':
            self.slu_classifier = torch.nn.Linear(
                self.decoder.output_size(),
                self.vocab_size,
            )
            self.slu_criterion = torch.nn.CrossEntropyLoss()

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

        # 2. CTC branch
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

        # 3. Decoder
        loss_ctc_dec, cer_ctc_dec, h_cls = self._calc_dec_loss(
            encoder_out,
            encoder_out_lens,
            text_bert,
            text_bert_lengths,
            text_bert_org,
            text_bert_org_lengths,
        )

        stats["loss_ctc_dec"] = (
            loss_ctc_dec.detach() if loss_ctc_dec is not None else None
        )
        stats["cer_ctc_dec"] = cer_ctc_dec

        # SLU
        if self.slu_type == 'ic':
            h_cls = self.slu_classifier(h_cls)
            loss_slu = self.slu_criterion(h_cls, text[:, 0])
            acc_slu = th_accuracy(
                h_cls,
                text[:, 0].unsqueeze(0),
                ignore_label=self.ignore_id,
            )

            stats["loss_slu"] = (
                loss_slu.detach() if loss_slu is not None else None
            )
            stats["acc_slu"] = acc_slu

        # 4. loss definition
        if self.asr_weight == 0.0:
            loss = loss_ctc_dec
        else:
            loss = (
                self.asr_weight * loss_ctc_enc + \
                (1 - self.asr_weight) * loss_ctc_dec
            )

        if self.slu_type == 'ic':
            loss = loss + self.slu_weight * loss_slu

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

    def _calc_dec_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        ys_pad_trg: torch.Tensor,
        ys_pad_trg_lens: torch.Tensor,
    ):
        # embed
        if self.embed_enc_layer == "linear":
            enc_emb = self.embed_enc(encoder_out)
        else:
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
                ys_pad,
                self.bert_mask_token,
                self.bert_pad_token,
                self.ignore_id,
            )

            # forward BERT
            predecoder_out, _ = self.predecoder(
                ys_in_pad, ys_pad_lens
            )

            # embed
            bert_emb = self.embed_bert(predecoder_out)
            bert_mask = (
                ~make_pad_mask(ys_pad_lens)[:, None, :]
            ).to(encoder_out.device)

            embs = torch.cat([enc_emb, bert_emb], dim=1)
            masks = torch.cat([enc_mask, bert_mask], dim=2)

        # forward decoder
        decoder_out, _, _ = self.decoder(
            embs,
            encoder_out_lens, # not used
            masks=masks,
        )

        # Calc CTC loss
        loss_ctc = self.ctc_dec(
            decoder_out[:, :encoder_out_lens.max()],
            encoder_out_lens,
            ys_pad_trg,
            ys_pad_trg_lens
        )

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator_dec is not None:
            ys_hat = self.ctc_dec.argmax(
                decoder_out[:, :encoder_out_lens.max()]
            ).data
            cer_ctc = self.error_calculator_dec(
                ys_hat.cpu(), ys_pad_trg.cpu(), is_ctc=True
            )

        # return loss_ctc, cer_ctc, decoder_out[:, encoder_out_lens.max()]
        return loss_ctc, cer_ctc, None

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

class BERTCTCInference(torch.nn.Module):
    """Mask-CTC-based non-autoregressive inference"""

    def __init__(
        self,
        asr_model: BERTCTCModel,
        n_iterations: int,
        length_init_with_asr: bool,
    ):
        super().__init__()

        self.encoder = asr_model.encoder
        self.ctc_enc = asr_model.ctc_enc

        self.decoder = asr_model.decoder
        self.ctc_dec = asr_model.ctc_dec

        asr_model.predecoder.reset_embeddings_position_ids() # IMPORTANT!!!
        self.bert = asr_model.predecoder
        self.bert_tokenizer = asr_model.predecoder.tokenizer
        self.bert_cls_token = asr_model.bert_cls_token
        self.bert_sep_token = asr_model.bert_sep_token
        self.bert_mask_token = asr_model.bert_mask_token

        self.embed_enc_layer = asr_model.embed_enc_layer
        self.embed_enc = asr_model.embed_enc
        self.embed_bert = asr_model.embed_bert

        self.tokenizer = None # defined in bin/asr_inference_maskctc.py:L112
        self.converter = TokenIDConverter(token_list=asr_model.token_list)

        self.n_iterations = n_iterations
        self.length_init_with_asr = length_init_with_asr

        self.slu_type = asr_model.slu_type
        if self.slu_type == 'ic':
            self.slu_classifier = asr_model.slu_classifier

    def ids2text(self, ids: List[int]):
        text = "".join(self.converter.ids2tokens(ids))
        return text

    def forward(
        self,
        enc_out: torch.Tensor,
        attn_save_dir: Optional[str],
    ) -> List[Hypothesis]:
        # greedy ctc outputs
        enc_out = enc_out.unsqueeze(0)
        if self.embed_enc_layer == "linear":
            enc_emb = self.embed_enc(enc_out)
        else:
            enc_emb, _ = self.embed_enc(enc_out, None)

        num_iter = self.n_iterations

        # encoder ASR
        if self.ctc_enc is not None and self.length_init_with_asr:
            ctc_enc_probs, ctc_enc_ids = torch.exp(
                self.ctc_enc.log_softmax(enc_out)
            ).max(dim=-1)
            y_enc_hat = torch.stack([x[0] for x in groupby(ctc_enc_ids[0])])
            y_enc_nonblank_idx = torch.nonzero(y_enc_hat != 0).squeeze(-1)

            text_enc = self.ids2text(
                y_enc_hat[y_enc_nonblank_idx].tolist()
            ).replace('▁', ' ')
            # ).replace('▁', ' ')[1:]
            logging.info("ctc:\n{}".format(text_enc))

            # convert encoder tokens to bert tokens
            bert_text_ids = self.bert_tokenizer(text_enc)['input_ids']
            if self.bert_tokenizer.unk_token_id in bert_text_ids:
                logging.warning("Unk token found: {}".format(
                    self.bert_tokenizer.convert_ids_to_tokens(bert_text_ids)
                ))

            y_bert_in = torch.Tensor(
                [bert_text_ids[1:-1]]
            ).long().to(y_enc_hat.device)
            if num_iter > 0:
                y_bert_in[0][:] = self.bert_mask_token
        else:
            y_bert_in = None

        # decoder ASR
        for t in range(1, num_iter + 1):
            if len(y_bert_in[0]) == 0:
                break

            if y_bert_in is not None:
                if y_bert_in[0][0] != self.bert_cls_token:
                    y_bert_in = torch.cat(
                        [
                            enc_out.new([self.bert_cls_token]).long(),
                            y_bert_in[0],
                            enc_out.new([self.bert_sep_token]).long(),
                        ]
                    ).unsqueeze(0)

                # logging.info("inp:\n{}".format(
                #     self.bert_tokenizer.decode(y_bert_in[0])
                # ))
                bert_out, _ = self.bert(y_bert_in, [y_bert_in.size(1)])

                bert_emb = self.embed_bert(bert_out)
                embs = torch.cat([enc_emb, bert_emb], dim=1)
            else:
                embs = enc_emb

            # forward decoder
            dec_out, _, _ = self.decoder(embs, [embs.size(1)])

            # attention plot
            if not attn_save_dir is None:
                for name, modu in self.decoder.named_modules():
                    if isinstance(modu, MultiHeadedAttention):
                        att_w = modu.attn.detach().cpu().numpy().squeeze(0)

                        w, h = plt.figaspect(1.0 / len(att_w))
                        fig = plt.Figure(figsize=(w * 1.3, h * 1.3))
                        axes = fig.subplots(1, len(att_w))

                        for ax, aw in zip(axes, att_w):
                            ax.imshow(aw.astype(numpy.float32), aspect="auto")
                            # ax.set_title(f"{k}_{id_}")
                            ax.set_xlabel("Input")
                            ax.set_ylabel("Output")
                            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

                        fig.savefig(
                            os.path.join(
                                attn_save_dir,
                                f"t{t}." + name + ".png",
                            )
                        )

            dec_enc_out = dec_out[:, :enc_emb.size(1)]
            ctc_dec_probs, ctc_dec_ids = torch.exp(
                self.ctc_dec.log_softmax(dec_enc_out)
            ).max(dim=-1)
            y_dec_hat = torch.stack([x[0] for x in groupby(ctc_dec_ids[0])])
            y_dec_nonblank_idx = torch.nonzero(y_dec_hat != 0).squeeze(-1)

            logging.info("dec:\n{}".format(
                self.bert_tokenizer.decode(y_dec_hat[y_dec_nonblank_idx])
            ))

            if t == num_iter or len(y_dec_nonblank_idx) == 0:
                y_bert_in = y_dec_hat[y_dec_nonblank_idx].unsqueeze(0)

                if self.slu_type == 'ic':
                    ic_label = self.slu_classifier(
                        dec_out[:, enc_emb.size(1)]
                    ).argmax(-1)

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

            mask_num = (len(y_dec_nonblank_idx) * (num_iter - t)) // num_iter
            mask_idx = torch.topk(
                probs_hat[y_dec_nonblank_idx], mask_num, dim=-1, largest=False
            )[1]

            y_bert_in = y_dec_hat[y_dec_nonblank_idx].unsqueeze(0)
            y_bert_in[0][mask_idx] = self.bert_mask_token

            logging.info("mask:{}".format(mask_num))
            logging.info("dec:\n{}".format(
                self.bert_tokenizer.decode(y_bert_in[0])
            ))

        # for TED2
        if len(y_bert_in[0]) > 0:
            if y_bert_in[0][0] == self.bert_cls_token:
                yseq = self.bert_tokenizer.decode(y_bert_in[0][1:-1].tolist())
            else:
                yseq = self.bert_tokenizer.decode(y_bert_in[0].tolist())

            # yseq = yseq.upper() # for librispeech

            if self.bert_tokenizer.name_or_path == "bert-base-chinese":
                yseq = yseq.replace(" ", "")

            logging.info("out:\n{}".format(
                yseq
            ))
            yseq = self.converter.tokens2ids(
                self.tokenizer.text2tokens(
                    yseq
                )
            )

            if self.slu_type == 'ic' and num_iter > 0:
                yseq = [int(ic_label[0])] + yseq
        else:
            yseq = y_bert_in[0].tolist()
            logging.info("out:\n{}".format("#No output"))


        yseq = torch.tensor(
            [0] + yseq + [0], device=y_bert_in.device
        )

        return Hypothesis(yseq=yseq)

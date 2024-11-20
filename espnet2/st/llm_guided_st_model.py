import logging
import random
from contextlib import contextmanager
from itertools import groupby
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from torch.nn.utils.rnn import pad_sequence
from typeguard import typechecked

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.llm.abs_llm import AbsLLM
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr_transducer.utils import get_transducer_task_io
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.e2e_asr_common import ErrorCalculator as ASRErrorCalculator
from espnet.nets.e2e_mt_common import ErrorCalculator as MTErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (  # noqa: H301
    LabelSmoothingLoss,
)

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class LLMGuidedSTModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: AbsDecoder,
        extra_asr_decoder: Optional[AbsDecoder],
        ctc: Optional[CTC],
        llm: AbsLLM,
        src_vocab_size: Optional[int],
        src_token_list: Optional[Union[Tuple[str, ...], List[str]]],
        asr_weight: float = 0.0,
        mtlalpha: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        report_bleu: bool = True,
        extract_feats_in_collect_stats: bool = True,
        ###
        src_sym_sos: str = "<sos/eos>",
        src_sym_eos: str = "<sos/eos>",
        is_encoder_eval: bool = True,
        is_llm_eval: bool = True,
    ):
        assert 0.0 <= asr_weight < 1.0, "asr_weight should be [0.0, 1.0)"
        assert 0.0 <= mtlalpha <= 1.0, "mtlalpha should be [0.0, 1.0]"

        super().__init__()

        self.sos = llm.start_of_response_token_id
        self.eos = llm.end_of_response_token_id
        assert src_sym_sos in src_token_list
        assert src_sym_eos in src_token_list
        self.src_sos = src_token_list.index(src_sym_sos)
        self.src_eos = src_token_list.index(src_sym_eos)
        logging.info(f"src_sos: {src_sym_sos}")
        logging.info(f"src_sos_id: {self.src_sos}")
        logging.info(f"src_eos: {src_sym_eos}")
        logging.info(f"src_eos_id: {self.src_eos}")
        # Blank id is assumed to be 0 for CTC training, which I think should be
        # OK for the current settings with llama2 (0:"<unk>") and llama3 (0:"!")
        self.blank_id = 0

        self.vocab_size = vocab_size
        self.src_vocab_size = src_vocab_size
        self.ignore_id = ignore_id
        self.asr_weight = asr_weight
        self.mtlalpha = mtlalpha
        self.token_list = token_list.copy()
        self.src_token_list = src_token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder

        ###
        self.is_encoder_eval = is_encoder_eval
        self.is_llm_eval = is_llm_eval

        self.decoder = decoder
        self.decoder.llm = llm
        self.decoder.embed = torch.nn.Linear(
            llm.output_size(),
            encoder.output_size(),
        )
        self.decoder.ctc = ctc # shares the same ctc instance
        ###

        self.st_use_transducer_decoder = False

        self.criterion_st = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        self.criterion_asr = LabelSmoothingLoss(
            size=src_vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # submodule for ASR task
        self.ctc = ctc # always use src ctc
        assert (
            src_token_list is not None
        ), "Missing src_token_list, cannot add asr module to st model"
        if self.mtlalpha < 1.0:
            self.extra_asr_decoder = extra_asr_decoder
        elif extra_asr_decoder is not None:
            logging.warning(
                "Not using extra_asr_decoder because "
                "mtlalpha is set as {} (== 1.0)".format(mtlalpha),
            )

        # MT error calculator
        self.mt_error_calculator = None
        if report_bleu:
            self.mt_error_calculator = MTErrorCalculator(
                token_list,
                "<space>", # Not used
                token_list[self.blank_id], # The first token
                report_bleu
            )

        # ASR error calculator
        self.asr_error_calculator = None
        if self.asr_weight > 0 and (report_cer or report_wer):
            assert (
                src_token_list is not None
            ), "Missing src_token_list, cannot add asr module to st model"
            self.asr_error_calculator = ASRErrorCalculator(
                src_token_list,
                "<space>", # Not used
                src_token_list[self.blank_id], # The first token
                report_cer,
                report_wer,
            )

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        src_text: Optional[torch.Tensor] = None,
        src_text_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch,)
            text: (Batch, Length)
            text_lengths: (Batch,)
            src_text: (Batch, length)
            src_text_lengths: (Batch,)
            kwargs: "utt_id" is among the input.
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)

        # additional checks with valid src_text
        if src_text is not None:
            assert src_text_lengths.dim() == 1, src_text_lengths.shape
            assert text.shape[0] == src_text.shape[0] == src_text_lengths.shape[0], (
                text.shape,
                src_text.shape,
                src_text_lengths.shape,
            )

        batch_size = speech.shape[0]
        text[text == -1] = self.ignore_id

        # for data-parallel
        text = text[:, : text_lengths.max()]
        if src_text is not None:
            src_text = src_text[:, : src_text_lengths.max()]

        # Set modules to eval mode
        if self.encoder.training and self.is_encoder_eval:
            self.encoder.eval()
        if self.decoder.llm is not None:
            if self.decoder.llm.lm.training and self.is_llm_eval:
                self.decoder.llm.lm.eval()

        # 1. Encoder
        st_encoder_out, st_encoder_out_lens = self.encode(speech, speech_lengths)
        asr_encoder_out, asr_encoder_out_lens = st_encoder_out, st_encoder_out_lens

        # 2a. CTC branch
        if self.asr_weight > 0:
            assert src_text is not None, "missing source text for asr sub-task of ST"

        if self.asr_weight > 0 and self.mtlalpha > 0:
            loss_asr_ctc, cer_asr_ctc = self._calc_asr_ctc_loss(
                asr_encoder_out, asr_encoder_out_lens, src_text, src_text_lengths
            )
        else:
            loss_asr_ctc, cer_asr_ctc = 0.0, None

        # 2b. Attention-decoder branch (extra ASR)
        if self.asr_weight > 0 and self.mtlalpha < 1.0:
            (
                loss_asr_att,
                acc_asr_att,
                cer_asr_att,
                wer_asr_att,
            ) = self._calc_asr_att_loss(
                asr_encoder_out,
                asr_encoder_out_lens,
                src_text,
                src_text_lengths,
            )
        else:
            loss_asr_att, acc_asr_att, cer_asr_att, wer_asr_att = 0.0, None, None, None

        # 2e. Attention-decoder branch (ST)
        loss_st_att, acc_st_att, bleu_st_att = self._calc_mt_att_loss(
            st_encoder_out,
            st_encoder_out_lens,
            text,
            text_lengths,
        )
        loss_st = loss_st_att

        # 3. Loss computation
        if self.asr_weight > 0:
            asr_ctc_weight = self.mtlalpha

            if asr_ctc_weight == 1.0:
                loss_asr = loss_asr_ctc
            elif asr_ctc_weight == 0.0:
                loss_asr = loss_asr_att
            else:
                loss_asr = (
                    asr_ctc_weight * loss_asr_ctc + (1 - asr_ctc_weight) * loss_asr_att
                )
            loss = (
                (1 - self.asr_weight) * loss_st
                + self.asr_weight * loss_asr
            )
        else:
            loss_asr = None
            loss = loss_st

        stats = dict(
            loss=loss.detach(),
            loss_asr=loss_asr.detach() if loss_asr is not None else loss_asr,
            loss_st_att=(
                loss_st_att.detach() if type(loss_st_att) is not float else loss_st_att
            ),
            loss_st=loss_st.detach(),
            acc_asr=acc_asr_att,
            acc=acc_st_att,
            cer_ctc=cer_asr_ctc,
            cer=cer_asr_att,
            wer=wer_asr_att,
            bleu=bleu_st_att,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        src_text: Optional[torch.Tensor] = None,
        src_text_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by st_inference.py

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
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

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

        return encoder_out, encoder_out_lens

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

    def _calc_mt_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(
            ys_pad, self.sos, self.eos, self.ignore_id, pad_input_with_eos=False
        )
        ys_in_lens = ys_pad_lens + 1
        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_st(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.mt_error_calculator is None:
            bleu_att = None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            bleu_att = self.mt_error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, bleu_att

    def _calc_asr_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(
            ys_pad, self.src_sos, self.src_eos, self.ignore_id
        )
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.extra_asr_decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_asr(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.src_vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.asr_error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.asr_error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_asr_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.asr_error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.asr_error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

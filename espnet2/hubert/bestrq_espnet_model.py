from contextlib import contextmanager
from distutils.version import LooseVersion
import logging
from pathlib import Path
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel


if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetSSLModel(AbsESPnetModel):
    """BEST-RQ Model"""

    def __init__(
        self,
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        extract_feats_in_collect_stats: bool = True,
        vector_size: int = 16,
        codebook_size: int = 8192,
        temporal_reduction: int = 4,
        mask_prob: float = 0.01,
        mask_length: int = 40,  # frames
        unmasked_region_weight: float = 0,
        apply_l2_normalization: bool = False,
        codebook_and_matrix_init_file: Union[Path, str] = None,
    ):
        assert check_argument_types()

        super().__init__()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder

        self.error_calculator = None

        self.criterion_ce = LabelSmoothingLoss(
            size=codebook_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        self.output_layer = torch.nn.Linear(encoder.output_size(), codebook_size)

        self.temporal_reduction = (
            temporal_reduction  # temporal reduction in the convolutional layers
        )
        self.codebook_size = codebook_size
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.unmasked_region_weight = unmasked_region_weight
        self.apply_l2_normalization = apply_l2_normalization

        self.ignore_id = ignore_id
        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

        if codebook_and_matrix_init_file is not None:
            init_stats = torch.load(
                codebook_and_matrix_init_file,
                map_location=lambda storage, loc: storage,
            )
            assert (
                init_stats["codebook"] is not None
                and init_stats["projection_matrix"] is not None
            ), "codebook_init is None."
            self.codebook_init = init_stats["codebook"]
            self.projection_matrix_init = init_stats["projection_matrix"]

        # Random projection matrix and codebook
        projection_matrix = torch.nn.Parameter(
            torch.zeros(frontend.output_size() * temporal_reduction, vector_size)
        )  # (reduced_feat_dim, vec_size)
        codebook = torch.nn.Parameter(
            torch.zeros(codebook_size, vector_size)
        )  # (codebook_size, vec_size)

        projection_matrix.requires_grad = False
        codebook.requires_grad = False

        self.register_parameter("projection_matrix", projection_matrix)
        self.register_parameter("codebook", codebook)

    def espnet_initialization_fn(self):
        if (
            getattr(self, "codebook_init", None) is not None
            and getattr(self, "projection_matrix_init", None) is not None
        ):
            self.projection_matrix.data = self.projection_matrix_init
            self.codebook.data = self.codebook_init
        else:
            torch.nn.init.xavier_uniform_(self.projection_matrix.data)
            torch.nn.init.normal_(self.codebook.data)
            self.projection_matrix.data = torch.nn.functional.normalize(
                self.projection_matrix.data, p=2, dim=1
            )
            self.codebook.data = torch.nn.functional.normalize(
                self.codebook.data, p=2, dim=1
            )

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        # Check that batch_size is unified
        batch_size = speech.shape[0]
        assert speech.shape[0] == speech_lengths.shape[0], (
            speech.shape,
            speech_lengths.shape,
        )

        # 1. Encoder
        feats, _, encoder_out, encoder_out_lens, time_masks = self.encode(
            speech, speech_lengths
        )
        prediction = self.output_layer(encoder_out)

        # 1.1 Feature Stacking
        reduced_feats, reduced_time_masks = self.stack_over_time(
            feats,
            time_masks,
            encoder_out_lens.max(),
        )

        # Get the labels
        # (batch, reduced_seq_len, vec_size)
        label, label_pad_mask = self.get_label(reduced_feats, encoder_out_lens)

        # 2. CrossEntropy loss
        loss = None
        stats = dict()

        # Masked region
        mmask = torch.logical_or(
            ~reduced_time_masks, label_pad_mask
        )  # mask for the masked region case, remove the unmasked region and the padding
        masked_loss = self.criterion_ce(
            prediction, label.masked_fill(mmask, self.ignore_id)
        )
        masked_acc = th_accuracy(
            prediction.view(-1, self.codebook_size),
            label.masked_fill(mmask, self.ignore_id),
            ignore_label=self.ignore_id,
        )

        prediction_idx = torch.masked_select(prediction.argmax(dim=-1), ~label_pad_mask)
        prediction_different_tokens = set(prediction_idx.tolist())
        prediction_coverage = len(prediction_different_tokens) / self.codebook_size

        label_coverage = len(set(label.view(-1).tolist())) / self.codebook_size

        # Unmasked region
        if not self.training or self.unmasked_region_weight > 0:
            umask = torch.logical_or(
                reduced_time_masks, label_pad_mask
            )  # mask for the unmasked region case, remove the masked region and the padding
            unmasked_loss = self.criterion_ce(
                prediction, label.masked_fill(umask, self.ignore_id)
            )

            unmasked_acc = th_accuracy(
                prediction.view(-1, self.codebook_size),
                label.masked_fill(umask, self.ignore_id),
                ignore_label=self.ignore_id,
            )

        else:
            unmasked_loss = torch.zeros_like(masked_loss)
            unmasked_acc = None

        loss = (
            1 - self.unmasked_region_weight
        ) * masked_loss + self.unmasked_region_weight * unmasked_loss

        # Collect total loss stats
        stats["loss"] = loss.detach()
        stats["masked_loss"] = masked_loss.detach()
        stats["unmasked_loss"] = unmasked_loss.detach()
        stats["masked_acc"] = masked_acc
        stats["unmasked_acc"] = unmasked_acc
        stats["prediction_coverage"] = prediction_coverage
        stats["label_coverage"] = label_coverage

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def stack_over_time(
        self,
        feats: torch.Tensor,
        masks: torch.Tensor,
        olen: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stack the input feature

        Args:
            feats: (Batch, Length, Dim)
            olen: output length
        """
        batch_size, _, f_dim = feats.shape
        stacked_feats = feats[:, : olen * self.temporal_reduction].reshape(
            batch_size, olen, self.temporal_reduction * f_dim
        )  # (batch, reduced_seq_len, tem_reduct * feat_dim)

        if masks is not None:
            stacked_masks = masks[:, : olen * self.temporal_reduction].reshape(
                batch_size, olen, self.temporal_reduction * masks.shape[-1]
            )  # (batch, reduced_seq_len, tem_reduct)
            stacked_masks = stacked_masks.any(dim=-1)  # (batch, reduced_seq_len)
        else:
            stacked_masks = None

        return stacked_feats, stacked_masks

    def get_label(
        self,
        feats: torch.Tensor,
        enc_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stack the input feature

        Args:
            feats: (Batch, Length, Dim)
            enc_lens: (Batch)
        """
        # (batch, reduced_seq_len, vec_size)
        vq_proj = torch.matmul(feats, self.projection_matrix)
        if not self.apply_l2_normalization:
            cdbk_distance = torch.cdist(vq_proj, self.codebook, p=1.0)
        else:
            cdbk_distance = torch.cdist(
                torch.nn.functional.normalize(vq_proj, p=2, dim=-1),
                self.codebook,
                p=1.0,
            )
        # (batch, reduced_seq_len, codebook_size) -> (batch, reduced_seq_len)
        labels = cdbk_distance.argmin(dim=2, keepdim=False)
        labels_pad_mask = make_pad_mask(enc_lens).to(labels.device)
        labels.masked_fill_(labels_pad_mask, self.ignore_id)
        return labels, labels_pad_mask

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
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
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        only_feature: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            only_feature: bool, only return the feature
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

            if only_feature:
                return feats, feats_lengths

            # 3. apply_mask
            masked_feats, masks = self.apply_mask(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            masked_feats, feats_lengths = self.preencoder(masked_feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        encoder_out, encoder_out_lens, _ = self.encoder(masked_feats, feats_lengths)

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

        return feats, feats_lengths, encoder_out, encoder_out_lens, masks

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

    def apply_mask(
        self,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adapted from specaug.

        Args:
            feats: (Batch, Length, Dim)
            speech_lengths: (Batch, )
        """
        org_size = feats.size()

        B, D = feats.shape[0:2]

        mask_pos_rand = torch.rand(B, D)

        mask = []
        aran = torch.arange(D)[None, :]
        for b in range(B):
            mask_pos = torch.nonzero(mask_pos_rand[b] <= self.mask_prob)
            tmp_mask = (mask_pos <= aran) * (aran <= (mask_pos + self.mask_length))
            mask.append(tmp_mask.any(dim=0))
        mask = torch.stack(mask, dim=0).to(feats.device)  # (B, Length, 1)

        # mask: (Batch, Length, 1)
        mask = mask.unsqueeze(2)

        normal_noise = torch.normal(0, 0.1, (torch.sum(mask), feats.shape[-1])).to(
            feats.device
        )

        if feats.requires_grad:
            feats = feats.masked_scatter(mask, normal_noise)
        else:
            feats = feats.masked_scatter_(mask, normal_noise)
        feats = feats.view(*org_size)

        return feats, mask

#!/usr/bin/env python3

"""Hugging Face Transformers PreDecoder."""

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.asr.predecoder.abs_predecoder import AbsPreDecoder
from typeguard import check_argument_types
from typing import Tuple

import copy
import logging
import torch

try:
    from transformers import AutoModel
    from transformers import AutoTokenizer

    is_transformers_available = True
except ImportError:
    is_transformers_available = False


class HuggingFaceTransformersPreDecoder(AbsPreDecoder):
    """Hugging Face Transformers PreDecoder."""

    def __init__(
        self,
        model_name_or_path: str,
    ):
        """Initialize the module."""
        assert check_argument_types()
        super().__init__()

        if not is_transformers_available:
            raise ImportError(
                "`transformers` is not available. Please install it via `pip install"
                " transformers` or `cd /path/to/espnet/tools && . ./activate_python.sh"
                " && ./installers/install_transformers.sh`."
            )

        self.transformer = AutoModel.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        assert self.transformer.config.model_type == "bert"

        self.pretrained_params = copy.deepcopy(self.transformer.state_dict())

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward."""
        mask = (~make_pad_mask(input_lengths)).to(input.device).int()

        args = {
            "input_ids": input,
            "attention_mask": mask,
            "return_dict": True,
        }

        output = self.transformer(**args).last_hidden_state

        return output, input_lengths

    def output_size(self) -> int:
        """Get the output size."""
        return self.transformer.config.hidden_size

    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def cls_token_id(self) -> int:
        return self.tokenizer.cls_token_id

    def sep_token_id(self) -> int:
        return self.tokenizer.sep_token_id

    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    def mask_token_id(self) -> int:
        return self.tokenizer.mask_token_id

    def token_list(self) -> int:
        sorted_dict = {
            k: v
            for k, v in sorted(
                self.tokenizer.vocab.items(), key=lambda item: item[1]
            )
        }
        return list(sorted_dict.keys())
        # return list(self.tokenizer.vocab.keys())

    def reset_embeddings_position_ids(self):
        model_params = self.transformer.state_dict()
        model_params.update(
            {'embeddings.position_ids' : self.pretrained_params['embeddings.position_ids']}
        )
        self.transformer.load_state_dict(model_params)
        logging.info("Reloaded embeddings.position_ids")

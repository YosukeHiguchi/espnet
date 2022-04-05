#!/usr/bin/env python3
from espnet2.tasks.bestrq import SSLTask


def get_parser():
    parser = SSLTask.get_parser()
    return parser


def main(cmd=None):
    r"""ASR training.

    Example:

        % python asr_train.py asr --print_config --optim adadelta \
                > conf/train_asr.yaml
        % python asr_train.py --config conf/train_asr.yaml
    """
    SSLTask.main(cmd=cmd)


if __name__ == "__main__":
    main()

#!/usr/bin/env python

# Author: Yifan Peng (Carnegie Mellon University)


import argparse
import os
import pathlib

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="List all book paths and split them"
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path to the dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to write the outputs"
    )
    parser.add_argument(
        "--num_outputs",
        type=int,
        required=True,
        help="Number of output files"
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    list_dir = pathlib.Path(args.root).glob('*/*')
    list_dir = [str(x.resolve()) for x in list_dir if x.is_dir()]  # all books

    assert len(list_dir) >= args.num_outputs

    new_lists = np.array_split(np.array(list_dir), args.num_outputs)
    for idx, books in enumerate(new_lists):
        with open(os.path.join(args.output_dir, f"book_path.{idx+1}"), 'w') as fp:
            for line in books:
                fp.write(line + '\n')

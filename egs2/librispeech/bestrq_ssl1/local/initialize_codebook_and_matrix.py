# -*- coding: utf-8 -*-

import argparse
import os
import torch


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100000, help="random seed.")
    parser.add_argument("--codebook_size", type=int, default=8192, help="codebook size.")
    parser.add_argument("--vector_size", type=int, default=16, help="projection dimension.")
    parser.add_argument("--input_dim", type=int, default=320, help="input dimension.")
    parser.add_argument("--output_dir", type=str, default="data/local", help="output directory for initialization.")

    return parser

def main(args):
    torch.manual_seed(args.seed)

    projection_matrix = torch.zeros(args.input_dim, args.vector_size)
    codebook = torch.zeros(args.codebook_size, args.vector_size)

    torch.nn.init.xavier_uniform_(projection_matrix)
    torch.nn.init.normal_(codebook)

    projection_matrix = torch.nn.functional.normalize(
        projection_matrix, p=2, dim=1
    )
    codebook = torch.nn.functional.normalize(
        codebook, p=2, dim=1
    )

    stats = dict(
        codebook = codebook,
        projection_matrix = projection_matrix,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(stats, f"{args.output_dir}/codebook_and_matrix_seed{args.seed}.pth")


if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)

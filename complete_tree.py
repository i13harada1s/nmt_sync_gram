import argparse
import itertools
import sys

from src import tree as treebank


def main():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        "-i",
        nargs="?",
        type=argparse.FileType("r"),
        default=sys.stdin,
        help="",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        nargs="?",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="",
    )
    parser.add_argument(
        "--text-path", 
        "-t", 
        type=str, 
        help="",
    )
    args = parser.parse_args()
    # fmt: on

    with args.input_path as i, args.output_path as o, open(args.text_path, "r") as t:
        for text_line, tree_line in zip(t, i):

            tkns = text_line.rstrip().split()
            tree = treebank.Tree.from_string(tree_line)

            assert len(tkns) == len(list(tree.terminals()))

            tree = tree.apply_terminals(lambda _, c=itertools.count(): tkns[next(c)])

            o.write(tree.to_string() + "\n")


if __name__ == "__main__":
    main()

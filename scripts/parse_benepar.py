import os
import re
import sys
import argparse

import nltk
import spacy
import benepar

from nltk.tree import Tree
from functools import reduce


SUPPORT_LANGS = ["en", "de"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", "-l", required=True, type=str, choices=SUPPORT_LANGS, help="target language.")
    #TODO: make properties file path
    # parser.add_argument("--peroperties-path", help="input file")
    parser.add_argument("--input", "-i", required=True, help="input file")
    parser.add_argument("--output", "-o", required=True, help="output file")
    parser.add_argument("--remove-root", action="store_true", help="")
    parser.add_argument("--sentence-segment", type=str, default='\n', help="split sentences")
    args = parser.parse_args()

    parser = benepar.Parser("benepar_en3")
        
    with open(args.input, 'r', encoding='utf8') as reader, \
         open(args.output, 'w', encoding='utf8') as writer:
        for line in reader:
            #TODO: how to get binary tree?
            input_sentence = benepar.InputSentence(words=line.split())
            tree = parser.parse(input_sentence)
            # print(tree_str)
            # tree = Tree.fromstring(tree_str)
            tree = binarize(tree)
            # tree.pretty_print()
            
if __name__ == "__main__":
    main()

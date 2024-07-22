import re
import typing

import numpy as np
from nltk.tree import Tree

from src import tree as treebank

def process_tree_string(tree_string: str):
    return re.sub("[ |\n]+", " ", tree_string)


def build_tree_from_distance(
    distance: np.ndarray,
    sentence: typing.Union[typing.List[str], np.ndarray],
    dummy_label: str = "NONE",
):
    """
    Builds a binary tree from a syntactic distance.
    """
    assert len(distance) == len(sentence)

    if len(distance) == 1:
        tree = Tree(dummy_label, [sentence[0]])
    else:
        idx_max = np.argmax(distance)
        tree = []
        if len(sentence[:idx_max]) > 0:
            tree = Tree(
                dummy_label,
                build_tree_from_distance(distance[:idx_max], sentence[:idx_max], dummy_label),
            )
        tmp = Tree(dummy_label, [sentence[idx_max]])
        if len(sentence[idx_max + 1 :]) > 0:
            tmp = Tree(
                dummy_label,
                [
                    tmp,
                    build_tree_from_distance(
                        distance[idx_max + 1 :], sentence[idx_max + 1 :], dummy_label
                    ),
                ],
            )
        if tree == []:
            tree = Tree(dummy_label, tmp)
        else:
            tree = Tree(dummy_label, [tree, tmp])

    return tree

def build_tree_from_distance_orig(
    distance: np.ndarray,
    sentence: typing.Union[typing.List[str], np.ndarray],
    dummy_label: str = "NONE",
):
    """
    Builds a binary tree from a syntactic distance.
    """
    assert len(distance) == len(sentence)

    if len(distance) == 1:
        tree = Tree(dummy_label, tuple([sentence[0]]))
    else:
        idx_max = np.argmax(distance)
        tree = []
        if len(sentence[:idx_max]) > 0:
            tree = treebank.Tree(
                dummy_label,
                tuple([
                    build_tree_from_distance(
                        distance[:idx_max], sentence[:idx_max], dummy_label
                    )
                ]),
            )
        tmp = treebank.Tree(dummy_label, tuple([sentence[idx_max]]))
        if len(sentence[idx_max + 1 :]) > 0:
            tmp = treebank.Tree(
                dummy_label,
                tuple([
                    tmp,
                    build_tree_from_distance(
                        distance[idx_max + 1 :], sentence[idx_max + 1 :], dummy_label
                    ),
                ]),
            )
        if tree:
            tree = treebank.Tree(dummy_label, tuple([tmp]))
        else:
            tree = treebank.Tree(dummy_label, tuple([tree, tmp]))

    return tree

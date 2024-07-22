from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, List, Literal, Optional, Tuple, Union


@dataclass(frozen=True)
class Tree:
    """Dataclass for a tree used for natural language parsing.

    Attributes:
        label (str): label of a node, e.g. NP.
        children (Tuple[Union[Tree, str], ...]): children of a node. a child is a Tree (non-terminal node) or str (terminal node).
        is_preterminal (bool): whether or not the node has a terminal child.
    """

    label: str
    children: Union[Tuple[Union[Tree, str], ...], List[Union[Tree, str]]]

    def __post_init__(self):
        if not isinstance(self.children, tuple):
            if isinstance(self.children, List):
                object.__setattr__(self, "children", tuple(self.children))
            else:
                raise TypeError("children must be a tuple")

        is_preterminal = any(isinstance(child, str) for child in self.children)
        if is_preterminal and len(self) != 1:
            raise ValueError("preterminal must consist of one terminal")
        if not self.children:
            raise ValueError("tree must contain at least one node")

    def __len__(self):
        return len(self.children)

    def __iter__(self):
        return iter(self.children)

    def __contains__(self, type_):
        for child in self.children:
            if isinstance(child, type_):
                return True
        return False

    @property
    def is_preterminal(self):
        return str in self

    @classmethod
    def from_string(cls, string: str, empty_root_label: Optional[str] = None) -> Tree:
        """Creating Tree from a string representation of a constituency tree.
        Args:
            s (str): a string representation of a constituency tree.
        Returns:
            Tree: Tree representing the input constituency tree.
        Examples:
            >>> t = Tree.from_string("(A (B b) (C c) (DE (D d) (E e)))")
        """
        return next(
            parse_trees(string, empty_root_label=empty_root_label, check_exact_one=True)
        )

    def to_string(self) -> str:
        """Formatting Tree into a string representation of a constituency tree.
        Returns:
            str: a string representation of a constituency tree.
        Examples:
            >>> t = Tree("A", [Tree("B", ["b"]), Tree("C", ["c"])])
            >>> print(t.to_string())
            (A (B b) (C c))
        """
        body = " ".join(
            child.to_string() if isinstance(child, Tree) else child
            for child in self.children
        )
        return f"({self.label} {body})"

    def traverse(self) -> Iterator[Tree]:
        """Iterator of the (Tree, Span) for a node.
        Yields:
            Iterator[Tuple[Tree, Span]]: Tuple of Tree and Span for a node.
        """

        def _traverse(tree):
            for child in tree.children:
                if isinstance(child, str):
                    break
                yield from _traverse(child)
            yield tree

        yield from _traverse(self)

    def terminals(self) -> Iterator[str]:
        """Yields terminals of symbols of the tree."""
        for child in self.children:
            if isinstance(child, Tree):
                yield from child.terminals()
            else:
                yield child

    def preterminals(self) -> Iterator[str]:
        """Yields preterminals of symbols of the tree."""
        for child in self.children:
            if isinstance(child, Tree):
                yield from child.preterminals()
            else:
                yield self.label
                break

    def nonterminals(self) -> Iterator[str]:
        """Yields nonterminals of symbols of the tree."""
        if not self.is_preterminal:
            yield self.label
            for child in self.children:
                if isinstance(child, Tree):
                    yield from child.nonterminals()

    def apply_terminals(self, fn: Callable[[str], str]) -> Tree:
        """Apply a function to each terminals."""
        if self.is_preterminal:
            return Tree(self.label, tuple([fn(child) for child in self.children]))
        else:
            nodes = [child.apply_terminals(fn) for child in self.children]
            return Tree(self.label, tuple(nodes))

    def apply_preterminals(self, fn: Callable[[str], str]) -> Tree:
        """Apply a function to each preterminal labels."""
        if self.is_preterminal:
            return Tree(fn(self.label), self.children)
        else:
            nodes = [child.apply_preterminals(fn) for child in self.children]
            return Tree(self.label, tuple(nodes))

    def apply_nonterminals(self, fn: Callable[[str], str]) -> Tree:
        """Apply a function to each terminal labels."""
        if self.is_preterminal:
            return Tree(self.label, self.children)
        else:
            nodes = [child.apply_nonterminals(fn) for child in self.children]
            return Tree(fn(self.label), tuple(nodes))

    def remove_empty_node(
        self, remove_function: Callable[[Union["Tree", str]], bool]
    ) -> Tree:
        """Remove empty nodes using remove_function
        Args:
            remove_function (Callable[[Union["Tree", str]], bool]):
                function judging whether node should be removed. If the
                `remove_function` returns `True`, the input node will be
                removed.
        Returns:
            Tree: Tree object of which empty nodes are removed
        """
        if self.is_preterminal:
            return Tree(self.label, self.children)
        else:
            nodes = [
                child.remove_empty_node(remove_function)
                for child in self.children
                if not remove_function(child)
            ]
            return Tree(self.label, tuple(nodes))

    def remove_empty_preterminal(self, empty_pt: str = "-NONE-") -> Tree:
        """Remove subtrees that have empty preterminals e.g. '-NONE-'
        Args:
            empty_pt (str): label representing empty preterminals
        Returns:
            Tree: tree whose empty preterminals are removed.
        Examples:
            >>> t = Tree.from_string("(A (B (-NONE- *b*) (C c)) (D d))")
            >>> t = t.remove_empty_preterminal("-NONE-")
            >>> print(t.to_string())
            (A (B (C c)) (D d))
        """
        # remove descendent nodes with only empty_pt

        def _should_remove_pt(node):
            # just removing (-NONE- *) is not sufficient.
            # For example, in a case like (VP (NP (-NONE- *-1)) (PP ...)),
            # it is required to remove NP, not -NONE-.
            if isinstance(node, str):
                return False
            return all([pt == empty_pt for pt in node.preterminals()])

        return self.remove_empty_node(_should_remove_pt)

    def remove_empty_terminal(self, empty_t_expr: List[str]) -> Tree:
        """Remove subtrees that have empty terminals detected with the
        regular expression.`
        Args:
            empty_t_expr (List[str]): regular expression detecting empty terminals
        Returns:
            Tree: tree whose empty terminals are removed.
        Examples:
            >>> t = Tree.from_string("(A (B *pro*) (C c) (D *))")
            >>> t = t.remove_empty_terminal(["\*", "\*.*\*"])
            >>> print(t.to_string())
            (A (C c))
        """
        # remove descendent nodes with only empty_t

        def _is_empty(terminal):
            return any([re.match(e, terminal) for e in empty_t_expr])

        def _should_remove_t(node):
            if isinstance(node, str):
                return False
            return all([_is_empty(terminal) for terminal in node.terminals()])

        return self.remove_empty_node(_should_remove_t)

    def merge_unaries(
        self,
        merge_preterminal: bool = False,
        merge_root: bool = False,
        separator: str = "+",
    ) -> Tree:
        """Merging unarie subtrees with a single child.
        Args:
            merge_preterminal (bool): `False` will not merge the preterminal nodes, by default `False`.
            merge_root (bool): `False` will not merge the root node, by default `False`.
            separetor (str): a symbol used for separating merged node labels.
        Examples:
            >>> t = Tree.from_string("(S (A (B (C (D foo)))))")
            >>> print(t.merge_unaries(merge_preterminal=False, merge_root=False).to_string())
            (S (A+B+C (D foo)))
            >>> print(t.merge_unaries(merge_preterminal=True, merge_root=False).to_string())
            (S (A+B+C+D foo))
            >>> print(t.merge_unaries(merge_preterminal=False, merge_root=True).to_string())
            (S+A+B+C (D foo))
            >>> print(t.merge_unaries(merge_preterminal=True, merge_root=True).to_string())
            (S+A+B+C+D foo)
        """
        if not merge_root:
            children = [
                child.merge_unaries(merge_preterminal, True, separator)
                for child in self.children
            ]
            return Tree(self.label, tuple(children))

        if self.is_preterminal:
            return Tree(self.label, self.children)
        else:
            if len(self.children) == 1:
                if self.children[0].is_preterminal and not merge_preterminal:
                    return Tree(self.label, self.children)
                new_label = self.label + separator + self.children[0].label
                new_tree = Tree(new_label, self.children[0].children)
                return new_tree.merge_unaries(merge_preterminal, True, separator)
            else:
                children = [
                    child.merge_unaries(merge_preterminal, True, separator)
                    for child in self.children
                ]
                return Tree(self.label, tuple(children))

    def split_unaries(self, separator="+") -> Tree:
        """Splitting merged unaries into the respective subtrees.
        Args:
            separator (str): a symbol used for merging unary subtrees (merge_unaries()).
        Examples:
            >>> t = Tree.from_string("(S (A (B (C (D foo)))))")
            >>> t = t.merge_unaries(merge_preterminal=True, merge_root=True)
            >>> print(t.to_string())
            (S+A+B+C+D foo)
            >>> t = t.split_unaries()
            >>> print(t.to_string())
            (S (A (B (C (D foo)))))
        """
        if separator in self.label:
            parent_label, child_label = self.label.split(separator, 1)
            child = Tree(child_label, self.children)
            parent = Tree(parent_label, (child,))
            children = [
                child.split_unaries(separator) if not isinstance(child, str) else child
                for child in parent.children
            ]
            return Tree(parent.label, tuple(children))
        else:
            children = [
                child.split_unaries(separator) if not isinstance(child, str) else child
                for child in self.children
            ]
            return Tree(self.label, tuple(children))

    def binarize(
        self,
        factor: Literal["left", "right"] = "left",
        child_num: int = 0,
        l_mark: str = "_",
        l_sep: str = "|",
        l_child_sep: str = "-",
        l_brackets: str = "<>",
    ) -> Tree:
        """Binarize the tree. There are two factorization methods, and labeling method is specified with child_num.
        Args:
            factor (str): The order of factorization. Only "left" and "right" are valid. Defaults to "right".
            child_num (int): Number of child labels for newly-introduced label. Defaults to 0. eg: 0 -> _A; 2 -> _A|<B-C>
            l_mark (str): Marker for newly-introduced label. Defaults to "_".
            l_sep (str): Separate symbol for label between parent and children. Defaults to "|".
            l_child_sep (str): Separate symbol for children. Defaults to "-".
            l_brackets (str): Brackets for describing children. Defaults to "<>".
        Returns:
            Tree: binarized tree.
        """

        def _get_factorized_label(parent, children):
            actual_child_num = min(child_num, len(children))

            if parent.label.startswith(l_mark):
                label_base = parent.label
            else:
                label_base = l_mark + parent.label

            if actual_child_num == 0:
                return label_base
            else:
                children_label = l_child_sep.join(
                    child.label for child in children[:actual_child_num]
                )
                return (
                    label_base + l_sep + l_brackets[0] + children_label + l_brackets[1]
                )

        if self.is_preterminal:
            return Tree(self.label, self.children)
        else:
            children = list(self.children)
            if len(self.children) >= 3:
                if factor == "right":
                    left_node = self.children[0]
                    right_node = Tree(
                        label=_get_factorized_label(self, self.children[1:]),
                        children=self.children[1:],
                    )
                elif factor == "left":
                    left_node = Tree(
                        label=_get_factorized_label(self, self.children[:-1]),
                        children=self.children[:-1],
                    )
                    right_node = self.children[-1]
                children = [left_node, right_node]
            nodes = [
                child.binarize(
                    factor, child_num, l_mark, l_sep, l_child_sep, l_brackets
                )
                for child in children
            ]
            return Tree(self.label, tuple(nodes))

    def unbinarize(
        self,
        child_num: int = 0,
        l_mark: str = "_",
        l_sep: str = "|",
        l_child_sep: str = "-",
        l_brackets: str = "<>",
    ):
        """"""

        def _is_factorized_label(node):
            return node.label.startswith(l_mark)

        if self.is_preterminal:
            return Tree(self.label, self.children)
        else:
            children = list()
            find_factor = False
            for child in self.children:
                if _is_factorized_label(child):
                    find_factor = True
                    children.extend([grandchild for grandchild in child.children])
                else:
                    children.append(child)

            if find_factor:
                return Tree(self.label, tuple(children)).unbinarize()

            children = [child.unbinarize() for child in self.children]
            return Tree(self.label, tuple(children))


_B_OPEN = "("
_B_CLOSE = ")"
_LABEL_P = r"[^\s{}{}]+".format(re.escape(_B_OPEN), re.escape(_B_CLOSE))
_LEAF_P = r"[^\s{}{}]+".format(re.escape(_B_OPEN), re.escape(_B_CLOSE))
_TOKEN_P = "|".join(
    [
        r"^\s*[^\s{}{}].*$".format(re.escape(_B_OPEN), re.escape(_B_CLOSE)),
        r"{}\s*({})?".format(re.escape(_B_OPEN), _LABEL_P),
        r"({})".format(_LEAF_P),
        re.escape(_B_CLOSE),
    ]
)
_TOKEN_RE = re.compile(_TOKEN_P, re.DOTALL)  # re.DOTALL for mathcing with comments


def parse_trees(
    text: Union[Iterable[str], str],
    empty_root_label: Optional[str] = None,
    check_exact_one: bool = False,
) -> Iterator["Tree"]:
    """Parsing a string representation of a consituency tree.
    Args:
        text (Union[Iterable[str], str]): input strings representing a
            constituency tree.
        empty_root_label (str): a label for the root node when the root
            does not the label.
        check_exact_one (bool): If True, it will check whether the
            `text` represents a single tree.
    Yields:
        Tree: the root of a tree
    Examples:
        >>> t = next(parse_trees('(A (B b) (C c))'))
        >>> print(t.to_string())
        (A (B b) (C c))
        >>> t = next(parse_trees("((B b) (C c))", empty_root_label="A"))
        >>> print(t.to_string())
        (A (B b) (C c))
        >>> t = next(parse_trees("(B b) (C c)"))
        >>> print(t.to_string())
        (B b)
        >>> t = next(parse_trees("(B b) (C c)", check_exact_one=True))
        ParseError: index (0, 6): expected '<end-of-string>' but got '(C'
    """
    context: List[str] = []
    stack: List[Tuple[str, List[Union[Tree, str]]]] = []
    last_tree = None
    iter_index = 0
    for s in [text] if isinstance(text, str) else text:
        context.append(s)
        for match in _TOKEN_RE.finditer(s):
            if check_exact_one and last_tree is not None:
                _parse_error(iter_index, context, match, "<end-of-string>")
            token = match.group()

            if token[0] == _B_OPEN:
                label = token[1:].lstrip()
                if label == "":
                    if len(stack) > 0:
                        _parse_error(iter_index, context, match, "<label>")
                    label = label if empty_root_label is None else empty_root_label
                stack.append((label, []))
            elif token == _B_CLOSE:
                if len(stack) == 0:
                    _parse_error(iter_index, context, match, _B_OPEN)
                label, children = stack.pop()

                if len(children) == 0:
                    _parse_error(iter_index, context, match, "<children>")
                if len(children) > 1 and any(isinstance(c, str) for c in children):
                    _parse_error(iter_index, context, match, "<single-terminal>")

                node = Tree(label, tuple(children))
                if len(stack) > 0:
                    stack[-1][1].append(node)
                    continue

                if last_tree is not None:
                    yield last_tree
                last_tree = node
                context = context[-2:]  # Keep at most 2 context items.
            else:
                if match.start() == 0 and match.end() == len(s):  # Comment line.
                    if len(stack) > 0:  # Do not allow comment inside a tree.
                        _parse_error(iter_index, context, match, _B_OPEN)
                    continue

                if len(stack) == 0:
                    _parse_error(iter_index, context, match, _B_OPEN)
                stack[-1][1].append(token)

        iter_index += 1

    if stack:
        _parse_error(iter_index, context, None, _B_CLOSE)
    elif check_exact_one and last_tree is None:
        _parse_error(iter_index, context, None, "<tree expression>")

    if last_tree is not None:
        yield last_tree


class ParseError(Exception):
    pass


def _parse_error(
    index: int, context: List[str], match: Union[re.Match, None], expected: str
):
    token = "<end-of-string>" if match is None else match.group()
    error_index = (index, len(context[-1]) if match is None else match.start())
    error_context = "\n".join(c.rstrip() for c in context)
    msg = (
        f"index {error_index}: expected {expected!r} but got {token!r}\n"
        f"{error_context}"
    )
    raise ParseError(msg)

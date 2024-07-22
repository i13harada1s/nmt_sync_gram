from absl.testing import absltest

from src import tree as treebank


class TreeTest(absltest.TestCase):
    def testTreeIO(self):
        """Test for `Tree.from_string()` and `Tree.to_string()`"""
        test_patterns = [
            (
                "(S (NP (PRP He)) (VP (VBD had) (NP (DT an) (NN idea))) (. .))",
                "(S (NP (PRP He)) (VP (VBD had) (NP (DT an) (NN idea))) (. .))",
            ),
            (
                "(S  (NP  (PRP  He  )  ) (VP  (VBD   had  )  (NP  (DT  an)  (NN  idea  )  )  )  (.  .  )  )", # with space
                "(S (NP (PRP He)) (VP (VBD had) (NP (DT an) (NN idea))) (. .))",
            )
        ]
        for testcase, expected in test_patterns:
            tree = next(treebank.parse_trees(testcase))
            self.assertEqual(expected, tree.to_string())

    def testTerminals(self):
        test_patterns = [
            (
                "(S (NP (PRP He)) (VP (VBD had) (NP (DT an) (NN idea))) (. .))",
                ["He", "had", "an", "idea", "."],
            ),
        ]
        for testcase, expected in test_patterns:
            tree = next(treebank.parse_trees(testcase))
            self.assertEqual(expected, list(tree.terminals()))

    def testPreTerminals(self):
        test_patterns = [
            (
                "(S (NP (PRP He)) (VP (VBD had) (NP (DT an) (NN idea))) (. .))",
                ["PRP", "VBD", "DT", "NN", "."],
            ),
        ]
        for testcase, expected in test_patterns:
            tree = next(treebank.parse_trees(testcase))
            self.assertEqual(expected, list(tree.preterminals()))

    def testNonTerminals(self):
        test_patterns = [
            (
                "(S (NP (PRP He)) (VP (VBD had) (NP (DT an) (NN idea))) (. .))",
                ["S", "NP", "VP", "NP"],
            ),
        ]
        for testcase, expected in test_patterns:
            tree = next(treebank.parse_trees(testcase))
            self.assertEqual(expected, list(tree.nonterminals()))

    def testApplyTerminals(self):
        test_patterns = [
            (
                "(S-1 (NP-2 (PRP-3 HE)) (VP-4 (VBD-5 HAD) (NP-6 (DT-7 AN) (NN-8 IDEA))) (.-9 .))",
                "(S-1 (NP-2 (PRP-3 he)) (VP-4 (VBD-5 had) (NP-6 (DT-7 an) (NN-8 idea))) (.-9 .))",
                lambda terminal: terminal.lower(),
            ),
        ]
        for testcase, expected, function in test_patterns:
            tree = next(treebank.parse_trees(testcase))
            tree = tree.apply_terminals(function)
            self.assertEqual(expected, tree.to_string())

    def testApplyPreTerminals(self):
        test_patterns = [
            (
                "(S-1 (NP-2 (PRP-3 HE)) (VP-4 (VBD-5 HAD) (NP-6 (DT-7 AN) (NN-8 IDEA))) (.-9 .))",
                "(S-1 (NP-2 (PRP HE)) (VP-4 (VBD HAD) (NP-6 (DT AN) (NN IDEA))) (. .))",
                lambda preterminal: preterminal.split("-")[0],
            ),
        ]
        for testcase, expected, function in test_patterns:
            tree = next(treebank.parse_trees(testcase))
            tree = tree.apply_preterminals(function)
            self.assertEqual(expected, tree.to_string())

    def testApplNonTerminals(self):
        test_patterns = [
            (
                "(S-1 (NP-2 (PRP-3 HE)) (VP-4 (VBD-5 HAD) (NP-6 (DT-7 AN) (NN-8 IDEA))) (.-9 .))",
                "(S (NP (PRP-3 HE)) (VP (VBD-5 HAD) (NP (DT-7 AN) (NN-8 IDEA))) (.-9 .))",
                lambda nonterminal: nonterminal.split("-")[0],
            ),
        ]
        for testcase, expected, function in test_patterns:
            tree = next(treebank.parse_trees(testcase))
            tree = tree.apply_nonterminals(function)
            self.assertEqual(expected, tree.to_string())

    def testRemoveEmptyPreterminal(self):
        test_patterns = [
            (
                "(S (A-SBJ a) (B (-NONE- *b*)) (C (D-OBJ d) (-NONE- *e*)))",
                "(S (A-SBJ a) (C (D-OBJ d)))",
                "-NONE-",
            ),
        ]    
        for testcase, expected, pattern in test_patterns:
            tree = next(treebank.parse_trees(testcase))
            tree = tree.remove_empty_preterminal(empty_pt=pattern)
            self.assertEqual(expected, tree.to_string())

    def testRemoveEmptyTerminal(self):
        test_patterns = [
            (
                "(IP-MAT (NP-SBJ;{JOSUKE} *pro*) (NP-PRD (PP (NP;{JOJO4} (N 本作)) (P-ROLE の)) (N 主人公)) (AX *) (DUMMY *ICH*-20) (PU 。))",
                "(IP-MAT (NP-PRD (PP (NP;{JOJO4} (N 本作)) (P-ROLE の)) (N 主人公)) (PU 。))",
                ["\*", "\*.*\*", "\*ICH\*-\d+"],
            ),
        ]
        for testcase, expected, regex in test_patterns:
            tree = next(treebank.parse_trees(testcase))
            tree = tree.remove_empty_terminal(empty_t_expr=regex)
            self.assertEqual(expected, tree.to_string())

    def test_merge_unaries(self):
        test_patterns = [
            (
                "(S (A (B (C foo) (D bar))) (E (F baz)))",
                "(S (A+B (C foo) (D bar)) (E (F baz)))",
                [True, True, False, True, False, False],
                False,
                False,
                "+",
            ),
            (
                "(S (A (B (C foo) (D bar))) (E (F baz)))",
                "(S (A+B (C foo) (D bar)) (E+F baz))",
                [True, True, False, True, False],
                True,
                False,
                "+",
            ),
            (
                "(S (A (B (C (D foo)))))",
                "(S (A+B+C (D foo)))",
                [True, False, False],
                False,
                False,
                "+",
            ),
            (
                "(S (A (B (C (D foo)))))",
                "(S (A+B+C+D foo))",
                [True, False],
                True,
                False,
                "+",
            ),
            (
                "(S (A (B (C (D foo)))))",
                "(S+A+B+C (D foo))",
                [True, False],
                False,
                True,
                "+",
            ),
            (
                "(S (A (B (C (D foo)))))", 
                "(S+A+B+C+D foo)", 
                [True], 
                True, 
                True,
                "+",
            ),
            (
                "(S (A (B (C foo) (D bar))) (E (F baz)))",
                "(S (A-B (C foo) (D bar)) (E-F baz))",
                [True, True, False, True, False],
                True,
                False,
                "-",
            ),
        ]

        for (
            testcase,
            expected,
            expected_preterminal,
            merge_preterminal,
            merge_root,
            separator,
        ) in test_patterns:
            tree = next(treebank.parse_trees(testcase))
            tree = tree.merge_unaries(
                merge_preterminal=merge_preterminal, merge_root=merge_root, separator=separator
            )
            self.assertEqual(expected, tree.to_string())
            self.assertEqual(
                [node.is_preterminal for node in tree.traverse()],
                expected_preterminal,
            )

    def test_split_unaries(self):
        test_patterns = [
            (
                "(S (A+B (C foo) (D bar)) (E (F baz)))",
                "(S (A (B (C foo) (D bar))) (E (F baz)))",
                [True, True, False, False, True, False, False],
                "+",
            ),
            (
                "(S (A+B (C foo) (D bar)) (E+F baz))",
                "(S (A (B (C foo) (D bar))) (E (F baz)))",
                [True, True, False, False, True, False, False],
                "+",
            ),
            (
                "(S (A+B+C (D foo)))",
                "(S (A (B (C (D foo)))))",
                [True, False, False, False, False],
                "+",
            ),
            (
                "(S (A+B+C+D foo))",
                "(S (A (B (C (D foo)))))",
                [True, False, False, False, False],
                "+",
            ),
            (
                "(S+A+B+C (D foo))",
                "(S (A (B (C (D foo)))))",
                [True, False, False, False, False],
                "+",
            ),
            (
                "(S (A-B (C foo) (D bar)) (E-F baz))",
                "(S (A (B (C foo) (D bar))) (E (F baz)))",
                [True, True, False, False, True, False, False],
                "-",
            ),
            (
                "(S (A+B (C foo) (D bar)) (E+F baz))",
                "(S (A+B (C foo) (D bar)) (E+F baz))",
                [True, True, False, True, False],
                "-",
            ),
        ]
        for (testcase, expected, expected_preterminal, separator) in test_patterns:
            tree = next(treebank.parse_trees(testcase))
            tree = tree.split_unaries(separator=separator)
            self.assertEqual(expected, tree.to_string())
            self.assertEqual(
                [node.is_preterminal for node in tree.traverse()],
                expected_preterminal,
            )

    def test_binarize(self):
        test_patterns = [
            (
                "(S (A (B b) (C c) (D d)) (E e) (F f))",
                "(S (A (B b) (_A (C c) (D d))) (_S (E e) (F f)))",
                "right",
                0,
            ),
            (
                "(S (A (B b) (C c) (D d)) (E e) (F f))",
                "(S (A (B b) (_A|<C> (C c) (D d))) (_S|<E> (E e) (F f)))",
                "right",
                1,
            ),
            (
                "(S (A (B b) (C c) (D d)) (E e) (F f))",
                "(S (_S|<A-E> (A (_A|<B-C> (B b) (C c)) (D d)) (E e)) (F f))",
                "left",
                2,
            ),
            (
                "(S (A (B b) (C c) (D d)) (E e) (F f))",
                "(S (A (B b) (_A|<C-D> (C c) (D d))) (_S|<E-F> (E e) (F f)))",
                "right",
                3,
            ),
            (
                "(S (A (B b) (C c) (D d)) (E e) (F f))",
                "(S (_S|<A-E> (A (_A|<B-C> (B b) (C c)) (D d)) (E e)) (F f))",
                "left",
                3,
            ),
            (
                "(TOP (S (PP (IN At) (NP (DT this) (NN point))) (, ,) (NP (DT the) (NNP Dow)) (VP (VBD was) (ADVP (RB down) (NP (QP (RB about) (CD 35)) (NNS points)))) (. .)))",
                "(TOP (S (_S|<PP-,-NP-VP> (_S|<PP-,-NP-VP>|<PP-,-NP> (_S|<PP-,-NP-VP>|<PP-,-NP>|<PP-,> (PP (IN At) (NP (DT this) (NN point))) (, ,)) (NP (DT the) (NNP Dow))) (VP (VBD was) (ADVP (RB down) (NP (QP (RB about) (CD 35)) (NNS points))))) (. .)))",
                "left",
                5,
            ),
        ]
        for testcase, expected, factor, child_num in test_patterns:
            tree = next(treebank.parse_trees(testcase))
            tree = tree.binarize(factor=factor, child_num=child_num)
            self.assertEqual(tree.to_string(), expected)

    def test_unbinarize(self):
        test_patterns = [
            (
                "(S (A (B b) (_A (C c) (D d))) (_S (E e) (F f)))",
                "(S (A (B b) (C c) (D d)) (E e) (F f))",
            ),
            (
                "(S (A (B b) (_A|<C> (C c) (D d))) (_S|<E> (E e) (F f)))",
                "(S (A (B b) (C c) (D d)) (E e) (F f))",
            ),
            (
                "(S (_S|<A-E> (A (_A|<B-C> (B b) (C c)) (D d)) (E e)) (F f))",
                "(S (A (B b) (C c) (D d)) (E e) (F f))",
            ),
            (
                "(S (A (B b) (_A|<C-D> (C c) (D d))) (_S|<E-F> (E e) (F f)))",
                "(S (A (B b) (C c) (D d)) (E e) (F f))",
            ),
            (
                "(S (_S|<A-E> (A (_A|<B-C> (B b) (C c)) (D d)) (E e)) (F f))",
                "(S (A (B b) (C c) (D d)) (E e) (F f))",
            ),
            (
                "(TOP (S (_S|<PP-,-NP-VP> (_S|<PP-,-NP-VP>|<PP-,-NP> (_S|<PP-,-NP-VP>|<PP-,-NP>|<PP-,> (PP (IN At) (NP (DT this) (NN point))) (, ,)) (NP (DT the) (NNP Dow))) (VP (VBD was) (ADVP (RB down) (NP (QP (RB about) (CD 35)) (NNS points))))) (. .)))",
                "(TOP (S (PP (IN At) (NP (DT this) (NN point))) (, ,) (NP (DT the) (NNP Dow)) (VP (VBD was) (ADVP (RB down) (NP (QP (RB about) (CD 35)) (NNS points)))) (. .)))",
            ),
        ]
        for testcase, expected in test_patterns:
            tree = next(treebank.parse_trees(testcase))
            tree = tree.unbinarize()
            self.assertEqual(tree.to_string(), expected)


if __name__ == "__main__":
    absltest.main()

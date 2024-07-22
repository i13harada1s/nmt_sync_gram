from absl.testing import absltest

import numpy

from src.tree_utils import build_tree_from_distance, build_tree_from_distance_orig
 

class TreeUtilsTest(absltest.TestCase):
    def test_build_tree_from_distance(self):
        
        sentence = "she enjoys playing tennis .".split()
        distance = numpy.array([5, 3, 2, 1, 4])
        
        tree = build_tree_from_distance(distance=distance, sentence=sentence, dummy_label="@")
        
        expected = "(@ (@ she) (@ (@ (@ enjoys) (@ (@ playing) (@ tennis))) (@ .)))"
        
        self.assertEqual(expected, tree._pformat_flat("", "()", False))

        
    # def test_build_tree_from_distance_orig(self):
        
    #     sentence = "she enjoys playing tennis .".split()
    #     distance = numpy.array([5, 3, 2, 1, 4])
        
    #     tree = build_tree_from_distance_orig(distance=distance, sentence=sentence, dummy_label="@")
        
    #     expected = "(@ (@ she) (@ (@ (@ enjoys) (@ (@ playing) (@ tennis))) (@ .)))"
        
    #     # self.assertEqual(expected, tree._pformat_flat("", "()", False))
    #     self.assertEqual(expected, tree.to_string())

if __name__ == "__main__":
    absltest.main()

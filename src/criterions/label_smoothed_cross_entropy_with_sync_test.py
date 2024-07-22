import numpy
import torch
from absl.testing import absltest
from src.criterions.label_smoothed_cross_entropy_with_sync import hinge_loss, rank_loss


class HingeLossTest(absltest.TestCase):
    def test_loss(self):
        # https://github.com/keras-team/keras/blob/2c48a3b38b6b6139be2da501982fd2f61d7d48fe/keras/losses_test.py#L1314

        y_pred = torch.Tensor([[-0.3, 0.2, -0.1, 1.6], [-0.25, -1.0, 0.5, 0.6]])
        y_true = torch.Tensor([[-1.0, 1.0, -1.0, 1.0], [-1.0, -1.0, 1.0, 1.0]])

        #            y_true * y_pred = [[0.3, 0.2, 0.1, 1.6], [0.25, 1, 0.5, 0.6]]
        #        1 - y_true * y_pred = [[0.7, 0.8, 0.9, -0.6], [0.75, 0, 0.5, 0.4]]
        # max(0, 1 - y_true * y_pred) = [[0.7, 0.8, 0.9, 0], [0.75, 0, 0.5, 0.4]]
        #                        loss = [[0.7 + 0.8 + 0.9 + 0] / 4, [0.75 + 0 + 0.5 + 0.4] / 4]
        #                             = [0.6, 0.4125]
        #                   mean_loss = [0.6 + 0.4125] / 2

        loss = hinge_loss(y_pred, y_true, reduction="mean")

        self.assertAlmostEqual(0.506, loss.item(), 3)


class RankLossTest(absltest.TestCase):
    def test_loss(self):
        y_pred = torch.Tensor([[-0.3, 0.2, -0.1, 1.6], [-0.25, -1.0, 0.5, 0.6]])
        y_true = torch.Tensor([[-1.0, 1.0, -1.0, 1.0], [-1.0, -1.0, 1.0, 1.0]])
        
        # rank_loss = sum_(j>i){max(0, 1 - sign(y_true_i - y_true_j)(y_pred_i - y_pred_j))}
        
        # y_pred_i - y_pred_j = [
        #     [
        #         [0, -0.5, -0.2, -1.9], 
        #         [0.5, 0, 0.3, -1.4], 
        #         [0.2, -0.3, 0, -1.7], 
        #         [1.9, 1.4, 1.7, 0],
        #     ], 
        #     [
        #         [0, 0.75, -0.75, -0.85], 
        #         [-0.75, 0, -1.5, -1.6], 
        #         [0.75, 1.5, 0, -0.1], 
        #         [0.85, 1.6, 0.1, 0],
        #     ]
        # ]
        
        # y_true_i - y_true_j = [
        #     [
        #         [0, -2, 0, -2], 
        #         [2, 0, 2, 0], 
        #         [0, -2, 0, -2], 
        #         [2, 0, 2, 0],
        #     ], 
        #     [
        #         [0, 0, -2, -2],
        #         [0, 0, -2, -2],
        #         [2, 2, 0, 0], 
        #         [2, 2, 0, 0],
        #     ]
        # ]
        
        # sign(y_true_i - y_true_j) = [
        #     [
        #         [0, -1, 0, -1], 
        #         [1, 0, 1, 0], 
        #         [0, -1, 0, -1], 
        #         [1, 0, 1, 0]
        #     ], 
        #     [
        #         [0, 0, -1, -1], 
        #         [0, 0, -1, -1], 
        #         [1, 1, 0, 0], 
        #         [1, 1, 0, 0]
        #     ]
        # ]
        
        # sign(y_true_i - y_true_j)(y_pred_i - y_pred_j) = [
        # [
        #         [0, 0.5, 0, 1.9], 
        #         [0.5, 0, 0.3, 0], 
        #         [0, 0.3, 0, 1.7], 
        #         [1.9, 0, 1.7, 0],
        #     ], 
        #     [
        #         [0, 0, 0.75, 0.85], 
        #         [0, 0, 1.5, 1.6], 
        #         [0.75, 1.5, 0, 0], 
        #         [0.85, 1.6, 0, 0],
        #     ]
        # ]
        
        # 1 - sign(y_true_i - y_true_j)(y_pred_i - y_pred_j) = [
        # [
        #         [1, 0.5, 1, -0.9], 
        #         [0.5, 1, 0.7, 1], 
        #         [1, 0.7, 1, -0.7], 
        #         [-0.9, 1, -0.7, 1],
        #     ], 
        #     [
        #         [1, 1, 0.25, 0.15], 
        #         [1, 1, -0.5, -0.6], 
        #         [0.25, -0.5, 1, 1], 
        #         [0.15, -0.6, 1, 1],
        #     ]
        # ]
        
        # (j>i)max(0, 1 - sign(y_true_i - y_true_j)(y_pred_i - y_pred_j)) = [
        # [
        #         [0, 0, 0, 0], 
        #         [0.5, 0, 0, 0], 
        #         [1, 0.7, 0, 0], 
        #         [0, 1, 0, 0],
        #     ], 
        #     [
        #         [0, 0, 0, 0], 
        #         [1, 0, 0, 0], 
        #         [0.25, 0, 0, 0], 
        #         [0.15, 0, 1, 0],
        #     ]
        # ]

        # NOTE: average with the number of valid numbers
        # loss = [
        #     [0 + 0 + 0 + 0 + 0.5 + 0 + 0 + 0 + 1 + 0.7 + 0 + 0 + 0 + 1 + 0 + 0] / 6,
        #     [0 + 0 + 0 + 0 + 1 + 0 + 0 + 0 + 0.25 + 0 + 0 + 0 + 0.15 + 0 + 1 + 0] / 6
        # ]
        #  = [[3.2/6=0.533], [2.4/6=0.4]]
        #  = [(0.533+0.4)/2] = 0.4666
        
        loss = rank_loss(y_pred.unsqueeze(-1), y_true.unsqueeze(-1), reduction="mean")

        self.assertAlmostEqual(0.4666, loss.item(), 3)


if __name__ == "__main__":
    absltest.main()

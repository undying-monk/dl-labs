import unittest

import torch

from models.model_torch import wrap_yolo_loss


class IgnoreMaskLossTests(unittest.TestCase):
    def test_overlapping_non_responsible_anchor_is_excluded_from_no_object_loss(self):
        loss_fn = wrap_yolo_loss(
            loss_weight=[0, 0, 1, 0],
            anchors=torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
            stride=32,
            ignore_threshold=0.5,
        )
        predictions = torch.zeros(1, 1, 1, 2, 6)
        predictions[0, 0, 0, 1, 4] = 10.0

        targets = torch.zeros(1, 1, 1, 2, 6)
        targets[0, 0, 0, 0] = torch.tensor([0.5, 0.5, 0.0, 0.0, 1.0, 1.0])

        loss = loss_fn(predictions, targets)

        self.assertEqual(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()

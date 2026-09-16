import unittest

import torch

from models.model_torch import decode_batch_targets


class MetricTargetDecodingTests(unittest.TestCase):
    def test_decoder_preserves_overlapping_ground_truth_without_prediction_postprocessing(self):
        anchors = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        targets = torch.zeros(1, 1, 1, 2, 7)
        targets[0, 0, 0, 0] = torch.tensor([0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 1.0])
        targets[0, 0, 0, 1] = torch.tensor([0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 0.0])

        metric_targets = decode_batch_targets(targets, anchors, stride=32)

        self.assertEqual(len(metric_targets), 1)
        self.assertEqual(set(metric_targets[0]), {"boxes", "labels"})
        self.assertEqual(metric_targets[0]["boxes"].shape, (2, 4))
        self.assertTrue(
            torch.allclose(
                metric_targets[0]["boxes"],
                torch.tensor([[0.0, 0.0, 32.0, 32.0], [0.0, 0.0, 32.0, 32.0]]),
            )
        )
        self.assertTrue(torch.equal(metric_targets[0]["labels"], torch.tensor([1, 0])))


if __name__ == "__main__":
    unittest.main()

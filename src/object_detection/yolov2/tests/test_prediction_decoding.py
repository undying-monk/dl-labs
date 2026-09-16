import unittest

import numpy as np
import torch

from utils.util import decode_batch_predictions, decode_predictions


class PredictionDecodingTests(unittest.TestCase):
    def setUp(self):
        self.anchors = np.array([[1.0, 1.0]], dtype=np.float32)

    def test_batch_decoder_applies_sigmoid_to_center_offsets(self):
        predictions = torch.tensor([[[[[0.0, 0.0, 0.0, 0.0, 10.0, 10.0]]]]])

        decoded = decode_batch_predictions(
            predictions,
            self.anchors,
            stride=32,
            objectness_threshold=0.01,
            conf_threshold=0.05,
        )

        self.assertTrue(torch.allclose(decoded[0]["boxes"], torch.tensor([[0.0, 0.0, 32.0, 32.0]])))

    def test_single_decoder_applies_sigmoid_to_center_offsets(self):
        prediction = torch.tensor([[[[0.0, 0.0, 0.0, 0.0, 1.0, 1.0]]]])

        boxes, _, _ = decode_predictions(
            prediction,
            self.anchors,
            stride=32,
            conf_threshold=0.05,
        )

        self.assertTrue(torch.allclose(boxes, torch.tensor([[0.0, 0.0, 32.0, 32.0]])))


if __name__ == "__main__":
    unittest.main()

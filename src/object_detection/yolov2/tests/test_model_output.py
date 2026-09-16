import unittest

import torch
from torch import nn

from object_detection.yolov2.models.model_torch import YOLOv2


class ModelOutputTests(unittest.TestCase):
    def test_detection_projection_emits_unconstrained_output_values(self):
        model = YOLOv2(num_anchors=5, num_classes=20)
        projection = model.stage7[-1]

        self.assertIsInstance(projection, nn.Conv2d)

        with torch.no_grad():
            projection.weight.zero_()
            projection.bias.fill_(-2.0)
            output = projection(torch.zeros(2, 1024, 13, 13))

        self.assertEqual(output.shape, (2, 125, 13, 13))
        self.assertTrue(torch.equal(output, torch.full_like(output, -2.0)))


if __name__ == "__main__":
    unittest.main()

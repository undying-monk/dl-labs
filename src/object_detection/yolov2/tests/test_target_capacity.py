import unittest

import torch

from datasets.dataset import YoloV2GridTransform
from utils.util import encode_yolo_target


class TargetCapacityTests(unittest.TestCase):
    def setUp(self):
        self.anchors = torch.tensor([[1.0, 1.0]])
        self.target = {
            "boxes": torch.tensor([[1.0, 1.0, 31.0, 31.0], [2.0, 2.0, 30.0, 30.0]]),
            "labels": [0, 0],
            "image_id": "collision-example",
        }

    def test_dataset_encoder_fails_when_a_cell_exceeds_anchor_capacity(self):
        transform = YoloV2GridTransform(
            base_shape=(1, 1, 1, 6),
            anchors=self.anchors.tolist(),
            grid_width=32,
            grid_height=32,
        )

        with self.assertRaisesRegex(ValueError, "all 1 anchors are occupied"):
            transform(self.target)

    def test_utility_encoder_fails_when_a_cell_exceeds_anchor_capacity(self):
        with self.assertRaisesRegex(ValueError, "all 1 anchors are occupied"):
            encode_yolo_target(
                self.target,
                self.anchors,
                grid_width=32,
                grid_height=32,
                grid_shape=(1, 1),
                num_classes=1,
            )


if __name__ == "__main__":
    unittest.main()

import unittest

import torch

from datasets.dataset import YoloV2GridTransform
from utils.util import encode_yolo_target


class TargetEncodingAnchorAssignmentTests(unittest.TestCase):
    def setUp(self):
        self.anchors = torch.tensor([[1.0, 1.0], [3.0, 3.0]])
        self.box = torch.tensor([[100.0, 100.0, 132.0, 132.0]])
        self.target = {"boxes": self.box, "labels": [0], "image_id": "example"}

    def assert_small_anchor_is_assigned(self, encoded_target):
        self.assertEqual(encoded_target.shape, (13, 13, 2, 6))
        self.assertEqual(encoded_target[3, 3, 0, 4].item(), 1.0)
        self.assertEqual(encoded_target[3, 3, 1, 4].item(), 0.0)

    def test_dataset_grid_transform_matches_anchors_using_box_dimensions(self):
        transform = YoloV2GridTransform(
            base_shape=(13, 13, 2, 6),
            anchors=self.anchors.tolist(),
            grid_width=32,
            grid_height=32,
        )

        encoded_target = transform(self.target)

        self.assert_small_anchor_is_assigned(encoded_target)

    def test_utility_encoder_matches_anchors_using_box_dimensions(self):
        encoded_target = encode_yolo_target(
            self.target,
            self.anchors,
            grid_width=32,
            grid_height=32,
            grid_shape=(13, 13),
            num_classes=1,
        )

        self.assert_small_anchor_is_assigned(encoded_target)


if __name__ == "__main__":
    unittest.main()

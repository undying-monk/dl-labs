import unittest

import torch

from datasets.dataset import YoloV2GridTransform


class VectorizedTargetEncodingTests(unittest.TestCase):
    def test_same_cell_objects_use_distinct_anchor_slots(self):
        transform = YoloV2GridTransform(
            base_shape=(1, 1, 2, 7),
            anchors=[[1.0, 1.0], [2.0, 2.0]],
            grid_width=32,
            grid_height=32,
        )
        target = {
            "boxes": torch.tensor([[1.0, 1.0, 31.0, 31.0], [2.0, 2.0, 30.0, 30.0]]),
            "labels": [0, 1],
            "image_id": "same-cell",
        }

        encoded = transform(target)

        self.assertEqual(encoded.shape, (1, 1, 2, 7))
        self.assertEqual(encoded[..., 4].sum().item(), 2.0)
        self.assertTrue(torch.equal(encoded[0, 0, :, 5:].argmax(dim=-1), torch.tensor([0, 1])))


if __name__ == "__main__":
    unittest.main()

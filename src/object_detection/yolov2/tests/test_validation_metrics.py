import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.model_torch import test_loop, wrap_yolo_loss


class StaticYoloModel(nn.Module):
    def __init__(self, prediction):
        super().__init__()
        self.register_buffer("prediction", prediction)

    def forward(self, images):
        return self.prediction.expand(images.shape[0], -1, -1, -1, -1)


class ValidationMetricTests(unittest.TestCase):
    def test_validation_reports_detection_metrics_without_anchor_classification_metrics(self):
        anchors = np.array([[1.0, 1.0]], dtype=np.float32)
        prediction = torch.tensor([[[[[0.0, 0.0, 0.0, 0.0, 10.0, 10.0]]]]])
        targets = torch.tensor([[[[[0.5, 0.5, 0.0, 0.0, 1.0, 1.0]]]]])
        dataloader = DataLoader(TensorDataset(torch.zeros(1, 3, 32, 32), targets), batch_size=1)
        loss_fn = wrap_yolo_loss(anchors=anchors)

        history = test_loop(
            dataloader,
            StaticYoloModel(prediction),
            loss_fn,
            num_classes=1,
            anchors=anchors,
            device="cpu",
        )

        self.assertTrue({"map", "map_50", "map_75", "mar_100"}.issubset(history))
        self.assertNotIn("accuracy", history)
        self.assertNotIn("precision", history)
        self.assertNotIn("recall", history)
        self.assertNotIn("f1", history)
        self.assertEqual(history["map_50"], 1.0)


if __name__ == "__main__":
    unittest.main()

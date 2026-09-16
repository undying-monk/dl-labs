import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.model_torch import train_loop


class FixedPredictionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.prediction = nn.Parameter(torch.tensor([[[[[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]]]]))

    def forward(self, images):
        return self.prediction.expand(images.shape[0], -1, -1, -1, -1)


class TrainingMetricTests(unittest.TestCase):
    def test_correct_metric_uses_positive_object_count_as_its_denominator(self):
        images = torch.zeros(2, 3, 32, 32)
        targets = torch.zeros(2, 1, 1, 1, 7)
        targets[..., 4] = 1.0
        targets[..., 5] = 1.0
        dataloader = DataLoader(TensorDataset(images, targets), batch_size=2)
        model = FixedPredictionModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

        history = train_loop(
            dataloader,
            model,
            F.mse_loss,
            optimizer,
            batch_size=2,
            num_classes=2,
            device="cpu",
        )

        self.assertEqual(history["correct"], 1.0)


if __name__ == "__main__":
    unittest.main()

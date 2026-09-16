import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from models import model_torch


class TrainingResumeTests(unittest.TestCase):
    def test_training_saves_latest_checkpoint_and_resumes_at_the_next_epoch(self):
        train_result = {"loss": 1.0, "correct": 0.5, "accuracy": 0.5}
        validation_result = {
            "loss": 1.0,
            "map": 0.25,
            "map_50": 0.5,
            "map_75": 0.25,
            "mar_100": 0.5,
            "map_metric": {"map": torch.tensor(0.25)},
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            latest_path = Path(temporary_directory) / "latest.pth"
            best_path = Path(temporary_directory) / "best.pth"

            model = nn.Linear(1, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            with patch.object(model_torch, "train_loop", return_value=train_result), patch.object(
                model_torch, "test_loop", return_value=validation_result
            ):
                history = model_torch.train_model(
                    1,
                    model,
                    [],
                    [],
                    None,
                    optimizer,
                    None,
                    1,
                    1,
                    [],
                    "cpu",
                    checkpoint_path=latest_path,
                    best_checkpoint_path=best_path,
                )

            self.assertTrue(latest_path.exists())
            self.assertTrue(best_path.exists())
            self.assertEqual(len(history["train_loss"]), 1)

            resumed_model = nn.Linear(1, 1)
            resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=0.1)
            with patch.object(model_torch, "train_loop", return_value=train_result) as train_loop, patch.object(
                model_torch, "test_loop", return_value=validation_result
            ):
                resumed_history = model_torch.train_model(
                    2,
                    resumed_model,
                    [],
                    [],
                    None,
                    resumed_optimizer,
                    None,
                    1,
                    1,
                    [],
                    "cpu",
                    checkpoint_path=latest_path,
                    best_checkpoint_path=best_path,
                )

            self.assertEqual(train_loop.call_count, 1)
            self.assertEqual(len(resumed_history["train_loss"]), 2)
            self.assertEqual(torch.load(latest_path, weights_only=False)["epoch"], 1)


if __name__ == "__main__":
    unittest.main()

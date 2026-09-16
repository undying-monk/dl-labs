import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from models.model_torch import load_checkpoint, save_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trip_restores_model_and_training_metadata(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
        expected_weight = model.weight.detach().clone()

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "yolov2.pth"
            save_checkpoint(
                checkpoint_path,
                epoch=3,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics={"best_val_map": 0.42},
            )

            with torch.no_grad():
                model.weight.zero_()

            checkpoint = load_checkpoint(
                checkpoint_path,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                map_location="cpu",
            )

        self.assertTrue(torch.equal(model.weight, expected_weight))
        self.assertEqual(checkpoint["format_version"], 1)
        self.assertEqual(checkpoint["epoch"], 3)
        self.assertEqual(checkpoint["metrics"]["best_val_map"], 0.42)


if __name__ == "__main__":
    unittest.main()

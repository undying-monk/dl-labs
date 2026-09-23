import unittest
import torch

from segmentation.u_net.models.model_torch import UNet

class ModelOutputTests(unittest.TestCase):
    def test_model_output_values(self):
        model = UNet(num_classes=2)

        # Batch of 2 RGB images
        images = torch.randn(2, 3, 256, 256)

        with torch.no_grad():
            logits = model(images)
        masks = logits.argmax(dim=1)

        self.assertEqual(masks.shape, (2, 256, 256))

if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

from datasets.dataset import get_dataloader_kwargs


class DataLoaderConfigurationTests(unittest.TestCase):
    @patch("datasets.dataset.os.cpu_count", return_value=8)
    def test_cuda_uses_capped_worker_count_and_pinned_memory(self, _):
        kwargs = get_dataloader_kwargs("cuda")

        self.assertEqual(kwargs["num_workers"], 4)
        self.assertTrue(kwargs["pin_memory"])
        self.assertTrue(kwargs["persistent_workers"])
        self.assertEqual(kwargs["prefetch_factor"], 2)

    @patch("datasets.dataset.os.cpu_count", return_value=1)
    def test_cpu_keeps_memory_unpinned(self, _):
        kwargs = get_dataloader_kwargs("cpu")

        self.assertEqual(kwargs["num_workers"], 1)
        self.assertFalse(kwargs["pin_memory"])


if __name__ == "__main__":
    unittest.main()

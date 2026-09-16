import unittest
from pathlib import Path


class VocSplitTests(unittest.TestCase):
    def test_official_train_and_validation_splits_are_disjoint_and_cover_trainval(self):
        split_dir = Path(__file__).parents[1] / "data" / "VOC2007" / "ImageSets" / "Main"

        def read_ids(name):
            return {line.strip() for line in (split_dir / name).read_text().splitlines() if line.strip()}

        train_ids = read_ids("train.txt")
        val_ids = read_ids("val.txt")
        trainval_ids = read_ids("trainval.txt")

        self.assertTrue(train_ids.isdisjoint(val_ids))
        self.assertEqual(train_ids | val_ids, trainval_ids)


if __name__ == "__main__":
    unittest.main()

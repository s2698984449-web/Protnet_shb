import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from PIL import Image

from data_proc.baseline_data import B0DetectionDataset, build_b0_index, sample_b0_epoch
from data_proc.detection import UnifiedDetectionAdapter


class B0DataTest(unittest.TestCase):
    def test_index_keeps_meta_train_and_excludes_official_conflict_view(self):
        group = {
            "family_id": "realiad:p",
            "dataset_id": "realiad",
            "product_id": "p",
            "physical_group_id": "p_S1",
            "outer_split": "train",
            "meta_split": "train",
            "evaluation_stage": "inner_oof",
            "n_images": "2",
            "image_ids": "defect|normal",
            "image_paths": "defect.png|normal.png",
            "mask_paths": "defect-mask.png",
        }
        manifest = {
            image_id: {
                "image_id": image_id,
                "dataset_id": "realiad",
                "product_id": "p",
                "physical_sample_id": "S1",
                "normal_flag": str(normal),
                "image_path": f"{image_id}.png",
                "mask_path": "" if normal else "defect-mask.png",
                "defect_labels": "" if normal else "scratch",
            }
            for image_id, normal in [("defect", 0), ("normal", 1)]
        }
        meta_val = {**group, "meta_split": "validation", "family_id": "realiad:v"}
        split = {
            "dataset_id": "realiad",
            "group_key": "{category}_{sample_id}",
            "conflict_policy": "preserve_official_test",
            "excluded_official_train_image_ids": ["normal"],
        }
        adapter = SimpleNamespace(manifest=manifest)
        groups = [group, meta_val]
        index = build_b0_index(adapter, groups, {"realiad:p"}, split)
        self.assertEqual([row["image_id"] for row in index], ["defect"])
        with self.assertRaises(AssertionError):
            build_b0_index(adapter, groups, {"realiad:v"}, split)

        manifest["defect"]["image_path"] = "changed.png"
        with self.assertRaises(AssertionError):
            build_b0_index(adapter, groups, {"realiad:p"}, split)

    def test_sampling_covers_families_and_prefers_positive_view(self):
        index = []
        for domain in ("ds1", "ds2"):
            for family in ("a", "b"):
                for group, normal in [("mixed", 1), ("mixed", 0), ("good", 1)]:
                    index.append({
                        "dataset_id": domain,
                        "family_id": f"{domain}:{family}",
                        "product_id": family,
                        "physical_group_id": group,
                        "normal_flag": normal,
                    })

        sampling = {
            "groups_per_dataset": 4,
            "positive_group_fraction": 0.5,
            "max_views_per_group": 1,
        }
        first = sample_b0_epoch(index, seed=9, epoch=0, sampling=sampling)
        self.assertEqual(first, sample_b0_epoch(index, seed=9, epoch=0, sampling=sampling))
        self.assertEqual(len(first), len(set(first)))
        chosen = [index[i] for i in first]
        self.assertEqual(Counter(row["family_id"] for row in chosen), {
            "ds1:a": 2, "ds1:b": 2, "ds2:a": 2, "ds2:b": 2,
        })
        self.assertTrue(all(
            bool(row["normal_flag"]) == (row["physical_group_id"] == "good")
            for row in chosen
        ))

    def test_tiny_defect_does_not_become_negative_training_example(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            mask = Image.new("L", (8, 8))
            mask.putpixel((2, 2), 255)
            mask.save(root / "tiny.png")
            adapter = UnifiedDetectionAdapter.__new__(UnifiedDetectionAdapter)
            adapter.config = {"classes": {"defect": 1}}
            adapter.dataset_roots = {"mvtec": root}
            adapter.mask_config = {
                "connectivity": 8,
                "min_component_area": 10,
                "thresholds": {"mvtec": "nonzero"},
            }
            row = {
                "image_id": "tiny",
                "dataset_id": "mvtec",
                "normal_flag": 0,
                "image_path": "not_opened.png",
                "mask_path": "tiny.png",
                "defect_labels": "scratch",
            }
            dataset = B0DetectionDataset(adapter, [row])
            self.assertEqual(dataset[0], (None, {"image_id": "tiny", "dataset_id": "mvtec"}))

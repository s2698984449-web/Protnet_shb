"""数据分组配置决定训练 Episode 来源。"""

from __future__ import annotations

import unittest

from data_proc.fomaml_episodes import filter_family_ids, load_fomaml_data_config, load_training_episodes


class FomamlDataProfileTest(unittest.TestCase):
    def test_active_profile_is_mvtec_mode(self):
        profile = load_fomaml_data_config()
        self.assertEqual(profile["profile"], "mvtec_mode_v1")
        self.assertEqual(profile["episode_source"], "mode_manifest")
        self.assertEqual(profile["dataset_ids"], ["mvtec"])
        self.assertFalse(profile["replaces_three_dataset_protocol"])

    def test_family_filter_keeps_only_configured_datasets(self):
        families = ["mvtec:bottle", "visa:candle", "realiad:pcb", "mvtec:cable"]
        self.assertEqual(
            filter_family_ids(families, {"dataset_ids": ["mvtec"]}),
            ["mvtec:bottle", "mvtec:cable"],
        )
        self.assertEqual(filter_family_ids(families, {"dataset_ids": None}), families)

    def test_mvtec_mode_loads_fold0_source_families_only(self):
        from common.io import load_verified_research_splits

        _outer, _meta, inner = load_verified_research_splits()
        fold = next(item for item in inner["folds"] if item["fold_id"] == 0)
        adapter, episodes, scope = load_training_episodes(0, fold["theta0_train_family_ids"])
        self.assertIsNotNone(adapter)
        self.assertEqual(scope["data_profile"], "mvtec_mode_v1")
        self.assertEqual(scope["episode_source"], "mode_manifest")
        self.assertEqual(len(scope["train_family_ids"]), 8)
        self.assertTrue(all(family.startswith("mvtec:") for family in scope["train_family_ids"]))
        self.assertTrue(all(item["dataset_id"] == "mvtec" for item in episodes))
        self.assertTrue(all(item["episode_id"].startswith("mode:") for item in episodes))
        self.assertTrue(all(item["family_id"] in scope["train_family_ids"] for item in episodes))
        self.assertFalse(
            set(scope["train_family_ids"]) & set(scope["oof_target_family_ids"])
        )
        sample = episodes[0]
        self.assertIn(sample["support_mode"], sample["episode_classes"])
        self.assertGreaterEqual(len(sample["support"]), 1)
        self.assertGreaterEqual(len(sample["query_defect"]), 1)
        self.assertIn("reference", sample)
        # Query 异模式缺陷也必须进入正样本类别，不能只留 Support 模式。
        self.assertGreaterEqual(len(sample["episode_classes"]), 1)

    def test_three_dataset_profile_still_loads_frozen_oof(self):
        from common.io import load_verified_research_splits

        _outer, _meta, inner = load_verified_research_splits()
        fold = next(item for item in inner["folds"] if item["fold_id"] == 0)
        _adapter, episodes, scope = load_training_episodes(
            0, fold["theta0_train_family_ids"], profile="three_dataset_oof_v1",
        )
        self.assertEqual(scope["data_profile"], "three_dataset_oof_v1")
        self.assertEqual(len(scope["train_family_ids"]), 27)
        self.assertTrue(any(item["episode_id"].startswith("oof:") for item in episodes))
        self.assertFalse(any(item["episode_id"].startswith("mode:") for item in episodes))


if __name__ == "__main__":
    unittest.main()

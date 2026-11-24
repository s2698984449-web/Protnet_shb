"""缺陷模式感知采样。只用人工物理组，不改冻结划分。"""

from __future__ import annotations

import unittest

from data_proc.mode_episodes import build_mode_episodes


def _protocol(**overrides):
    protocol = {
        "version": "mode_v1",
        "scope": "controlled_validation",
        "seed": 7,
        "same_query_ratio": 0.6,
        "other_query_ratio": 0.4,
        "k_shot": 5,
        "q_defect_groups": 5,
        "n_reference_groups": 5,
        "q_normal_groups": 5,
        "max_episodes_per_mode": 1,
    }
    protocol.update(overrides)
    return protocol


def _row(product, group_id, role, labels, image_id, dataset="mvtec"):
    return {
        "dataset_id": dataset,
        "family_id": f"{dataset}:{product}",
        "product_id": product,
        "physical_group_id": group_id,
        "episode_pool_role": role,
        "defect_labels": labels,
        "image_ids": image_id,
    }


def _pool(product, role, label, count, prefix):
    return [
        _row(product, f"{prefix}-{index}", role, label, f"img-{prefix}-{index}")
        for index in range(count)
    ]


def _balanced_rows():
    rows = []
    for label, mark in (("broken_large", "a"), ("broken_small", "b"), ("contamination", "c")):
        rows.extend(_pool("bottle", "SUPPORT", label, 5, f"s-{mark}"))
        rows.extend(_pool("bottle", "QUERY_DEFECT", label, 5, f"q-{mark}"))
    rows.extend(_pool("bottle", "R", "", 15, "r"))
    rows.extend(_pool("bottle", "QUERY_NORMAL", "", 15, "n"))
    return rows


class ModeEpisodeTest(unittest.TestCase):
    def test_query_defect_split_is_three_same_and_two_other(self):
        episodes, skipped = build_mode_episodes(_balanced_rows(), _protocol(), dataset_id="mvtec")
        self.assertEqual(skipped, [])
        self.assertEqual([item["support_mode"] for item in episodes], ["broken_large", "broken_small", "contamination"])
        for episode in episodes:
            self.assertEqual(len(episode["support_group_ids"]), 5)
            self.assertEqual(episode["query_defect_modes"].count(episode["support_mode"]), 3)
            self.assertEqual(len(episode["query_defect_modes"]) - 3, 2)
            self.assertAlmostEqual(episode["actual_same_ratio"], 0.6)
            self.assertAlmostEqual(episode["actual_other_ratio"], 0.4)
            self.assertFalse(episode["degraded"])
            others = [label for label in episode["query_defect_modes"] if label != episode["support_mode"]]
            self.assertEqual(len(set(others)), 2)

    def test_single_mode_degrades_without_padding(self):
        rows = []
        rows.extend(_pool("tile", "SUPPORT", "scratch", 6, "s"))
        rows.extend(_pool("tile", "QUERY_DEFECT", "scratch", 6, "q"))
        rows.extend(_pool("tile", "R", "", 6, "r"))
        rows.extend(_pool("tile", "QUERY_NORMAL", "", 6, "n"))
        episodes, _skipped = build_mode_episodes(rows, _protocol(), dataset_id="mvtec")
        self.assertEqual(len(episodes), 1)
        episode = episodes[0]
        self.assertEqual(episode["query_defect_modes"], ["scratch", "scratch", "scratch"])
        self.assertEqual(episode["actual_other_ratio"], 0.0)
        self.assertIn("single_mode", episode["degradation_reason"])
        self.assertEqual(len(set(episode["query_defect_group_ids"])), 3)

    def test_short_support_does_not_duplicate_groups(self):
        rows = []
        rows.extend(_pool("cable", "SUPPORT", "hole", 2, "s"))
        rows.extend(_pool("cable", "QUERY_DEFECT", "hole", 4, "qh"))
        rows.extend(_pool("cable", "QUERY_DEFECT", "cut", 4, "qc"))
        rows.extend(_pool("cable", "R", "", 6, "r"))
        rows.extend(_pool("cable", "QUERY_NORMAL", "", 6, "n"))
        rows.extend(_pool("pill", "QUERY_DEFECT", "crack", 3, "pq"))
        episodes, skipped = build_mode_episodes(rows, _protocol(max_episodes_per_mode=2), dataset_id="mvtec")
        self.assertEqual(len(episodes), 1)
        episode = episodes[0]
        self.assertEqual(len(episode["support_group_ids"]), 2)
        self.assertIn("insufficient_support", episode["degradation_reason"])
        self.assertTrue(any(item["reason"] == "no_support_mode" for item in skipped))
        self.assertTrue(any(item.get("reason") == "insufficient_support" and item.get("available_support") == 0 for item in skipped))
        ids = (
            episode["support_group_ids"] + episode["query_defect_group_ids"]
            + episode["query_normal_group_ids"] + episode["reference_group_ids"]
        )
        self.assertEqual(len(ids), len(set(ids)))

    def test_support_query_and_reference_stay_in_their_role_pools(self):
        rows = _balanced_rows()
        rows.append(_row("bottle", "query-only-large", "QUERY_DEFECT", "broken_large", "img-query-only"))
        episodes, _skipped = build_mode_episodes(rows, _protocol(), dataset_id="mvtec")
        by_role = {}
        for row in rows:
            by_role.setdefault(row["episode_pool_role"], set()).add(row["physical_group_id"])
        for episode in episodes:
            support = set(episode["support_group_ids"])
            query = set(episode["query_defect_group_ids"]) | set(episode["query_normal_group_ids"])
            reference = set(episode["reference_group_ids"])
            self.assertFalse(support & query or support & reference or query & reference)
            self.assertTrue(support <= by_role["SUPPORT"])
            self.assertTrue(set(episode["query_defect_group_ids"]) <= by_role["QUERY_DEFECT"])
            self.assertTrue(set(episode["query_normal_group_ids"]) <= by_role["QUERY_NORMAL"])
            self.assertTrue(reference <= by_role["R"])
            self.assertNotIn("query-only-large", support)
            images = [
                image_id
                for key in ("support", "query_defect", "query_normal", "reference")
                for unit in episode[key]
                for image_id in unit["image_ids"]
            ]
            self.assertEqual(len(images), len(set(images)))

    def test_same_seed_rebuilds_the_same_episodes(self):
        rows = _balanced_rows()
        rows.extend(_row("visa-product", f"v-{index}", "SUPPORT", "anomaly", f"vimg-{index}", dataset="visa") for index in range(6))
        first, first_skipped = build_mode_episodes(rows, _protocol(), dataset_id="mvtec")
        second, second_skipped = build_mode_episodes(rows, _protocol(), dataset_id="mvtec")
        self.assertEqual(first, second)
        self.assertEqual(first_skipped, second_skipped)
        self.assertTrue(all(item["dataset_id"] == "mvtec" for item in first))
        again = build_mode_episodes(rows, _protocol(seed=8), dataset_id="mvtec")[0]
        self.assertNotEqual(first, again)


if __name__ == "__main__":
    unittest.main()

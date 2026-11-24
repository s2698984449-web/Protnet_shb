from __future__ import annotations

import unittest

import torch

from training.meta_transfer import (
    ModuleSelector,
    average_descriptors,
    build_selector_episodes,
    hard_select,
    leave_one_product_out,
    oracle_source,
    rank_null_loss,
    reliability_gate,
    utility_rows_from_holdout,
)


def _config():
    return {
        "stage_a_epochs": 25,
        "stage_b_epochs": 10,
        "learning_rate": 0.08,
        "b_weight_decay": 0.0,
        "temperature_start": 1.0,
        "temperature_end": 0.5,
        "eps_loss": 0.001,
        "hidden": 16,
        "seed": 3,
        "g3_requires_positive_module_gain": True,
        "descriptor": "v0",
        "primary_stage": "after",
    }


def _episode(product, episode_id, feature_source):
    modules_a = {
        "Feature": {"utility": 1.0 if feature_source == "a" else -1.0, "ap50_delta": 0.2 if feature_source == "a" else -0.2},
        "Where": {"utility": -1.0 if feature_source == "a" else 1.0, "ap50_delta": -0.2 if feature_source == "a" else 0.2},
        "Box": {"utility": -1.0 if feature_source == "a" else 1.0, "ap50_delta": -0.2 if feature_source == "a" else 0.2},
        "Class": {"utility": -1.0 if feature_source == "a" else 1.0, "ap50_delta": -0.2 if feature_source == "a" else 0.2},
    }
    modules_b = {
        name: {"utility": -row["utility"], "ap50_delta": -row["ap50_delta"]}
        for name, row in modules_a.items()
    }
    return {
        "episode_id": episode_id,
        "product": product,
        "null_loss": 1.0,
        "target": [1.0, 0.0, 0.0, 0.0],
        "sources": [
            {"source": "a", "source_descriptor": [1.0, 0.0, 0.0, 0.0], "modules": modules_a},
            {"source": "b", "source_descriptor": [0.0, 1.0, 0.0, 0.0], "modules": modules_b},
        ],
    }


class SelectorTest(unittest.TestCase):
    def test_module_terms_sum_to_zero_and_null_stays_unmoved(self):
        torch.manual_seed(1)
        module = ModuleSelector(4, hidden=8)
        terms = module.centered_module_terms(torch.zeros(4), torch.eye(2, 4))
        self.assertTrue(torch.allclose(terms.sum(dim=-1), torch.zeros(2), atol=1e-6))
        self.assertEqual(hard_select({"a": -0.2, "b": 0.0}), "Null")
        self.assertEqual(hard_select({"b": 0.4, "a": 0.4}), "a")

    def test_missing_utility_is_not_treated_as_zero(self):
        scores = torch.tensor([0.5, -0.2], requires_grad=True)
        loss = rank_null_loss(scores, [None, -1.0])
        loss.backward()
        self.assertEqual(scores.grad[0].item(), 0.0)
        self.assertGreater(scores.grad[1].item(), 0.0)

    def test_held_out_product_can_pick_different_modules(self):
        episodes = [
            _episode("p1", "e1", "a"),
            _episode("p2", "e2", "a"),
        ]
        trained = leave_one_product_out(episodes, _config())
        choices = {
            (row["product"], row["module"]): row["choice"]
            for row in trained["variants"]["rank_cost"]["choices"]
        }
        self.assertEqual(choices[("p1", "Feature")], "a")
        self.assertEqual(choices[("p1", "Where")], "b")
        self.assertEqual(choices[("p2", "Class")], "b")
        self.assertGreater(trained["g3_module_gain"], 0.0)
        self.assertTrue(trained["module_level_claim"])
        self.assertIn("Feature:a", trained["products"]["p1"]["contributed"])

    def test_single_measurement_does_not_pass_reliability(self):
        gate = reliability_gate(1)
        self.assertFalse(gate["passed"])
        oracle = oracle_source(
            [
                {"episode_id": "e1", "source": "a", "utility": 1.0},
                {"episode_id": "e1", "source": "b", "utility": -1.0},
                {"episode_id": "e2", "source": "a", "utility": -0.5},
                {"episode_id": "e2", "source": "b", "utility": 0.2},
            ],
            {"e1"},
            {"e2"},
        )
        self.assertEqual(oracle["oracle_source"], "a")
        self.assertAlmostEqual(oracle["eval_mean"], -0.5)

    def test_disallowed_source_is_omitted_instead_of_zero(self):
        payload = {
            "arms": {"null": {"episodes": {"e1": {
                "dataset": "mvtec", "product_id": "capsule", "family_id": "mvtec:capsule",
                "source_available_mask": [1, 1, 0],
                "after": {"query_detection_loss": 0.4},
            }}}},
            "comparison": {
                "common_proposal_delta_vs_null": {"Feature": {"episodes": {"e1": {"after": {
                    "utility": 0.1, "ap50_delta": 0.02,
                }}}}},
                "delta_vs_null": {
                    module: {"episodes": {"e1": {"after": {"utility": None, "ap50_delta": 0.0}}}}
                    for module in ("Where", "Box", "Class")
                },
            },
        }
        kept = utility_rows_from_holdout(payload, "mvtec:bottle", "mvtec:bottle", ["Null", "mvtec:bottle", "mvtec:cable"])
        self.assertEqual(len(kept), 4)
        dropped = utility_rows_from_holdout(payload, "mvtec:cable", "mvtec:cable", ["Null", "mvtec:bottle", "mvtec:cable"])
        self.assertEqual(dropped, [])
        episodes = build_selector_episodes(
            kept,
            {"e1": [0.0, 1.0]},
            {"mvtec:bottle": [1.0, 0.0]},
        )
        self.assertIsNone(episodes[0]["sources"][0]["modules"]["Where"]["utility"])
        self.assertEqual(average_descriptors([torch.zeros(2), torch.ones(2)]), [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()

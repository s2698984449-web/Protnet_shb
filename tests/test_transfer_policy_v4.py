"""条件化迁移 v4：反事实效用更新选卡/Null/ρ。"""

from __future__ import annotations

import unittest

import torch

from training.experience_bank import (
    GammaController, apply_counterfactual_policy, plan_module_borrow, update_card_utilities,
)
from training.retrieval_memory import MODULES
from training.transfer_policy import (
    POLICY_KIND, counterfactual_policy_loss, decide_inject, mix_cards_from_logits,
    null_probability_from_logit, smooth_utility_features,
)


def _bank():
    layout = [{"name": "backbone.weight", "shape": [4], "offset": 0, "numel": 4}]
    unit = torch.tensor([0.0, 3.0, 0.0, 4.0])
    unit = unit / unit.norm()
    card = {
        "unit": unit, "valid": True, "low_sample": True, "coherence": 1.0,
        "member_count": 1, "families": ["mvtec:bottle"], "task_ids": ["a"],
        "utility": {"mean": 0.8, "ema": 0.8, "variance": 0.2, "count": 20, "neg_rate": 0.1},
    }
    return {
        "manifest": {"fold_id": 0, "requested_tasks_per_family": 8, "tasks": ["a"]},
        "near_duplicates": [],
        "theta0_sha256": "theta0",
        "theta_sha256": "theta",
        "preprocess_sha256": "prep",
        "split_sha256": "split",
        "experience_groups": 2,
        "clusters_requested": 2,
        "top_mass_threshold": 0.8,
        "max_top_k": 4,
        "temperature": 1.0,
        "groups": {"mvtec:bottle": 0, "mvtec:cable": 1},
        "layouts": {module: list(layout) for module in MODULES},
        "pca": {
            name: {"mean": torch.zeros(4), "components": torch.eye(4), "singular_values": torch.ones(4)}
            for name in ("Appearance", "Morphology", "Difference")
        },
        "centers": {
            name: torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
            for name in ("Appearance", "Morphology", "Difference")
        },
        "center_valid": {name: torch.tensor([True, True]) for name in ("Appearance", "Morphology", "Difference")},
        "anchors": {module: 1.0 for module in MODULES},
        "keep_ratio": {module: 1.0 for module in MODULES},
        "next_keep_ratio": {module: 1.0 for module in MODULES},
        "trim_calibration": {},
        "signatures": {name: torch.zeros(1, 4) for name in ("Appearance", "Morphology", "Difference")},
        "assignments": [],
        "cards_memory": {(module, 0, 0): dict(card) for module in MODULES},
        "card_index": {
            f"{module}/0/0": {
                "module": module, "cluster": 0, "group": 0, "valid": True,
                "utility": dict(card["utility"]), "compatibility": {},
                "families": ["mvtec:bottle"], "member_count": 1,
            }
            for module in MODULES
        },
        "directory": "synthetic",
    }


class TransferPolicyV4Test(unittest.TestCase):
    def test_policy_kind_is_v4(self):
        self.assertEqual(POLICY_KIND, "conditional_transfer_v4")
        controller = GammaController()
        self.assertEqual(controller.state_dict()["kind"], POLICY_KIND)
        with self.assertRaises(ValueError):
            controller.load_state_dict({"kind": "conditional_transfer_v3", "policy": {}})

    def test_soft_mix_keeps_null_outside_softmax(self):
        units = torch.eye(2)
        mask = torch.tensor([True, True])
        mix, weights = mix_cards_from_logits(torch.tensor([2.0, 0.0]), units, mask)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=5)
        null = null_probability_from_logit(torch.tensor(0.0))
        effective = (1.0 - float(null)) * weights
        self.assertAlmostEqual(float(effective.sum()), 1.0 - float(null), places=5)
        self.assertTrue(
            decide_inject(
                allow_borrow=True, retrieved_valid=True, null_weight=null,
                applied_l2=1.0, step=1.0, discrete_null=False, null_threshold=0.5,
            )
        )

    def test_counterfactual_positive_utility_updates_policy(self):
        bank = _bank()
        key = torch.tensor([4.0, 0.0, 0.0, 0.0])
        controller = GammaController()
        plan = plan_module_borrow(
            bank, "Backbone", key, 1, {"mvtec:bottle"}, controller, True, support_step_l2=1.0,
        )
        self.assertTrue(plan["delta_vector"].requires_grad)
        before = {name: float(param.detach().sum()) for name, param in controller.policy.policy.named_parameters()}
        result = apply_counterfactual_policy(controller, [plan], utility=0.5)
        self.assertAlmostEqual(result["utility"], 0.5)
        after = {name: float(param.detach().sum()) for name, param in controller.policy.policy.named_parameters()}
        self.assertTrue(any(after[name] != before[name] for name in before))

    def test_counterfactual_loss_sign(self):
        bank = _bank()
        key = torch.tensor([4.0, 0.0, 0.0, 0.0])
        plan = plan_module_borrow(
            bank, "Backbone", key, 1, {"mvtec:bottle"}, GammaController(), True, support_step_l2=1.0,
        )
        pos = counterfactual_policy_loss([plan], 1.0, {"Backbone": 2.0})
        neg = counterfactual_policy_loss([plan], -1.0, {"Backbone": 2.0})
        self.assertLess(float(pos.detach()), 0.0)
        self.assertGreater(float(neg.detach()), 0.0)

    def test_card_utility_uses_counterfactual_signal(self):
        bank = _bank()
        key = torch.tensor([4.0, 0.0, 0.0, 0.0])
        plan = plan_module_borrow(
            bank, "Backbone", key, 1, {"mvtec:bottle"}, GammaController(), True, support_step_l2=1.0,
        )
        before = dict(bank["card_index"]["Backbone/0/0"]["utility"])
        update_card_utilities(bank, [plan], {"Backbone": -1.0})
        after = bank["card_index"]["Backbone/0/0"]["utility"]
        self.assertNotEqual(after["count"], before["count"])
        self.assertEqual(bank["combo_utility_log"][-1]["kind"], "module_combo_counterfactual")

    def test_utility_smoothing_still_used(self):
        cold = smooth_utility_features({"mean": 1.0, "ema": 1.0, "variance": 0.0, "count": 1, "neg_rate": 0.0})
        warm = smooth_utility_features({"mean": 1.0, "ema": 1.0, "variance": 0.5, "count": 40, "neg_rate": 0.2})
        self.assertLess(float(cold[0]), float(warm[0]))


if __name__ == "__main__":
    unittest.main()

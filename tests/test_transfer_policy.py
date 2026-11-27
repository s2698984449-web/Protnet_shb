"""条件化迁移 v5.1：硬 z、模块 LOO/Add-One、任务条件 Null、force-close。"""

from __future__ import annotations

import unittest

import torch

from training.experience_bank import (
    GammaController, apply_counterfactual_policy, apply_force_close_and_replan, force_close_plan,
    plan_module_borrow, plan_modules_in_graph_order, update_module_contributions,
)
from training.retrieval_memory import MODULES
from training.transfer_policy import (
    POLICY_KIND, UPSTREAM_OF, ApUtilityHead, counterfactual_policy_loss, decide_inject,
    encode_upstream_state, hard_borrow_gate, mix_cards_from_logits, plan_decision_features,
    risk_constrained_score, smooth_utility_features,
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


def _pad_inputs(controller, *, task_key, module="Backbone", upstream=None):
    k = controller.policy.max_candidates
    sims = torch.tensor([0.9] + [0.0] * (k - 1), dtype=torch.float32)
    norms = torch.tensor([1.0] + [0.0] * (k - 1), dtype=torch.float32)
    diffs = torch.tensor([0.1] + [1.0] * (k - 1), dtype=torch.float32)
    utils = torch.zeros(k, 4)
    utils[0] = torch.tensor([0.5, 0.2, 1.0, 0.1])
    mask = torch.tensor([True] + [False] * (k - 1))
    if upstream is None:
        upstream = torch.zeros(controller.policy.policy.upstream_dim)
    return controller.policy.policy(
        task_key, sims, norms, diffs, utils, module, mask, upstream,
    )


class TransferPolicyTest(unittest.TestCase):
    def test_policy_kind_rejects_old(self):
        self.assertEqual(POLICY_KIND, "conditional_transfer_v51")
        controller = GammaController()
        self.assertEqual(controller.state_dict()["kind"], POLICY_KIND)
        with self.assertRaises(ValueError):
            controller.load_state_dict({"kind": "conditional_transfer_v5", "policy_state": {}})

    def test_hard_gate_zero_inject(self):
        z_ste, z_hard, p_null = hard_borrow_gate(torch.tensor(8.0), threshold=0.5)
        self.assertGreater(float(p_null), 0.9)
        self.assertEqual(float(z_hard), 0.0)
        self.assertEqual(float(z_ste), 0.0)
        self.assertFalse(
            decide_inject(
                allow_borrow=True, retrieved_valid=True, z_hard=float(z_hard),
                applied_l2=0.0, step=1.0,
            )
        )

    def test_hard_gate_ste_has_grad(self):
        logit = torch.tensor(0.0, requires_grad=True)
        z_ste, _, _ = hard_borrow_gate(logit, threshold=0.5)
        (z_ste * 2.0).backward()
        self.assertIsNotNone(logit.grad)

    def test_z0_add_one_utility_reaches_null_logit(self):
        bank = _bank()
        key = torch.tensor([4.0, 0.0, 0.0, 0.0])
        controller = GammaController()
        with torch.no_grad():
            controller.policy.policy.null_head.bias.fill_(8.0)
        plan = plan_module_borrow(
            bank, "Backbone", key, 1, {"mvtec:bottle"}, controller, True, support_step_l2=1.0,
        )
        self.assertFalse(plan["inject"])
        before = float(controller.policy.policy.null_head.bias.detach())
        apply_counterfactual_policy(controller, [plan], {"Backbone": 0.5})
        after = float(controller.policy.policy.null_head.bias.detach())
        self.assertLess(after, before)

    def test_no_strength_boost_and_rho_can_vanish(self):
        controller = GammaController()
        self.assertFalse(hasattr(controller.policy, "strength_boost"))
        self.assertFalse(hasattr(controller.policy, "rho_floor"))
        raw = torch.tensor(-20.0)
        self.assertLess(float(controller.policy.effective_rho(raw, "Backbone")), 1e-6)

    def test_module_utilities_not_broadcast(self):
        bank = _bank()
        key = torch.tensor([4.0, 0.0, 0.0, 0.0])
        controller = GammaController()
        with torch.no_grad():
            controller.policy.policy.null_head.bias.fill_(-8.0)
        plan = plan_module_borrow(
            bank, "Backbone", key, 1, {"mvtec:bottle"}, controller, True, support_step_l2=1.0,
        )
        before = dict(bank["card_index"]["Backbone/0/0"]["utility"])
        update_module_contributions(bank, {"Backbone": 1.0}, [plan])
        after = bank["card_index"]["Backbone/0/0"]["utility"]
        self.assertEqual(after["count"], before["count"])
        self.assertEqual(bank["module_contribution_log"][-1]["kind"], "module_loo")

    def test_counterfactual_uses_per_module_utility(self):
        bank = _bank()
        key = torch.tensor([4.0, 0.0, 0.0, 0.0])
        controller = GammaController()
        with torch.no_grad():
            controller.policy.policy.null_head.bias.fill_(-8.0)
        plan = plan_module_borrow(
            bank, "Backbone", key, 1, {"mvtec:bottle"}, controller, True, support_step_l2=1.0,
        )
        if not plan["inject"]:
            self.skipTest("合成库未注入")
        pos = counterfactual_policy_loss([plan], {"Backbone": 1.0})
        neg = counterfactual_policy_loss([plan], {"Backbone": -1.0})
        self.assertLess(float(pos.detach()), 0.0)
        self.assertGreater(float(neg.detach()), 0.0)

    def test_card_softmax_independent_of_z(self):
        units = torch.eye(2)
        mask = torch.tensor([True, True])
        _, weights = mix_cards_from_logits(torch.tensor([2.0, 0.0]), units, mask)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=5)

    def test_ap_head_and_risk_score(self):
        head = ApUtilityHead(module_count=5, hidden=32)
        features = plan_decision_features([
            {"module": module, "rho": 0.1, "z_hard": 1.0, "null_probability": 0.2, "sim_max": 0.5}
            for module in MODULES
        ])
        pred = head(features)
        self.assertIn("mu_ap50", pred)
        self.assertIsInstance(risk_constrained_score(0.02, -1.0, risk_lambda=1.0), float)

    def test_loo_rotation_covers_all_modules(self):
        controller = GammaController()
        seen = []
        for _ in range(len(MODULES)):
            seen.extend(controller.policy.next_loo_modules())
        self.assertEqual(set(seen), set(MODULES))

    def test_utility_smoothing(self):
        cold = smooth_utility_features({"mean": 1.0, "ema": 1.0, "variance": 0.0, "count": 1, "neg_rate": 0.0})
        warm = smooth_utility_features({"mean": 1.0, "ema": 1.0, "variance": 0.5, "count": 40, "neg_rate": 0.2})
        self.assertLess(float(cold[0]), float(warm[0]))

    def test_null_uses_task_key(self):
        controller = GammaController()
        key_a = torch.zeros(controller.policy.key_dim)
        key_a[0] = 1.0
        key_b = torch.zeros(controller.policy.key_dim)
        key_b[1] = 1.0
        _, null_a, _ = _pad_inputs(controller, task_key=key_a)
        _, null_b, _ = _pad_inputs(controller, task_key=key_b)
        self.assertNotAlmostEqual(float(null_a.detach()), float(null_b.detach()), places=5)

    def test_null_mask_mean_empty_candidates(self):
        controller = GammaController()
        key = torch.zeros(controller.policy.key_dim)
        key[0] = 1.0
        k = controller.policy.max_candidates
        _, null_empty, _ = controller.policy.policy(
            key, torch.zeros(k), torch.zeros(k), torch.ones(k), torch.zeros(k, 4),
            "Backbone", torch.zeros(k, dtype=torch.bool),
            torch.zeros(controller.policy.policy.upstream_dim),
        )
        self.assertTrue(torch.isfinite(null_empty))

    def test_force_close_zeros_gate_and_replans_downstream(self):
        bank = _bank()
        controller = GammaController()
        with torch.no_grad():
            controller.policy.policy.null_head.bias.fill_(-8.0)
        keys = {
            "Appearance": torch.tensor([4.0, 0.0, 0.0, 0.0]),
            "Morphology": torch.tensor([0.0, 4.0, 0.0, 0.0]),
            "Difference": torch.tensor([0.0, 0.0, 4.0, 0.0]),
        }
        step_l2 = {module: 1.0 for module in MODULES}
        plans = plan_modules_in_graph_order(
            bank, keys, 1, {"mvtec:bottle"}, controller, True, step_l2,
        )
        by_mod = {plan["module"]: plan for plan in plans}
        force_close_plan(by_mod["Backbone"])
        self.assertAlmostEqual(float(by_mod["Backbone"]["borrow_gate"].detach()), 0.0, places=5)
        closed = apply_force_close_and_replan(
            plans, ["Where"],
            bank=bank, keys=keys, excluded_group=1, allowed_families={"mvtec:bottle"},
            controller=controller, allow_borrow=True, step_l2=step_l2,
        )
        self.assertEqual(closed, ["Where"])
        after = {plan["module"]: plan for plan in plans}
        slot = controller.policy.key_dim + 2
        # Where 是 RoiFeature 上游第二槽：Backbone 槽后的 z
        self.assertAlmostEqual(float(after["RoiFeature"]["upstream_state"][2 * slot - 1]), 0.0, places=5)
        for module in MODULES:
            if "Where" in (UPSTREAM_OF.get(module) or ()):
                self.assertIsNot(after[module], by_mod[module])

    def test_force_close_prob_config(self):
        controller = GammaController()
        self.assertAlmostEqual(float(controller.policy.force_close_prob), 0.25, places=5)


if __name__ == "__main__":
    unittest.main()

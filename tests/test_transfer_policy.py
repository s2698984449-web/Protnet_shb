"""条件化迁移 v5.3：Support 对齐特征、一阶稠密标签闸门、硬 z / LOO / force-close。"""

from __future__ import annotations

import unittest

import torch

from training.experience_bank import (
    GammaController, apply_counterfactual_policy, apply_force_close_and_replan, force_close_plan,
    plan_module_borrow, plan_modules_in_graph_order, update_module_contributions,
)
from training.retrieval_memory import MODULES
from training.transfer_policy import (
    ALIGN_FEATURE_DIM, POLICY_KIND, UPSTREAM_OF, ApUtilityHead, card_pair_policy_loss,
    card_probe_policy_loss, card_support_align_features, counterfactual_policy_loss, decide_inject,
    encode_upstream_state, first_order_card_listwise_loss, first_order_gate_loss,
    first_order_gate_report, hard_borrow_gate, mix_cards_from_logits, null_sign_policy_loss,
    plan_decision_features, rho_pair_policy_loss, rho_probe_policy_loss, risk_constrained_score,
    smooth_utility_features, utility_ap_sign_agreement,
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
        self.assertEqual(POLICY_KIND, "conditional_transfer_v53")
        controller = GammaController()
        self.assertEqual(controller.state_dict()["kind"], POLICY_KIND)
        with self.assertRaises(ValueError):
            controller.load_state_dict({"kind": "conditional_transfer_v52", "policy_state": {}})

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
        _, sharp = mix_cards_from_logits(torch.tensor([2.0, 0.0]), units, mask, temperature=0.5)
        self.assertGreater(float(sharp[0]), float(weights[0]))

    def test_null_sign_pushes_past_margin(self):
        logit = torch.tensor(0.0, requires_grad=True)
        plan = {"null_logit": logit}
        refuse = null_sign_policy_loss(plan, -0.1, refuse_weight=2.0, accept_weight=1.0, margin=0.5)
        refuse.backward()
        # 最小化：负梯度抬高 null_logit → Null
        self.assertLess(float(logit.grad), 0.0)
        logit2 = torch.tensor(0.0, requires_grad=True)
        accept = null_sign_policy_loss(
            {"null_logit": logit2}, 0.1, refuse_weight=2.0, accept_weight=1.0, margin=0.5,
        )
        accept.backward()
        # 正梯度压低 null_logit → Borrow
        self.assertGreater(float(logit2.grad), 0.0)

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
        self.assertEqual(controller.policy.rho_probes, [0.1, 0.2, 0.3])
        self.assertTrue(controller.policy.rho_pairwise)
        self.assertTrue(controller.policy.card_probe)
        self.assertTrue(controller.policy.card_pairwise)
        self.assertTrue(controller.policy.probe_alternate)
        self.assertAlmostEqual(float(controller.policy.rho_deadzone_rel), 0.05, places=5)
        self.assertAlmostEqual(float(controller.policy.null_sign_refuse_weight), 2.0, places=5)
        self.assertAlmostEqual(float(controller.policy.null_sign_accept_weight), 1.0, places=5)
        self.assertAlmostEqual(float(controller.policy.null_sign_margin), 0.5, places=5)
        self.assertAlmostEqual(float(controller.policy.card_temperature), 0.5, places=5)
        self.assertAlmostEqual(float(controller.policy.card_pair_loss_weight), 2.0, places=5)
        self.assertTrue(controller.policy.fo_enabled)
        self.assertTrue(controller.policy.fo_feed_policy)
        self.assertFalse(controller.policy.fo_supervise)
        self.assertAlmostEqual(float(controller.policy.fo_null_sign_scale), 0.0, places=5)
        self.assertEqual(int(controller.policy.policy.align_dim), ALIGN_FEATURE_DIM)
        self.assertFalse(controller.policy.ap_utility_enabled)
        self.assertEqual(int(controller.policy.ap_utility_every_n_steps), 0)
        self.assertEqual(
            controller.policy.rho_probe_pairs(),
            [(0.1, 0.2), (0.2, 0.3), (0.1, 0.3)],
        )

    def test_align_features_orthogonal_component(self):
        g = torch.tensor([1.0, 0.0, 0.0, 0.0])
        d = torch.tensor([1.0, 1.0, 0.0, 0.0])
        feats = card_support_align_features(g, d)
        self.assertEqual(int(feats.numel()), ALIGN_FEATURE_DIM)
        self.assertAlmostEqual(float(feats[0]), float(1.0 / (2.0 ** 0.5)), places=5)
        self.assertAlmostEqual(float(feats[2]), float(1.0 / (2.0 ** 0.5)), places=5)

    def test_align_input_columns_zero_initialized(self):
        controller = GammaController()
        trunk = controller.policy.policy.trunk[0]
        old = int(controller.policy.policy._trunk_old_in)
        self.assertTrue(torch.all(trunk.weight[:, old:] == 0).item())
        null_in = controller.policy.policy.null_direct[0]
        null_old = int(controller.policy.policy._null_old_in)
        self.assertTrue(torch.all(null_in.weight[:, null_old:] == 0).item())

    def test_first_order_listwise_and_gate_losses(self):
        logits = torch.tensor([0.0, 0.0, 0.0, 0.0], requires_grad=True)
        null_logit = torch.tensor(0.0, requires_grad=True)
        mask = torch.tensor([True, True, False, False])
        utilities = torch.tensor([0.2, -0.1, 0.0, 0.0])
        plan = {"card_logits": logits, "null_logit": null_logit, "candidate_mask": mask}
        card_loss = first_order_card_listwise_loss(plan, utilities, tau=0.1)
        card_loss.backward(retain_graph=True)
        self.assertLess(float(logits.grad[0]), float(logits.grad[1]))
        gate_loss = first_order_gate_loss(plan, utilities, deadzone=1e-4)
        gate_loss.backward()
        # max_u>0 → 目标 borrow → −null_logit 升高 → null_logit 梯度为正被压低
        self.assertGreater(float(null_logit.grad), 0.0)

    def test_first_order_gate_report_sign_and_corr(self):
        align = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
        fo_u = torch.tensor([0.5, -0.4, 0.0, 0.0])
        plan = {
            "module": "Backbone",
            "fo_module_utility": 0.3,
            "fo_card_utilities": fo_u,
            "candidate_align": align,
            "candidate_mask": torch.tensor([True, True, False, False]),
            "candidate_align_selected": 1.0,
        }
        report = first_order_gate_report([plan], {"Backbone": 0.2})
        self.assertEqual(report["n_comparable"], 1)
        self.assertAlmostEqual(float(report["sign_agree_rate"]), 1.0, places=5)
        self.assertIsNotNone(report["modules"]["Backbone"]["corr_cos_u"])
        self.assertGreater(float(report["modules"]["Backbone"]["corr_cos_u"]), 0.9)

    def test_rho_pair_prefers_better_intensity_both_directions(self):
        bank = _bank()
        key = torch.tensor([4.0, 0.0, 0.0, 0.0])

        def _rho_after(pair_probe):
            controller = GammaController()
            with torch.no_grad():
                controller.policy.policy.null_head.bias.fill_(-8.0)
                controller.policy.policy.rho_head.bias.fill_(0.0)
            plan = plan_module_borrow(
                bank, "Backbone", key, 1, {"mvtec:bottle"}, controller, True, support_step_l2=1.0,
            )
            before = float(plan["rho"].detach())
            apply_counterfactual_policy(controller, [plan], {}, rho_probe=pair_probe)
            plan2 = plan_module_borrow(
                bank, "Backbone", key, 1, {"mvtec:bottle"}, controller, True, support_step_l2=1.0,
            )
            return before, float(plan2["rho"].detach())

        # 0.2 优于 0.1 → ρ 应上调
        before, after = _rho_after({
            "module": "Backbone", "pairwise": True,
            "rho_a": 0.1, "utility_a": 0.03, "rho_b": 0.2, "utility_b": 0.08, "eps": 0.01,
        })
        self.assertGreater(after, before)
        # 0.1 优于 0.2 → ρ 应下调（相对当前偏向更小档）
        before, after = _rho_after({
            "module": "Backbone", "pairwise": True,
            "rho_a": 0.1, "utility_a": 0.08, "rho_b": 0.2, "utility_b": 0.03, "eps": 0.01,
        })
        self.assertLess(after, before)

    def test_rho_pair_deadzone_skips_update(self):
        rho = torch.tensor(0.15, requires_grad=True)
        with self.assertRaises(ValueError):
            rho_pair_policy_loss({"rho": rho}, 0.1, 0.03, 0.2, 0.035, eps=0.01)

    def test_rho_pair_loss_has_grad_to_rho(self):
        rho = torch.tensor(0.15, requires_grad=True)
        loss = rho_pair_policy_loss({"rho": rho}, 0.1, 0.03, 0.2, 0.08, eps=0.01)
        loss.backward()
        self.assertIsNotNone(rho.grad)
        self.assertGreater(abs(float(rho.grad)), 0.0)

    def test_card_probe_loss_only_touches_target_logit(self):
        logits = torch.tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)
        loss = card_probe_policy_loss({"card_logits": logits}, 1.0, 2)
        loss.backward()
        self.assertAlmostEqual(float(logits.grad[2]), -1.0, places=5)
        self.assertTrue(torch.all(logits.grad[[0, 1, 3]] == 0).item())

    def test_card_pair_ranks_winner_over_loser(self):
        logits = torch.tensor([0.0, 0.0, 0.0, 0.0], requires_grad=True)
        loss = card_pair_policy_loss({"card_logits": logits}, 0, 0.08, 1, 0.02, eps=0.01)
        loss.backward()
        self.assertLess(float(logits.grad[0]), float(logits.grad[1]))

    def test_utility_ap_sign_agreement_helper(self):
        agree = utility_ap_sign_agreement(0.1, 0.05)
        self.assertTrue(agree["comparable"])
        self.assertTrue(agree["agree"])
        disagree = utility_ap_sign_agreement(0.1, -0.05)
        self.assertFalse(disagree["agree"])
        noisy = utility_ap_sign_agreement(0.001, 0.05, eps_u=0.01)
        self.assertFalse(noisy["comparable"])


if __name__ == "__main__":
    unittest.main()

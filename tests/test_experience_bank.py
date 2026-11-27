"""检索式经验库的签名、分组、落盘和检索。不启动检测器训练。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from training.checkpoints import save_checkpoint
from training.experience_bank import (
    GammaController, InnerLRController, allowed_source_families, apply_alpha_proxy, apply_counterfactual_policy,
    assert_compatible_inner_lr, assert_sources_allowed, assign_experience_groups, attach_first_order_ratio,
    choose_vector_storage,
    _discard_task_deltas, decode_vector, frozen_retrieval_structure, gate_borrow_by_support,
    group_prototypes,
    holdout_borrow_plan,
    inner_lr_manifest_fields, layered_sketch_indices, latest_experience_bank, limit_role_views,
    load_experience_bank, own_update_from_borrowed_init, pack_named, plan_module_borrow, query_exposure,
    refresh_deltas_complete, refresh_experience_cards, retrieval_fomaml_step, retrieval_holdout_pair,
    retrieve_module_experience, save_experience_bank, select_bank_tasks, sketch_packed, sketch_values,
    task_keys_from_groups, task_signature, trim_packed, unpack_named, views_geometry,
)
from training.retrieval_memory import MODULES, geometry_descriptor, trim_vector


class ExperienceBankTest(unittest.TestCase):
    def test_same_object_families_share_a_group(self):
        families = [
            "mvtec:bottle", "mvtec:toothbrush", "realiad:toothbrush",
            "mvtec:cable", "mvtec:transistor", "realiad:transistor1",
        ]
        groups = assign_experience_groups(families, 3, [
            ["mvtec:toothbrush", "realiad:toothbrush"],
            ["mvtec:transistor", "realiad:transistor1"],
        ])
        self.assertEqual(groups["mvtec:toothbrush"], groups["realiad:toothbrush"])
        self.assertEqual(groups["mvtec:transistor"], groups["realiad:transistor1"])
        self.assertEqual(set(groups.values()), {0, 1, 2})
        self.assertEqual(groups, assign_experience_groups(list(reversed(families)), 3, [
            ["realiad:toothbrush", "mvtec:toothbrush"],
            ["mvtec:transistor", "realiad:transistor1"],
        ]))

    def test_bank_tasks_keep_distinct_support_sets(self):
        def episode(index, support):
            return {
                "episode_id": f"oof:mvtec:bottle:{index:03d}",
                "family_id": "mvtec:bottle",
                "support": [{"physical_group_id": group} for group in support],
            }

        episodes = [
            episode(0, ["a", "b"]),
            episode(1, ["b", "a"]),
            episode(2, ["c", "d"]),
        ]
        selected, report = select_bank_tasks(episodes, ["mvtec:bottle"], 8)
        self.assertEqual([item["episode_id"] for item in selected], [
            "oof:mvtec:bottle:000", "oof:mvtec:bottle:002",
        ])
        self.assertEqual(report[0]["available"], 3)
        self.assertEqual(report[0]["selected"], 2)
        self.assertLess(report[0]["selected"], report[0]["requested"])

    def test_signature_uses_groups_not_query(self):
        defect = torch.zeros(2, 8)
        defect[:, 0] = torch.tensor([1.0, 3.0])
        support = group_prototypes([{
            "physical_group_id": "s",
            "f_img": torch.zeros(8),
            "f_def": defect,
            "boxes": torch.tensor([[0.0, 0.0, 2.0, 4.0], [0.0, 0.0, 4.0, 2.0]]),
            "hw": (8, 8),
        }])
        reference = group_prototypes([{
            "physical_group_id": "r",
            "f_img": torch.ones(8),
            "f_def": torch.zeros(0, 8),
            "boxes": torch.zeros(0, 4),
            "hw": (8, 8),
        }])
        keys = task_keys_from_groups(support, reference)
        self.assertEqual(tuple(keys["Appearance"].shape), (8,))
        self.assertEqual(tuple(keys["Difference"].shape), (8,))
        self.assertEqual(tuple(keys["Morphology"].shape), (14,))
        self.assertTrue(torch.allclose(keys["proto_def"][:1], torch.tensor([2.0])))
        boxes = [torch.tensor([[0.0, 0.0, 10.0, 5.0], [1.0, 1.0, 4.0, 8.0]]), torch.zeros(0, 4)]
        self.assertTrue(torch.allclose(views_geometry(boxes, [(20, 30), (20, 30)]), geometry_descriptor(boxes, (20, 30))))

    def test_one_view_per_group_keeps_annotated_support(self):
        records = [
            {"split_role": "SUPPORT", "physical_group_id": "s", "image_id": "b", "annotations": []},
            {"split_role": "SUPPORT", "physical_group_id": "s", "image_id": "a", "annotations": [{"bbox": [0, 0, 1, 1]}]},
            {"split_role": "R", "physical_group_id": "r", "image_id": "c"},
            {"split_role": "R", "physical_group_id": "r", "image_id": "d"},
        ]
        chosen = limit_role_views(records, 1)
        self.assertEqual([row["image_id"] for row in chosen], ["a", "c"])

    def test_target_family_is_rejected_and_query_exposure_is_reported(self):
        scope = {
            "theta0_train_family_ids": ["mvtec:bottle"],
            "oof_target_family_ids": ["mvtec:capsule"],
            "meta_val_family_ids": ["visa:candle"],
            "outer_test_family_ids": ["mvtec:hazelnut"],
        }
        assert_sources_allowed(["mvtec:bottle"], scope)
        with self.assertRaises(AssertionError):
            assert_sources_allowed(["mvtec:capsule"], scope)
        exposure = query_exposure(
            [{"family_id": "mvtec:bottle", "physical_group_id": "q"}],
            [{"family_id": "mvtec:bottle", "query_defect": [{"physical_group_id": "q"}], "query_normal": []}],
        )
        self.assertEqual(exposure["query_seen_by_theta0"], "true")
        self.assertEqual(exposure["overlap_groups"], 1)

    def test_storage_roundtrip_picks_the_smaller_recoverable_format(self):
        sparse = torch.zeros(32)
        sparse[1] = 4
        sparse[8] = -2
        encoded = choose_vector_storage(sparse)
        self.assertEqual(encoded["format"], "sparse_fp16")
        restored = decode_vector({key: value for key, value in encoded.items() if key not in {"bytes", "cosine", "relative_error"}})
        self.assertGreaterEqual(float(torch.nn.functional.cosine_similarity(sparse, restored, dim=0)), 0.999)
        dense = torch.linspace(-1, 1, 32)
        self.assertEqual(choose_vector_storage(dense)["format"], "dense_fp16")

    def test_save_load_and_retrieve_cannot_read_the_excluded_group(self):
        layout = [{"name": "weight", "shape": [4], "offset": 0, "numel": 4}]
        unit = torch.tensor([0.0, 3.0, 0.0, 4.0])
        unit = unit / unit.norm()
        components = torch.eye(4)
        bank = {
            "manifest": {"fold_id": 0, "tasks": ["a"]},
            "near_duplicates": [],
            "theta0_sha256": "theta0",
            "theta_sha256": "theta",
            "preprocess_sha256": "preprocess",
            "split_sha256": "split",
            "experience_groups": 2,
            "clusters_requested": 2,
            "top_mass_threshold": 0.8,
            "max_top_k": 4,
            "temperature": 1.0,
            "groups": {"mvtec:bottle": 0},
            "layouts": {module: layout for module in MODULES},
            "pca": {name: {"mean": torch.zeros(4), "components": components} for name in ("Appearance", "Morphology", "Difference")},
            "centers": {name: torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]) for name in ("Appearance", "Morphology", "Difference")},
            "center_valid": {name: torch.tensor([True, False]) for name in ("Appearance", "Morphology", "Difference")},
            "anchors": {name: 1.0 for name in MODULES},
            "keep_ratio": {name: 1.0 for name in MODULES},
            "next_keep_ratio": {name: 1.0 for name in MODULES},
            "trim_calibration": {name: {"keep_ratio": 1.0, "calibrated": False, "scores": []} for name in MODULES},
            "signatures": {name: torch.zeros(1, 4) for name in ("Appearance", "Morphology", "Difference")},
            "assignments": [],
            "cards_memory": {
                ("Backbone", 0, 0): {
                    "unit": unit, "valid": True, "low_sample": True, "coherence": 1.0,
                    "member_count": 1, "families": ["mvtec:bottle"], "task_ids": ["a"],
                },
            },
        }
        key = torch.tensor([4.0, 0.0, 0.0, 0.0])
        own = retrieve_module_experience(bank, "Backbone", key, 0)
        self.assertFalse(own["valid"])
        self.assertNotIn(0, own["source_groups"])
        external = retrieve_module_experience(bank, "Backbone", key, None)
        self.assertTrue(external["valid"])
        self.assertEqual(external["source_groups"], [0])
        self.assertGreater(float(torch.dot(external["vector"], unit)), 0.999)
        with tempfile.TemporaryDirectory() as directory:
            before = retrieve_module_experience(bank, "Backbone", key, None)
            save_experience_bank(Path(directory), bank)
            # 落盘后内存 unit 被清空；校验必须用保存前快照，不能再读 cards_memory。
            self.assertEqual(int(bank["cards_memory"][("Backbone", 0, 0)]["unit"].numel()), 0)
            self.assertFalse(retrieve_module_experience(bank, "Backbone", key, None)["valid"])
            loaded = load_experience_bank(directory, "theta")
            with self.assertRaises(ValueError):
                load_experience_bank(directory, "other-theta")
            reloaded = retrieve_module_experience(loaded, "Backbone", key, None)
            self.assertTrue(before["valid"])
            self.assertTrue(reloaded["valid"])
            self.assertTrue(torch.allclose(reloaded["vector"], before["vector"], atol=1e-3))
            blocked = retrieve_module_experience(loaded, "Backbone", key, 0)
            self.assertFalse(blocked["valid"])
        indices = layered_sketch_indices(layout, 3, 7)
        self.assertEqual(sum(int(item["index"].numel()) for item in indices), 3)


def _retrieval_bank():
    layout = [{"name": "backbone.weight", "shape": [4], "offset": 0, "numel": 4}]
    unit = torch.tensor([0.0, 3.0, 0.0, 4.0])
    unit = unit / unit.norm()
    components = torch.eye(4)
    card = {
        "unit": unit, "valid": True, "low_sample": True, "coherence": 1.0,
        "member_count": 1, "families": ["mvtec:bottle"], "task_ids": ["a"],
    }
    return {
        "manifest": {"fold_id": 0, "requested_tasks_per_family": 8, "tasks": ["a"]},
        "near_duplicates": [],
        "theta0_sha256": "theta0",
        "theta_sha256": "theta",
        "preprocess_sha256": "preprocess",
        "split_sha256": "split",
        "experience_groups": 2,
        "clusters_requested": 2,
        "top_mass_threshold": 0.8,
        "max_top_k": 4,
        "temperature": 1.0,
        "groups": {"mvtec:bottle": 0, "mvtec:cable": 1},
        "layouts": {module: layout for module in MODULES},
        "pca": {name: {"mean": torch.zeros(4), "components": components} for name in ("Appearance", "Morphology", "Difference")},
        "centers": {name: torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]) for name in ("Appearance", "Morphology", "Difference")},
        "center_valid": {name: torch.tensor([True, False]) for name in ("Appearance", "Morphology", "Difference")},
        "anchors": {name: 1.0 for name in MODULES},
        "keep_ratio": {name: 1.0 for name in MODULES},
        "next_keep_ratio": {name: 1.0 for name in MODULES},
        "trim_calibration": {name: {"keep_ratio": 1.0, "calibrated": False, "scores": []} for name in MODULES},
        "signatures": {name: torch.zeros(1, 4) for name in ("Appearance", "Morphology", "Difference")},
        "assignments": [
            {"task_id": "oof:mvtec:bottle:000", "family_id": "mvtec:bottle", "group": 0, "cluster": {"Appearance": 0, "Morphology": 0, "Difference": 0}, "seed": 1, "support_image_ids": ["s"]},
            {"task_id": "oof:mvtec:cable:000", "family_id": "mvtec:cable", "group": 1, "cluster": {"Appearance": 0, "Morphology": 0, "Difference": 0}, "seed": 2, "support_image_ids": ["t"]},
        ],
        "cards_memory": {(module, 0, 0): dict(card) for module in MODULES},
        "directory": "synthetic",
    }


class RetrievalWiringTest(unittest.TestCase):
    def test_source_mask_drops_cards_from_unavailable_families(self):
        bank = _retrieval_bank()
        key = torch.tensor([4.0, 0.0, 0.0, 0.0])
        blocked = retrieve_module_experience(bank, "Backbone", key, None, allowed_families={"mvtec:cable"})
        self.assertFalse(blocked["valid"])
        opened = retrieve_module_experience(bank, "Backbone", key, None, allowed_families={"mvtec:bottle"})
        self.assertTrue(opened["valid"])
        self.assertEqual(opened["source_groups"], [0])

    def test_closed_borrow_skips_proxy_and_open_updates_policy(self):
        bank = _retrieval_bank()
        key = torch.tensor([4.0, 0.0, 0.0, 0.0])
        episode = {
            "episode_id": "oof:mvtec:cable:000", "family_id": "mvtec:cable",
            "source_available_action_ids": ["Null", "mvtec:bottle", "mvtec:cable"],
        }
        self.assertEqual(allowed_source_families(episode), {"mvtec:bottle"})
        controller = GammaController()
        closed = [
            plan_module_borrow(
                bank, module, key, 1, {"mvtec:bottle"}, controller, False, support_step_l2=1.0,
            )
            for module in MODULES
        ]
        self.assertTrue(all(plan["rho"].requires_grad for plan in closed))
        self.assertTrue(all(float(plan["rho"].detach()) <= controller.rho_max for plan in closed))
        self.assertEqual({plan["inject"] for plan in closed}, {False})
        before = {name: float(param.detach().sum()) for name, param in controller.policy.policy.named_parameters()}
        # 关闭借用时训练步不调用策略更新（空 module_utilities）
        apply_counterfactual_policy(controller, closed, {})
        after = {name: float(param.detach().sum()) for name, param in controller.policy.policy.named_parameters()}
        self.assertEqual(after, before)
        with torch.no_grad():
            controller.policy.policy.null_head.bias.fill_(-8.0)
        opened = [
            plan_module_borrow(
                bank, module, key, 1, {"mvtec:bottle"}, controller, True, support_step_l2=1.0,
            )
            for module in MODULES
        ]
        self.assertTrue(any(plan["inject"] for plan in opened))
        utilities = {plan["module"]: 0.8 for plan in opened if plan["inject"]}
        apply_counterfactual_policy(controller, opened, utilities)
        after_open = {name: float(param.detach().sum()) for name, param in controller.policy.policy.named_parameters()}
        self.assertTrue(any(after_open[name] != before[name] for name in before))

    def test_counterfactual_utility_updates_policy(self):
        bank = _retrieval_bank()
        key = torch.tensor([4.0, 0.0, 0.0, 0.0])
        helpful = GammaController()
        with torch.no_grad():
            helpful.policy.policy.null_head.bias.fill_(-8.0)
        helpful_plan = [
            plan_module_borrow(
                bank, "Backbone", key, 1, {"mvtec:bottle"}, helpful, True, support_step_l2=1.0,
            )
        ]
        before = float(helpful_plan[0]["rho"].detach())
        result = apply_counterfactual_policy(helpful, helpful_plan, {"Backbone": 1.0})
        self.assertAlmostEqual(result["module_utilities"]["Backbone"], 1.0)
        after_plan = plan_module_borrow(
            bank, "Backbone", key, 1, {"mvtec:bottle"}, helpful, True, support_step_l2=1.0,
        )
        self.assertNotEqual(float(after_plan["rho"].detach()), before)

    def test_module_delta_excludes_the_borrowed_increment(self):
        names = (
            "backbone.weight", "rpn.weight", "roi_heads.box_head.weight",
            "roi_heads.box_predictor.bbox_pred.weight", "roi_heads.box_predictor.cls_score.weight",
        )
        before = {name: torch.zeros(2) for name in names}
        borrow = {"backbone.weight": torch.tensor([0.4, -0.2])}
        learned = {"backbone.weight": torch.tensor([0.05, 0.01])}
        after = {
            name: before[name] + borrow.get(name, torch.zeros(2)) + learned.get(name, torch.zeros(2))
            for name in names
        }
        delta = own_update_from_borrowed_init(before, after, borrow)
        self.assertTrue(torch.allclose(delta["Backbone"]["backbone.weight"], learned["backbone.weight"]))
        self.assertTrue(torch.allclose(delta["Where"]["rpn.weight"], torch.zeros(2)))
        self.assertIn("RoiFeature", delta)

    def test_holdout_plan_rejects_query_and_does_not_update_gamma(self):
        class Guard(dict):
            def __getitem__(self, key):
                if "query" in str(key):
                    raise AssertionError(key)
                return dict.__getitem__(self, key)

        bank = _retrieval_bank()
        key = torch.tensor([4.0, 0.0, 0.0, 0.0])
        keys = {name: key for name in ("Appearance", "Morphology", "Difference")}
        episode = Guard({
            "episode_id": "holdout:capsule", "family_id": "mvtec:capsule",
            "source_available_action_ids": ["Null", "mvtec:bottle", "mvtec:capsule"],
        })
        controller = GammaController()
        before = {name: float(param.detach().sum()) for name, param in controller.policy.policy.named_parameters()}
        with self.assertRaises(AssertionError):
            holdout_borrow_plan(bank, episode, keys, controller, [{"split_role": "QUERY_DEFECT"}])
        plans = holdout_borrow_plan(
            bank, episode, keys, controller, [{"split_role": "SUPPORT"}, {"split_role": "R"}],
            step_l2={module: 1.0 for module in MODULES},
        )
        self.assertTrue(any(plan["valid"] for plan in plans))
        after = {name: float(param.detach().sum()) for name, param in controller.policy.policy.named_parameters()}
        self.assertEqual(after, before)
        from training.meta_transfer import retrieval_holdout_plan

        with self.assertRaises(AssertionError):
            retrieval_holdout_plan(bank, episode, keys, controller, [{"split_role": "QUERY_NORMAL"}])

    def test_null_option_can_block_injection(self):
        bank = _retrieval_bank()
        key = torch.tensor([4.0, 0.0, 0.0, 0.0])
        controller = GammaController()
        with torch.no_grad():
            controller.policy.policy.null_head.weight.zero_()
            controller.policy.policy.null_head.bias.fill_(20.0)
        plans = [
            plan_module_borrow(
                bank, module, key, 1, {"mvtec:bottle"}, controller, True, support_step_l2=1.0,
            )
            for module in MODULES
        ]
        self.assertTrue(all(float(plan["null_probability"]) > 0.9 for plan in plans))
        self.assertEqual({plan["inject"] for plan in plans}, {False})
        self.assertTrue(all(plan["named"] == {} for plan in plans))
        discrete = plan_module_borrow(
            bank, "Backbone", key, 1, {"mvtec:bottle"}, controller, True,
            support_step_l2=1.0,
        )
        self.assertEqual(discrete["null_decision"], "null")
        self.assertFalse(discrete["inject"])

    def test_first_order_ratio_compares_borrow_to_own_step(self):
        plan = {
            "module": "Backbone", "inject": True, "named": {},
            "vector": torch.tensor([1.0, 0.0]), "gamma": torch.tensor(0.5),
            "rho": torch.tensor(0.5), "anchor": torch.tensor(2.0), "utility_reject": "",
        }
        attach_first_order_ratio(
            [plan],
            {"Backbone": torch.tensor([4.0, 0.0])},
            {"Backbone": torch.tensor([1.0, 0.0])},
            {"Backbone": 0.25},
        )
        # R = ⟨g_Q, ρ ‖α g_S‖ û⟩ / ⟨g_Q, α g_S⟩
        self.assertAlmostEqual(plan["query_dot_borrow"], 0.5)
        self.assertAlmostEqual(plan["query_dot_step"], 1.0)
        self.assertAlmostEqual(plan["first_order_ratio"], 0.5)
        self.assertAlmostEqual(plan["candidate_borrow_l2"], 0.5)
        self.assertAlmostEqual(plan["alpha_grad_l2"], 1.0)

    def test_support_alignment_is_diagnostic_only(self):
        uphill = {
            "module": "Backbone", "inject": True, "named": {"w": torch.ones(2)},
            "vector": torch.tensor([1.0, 0.0]), "gamma": torch.tensor(0.5), "rho": torch.tensor(0.5),
            "anchor": torch.tensor(2.0), "utility_reject": "", "applied_borrow_l2": 1.0,
        }
        gate_borrow_by_support([uphill], {"Backbone": torch.tensor([1.0, 0.0])})
        self.assertTrue(uphill["inject"])
        self.assertEqual(uphill["named"], {"w": uphill["named"]["w"]})
        self.assertGreater(uphill["support_dot"], 0.0)
        downhill = {
            "module": "Backbone", "inject": True, "named": {"w": torch.tensor([-1.0, 0.0])},
            "vector": torch.tensor([1.0, 0.0]), "gamma": torch.tensor(0.5), "rho": torch.tensor(0.5),
            "anchor": torch.tensor(2.0), "utility_reject": "", "applied_borrow_l2": 1.0,
        }
        gate_borrow_by_support([downhill], {"Backbone": torch.tensor([-4.0, 0.0])})
        self.assertTrue(downhill["inject"])
        self.assertLess(downhill["support_dot"], 0.0)

    def test_dense_fp16_rejected_when_too_many_elements_flush_to_zero(self):
        values = torch.ones(100)
        values[:20] = 1e-8
        self.assertEqual(choose_vector_storage(values)["format"], "dense_fp32")

    def test_refresh_keeps_existing_clusters_and_does_not_pad_tasks(self):
        import inspect

        bank = _retrieval_bank()
        structure = frozen_retrieval_structure(bank)
        self.assertEqual(structure["effective_tasks"], 2)
        self.assertEqual(structure["requested_tasks_per_family"], 8)
        self.assertLess(structure["effective_tasks"], structure["requested_tasks_per_family"])
        self.assertEqual(structure["assignments"][0]["cluster"]["Appearance"], 0)
        bank["assignments"].append(dict(bank["assignments"][0]))
        with self.assertRaises(AssertionError):
            frozen_retrieval_structure(bank)
        source = inspect.getsource(refresh_experience_cards)
        self.assertNotIn("fit_fixed_pca", source)
        self.assertNotIn("spherical_kmeans", source)
        self.assertNotIn("_fit_keys", source)

    def test_default_bank_is_the_newest_one(self):
        import os

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            older = root / "fold_0" / "older"
            newer = root / "fold_0" / "newer"
            plain = root / "fold_0" / "plain"
            for path in (older, newer, plain):
                path.mkdir(parents=True)
            save_checkpoint(older / "module_memory.pt", {"kind": "experience_bank"})
            save_checkpoint(newer / "module_memory.pt", {"kind": "experience_bank"})
            save_checkpoint(plain / "module_memory.pt", {"kind": "module_delta"})
            os.utime(older / "module_memory.pt", (1_000, 1_000))
            os.utime(newer / "module_memory.pt", (2_000, 2_000))
            self.assertEqual(latest_experience_bank(0, root), newer)
            with self.assertRaises(FileNotFoundError):
                latest_experience_bank(1, root)

    def test_inner_lrs_start_at_one_hundredth_and_follow_the_query_dot(self):
        controller = InnerLRController()
        values, hits = controller.current()
        self.assertEqual(set(values), set(MODULES))
        for module in MODULES:
            self.assertAlmostEqual(values[module], 0.01, places=6)
            self.assertEqual(hits[module], "")
        controller.raw["Backbone"].data.fill_(-30)
        controller.raw["Where"].data.fill_(5)
        values, hits = controller.current()
        self.assertEqual(hits["Backbone"], "min")
        self.assertEqual(values["Backbone"], controller.minimum)
        self.assertEqual(hits["Where"], "max")
        self.assertAlmostEqual(values["Where"], controller.maximum, places=6)
        fresh = InnerLRController()
        before = fresh.current()[0]
        support = {module: torch.zeros(3) for module in MODULES}
        query_up = {module: torch.zeros(3) for module in MODULES}
        query_down = {module: torch.zeros(3) for module in MODULES}
        support["Backbone"] = torch.tensor([1.0, 0.0, 0.0])
        support["Box"] = torch.tensor([0.0, 1.0, 0.0])
        query_up["Backbone"] = torch.tensor([1.0, 0.0, 0.0])
        query_down["Box"] = torch.tensor([0.0, -2.0, 0.0])
        apply_alpha_proxy(fresh, support, query_up)
        apply_alpha_proxy(fresh, support, query_down)
        after = fresh.current()[0]
        self.assertGreater(after["Backbone"], before["Backbone"])
        self.assertLess(after["Box"], before["Box"])
        self.assertAlmostEqual(after["Where"], before["Where"], places=6)
        self.assertAlmostEqual(after["Class"], before["Class"], places=6)

    def test_no_borrow_does_not_write_history_into_theta(self):
        bank = _retrieval_bank()
        key = torch.tensor([4.0, 0.0, 0.0, 0.0])
        controller = GammaController()
        closed = [
            plan_module_borrow(
                bank, module, key, 1, {"mvtec:bottle"}, controller, False, support_step_l2=1.0,
            )
            for module in MODULES
        ]
        theta = {"backbone.weight": torch.ones(4)}
        for plan in closed:
            self.assertFalse(plan["inject"])
            self.assertEqual(plan["named"], {})
            for name, value in plan["named"].items():
                theta[name] = theta[name] + value
        self.assertTrue(torch.equal(theta["backbone.weight"], torch.ones(4)))
        import inspect

        source = inspect.getsource(retrieval_fomaml_step)
        self.assertIn("query_counterfactual_utility", source)
        self.assertIn("retrieval_support_adapt(", source)
        self.assertIn("retrieval_support_adapt(", inspect.getsource(retrieval_holdout_pair))
        signature = inspect.getsource(task_signature)
        self.assertIn('roles=("reference", "support")', signature)
        self.assertIn("group_prototypes(support_views), group_prototypes(reference_views)", signature)
        self.assertNotIn("query_defect", inspect.getsource(retrieval_holdout_pair))

    def test_bank_snapshot_records_theta_and_alpha(self):
        values = {module: 0.01 for module in MODULES}
        fields = inner_lr_manifest_fields({"inner_lr": 0.01, "inner_lr_snapshot": values})
        self.assertEqual(fields["inner_lr_source"], "module_snapshot")
        self.assertIn("alpha_m_snapshot", fields["delta_definition"])
        fixed = inner_lr_manifest_fields({"inner_lr": 0.01})
        self.assertEqual(fixed["inner_lr_source"], "fixed_inner_lr")
        self.assertIsNone(fixed["inner_lr_snapshot"])
        bank = _retrieval_bank()
        bank["manifest"] = {**bank["manifest"], **fields, "theta_sha256": bank["theta_sha256"]}
        with tempfile.TemporaryDirectory() as directory:
            save_experience_bank(directory, bank)
            loaded = load_experience_bank(directory, bank["theta_sha256"])
            self.assertEqual(loaded["theta_sha256"], bank["theta_sha256"])
            self.assertEqual(loaded["manifest"]["inner_lr_snapshot"], values)
            assert_compatible_inner_lr(loaded, values)
            with self.assertRaises(ValueError):
                load_experience_bank(directory, "other-theta")
        fixed_bank = _retrieval_bank()
        fixed_bank["manifest"] = {**fixed_bank["manifest"], **fixed}
        with self.assertRaises(ValueError):
            assert_compatible_inner_lr(fixed_bank, values)

    def test_holdout_uses_the_training_adapt_function(self):
        import inspect

        from training.meta_transfer import retrieval_holdout_pair as wrapped
        from training.meta_transfer import retrieval_holdout_query_plan

        self.assertIn("adapt_pair", inspect.getsource(wrapped))
        self.assertNotIn("query_mode_slices", inspect.getsource(retrieval_holdout_pair))
        episode = {
            "support_mode": "broken_large",
            "query_defect_group_ids": ["q1", "q2", "q3", "q4", "q5"],
            "query_defect_modes": ["broken_large", "broken_large", "broken_large", "crack", "contamination"],
            "query_normal_group_ids": ["n1"],
        }
        plan = retrieval_holdout_query_plan(episode)
        self.assertFalse(plan["query_used_for_retrieval"])
        self.assertEqual(plan["same_mode_ap50_group_ids"], ["q1", "q2", "q3"])
        self.assertEqual(plan["other_mode_ap50_group_ids"], ["q4", "q5"])
        self.assertEqual(plan["overall_ap50_group_ids"], ["q1", "q2", "q3", "q4", "q5", "n1"])

    def test_trim_and_sketch_packed_match_named_path(self):
        layout = [
            {"name": "a", "offset": 0, "numel": 8, "shape": (2, 4)},
            {"name": "b", "offset": 8, "numel": 4, "shape": (4,)},
        ]
        named = {"a": torch.randn(2, 4), "b": torch.randn(4)}
        flat = pack_named(named, layout)
        for ratio in (1.0, 0.5, 0.25):
            via_named = pack_named(
                {name: trim_vector(value, ratio) for name, value in unpack_named(flat, layout).items()},
                layout,
            )
            self.assertTrue(torch.allclose(trim_packed(flat, layout, ratio), via_named))
        indices = layered_sketch_indices(layout, 6, 0)
        for ratio in (1.0, 0.5):
            self.assertTrue(torch.allclose(
                sketch_packed(flat, layout, indices, ratio),
                sketch_values(unpack_named(flat, layout), indices, ratio),
            ))

    def test_refresh_deltas_complete_requires_all_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assignments = [{"task_id": "a:b:1"}, {"task_id": "a:b:2"}]
            self.assertFalse(refresh_deltas_complete(root, assignments))
            (root / "task_deltas").mkdir()
            save_checkpoint(root / "task_deltas" / "a_b_1.pt", {"x": 1})
            self.assertFalse(refresh_deltas_complete(root, assignments))
            save_checkpoint(root / "task_deltas" / "a_b_2.pt", {"x": 1})
            self.assertTrue(refresh_deltas_complete(root, assignments))
            _discard_task_deltas(root)
            self.assertFalse((root / "task_deltas").exists())
            self.assertFalse(refresh_deltas_complete(root, assignments))


if __name__ == "__main__":
    unittest.main()

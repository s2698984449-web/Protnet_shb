from __future__ import annotations

import unittest

import torch
from torchvision.models.detection._utils import BoxCoder

from training.fomaml_roi import (
    ROIExample,
    SKETCH_DIM,
    SupportScoreCache,
    _eligible_proposal_indices,
    _mean_optional,
    _remove_ignored_proposals,
    cosine_neg_grad,
    eval_support_view,
    fixed_proposal_detection_loss,
    live_detection_loss,
    module_count_sketch,
    prepare_roi_example,
    proposal_recall,
    reuse_weight,
    roi_detection_loss,
    validate_detection_loss_settings,
)
from training.theta0 import (
    _ParameterStore,
    _append_val_csv,
    _assert_meta_val_episode,
    _capture_rng_state,
    _fomaml_phase,
    _hydrate_library,
    _index_module_memory,
    _remember_module_delta,
    _rng_from_checkpoint,
    _rng_to_checkpoint,
    _spill_episode_delta,
    fomaml_resume_fingerprint,
    _mean_detection_loss,
    _prescreen_sketches,
    _rank_sources,
    _restore_rng_state,
    _source_sketch,
    choose_val_checkpoint,
    load_meta_val_episodes,
    prescreen_audit_due,
    rehearse_holdout_load,
    select_meta_val_episodes,
    select_module_initialization,
    split_parameter_delta,
)
from training.meta_transfer import (
    LoRALinear,
    _checked_block,
    _checkpoint_models,
    _module_parameter_names,
    classify_contributions,
    final_product_library,
    fold_source_products,
    inject_predictor_delta,
    load_innovation1_modules,
    lora_generator,
    select_delta_modules,
)
from common.io import load_verified_research_splits


class _Predictor(torch.nn.Module):
    def __init__(self, shift=0.0):
        super().__init__()
        self.shift = shift
        self.linear = torch.nn.Linear(4, 10)

    def forward(self, features):
        output = self.linear(features)
        return output[:, :2] + self.shift, output[:, 2:]


def _settings():
    return {
        "weights": {
            "roi": 0.25,
            "positive": 1.0,
            "ranking": 1.0,
            "hard_negative": 1.0,
            "localization": 0.5,
        },
        "positive_iou": 0.5,
        "negative_iou": 0.5,
        "working_logit": 0.0,
        "ranking_margin": 0.2,
        "hard_negative_top_k": 2,
    }


def _example(normal: bool) -> ROIExample:
    boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]],
    )
    return ROIExample(
        features=torch.randn(2, 4),
        labels=torch.tensor([0, 0]) if normal else torch.tensor([1, 0]),
        regression_targets=torch.zeros(2, 4),
        sampled_boxes=boxes,
        proposal_features=torch.randn(2, 4),
        proposal_boxes=boxes,
        target_boxes=torch.zeros(0, 4) if normal else boxes[:1],
        ignore_boxes=torch.zeros(0, 4),
        normal_flag=normal,
    )


class FomamlLossTest(unittest.TestCase):
    def test_detection_loss_is_finite_for_positive_image(self):
        predictor = _Predictor()
        loss = roi_detection_loss(
            predictor,
            _example(normal=False),
            BoxCoder((10.0, 10.0, 5.0, 5.0)),
            _settings(),
            "cpu",
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(predictor.linear.weight.grad.abs().sum().item(), 0.0)
        averaged = _mean_detection_loss(
            _Predictor(), [_example(normal=False)], BoxCoder((10.0, 10.0, 5.0, 5.0)), _settings(), "cpu",
        )
        self.assertTrue(torch.isfinite(averaged))

    def test_detection_loss_matches_every_ground_truth_box(self):
        boxes = torch.tensor([
            [0.0, 0.0, 10.0, 10.0],
            [40.0, 40.0, 54.0, 54.0],
            [80.0, 0.0, 88.0, 8.0],
        ])
        example = ROIExample(
            features=torch.randn(3, 4),
            labels=torch.tensor([1, 1, 0]),
            regression_targets=torch.zeros(3, 4),
            sampled_boxes=boxes,
            proposal_features=torch.randn(3, 4),
            proposal_boxes=boxes,
            target_boxes=boxes[:2],
            ignore_boxes=torch.zeros(0, 4),
            normal_flag=False,
        )
        loss = roi_detection_loss(
            _Predictor(),
            example,
            BoxCoder((10.0, 10.0, 5.0, 5.0)),
            _settings(),
            "cpu",
        )
        self.assertTrue(torch.isfinite(loss))

    def test_detection_loss_is_finite_for_normal_image(self):
        loss = roi_detection_loss(
            _Predictor(),
            _example(normal=True),
            BoxCoder((10.0, 10.0, 5.0, 5.0)),
            _settings(),
            "cpu",
        )
        self.assertTrue(torch.isfinite(loss))

    def test_eligible_indices_preserve_filtered_proposal_order(self):
        proposals = torch.tensor([
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 30.0, 30.0],
            [40.0, 40.0, 50.0, 50.0],
        ])
        positive = proposals[:1]
        ignore = proposals[1:2]
        indices = _eligible_proposal_indices(proposals, positive, ignore, 0.5, 0.5)
        filtered = _remove_ignored_proposals(proposals, positive, ignore, 0.5, 0.5)
        self.assertTrue(torch.equal(proposals[indices], filtered))
        self.assertTrue(torch.equal(indices, torch.tensor([0, 2])))

    def test_invalid_loss_settings_are_rejected(self):
        settings = _settings()
        settings["negative_iou"] = 0.25
        with self.assertRaises(ValueError):
            validate_detection_loss_settings(settings)

    def test_margin_ignores_shared_logit_shift_and_class_logit_does_not(self):
        torch.manual_seed(0)
        example = _example(normal=False)
        coder = BoxCoder((10.0, 10.0, 5.0, 5.0))
        base = _Predictor(0.0)
        shifted = _Predictor(5.0)
        shifted.linear.load_state_dict(base.linear.state_dict())
        margin = _settings()
        margin["defect_score"] = "margin"
        legacy = _settings()
        legacy["defect_score"] = "class_logit"
        margin_base = roi_detection_loss(base, example, coder, margin, "cpu")
        margin_shifted = roi_detection_loss(shifted, example, coder, margin, "cpu")
        legacy_base = roi_detection_loss(base, example, coder, legacy, "cpu")
        legacy_shifted = roi_detection_loss(shifted, example, coder, legacy, "cpu")
        self.assertTrue(torch.allclose(margin_base, margin_shifted))
        self.assertGreater(abs(legacy_base.item() - legacy_shifted.item()), 1e-4)

    def test_missing_defect_score_keeps_class_logit_behavior(self):
        torch.manual_seed(0)
        example = _example(normal=False)
        coder = BoxCoder((10.0, 10.0, 5.0, 5.0))
        predictor = _Predictor()
        omitted = roi_detection_loss(predictor, example, coder, _settings(), "cpu")
        legacy = _settings()
        legacy["defect_score"] = "class_logit"
        explicit = roi_detection_loss(predictor, example, coder, legacy, "cpu")
        self.assertTrue(torch.equal(omitted, explicit))

    def test_predictor_delta_is_added_before_support_adaptation(self):
        base = {"cls_score.weight": torch.zeros(2, 2), "bbox_pred.weight": torch.ones(2, 2)}
        delta = {"roi_heads.box_predictor.cls_score.weight": torch.full((2, 2), 0.25)}
        injected = inject_predictor_delta(base, delta)
        self.assertTrue(torch.equal(injected["cls_score.weight"], torch.full((2, 2), 0.25)))
        self.assertTrue(torch.equal(injected["bbox_pred.weight"], torch.ones(2, 2)))
        bare = inject_predictor_delta(base, {"bbox_pred.weight": torch.full((2, 2), 0.5)})
        self.assertTrue(torch.equal(bare["bbox_pred.weight"], torch.full((2, 2), 1.5)))

    def test_only_the_latest_family_delta_stays_in_memory(self):
        import tempfile
        from pathlib import Path

        module_memory = {}
        library = {}
        with tempfile.TemporaryDirectory() as temporary:
            parts = Path(temporary)
            first = {"product_id": "bottle", "delta": {"Feature": {"backbone.weight": torch.ones(2, 2)}}, "sketch": {}}
            second = {"product_id": "bottle", "delta": {"Feature": {"backbone.weight": torch.full((2, 2), 2.0)}}, "sketch": {}}
            _remember_module_delta(module_memory, library, "mvtec:bottle", "oof:mvtec:bottle:000", first, parts)
            _remember_module_delta(module_memory, library, "mvtec:bottle", "oof:mvtec:bottle:001", second, parts)
            self.assertNotIn("delta", module_memory["oof:mvtec:bottle:000"])
            self.assertTrue((parts / "oof_mvtec_bottle_000.pt").is_file())
            self.assertTrue(torch.equal(
                library["mvtec:bottle"]["delta"]["Feature"]["backbone.weight"], torch.full((2, 2), 2.0),
            ))
            reloaded = final_product_library(module_memory, list(module_memory), parts)
            self.assertTrue(torch.equal(
                reloaded["mvtec:bottle"]["delta"]["Feature"]["backbone.weight"], torch.full((2, 2), 2.0),
            ))

    def test_resume_reloads_rng_optimizer_and_library(self):
        import random
        import tempfile
        from pathlib import Path

        from training.checkpoints import _abandoned_run, load_checkpoint, save_checkpoint

        module_memory = {}
        library = {}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = root / "module_memory_parts"
            first = {"product_id": "bottle", "delta": {"Feature": {"backbone.weight": torch.ones(2, 2)}}, "sketch": {}}
            second = {"product_id": "bottle", "delta": {"Feature": {"backbone.weight": torch.full((2, 2), 2.0)}}, "sketch": {}}
            _remember_module_delta(module_memory, library, "mvtec:bottle", "oof:mvtec:bottle:000", first, parts)
            _remember_module_delta(module_memory, library, "mvtec:bottle", "oof:mvtec:bottle:001", second, parts)
            _spill_episode_delta(parts, "oof:mvtec:bottle:001", module_memory["oof:mvtec:bottle:001"], release=False)
            self.assertIn("delta", module_memory["oof:mvtec:bottle:001"])
            indexed = _index_module_memory(module_memory)
            self.assertNotIn("delta", indexed["oof:mvtec:bottle:001"])
            probe = torch.nn.Linear(2, 2, bias=False)
            optimizer = torch.optim.SGD(probe.parameters(), lr=0.1, momentum=0.9)
            probe.weight.grad = torch.ones_like(probe.weight)
            optimizer.step()
            random.random()
            rng = _rng_to_checkpoint(_capture_rng_state())
            expected = torch.rand(1)
            save_checkpoint(root / "resume.pt", {
                "rng": rng, "optimizer": optimizer.state_dict(), "module_memory": indexed,
                "library_index": {"mvtec:bottle": "oof:mvtec:bottle:001"},
            })
            self.assertFalse(_abandoned_run(root))
            loaded = load_checkpoint(root / "resume.pt")
            _restore_rng_state(_rng_from_checkpoint(loaded["rng"]))
            self.assertTrue(torch.equal(torch.rand(1), expected))
            restored = torch.nn.Linear(2, 2, bias=False)
            restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.1, momentum=0.9)
            restored_optimizer.load_state_dict(loaded["optimizer"])
            momentum = restored_optimizer.state[restored.weight]["momentum_buffer"]
            self.assertTrue(torch.equal(momentum, optimizer.state[probe.weight]["momentum_buffer"]))
            hydrated = _hydrate_library(loaded["module_memory"], loaded["library_index"], parts)
            self.assertTrue(torch.equal(
                hydrated["mvtec:bottle"]["delta"]["Feature"]["backbone.weight"], torch.full((2, 2), 2.0),
            ))

    def test_resume_fingerprint_tracks_switches(self):
        base = {
            "supervised_theta0_sha256": "a", "fomaml_episode_sha256": "b", "seed": 1, "fold_id": 0,
            "fomaml_settings": {"prescreen_k": 4, "val_ap50": True, "loss": {"weights": {"roi": 0.25}}},
        }
        other = {
            **base,
            "fomaml_settings": {"prescreen_k": 0, "val_ap50": True, "loss": {"weights": {"roi": 0.25}}},
        }
        self.assertNotEqual(fomaml_resume_fingerprint(base), fomaml_resume_fingerprint(other))

    def test_task_memory_keeps_the_last_episode_of_each_product(self):
        episodes = {
            "oof:mvtec:bottle:000": {"product_id": "bottle", "delta": {"Box": 1}},
            "oof:mvtec:bottle:001": {"product_id": "bottle", "delta": {"Box": 2}},
            "oof:realiad:toothbrush:000": {"product_id": "toothbrush", "delta": {"Box": 3}},
        }
        order = ["oof:mvtec:bottle:000", "oof:realiad:toothbrush:000", "oof:mvtec:bottle:001"]
        library = final_product_library(episodes, order)
        self.assertEqual(library["mvtec:bottle"]["episode_id"], "oof:mvtec:bottle:001")
        self.assertEqual(library["mvtec:bottle"]["delta"]["Box"], 2)
        self.assertEqual(library["realiad:toothbrush"]["episode_id"], "oof:realiad:toothbrush:000")
        with self.assertRaises(ValueError):
            final_product_library(episodes, order[:-1])

    def test_lora_starts_at_zero_and_matches_across_arms(self):
        linear = torch.nn.Linear(4, 2)
        first = LoRALinear(linear, 4, 4.0, lora_generator(7, "episode"))
        second = LoRALinear(linear, 4, 4.0, lora_generator(7, "episode"))
        features = torch.randn(3, 4)
        self.assertTrue(torch.allclose(first(features), linear(features)))
        self.assertTrue(torch.equal(first.A, second.A))
        self.assertTrue(torch.equal(first.B, second.B))
        manifest = {"modules": {"Feature": {}, "Where": {}, "Box": {}, "Class": {}}}
        self.assertEqual(select_delta_modules(manifest, ["Where", "Feature"]), ["Where", "Feature"])
        with self.assertRaises(ValueError):
            select_delta_modules(manifest, ["Feature", "Feature"])

    def test_contribution_keeps_partial_gains_and_non_gains(self):
        mixed = classify_contributions({"visa:chewinggum": 0.02, "realiad:pcb": -0.01})
        self.assertFalse(mixed["require_all_targets"])
        self.assertEqual(mixed["contributed_products"], ["visa:chewinggum"])
        self.assertEqual(mixed["not_selected_products"], ["realiad:pcb"])
        self.assertEqual(mixed["outcome"], "partial_contribution")
        none = classify_contributions({"mvtec:capsule": -0.1, "mvtec:zipper": 0.0})
        self.assertEqual(none["contributed_products"], [])
        self.assertEqual(none["outcome"], "source_selection")

    def test_reuse_weight_stays_below_full_copy(self):
        self.assertAlmostEqual(reuse_weight(1.0, 0.8, 0.5, higher_is_better=False), 0.2)
        self.assertAlmostEqual(reuse_weight(1.0, 0.0, 0.5, higher_is_better=False), 0.5)
        self.assertEqual(reuse_weight(1.0, 1.2, 0.5, higher_is_better=False), 0.0)
        self.assertEqual(reuse_weight(0.2, 0.2, 0.5, higher_is_better=True), 0.0)
        self.assertAlmostEqual(reuse_weight(0.2, 0.9, 0.5, higher_is_better=True), 0.5)
        with self.assertRaises(ValueError):
            reuse_weight(1.0, 0.0, 1.0, higher_is_better=False)

    def test_module_delta_is_only_the_step_itself(self):
        before = {
            "backbone.weight": torch.zeros(2),
            "rpn.weight": torch.ones(2),
            "roi_heads.box_head.weight": torch.zeros(2),
            "roi_heads.box_predictor.bbox_pred.weight": torch.zeros(2),
            "roi_heads.box_predictor.cls_score.weight": torch.zeros(2),
        }
        after = {name: value + 0.25 for name, value in before.items()}
        delta = split_parameter_delta(before, after)
        self.assertEqual(set(delta), {"Feature", "Where", "Box", "Class"})
        self.assertTrue(torch.equal(delta["Feature"]["backbone.weight"], torch.full((2,), 0.25)))
        self.assertTrue(torch.equal(delta["Where"]["rpn.weight"], torch.full((2,), 0.25)))
        self.assertNotIn("borrowed", delta)

    def test_live_loss_reaches_feature_where_box_and_class(self):
        from torchvision.models.detection import fasterrcnn_resnet50_fpn

        torch.manual_seed(0)
        model = fasterrcnn_resnet50_fpn(
            weights=None, weights_backbone=None, num_classes=2, min_size=128, max_size=128,
        )
        model.train()
        record = {
            "image_id": "synthetic",
            "normal_flag": 0,
            "annotations": [{"bbox": [16.0, 16.0, 112.0, 112.0]}],
            "ignore_annotations": [],
            "image_tensor": torch.rand(3, 128, 128),
        }
        loss = live_detection_loss(
            model, record, torch.device("cpu"), _settings(), 1.0, 0.5, checkpoint_graph=True,
        )
        loss.backward()
        totals = {"Feature": 0.0, "Where": 0.0, "Box": 0.0, "Class": 0.0}
        prefixes = {
            "Feature": ("backbone.", "roi_heads.box_head."),
            "Where": ("rpn.",),
            "Box": ("roi_heads.box_predictor.bbox_pred.",),
            "Class": ("roi_heads.box_predictor.cls_score.",),
        }
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            for label, group in prefixes.items():
                if name.startswith(group):
                    totals[label] += float(parameter.grad.detach().abs().sum())
        for label, total in totals.items():
            self.assertGreater(total, 0.0, label)

    def test_parameter_store_keeps_momentum_and_detached_inner_delta(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Linear(1, 1, bias=False)
                self.rpn = torch.nn.Linear(1, 1, bias=False)
                self.roi_heads = torch.nn.Module()
                self.roi_heads.box_head = torch.nn.Linear(1, 1, bias=False)
                self.roi_heads.box_predictor = torch.nn.Module()
                self.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(1, 1, bias=False)
                self.roi_heads.box_predictor.cls_score = torch.nn.Linear(1, 1, bias=False)

        model = Toy()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.5)
        store = _ParameterStore(model)
        for slot in store.slots.values():
            for buffer in slot:
                self.assertFalse(buffer.is_pinned())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        store.capture("primary")
        original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        momentum = [optimizer.state[parameter]["momentum_buffer"] for parameter in model.parameters()]
        store.restore("primary")
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter, original[name]), name)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        for buffer in momentum:
            self.assertTrue(torch.equal(buffer, torch.full_like(buffer, 1.9)))

        store.capture("before")
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(0.25)
        store.capture("after")
        delta = split_parameter_delta(store.slot_dict("before"), store.slot_dict("after"))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        store.capture("after")
        self.assertTrue(torch.equal(
            delta["Feature"]["backbone.weight"], torch.tensor([[0.25]]),
        ))
        self.assertTrue(torch.equal(
            delta["Where"]["rpn.weight"], torch.tensor([[0.25]]),
        ))
        self.assertTrue(torch.equal(
            delta["Box"]["roi_heads.box_predictor.bbox_pred.weight"], torch.tensor([[0.25]]),
        ))
        self.assertTrue(torch.equal(
            delta["Class"]["roi_heads.box_predictor.cls_score.weight"], torch.tensor([[0.25]]),
        ))
        self.assertEqual(
            set(delta["Feature"]), {"backbone.weight", "roi_heads.box_head.weight"},
        )

    def test_score_cache_reuses_upstream_and_matches_original_rules(self):
        from torchvision.models.detection import fasterrcnn_resnet50_fpn

        torch.manual_seed(0)
        device = torch.device("cpu")
        model = fasterrcnn_resnet50_fpn(
            weights=None, weights_backbone=None, num_classes=2, min_size=128, max_size=128,
        )
        model.eval()
        record = {
            "image_id": "synthetic",
            "normal_flag": 0,
            "annotations": [{"bbox": [16.0, 16.0, 96.0, 96.0]}],
            "ignore_annotations": [],
            "image_tensor": torch.rand(3, 128, 128),
        }
        records = [record]
        settings = {"loss": _settings(), "ignore_iou": 0.5, "reuse_weight_cap": 0.5}
        counts = {"backbone": 0, "rpn": 0}

        def count_backbone(_module, _inputs, _output):
            counts["backbone"] += 1

        def count_rpn(_module, _inputs, _output):
            counts["rpn"] += 1

        backbone_hook = model.backbone.register_forward_hook(count_backbone)
        rpn_hook = model.rpn.register_forward_hook(count_rpn)
        cache = SupportScoreCache(model, device, settings)
        try:
            with torch.no_grad():
                first = cache.score_where(records)
                self.assertIsNotNone(first)
                after_first = dict(counts)
                second = cache.score_where(records)
                self.assertEqual(second, first)
                self.assertEqual(counts, after_first)
                cache.mark("Where", ("candidate", "mvtec:alpha"))
                cache.score_where(records)
                self.assertEqual(counts["backbone"], after_first["backbone"])
                self.assertGreater(counts["rpn"], after_first["rpn"])
                marked_rpn = counts["rpn"]
                cache.mark("Feature", ("candidate", "mvtec:alpha"))
                cache.score_where(records)
                self.assertGreater(counts["backbone"], after_first["backbone"])
                self.assertGreater(counts["rpn"], marked_rpn)
            with torch.enable_grad():
                with self.assertRaises(RuntimeError):
                    cache.score_where(records)
        finally:
            cache.close()
            backbone_hook.remove()
            rpn_hook.remove()

        def shift_box(amount):
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name.startswith("roi_heads.box_predictor.bbox_pred."):
                        parameter.add_(amount)

        torch.manual_seed(8)
        with torch.no_grad():
            legacy_feature = _legacy_score(model, "Feature", records, device, settings)
            legacy_where = _legacy_score(model, "Where", records, device, settings)
            legacy_head = _legacy_score(model, "Box", records, device, settings)
            shift_box(0.01)
            legacy_shifted = _legacy_score(model, "Box", records, device, settings)
            shift_box(-0.01)
        direct = SupportScoreCache(model, device, settings)
        try:
            torch.manual_seed(8)
            with torch.no_grad():
                self.assertAlmostEqual(direct.score_feature(records), legacy_feature, places=5)
                self.assertAlmostEqual(direct.score_where(records), legacy_where, places=5)
                self.assertAlmostEqual(direct.score_head(records), legacy_head, places=5)
                direct.mark("Box", ("candidate", "mvtec:up"))
                shift_box(0.01)
                self.assertAlmostEqual(direct.score_head(records), legacy_shifted, places=5)
                shift_box(-0.01)
        finally:
            direct.close()

        library = {
            "mvtec:up": {"Box": _predictor_delta(model, "roi_heads.box_predictor.bbox_pred.", 0.01)},
            "mvtec:down": {"Box": _predictor_delta(model, "roi_heads.box_predictor.bbox_pred.", -0.01)},
        }
        initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        torch.manual_seed(4)
        with torch.no_grad():
            new_chosen, new_choices, new_weights = select_module_initialization(
                model, library, "visa:target", records, device, settings,
            )
        new_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        model.load_state_dict(initial)
        torch.manual_seed(4)
        with torch.no_grad():
            _legacy_choices, legacy_weights = _legacy_select(
                model, library, "visa:target", records, device, settings,
            )
        self.assertEqual(new_choices, _legacy_choices)
        for name in ("Feature", "Where", "Box", "Class"):
            self.assertAlmostEqual(new_weights[name], legacy_weights[name], places=5)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.allclose(value, new_state[key], rtol=1e-5, atol=1e-6), key)
        self.assertLessEqual(max(new_weights.values()), 0.5)
        self.assertTrue(all(weight < 1.0 for weight in new_weights.values()))
        if new_chosen:
            self.assertNotIn("Class", new_chosen)
            self.assertNotIn("Feature", new_chosen)
            self.assertNotIn("Where", new_chosen)

    def test_count_sketch_ranks_the_descent_direction(self):
        aligned = {
            "backbone.weight": torch.randn(6, 3),
            "roi_heads.box_head.weight": torch.randn(2, 2),
        }
        opposite = {name: -value for name, value in aligned.items()}
        grad = {name: -value.clone() for name, value in aligned.items()}
        first = module_count_sketch(aligned)
        self.assertEqual(first.dtype, torch.float16)
        self.assertEqual(tuple(first.shape), (SKETCH_DIM,))
        self.assertTrue(torch.equal(first, module_count_sketch(aligned)))
        self.assertGreater(
            cosine_neg_grad(module_count_sketch(grad), first),
            cosine_neg_grad(module_count_sketch(grad), module_count_sketch(opposite)),
        )
        self.assertEqual(cosine_neg_grad(torch.zeros(SKETCH_DIM), torch.zeros(SKETCH_DIM)), -1.0)
        grad_sketches = {
            "Feature": module_count_sketch(grad),
            "Where": torch.zeros(SKETCH_DIM, dtype=torch.float16),
        }
        library = {
            "mvtec:z": {"delta": {"Feature": opposite}},
            "mvtec:a": {"Feature": aligned},
            "mvtec:same": {"delta": {"Feature": aligned}, "sketch": {"Feature": first}},
        }
        ranking = _rank_sources(library, "mvtec:same", grad_sketches)
        self.assertEqual(ranking["Feature"], ["mvtec:a", "mvtec:z"])
        self.assertEqual(ranking["Where"], [])
        tied = {
            "mvtec:z": {"Feature": aligned},
            "mvtec:a": {"Feature": {name: value.clone() for name, value in aligned.items()}},
        }
        self.assertEqual(_rank_sources(tied, "other", grad_sketches)["Feature"], ["mvtec:a", "mvtec:z"])

    def test_four_module_checkpoint_uses_parameter_names(self):
        class _Detector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Linear(1, 1, bias=False)
                self.backbone.register_buffer("running_mean", torch.zeros(1))
                self.rpn = torch.nn.Linear(1, 1, bias=False)
                self.roi_heads = torch.nn.Module()
                self.roi_heads.add_module("box_head", torch.nn.Linear(1, 1, bias=False))
                predictor = torch.nn.Module()
                predictor.add_module("bbox_pred", torch.nn.Linear(1, 1, bias=False))
                predictor.add_module("cls_score", torch.nn.Linear(1, 1, bias=False))
                self.roi_heads.add_module("box_predictor", predictor)
                self.extra = torch.nn.Linear(1, 1, bias=False)

        model = _Detector()
        feature_names = _module_parameter_names(model, "Feature")
        self.assertEqual(feature_names, {"backbone.weight", "roi_heads.box_head.weight"})
        base = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        adapted = {name: tensor.clone() for name, tensor in base.items()}
        adapted["backbone.weight"] = adapted["backbone.weight"] + 0.1
        adapted["backbone.running_mean"] = adapted["backbone.running_mean"] + 1
        _checkpoint_models({"model": base}, {"model": adapted}, model, four_modules=True)
        with self.assertRaises(ValueError):
            _checkpoint_models({"model": base}, {"model": adapted}, model, four_modules=False)
        outside = {name: tensor.clone() for name, tensor in base.items()}
        outside["extra.weight"] = outside["extra.weight"] + 0.1
        with self.assertRaises(ValueError):
            _checkpoint_models({"model": base}, {"model": outside}, model, four_modules=True)
        block = {name: base[name].clone() for name in feature_names}
        checked = _checked_block(block, feature_names, "mvtec:bottle", "Feature", base)
        self.assertEqual(set(checked), feature_names)
        with self.assertRaises(ValueError):
            _checked_block({**block, "backbone.running_mean": base["backbone.running_mean"]}, feature_names, "mvtec:bottle", "Feature", base)

    def test_prescreen_keeps_weights_and_can_skip_feature_sources(self):
        from torchvision.models.detection import fasterrcnn_resnet50_fpn

        torch.manual_seed(0)
        device = torch.device("cpu")
        model = fasterrcnn_resnet50_fpn(
            weights=None, weights_backbone=None, num_classes=2, min_size=128, max_size=128,
        )
        model.eval()
        record = {
            "image_id": "synthetic",
            "normal_flag": 0,
            "annotations": [{"bbox": [16.0, 16.0, 96.0, 96.0]}],
            "ignore_annotations": [],
            "image_tensor": torch.rand(3, 128, 128),
        }
        settings = {
            "loss": _settings(), "where_loss_weight": 1.0, "ignore_iou": 0.5,
            "gradient_checkpointing": True, "reuse_weight_cap": 0.5,
        }
        torch.manual_seed(3)
        rng = torch.get_rng_state()
        parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        buffers = {name: buf.detach().clone() for name, buf in model.named_buffers()}
        sketches = _prescreen_sketches(model, [record], device, settings)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter, parameters[name]), name)
            self.assertIsNone(parameter.grad)
        for name, buf in model.named_buffers():
            self.assertTrue(torch.equal(buf, buffers[name]), name)
        for label in ("Feature", "Where", "Box", "Class"):
            self.assertEqual(tuple(sketches[label].shape), (SKETCH_DIM,))
            self.assertEqual(sketches[label].dtype, torch.float16)

        feature_name = next(name for name, _parameter in model.named_parameters() if name.startswith("backbone."))
        library = {
            "mvtec:up": {
                "Feature": {feature_name: torch.full_like(parameters[feature_name].cpu(), 0.01)},
                "Box": _predictor_delta(model, "roi_heads.box_predictor.bbox_pred.", 0.01),
            },
            "mvtec:down": {
                "Feature": {feature_name: torch.full_like(parameters[feature_name].cpu(), -0.01)},
                "Box": _predictor_delta(model, "roi_heads.box_predictor.bbox_pred.", -0.01),
            },
        }
        counts = {"backbone": 0, "predictor": 0}

        def count_backbone(_module, _inputs, _output):
            counts["backbone"] += 1

        def count_predictor(_module, _inputs, _output):
            counts["predictor"] += 1

        backbone_hook = model.backbone.register_forward_hook(count_backbone)
        predictor_hook = model.roi_heads.box_predictor.register_forward_hook(count_predictor)
        initial = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        try:
            torch.manual_seed(4)
            _chosen, full_choices, _weights = select_module_initialization(
                model, library, "visa:target", [record], device, settings,
            )
            full_backbone = counts["backbone"]
            model.load_state_dict(initial)
            counts["backbone"] = 0
            counts["predictor"] = 0
            torch.manual_seed(4)
            _chosen, open_box, _weights = select_module_initialization(
                model, library, "visa:target", [record], device, settings,
                {"Feature": set(), "Where": set()},
            )
            open_predictor = counts["predictor"]
            empty_backbone = counts["backbone"]
            model.load_state_dict(initial)
            counts["predictor"] = 0
            torch.manual_seed(4)
            _chosen, closed_box, _weights = select_module_initialization(
                model, library, "visa:target", [record], device, settings,
                {"Feature": set(), "Where": set(), "Box": set()},
            )
        finally:
            backbone_hook.remove()
            predictor_hook.remove()
        self.assertGreater(full_backbone, empty_backbone)
        self.assertEqual(full_choices["Feature"] in {"Null", "mvtec:up", "mvtec:down"}, True)
        self.assertEqual(open_box["Feature"], "Null")
        self.assertEqual(open_box["Where"], "Null")
        self.assertGreater(open_predictor, counts["predictor"])
        self.assertEqual(closed_box["Box"], "Null")
        self.assertEqual(open_box["Class"], closed_box["Class"])

    def test_prescreen_audit_starts_after_two_k_sources_and_samples(self):
        self.assertFalse(prescreen_audit_due(8, 4, 10, 10, 0, 5))
        self.assertFalse(prescreen_audit_due(9, 4, 9, 10, 0, 5))
        self.assertTrue(prescreen_audit_due(9, 4, 10, 10, 0, 5))
        self.assertTrue(prescreen_audit_due(9, 4, 20, 10, 1, 5))
        self.assertFalse(prescreen_audit_due(9, 4, 20, 10, 5, 5))
        self.assertFalse(prescreen_audit_due(9, 0, 10, 10, 0, 5))

    def test_sketch_dim_mismatch_recomputes_from_delta(self):
        delta = {"backbone.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
        stored = module_count_sketch(delta, 32)
        memory = {"delta": {"Feature": delta}, "sketch": {"Feature": stored}}
        _source_sketch.warned = False
        self.assertTrue(torch.equal(_source_sketch(memory, "Feature", delta, 32), stored))
        short = {"delta": {"Feature": delta}, "sketch": {"Feature": torch.zeros(4, dtype=torch.float16)}}
        _source_sketch.warned = False
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rebuilt = _source_sketch(short, "Feature", delta, 32)
        self.assertEqual(tuple(rebuilt.shape), (32,))
        self.assertIn("警告", buffer.getvalue())
        self.assertTrue(torch.equal(rebuilt, module_count_sketch(delta, 32)))

    def test_meta_val_subset_excludes_outer_test_and_stays_fixed(self):
        from common.io import load_verified_research_splits

        _outer, meta, _inner = load_verified_research_splits()
        provenance = {
            "meta_val_family_ids": meta["meta_val_family_ids"],
            "outer_test_family_ids": _outer["outer_test_family_ids"],
        }
        episodes = load_meta_val_episodes(provenance, 0, 1)
        outer = set(_outer["outer_test_family_ids"])
        self.assertGreater(len(episodes), 4)
        self.assertTrue(all(row["bank"] == "meta_val" and row["family_id"] not in outer for row in episodes))
        first = select_meta_val_episodes(episodes, 4, 7)
        second = select_meta_val_episodes(episodes, 4, 7)
        self.assertEqual([row["episode_id"] for row in first], [row["episode_id"] for row in second])
        self.assertEqual(len(first), 4)
        foreign = dict(episodes[0])
        foreign["family_id"] = next(iter(outer))
        with self.assertRaises(ValueError):
            _assert_meta_val_episode(foreign, set(meta["meta_val_family_ids"]), outer)

    def test_val_checkpoint_keeps_theta0_unless_ap50_is_higher(self):
        self.assertEqual(choose_val_checkpoint([(0, 0.20), (1, 0.19), (2, 0.20)]), (0, "theta0_best"))
        self.assertEqual(choose_val_checkpoint([(0, 0.20), (1, 0.21)]), (1, "meta_val_ap50"))
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "meta_val_ap50.csv"
            _append_val_csv(path, 0, 0.2, 0.2, 0.5)
            _append_val_csv(path, 1, 0.2, 0.21, 0.4)
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], "epoch,theta0_ap50,current_ap50,meta_val_query_loss")
        self.assertEqual(len(lines), 3)

    def test_holdout_rehearsal_reports_failure_without_deleting_files(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            supervised = root / "supervised.pt"
            fomaml = root / "last.pt"
            memory = root / "module_memory.pt"
            supervised.write_bytes(b"supervised")
            fomaml.write_bytes(b"fomaml")
            memory.write_bytes(b"memory")
            with patch("training.meta_transfer.load_holdout", side_effect=RuntimeError("检查未通过")):
                message = rehearse_holdout_load(supervised, fomaml)
            self.assertIn("失败", message)
            self.assertIn("检查未通过", message)
            self.assertEqual(supervised.read_bytes(), b"supervised")
            self.assertEqual(fomaml.read_bytes(), b"fomaml")
            self.assertEqual(memory.read_bytes(), b"memory")
            with patch("training.meta_transfer.load_holdout", return_value={"sources": {}}):
                self.assertIn("通过", rehearse_holdout_load(supervised, fomaml))

    def test_rng_snapshot_restores_torch_state(self):
        torch.manual_seed(1)
        state = _capture_rng_state()
        torch.manual_seed(2)
        _restore_rng_state(state)
        self.assertTrue(torch.equal(torch.get_rng_state(), state["torch"]))

    def test_phase_log_names_the_episode_and_stage(self):
        import io
        from contextlib import redirect_stdout

        trace = {"epoch": 1, "step": 31, "episode": "oof:mvtec:bottle:000", "phase": "进入"}
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            _fomaml_phase(trace, "Feature", "sources=4")
            _fomaml_phase(None, "Where")
        text = buffer.getvalue().strip().splitlines()
        self.assertEqual(len(text), 1)
        self.assertIn("epoch=1", text[0])
        self.assertIn("step=31", text[0])
        self.assertIn("episode=oof:mvtec:bottle:000", text[0])
        self.assertIn("phase=Feature", text[0])
        self.assertIn("sources=4", text[0])
        self.assertEqual(trace["phase"], "Feature")

    def test_innovation1_uses_every_module_and_many_sources(self):
        self.assertEqual(load_innovation1_modules(), ["Feature", "Where", "Box", "Class"])
        sources = fold_source_products(0)
        self.assertGreater(len(sources), 1)
        _, _, inner = load_verified_research_splits()
        fold = next(row for row in inner["folds"] if row["fold_id"] == 0)
        self.assertTrue(set(sources).isdisjoint(fold["val_product_ids"]))


def _predictor_delta(model, prefix: str, scale: float) -> dict:
    return {
        name: torch.full_like(parameter.detach().cpu(), scale)
        for name, parameter in model.named_parameters()
        if name.startswith(prefix)
    }


def _restore_named(model, saved) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(saved[name])


def _apply_delta(model, delta, alpha: float = 1.0) -> None:
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in delta.items():
            parameter = parameters[name]
            parameter.add_(value.to(device=parameter.device, dtype=parameter.dtype), alpha=alpha)


def _legacy_score(model, module_name, records, device, settings):
    if module_name == "Feature":
        losses = []
        for record in records:
            view = eval_support_view(model, record, device, settings["ignore_iou"], keep_pyramid=False)
            losses.append(float(fixed_proposal_detection_loss(
                model, view, device, settings["loss"], settings["ignore_iou"],
            )))
        return _mean_optional(losses)
    if module_name == "Where":
        scores = []
        for record in records:
            view = eval_support_view(model, record, device, settings["ignore_iou"], keep_pyramid=True)
            scores.append(proposal_recall(model, view, device))
        return _mean_optional(scores)
    losses = []
    coder = model.roi_heads.box_coder
    for record in records:
        example = prepare_roi_example(model, record, device, settings["ignore_iou"])
        if example is None:
            continue
        losses.append(float(roi_detection_loss(
            model.roi_heads.box_predictor, example, coder, settings["loss"], device,
        )))
    return _mean_optional(losses)


def _legacy_select(model, library, family_id, records, device, settings):
    modules = ("Feature", "Where", "Box", "Class")
    saved = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
    chosen, choices, weights = {}, {}, {}
    model.eval()
    for module_name in modules:
        base = _legacy_score(model, module_name, records, device, settings)
        best_name, best_weight, best_delta = "Null", 0.0, None
        for source_family, memory in library.items():
            if source_family == family_id:
                continue
            delta = memory.get(module_name) or {}
            if not delta or base is None:
                continue
            _restore_named(model, saved)
            for block in chosen.values():
                _apply_delta(model, block)
            _apply_delta(model, delta)
            score = _legacy_score(model, module_name, records, device, settings)
            if score is None:
                continue
            weight = reuse_weight(
                base, score, settings["reuse_weight_cap"], higher_is_better=module_name == "Where",
            )
            if weight > best_weight:
                best_name, best_weight, best_delta = source_family, weight, delta
        _restore_named(model, saved)
        for block in chosen.values():
            _apply_delta(model, block)
        choices[module_name] = best_name
        weights[module_name] = best_weight
        if best_delta is not None and best_weight > 0:
            scaled = {name: value.mul(float(best_weight)) for name, value in best_delta.items()}
            chosen[module_name] = scaled
            _apply_delta(model, scaled)
    return choices, weights


if __name__ == "__main__":
    unittest.main()

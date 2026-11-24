from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from training.evaluation import _QueryMatches, evaluate_detector, operating_point_comparison, recall_at_normal_fp


_BOX = [0, 0, 10, 10]
_IGNORE = [20, 20, 30, 30]
_CONFIG = {
    "iou_thresholds": [0.5, 0.75],
    "working_iou": 0.5,
    "score_threshold": 0.5,
    "ignore_iou": 0.5,
    "batch_size": 2,
}


def _prediction(boxes, scores):
    return {
        "boxes": torch.tensor(boxes).reshape(-1, 4),
        "scores": torch.tensor(scores),
        "labels": torch.ones(len(scores), dtype=torch.int64),
    }


def _record(image_id, *, normal=False, product="test", positives=(), ignore=()):
    return {
        "image_id": str(image_id),
        "image_path": str(image_id),
        "dataset": "sample",
        "product_id": product,
        "split_role": "QUERY_NORMAL" if normal else "QUERY_DEFECT",
        "normal_flag": int(normal),
        "annotations": [{"bbox": box} for box in positives],
        "ignore_annotations": [{"bbox": box} for box in ignore],
    }


class _Detector(torch.nn.Module):
    def __init__(self, predictions):
        super().__init__()
        self.predictions = predictions
        self.seen = []

    def forward(self, images):
        assert not self.training and not torch.is_grad_enabled()
        image_ids = [int(image.item()) for image in images]
        self.seen.extend(image_ids)
        return [self.predictions[image_id] for image_id in image_ids]


class B0EvaluationTest(unittest.TestCase):
    def test_matched_recall_uses_one_score_threshold(self):
        normal = _QueryMatches(
            scores=np.array([0.9]),
            positive_iou=np.zeros((1, 0)),
            ignore_overlap=np.array([False]),
            pure_normal=True,
        )
        defect = _QueryMatches(
            scores=np.array([0.8]),
            positive_iou=np.array([[1.0]]),
            ignore_overlap=np.array([False]),
            pure_normal=False,
        )
        blocked = recall_at_normal_fp([normal, defect], 0.5, 0.0)
        allowed = recall_at_normal_fp([normal, defect], 0.5, 1.0)
        self.assertEqual(blocked["recall"], 0.0)
        self.assertEqual(allowed["recall"], 1.0)
        hit = _QueryMatches(
            scores=np.array([0.9]),
            positive_iou=np.array([[1.0]]),
            ignore_overlap=np.array([False]),
            pure_normal=False,
        )
        miss = _QueryMatches(
            scores=np.array([0.9]),
            positive_iou=np.array([[0.0]]),
            ignore_overlap=np.array([False]),
            pure_normal=False,
        )
        reference = {("visa", "chewinggum"): [hit], ("mvtec", "capsule"): [hit]}
        candidate = {("visa", "chewinggum"): [hit], ("mvtec", "capsule"): [miss]}
        summary = operating_point_comparison(reference, candidate, _CONFIG)
        self.assertEqual(summary["product_count"], 2)
        self.assertEqual(summary["ap50_positive_products"], 0)
        self.assertLess(summary["ap50_product_min"], summary["ap50_product_max"])

    def _evaluate(self, model, *episode_records):
        def build_records(episode, roles):
            self.assertEqual(roles, ("query_normal", "query_defect"))
            return episode["records"]

        adapter = SimpleNamespace(
            config={"classes": {"defect": 1}}, build_episode_records=build_records
        )
        episodes = [
            {"episode_id": str(index), "records": records}
            for index, records in enumerate(episode_records)
        ]
        with patch("training.evaluation.image_to_tensor", side_effect=lambda path: torch.tensor(int(path))):
            result = evaluate_detector(model, episodes, adapter, "cpu", _CONFIG)
        self.assertTrue(model.training)
        return result

    def test_ignore_duplicate_normal_and_ap_confidence(self):
        model = _Detector(
            {
                1: _prediction([_IGNORE, _BOX, _BOX], [0.99, 0.95, 0.9]),
                2: _prediction([_BOX], [0.8]),
                3: _prediction([_BOX], [0.3]),
            }
        )
        result = self._evaluate(
            model,
            [
                _record(1, positives=[_BOX], ignore=[_IGNORE]),
                _record(2, normal=True),
                _record(3, positives=[_BOX]),
            ],
        )["product_macro"]
        self.assertAlmostEqual(result["ap50"], (51 + 50 * 0.5) / 101)
        self.assertEqual(result["ap"], result["ap50"])
        self.assertAlmostEqual(result["precision"], 1 / 3)
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertEqual(result["normal_fp_per_image"], 1)

    def test_repeat_query_keeps_episode_scope_and_reuses_predictions(self):
        model = _Detector({1: _prediction([_BOX], [0.9]), 2: _prediction([], [])})
        normal = _record(2, normal=True)
        result = self._evaluate(
            model,
            [_record(1, positives=[_BOX]), normal],
            [_record(1, ignore=[_BOX]), normal],
        )
        self.assertEqual(model.seen, [1, 2])
        self.assertEqual(result["counts"]["query_image_occurrences"], 4)
        self.assertEqual(result["product_macro"]["precision"], 1)
        self.assertEqual(result["product_macro"]["ap50"], 1)

    def test_product_macro_with_empty_predictions_and_distinct_iou(self):
        model = _Detector({1: _prediction([[0, 0, 7, 10]], [0.9]), 2: _prediction([], [])})
        result = self._evaluate(
            model,
            [_record(1, product="one", positives=[_BOX])],
            [_record(2, product="two", positives=[_BOX, _IGNORE])],
        )
        self.assertEqual(result["product_macro"]["ap50"], 0.5)
        self.assertEqual(result["per_dataset"]["sample"]["ap"], 0.25)
        self.assertIsNone(result["product_macro"]["normal_fp_per_image"])


if __name__ == "__main__":
    unittest.main()

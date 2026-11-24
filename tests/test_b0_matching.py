"""验证检测器配置中的 RPN 相似度与锚框训练标签。"""

import math
import unittest

import torch
from torchvision.ops import box_iou

from models.detector import build_faster_rcnn


def _build_rpn(**model_config):
    return build_faster_rcnn({
        "architecture": "fasterrcnn_resnet50_fpn",
        "num_classes": 2,
        **model_config,
    }).rpn


class B0MatchingTest(unittest.TestCase):
    def test_default_hybrid_accounts_for_center_and_half_size(self):
        rpn = _build_rpn()
        ground_truth = torch.tensor([[0.0, 0.0, 4.0, 4.0]])
        candidates = torch.tensor([
            [0.0, 0.0, 4.0, 4.0],
            [3.0, 4.0, 7.0, 8.0],
            [-2.0, -2.0, 6.0, 6.0],
        ])
        expected = torch.tensor([[1.0, math.exp(-5 / 12.8), math.exp(-math.sqrt(8) / 12.8)]])
        torch.testing.assert_close(rpn.box_similarity(ground_truth, candidates), expected)

    def test_hybrid_score_preserves_stronger_iou_or_nwd(self):
        constant = 12.8
        rpn = _build_rpn(rpn_matching={"mode": "iou_nwd", "nwd_constant": constant})
        ground_truth = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 128.0, 128.0]])
        anchors = torch.tensor([[3.0, 0.0, 5.0, 2.0], [4.0, 0.0, 132.0, 128.0]])
        expected = torch.tensor([math.exp(-3 / constant), 124 / 132])
        torch.testing.assert_close(rpn.box_similarity(ground_truth, anchors).diagonal(), expected)

    def test_training_assignment_uses_hybrid_score_and_existing_thresholds(self):
        ground_truth = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        anchors = torch.tensor([
            [0.0, 0.0, 2.0, 2.0],
            [3.0, 0.0, 5.0, 2.0],
            [9.0, 0.0, 11.0, 2.0],
            [64.0, 0.0, 66.0, 2.0],
        ])
        rpn = _build_rpn(rpn_matching={"mode": "iou_nwd", "nwd_constant": 12.8})
        matcher = rpn.proposal_matcher
        self.assertEqual((matcher.high_threshold, matcher.low_threshold), (0.7, 0.3))
        self.assertEqual(matcher(box_iou(ground_truth, anchors))[1].item(), matcher.BELOW_LOW_THRESHOLD)
        labels, _ = rpn.assign_targets_to_anchors(
            [anchors, anchors], [{"boxes": ground_truth}, {"boxes": torch.empty(0, 4)}],
        )
        torch.testing.assert_close(labels[0], torch.tensor([1.0, 1.0, -1.0, 0.0]))
        torch.testing.assert_close(labels[1], torch.zeros(len(anchors)))

    def test_invalid_rpn_matching_config_fails(self):
        for config in (
            {"mode": "iou", "nwd_constant": 12.8},
            {"mode": "nwd", "nwd_constant": 12.8},
            {"mode": "iou_nwd", "nwd_constant": 0.0},
            {"mode": "iou_nwd", "nwd_constant": math.inf},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                _build_rpn(rpn_matching=config)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import torch

from training.meta_transfer import HoldoutFrame, summarize_proposals


class HoldoutProposalTest(unittest.TestCase):
    def test_proposal_summary_counts_gt_coverage(self):
        frame = HoldoutFrame(
            image_id="q",
            proposals=torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
            features=torch.zeros(2, 4),
            valid_size=(40, 40),
            original_size=(40, 40),
        )
        records = [[{
            "image_id": "q",
            "dataset": "sample",
            "product_id": "product",
            "normal_flag": 0,
            "annotations": [{"bbox": [0.0, 0.0, 10.0, 10.0]}],
        }]]
        result = summarize_proposals({"q": frame}, records)
        self.assertEqual(result["defect"]["gt_boxes"], 1)
        self.assertEqual(result["defect"]["gt_covered"], 1)
        self.assertEqual(result["defect"]["recall_at_50"], 1.0)


if __name__ == "__main__":
    unittest.main()

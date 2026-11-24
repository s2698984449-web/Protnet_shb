import unittest

import torch

from data_proc.detection_geometry import ShrinkOnlyDetectionTransform, image_geometry
from models.detector import build_faster_rcnn


def _build_b0_transform(**overrides):
    return build_faster_rcnn({
        "architecture": "fasterrcnn_resnet50_fpn",
        "num_classes": 2,
        **overrides,
    }).transform


class B0GeometryTest(unittest.TestCase):
    def test_short_and_long_caps_without_upscaling(self):
        for original_size, valid_size, padded_size in (
            ((769, 1001), (769, 1001), (800, 1024)),
            ((1600, 2000), (1024, 1280), (1024, 1280)),
            ((900, 1800), (768, 1536), (768, 1536)),
            ((1800, 900), (1536, 768), (1536, 768)),
        ):
            with self.subTest(original_size=original_size):
                geometry = image_geometry(original_size, 1024, 1536)
                self.assertEqual(geometry.original_size, original_size)
                self.assertEqual(geometry.valid_size, valid_size)
                self.assertEqual(geometry.padded_size, padded_size)

    def test_box_mapping_and_valid_boundary_clipping(self):
        geometry = image_geometry((1000, 1999), 1024, 1536)
        self.assertEqual(geometry.valid_size, (768, 1536))
        original = torch.tensor([[199.9, 200.0, 999.5, 750.0]])
        valid = torch.tensor([[153.6, 153.6, 768.0, 576.0]])
        torch.testing.assert_close(geometry.to_valid_boxes(original), valid)
        torch.testing.assert_close(geometry.to_original_boxes(valid), original)
        torch.testing.assert_close(
            geometry.to_original_boxes(torch.tensor([[-5.0, -7.0, 1560.0, 800.0]])),
            torch.tensor([[0.0, 0.0, 1999.0, 1000.0]]),
        )
        self.assertEqual(geometry.to_valid_boxes(torch.empty(0, 4)).shape, (0, 4))

    def test_b0_training_and_evaluation_geometry(self):
        transform = ShrinkOnlyDetectionTransform(96, 128, [0.0] * 3, [1.0] * 3)
        small = torch.ones(3, 37, 69)
        large = torch.ones(3, 192, 256)
        small_target = {"boxes": torch.empty(0, 4), "area": torch.empty(0)}
        large_target = {
            "boxes": torch.tensor([[40.0, 20.0, 150.0, 170.0]]),
            "area": torch.tensor([16500.0]),
        }
        target_boxes = large_target["boxes"].clone()
        target_area = large_target["area"].clone()
        batch, targets = transform([small, large], [small_target, large_target])
        self.assertEqual(batch.image_sizes, [(37, 69), (96, 128)])
        self.assertEqual(batch.tensors.shape, (2, 3, 96, 128))
        torch.testing.assert_close(batch.tensors[0, :, :37, :69], small)
        self.assertEqual(torch.count_nonzero(batch.tensors[0, :, 37:, :]).item(), 0)
        self.assertEqual(torch.count_nonzero(batch.tensors[0, :, :, 69:]).item(), 0)
        self.assertEqual(targets[0]["boxes"].shape, (0, 4))
        torch.testing.assert_close(targets[1]["boxes"], torch.tensor([[20.0, 10.0, 75.0, 85.0]]))
        torch.testing.assert_close(targets[1]["area"], torch.tensor([4125.0]))
        torch.testing.assert_close(large_target["boxes"], target_boxes)
        torch.testing.assert_close(large_target["area"], target_area)

        transform.eval()
        eval_batch, _ = transform([small, large])
        self.assertEqual(eval_batch.image_sizes, batch.image_sizes)
        eval_output = transform.postprocess(
            [{"boxes": target["boxes"]} for target in targets],
            eval_batch.image_sizes,
            [(37, 69), (192, 256)],
        )
        self.assertEqual(eval_output[0]["boxes"].shape, (0, 4))
        torch.testing.assert_close(eval_output[1]["boxes"], target_boxes)

    def test_default_b0_uses_shared_geometry_without_upscaling(self):
        transform = _build_b0_transform().eval()
        self.assertIsInstance(transform, ShrinkOnlyDetectionTransform)
        self.assertEqual((transform.min_size, transform.max_size), ((1024,), 1536))
        batch, _ = transform([torch.ones(3, 31, 47)])
        self.assertEqual(batch.image_sizes, [(31, 47)])
        self.assertEqual(batch.tensors.shape[-2:], (32, 64))

    def test_b0_rejects_legacy_geometry_overrides(self):
        for overrides in ({"min_size": 256}, {"max_size": 512}, {"resize_mode": "torchvision"}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                _build_b0_transform(**overrides)


if __name__ == "__main__":
    unittest.main()

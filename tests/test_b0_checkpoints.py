"""验证断点和最佳权重在打断后保持一致。"""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from training.checkpoints import (
    b0_run_directory,
    load_checkpoint,
    restore_best_checkpoint,
    save_checkpoint,
)


class B0CheckpointTest(TestCase):
    def test_failed_atomic_save_keeps_existing_checkpoint(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_checkpoint(path, {"epoch": 1})
            with patch("training.checkpoints.os.replace", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    save_checkpoint(path, {"epoch": 2})
            self.assertEqual(load_checkpoint(path), {"epoch": 1})
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_recovery_publishes_only_committed_best(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            best = root / "best.pt"
            pending = root / "best.pending.pt"
            save_checkpoint(best, {"epoch": 1})
            save_checkpoint(pending, {"epoch": 2})
            restore_best_checkpoint(root, best_epoch=1)
            self.assertFalse(pending.exists())
            self.assertEqual(load_checkpoint(best), {"epoch": 1})
            save_checkpoint(pending, {"epoch": 2})
            restore_best_checkpoint(root, best_epoch=2)
            self.assertFalse(pending.exists())
            self.assertEqual(load_checkpoint(best), {"epoch": 2})

    def test_concurrent_resume_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            with b0_run_directory(root, resume=False):
                with self.assertRaises(RuntimeError):
                    with b0_run_directory(root, resume=True):
                        self.fail("一个实验目录允许了两个写入者")
            with b0_run_directory(root, resume=True):
                self.assertTrue(root.is_dir())

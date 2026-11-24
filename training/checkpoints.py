"""B0 单运行互斥和原子保存。"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch


def _abandoned_run(directory: Path) -> bool:
    """启动阶段中断、还没有 last.pt 的目录可以重新开始。"""
    if not directory.is_dir():
        return False
    names = {path.name for path in directory.iterdir()}
    if "last.pt" in names:
        return False
    allowed = {".run.lock", "metrics.json", "meta_val_ap50.csv", "val_best.pt", "module_memory_parts"}
    if not names <= allowed:
        return False
    if names <= {".run.lock"}:
        return True
    metrics = directory / "metrics.json"
    if not metrics.is_file():
        return False
    try:
        payload = json.loads(metrics.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    # 没有 last.pt 就是没写完，允许同名重开。
    return not payload.get("complete")


@contextmanager
def b0_run_directory(directory: Path, resume: bool):
    if resume:
        if not directory.is_dir():
            raise FileNotFoundError(f"B0 实验目录不存在：{directory}")
    elif _abandoned_run(directory):
        pass
    else:
        directory.mkdir(parents=True, exist_ok=False)
    with (directory / ".run.lock").open("a+b") as lock:
        try:
            if os.name == "nt":
                import msvcrt

                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"B0 实验正在执行：{directory}") from error
        yield


@contextmanager
def _atomic_output(path: Path):
    output = NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False)
    temporary = Path(output.name)
    try:
        with output:
            yield output
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(path: Path, payload: dict) -> None:
    with _atomic_output(path) as output:
        torch.save(payload, output)


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def save_run_metrics(directory: Path, provenance: dict, history: list, best_epoch, best_ap50) -> None:
    payload = {
        "provenance": provenance, "best_epoch": best_epoch,
        "best_ap50": best_ap50, "history": history,
    }
    with _atomic_output(directory / "metrics.json") as output:
        output.write((json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def restore_best_checkpoint(directory: Path, best_epoch: int) -> None:
    """续训时补全上次已提交但尚未发布的最佳权重。"""
    pending = directory / "best.pending.pt"
    best = directory / "best.pt"
    if pending.is_file():
        if load_checkpoint(pending)["epoch"] == best_epoch:
            os.replace(pending, best)
        else:
            pending.unlink()
    if not best.is_file():
        raise FileNotFoundError(f"B0 续训缺少第 {best_epoch} 轮的最佳权重：{best}")

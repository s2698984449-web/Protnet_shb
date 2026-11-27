"""折内监督底座，以及从该底座继续的一阶元初始化。"""

from __future__ import annotations

import csv
import gc
import json
import os
import random
import re
from collections import defaultdict
from time import perf_counter
import numpy as np
import torch
from common.io import file_sha256, json_sha256, load_config, load_verified_research_splits, named_config_path, project_root
from data_proc.baseline_data import B0DetectionDataset
from models.detector import backbone_weight_manifest, build_faster_rcnn
from training.checkpoints import _atomic_output, b0_run_directory, save_checkpoint
from training.train_b0 import (
    _check_runtime, _code_version, _epoch_positions, _prepare_data,
    _sampling_summary, _train_epoch, _train_loader, _training_settings,
)
from importlib.metadata import version
from math import isfinite
from pathlib import Path
from learn2learn.algorithms import MAML
from common.io import file_sha256, json_sha256, load_config, named_config_path, project_root, read_jsonl
from data_proc.fomaml_episodes import (
    InsufficientEpisodeAnnotations, episode_records, load_fomaml_data_config, load_training_episodes,
)
from models.detector import b0_model_config, build_faster_rcnn
from training.checkpoints import _atomic_output, b0_run_directory, load_checkpoint, save_checkpoint
from training.fomaml_roi import (
    MODULE_PREFIXES,
    SKETCH_DIM,
    SupportScoreCache,
    cosine_neg_grad,
    live_detection_loss,
    module_count_sketch,
    prepare_roi_example,
    roi_detection_loss,
    reuse_weight,
    validate_detection_loss_settings,
)
from training.train_b0 import _check_runtime, _code_version


# theta0.py

def _fold_scope(fold_id: int) -> dict:
    outer, meta, inner = load_verified_research_splits()
    matches = [fold for fold in inner["folds"] if fold["fold_id"] == fold_id]
    if len(matches) != 1:
        raise ValueError(f"冻结的 OOF 划分中找不到唯一折编号 {fold_id}")
    fold = matches[0]
    source = set(fold["theta0_train_family_ids"])
    target = set(fold["target_family_ids"])
    meta_val = set(meta["meta_val_family_ids"])
    outer_test = set(outer["outer_test_family_ids"])
    if (
        not source or len(source) != len(fold["theta0_train_family_ids"])
        or source != set(fold["train_family_ids"]) or source != set(fold["delta_source_family_ids"])
        or target != set(fold["val_family_ids"]) or source & (target | meta_val | outer_test)
        or source != set(meta["meta_train_family_ids"]) - target
    ):
        raise AssertionError(f"折 {fold_id} 的 θ₀ 源族与冻结训练及留出边界不一致")
    return {
        "fold_id": fold_id, "fold_sha256": json_sha256(fold),
        "theta0_train_family_ids": sorted(source),
        "delta_source_family_ids": sorted(fold["delta_source_family_ids"]),
        "oof_target_family_ids": sorted(target),
        "meta_val_family_ids": sorted(meta_val), "outer_test_family_ids": sorted(outer_test),
    }


def _pilot_family_ids(index: list[dict]) -> set[str]:
    groups_by_family = defaultdict(dict)
    domains = {}
    for row in index:
        family = row["family_id"]
        domains[family] = row["dataset_id"]
        groups = groups_by_family[family]
        key = (row["product_id"], row["physical_group_id"])
        groups[key] = groups.get(key, False) or not row["normal_flag"]
    selected = {}
    for family, groups in sorted(groups_by_family.items()):
        if set(groups.values()) == {False, True}:
            selected.setdefault(domains[family], family)
    if set(selected) != set(domains.values()):
        raise AssertionError("θ₀ Pilot 某域没有同时含缺陷组与纯正常组的折内产品族")
    return set(selected.values())


def _run_training(adapter, index, plans, sampling, device, provenance, run_dir) -> dict:
    settings = provenance["settings"]
    seed = settings["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    model = build_faster_rcnn(settings["model"]).to(device)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=settings["learning_rate"], momentum=settings["momentum"], weight_decay=settings["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=settings["lr_milestones"], gamma=settings["lr_gamma"],
    )
    dataset = B0DetectionDataset(adapter, index)
    positions = []
    loader = _train_loader(dataset, positions, device, settings, persistent=settings.get("persistent_workers", False))
    history, trained_families = [], set()
    global_step = 0
    for epoch, plan in enumerate(plans, start=1):
        positions[:] = plan
        start = perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        trained = _train_epoch(model, optimizer, dataset, positions, device, settings, epoch, global_step, loader=loader)
        seconds = perf_counter() - start
        skipped = {row["image_id"] for row in trained["skipped_annotations"]}
        used = [index[position] for position in positions if index[position]["image_id"] not in skipped]
        if len(used) != trained["images"] or (provenance["mode"] == "pilot" and skipped):
            raise RuntimeError(f"θ₀ 第 {epoch} 轮实际训练样本与计划不符，跳过：{sorted(skipped)}")
        trained_families.update(row["family_id"] for row in used)
        global_step += trained["steps"]
        history.append({
            "epoch": epoch, "global_step": global_step, "lr": lr,
            "sampling": sampling[epoch - 1], "training": trained, "train_seconds": seconds,
            "train_image_ids": [row["image_id"] for row in used],
        })
        scheduler.step()
        print(f"theta0 fold={provenance['fold_id']} epoch={epoch} steps={global_step} train={seconds:.1f}s", flush=True)
    if trained_families != set(provenance["train_family_ids"]):
        raise AssertionError("实际获得梯度更新的产品族未完整覆盖本次 θ₀ 训练范围")

    checkpoint = {
        "model": model.state_dict(), "provenance": provenance, "epoch": len(history), "global_step": global_step,
    }
    metrics = {
        "complete": True, "provenance": provenance, "epoch": len(history),
        "global_step": global_step, "history": history,
    }
    save_checkpoint(run_dir / "last.pt", checkpoint)
    with _atomic_output(run_dir / "metrics.json") as handle:
        handle.write((json.dumps(metrics, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return metrics


def train_theta0(fold_id: int, run_name: str, pilot: bool, dry_run: bool) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_name):
        raise ValueError("run-name 只能使用字母、数字、连字符和下划线")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    scope = _fold_scope(fold_id)
    baseline_path = named_config_path("baseline.yaml")
    profile_path = named_config_path("b0_pretrained.yaml")
    theta0_path = named_config_path("theta0.yaml")
    config, profile, theta0 = (load_config(path) for path in (baseline_path, profile_path, theta0_path))
    config["training"].update(profile["training"])
    config["training"]["model"].update(profile["model"])
    training = theta0["training"]
    config["training"].update({**training, "sampling": {**config["training"]["sampling"], **training["sampling"]}})
    config["pilot"] = theta0["pilot"]
    settings = _training_settings(config, pilot)
    if settings["optimizer"] != "sgd" or settings["epochs"] < 1:
        raise ValueError("θ₀ 使用 SGD，训练轮数必须为正")

    adapter, raw_index, _, data = _prepare_data(config, settings, pilot=False)
    families = set(scope["theta0_train_family_ids"])
    index = [row for row in raw_index if row["family_id"] in families]
    if {row["family_id"] for row in index} != families:
        raise AssertionError("θ₀ 索引未完整覆盖冻结训练产品族")
    if pilot:
        families = _pilot_family_ids(index)
        index = [row for row in index if row["family_id"] in families]

    plans = [_epoch_positions(index, settings, epoch) for epoch in range(1, settings["epochs"] + 1)]
    sampling = [_sampling_summary(index, plan) for plan in plans]
    if pilot and any(
        counts != {"defect_views": 1, "normal_views": 1}
        for epoch in sampling for counts in epoch["by_dataset"].values()
    ):
        raise AssertionError("θ₀ Pilot 必须在每个域选一张缺陷和一张正常图")

    code = _code_version()
    for path in ("training/theta0.py", "scripts/train.py"):
        code["source_sha256"][path] = file_sha256(project_root() / path)
    provenance = {
        **data, **scope, "train_family_ids": sorted(families), "train_index_sha256": json_sha256(index),
        "mode": "pilot" if pilot else "fold_train_full", "seed": settings["seed"], "settings": settings,
        "config_sha256": file_sha256(baseline_path), "theta0_config_sha256": file_sha256(theta0_path),
        "detector_config_sha256": file_sha256(named_config_path("b0_v3.yaml")), "code": code,
        "initialization": {
            "profile": "pretrained", "profile_config_sha256": file_sha256(profile_path),
            "backbone": backbone_weight_manifest(settings["model"]),
        },
    }
    run_dir = project_root() / theta0["output_dir"] / f"fold_{fold_id}" / run_name
    summary = {
        **scope, "mode": provenance["mode"], "train_family_ids": sorted(families),
        "train_families": len(families), "allowed_train_families": len(scope["theta0_train_family_ids"]),
        "train_index_images": len(index), "epochs": settings["epochs"], "epoch_sampling": sampling,
        "output_dir": str(run_dir),
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if dry_run:
        return summary
    device = _check_runtime(settings)
    with b0_run_directory(run_dir, resume=False):
        return _run_training(adapter, index, plans, sampling, device, provenance, run_dir)

# theta0_fomaml.py

def _phase_examples(detector, records, device, ignore_iou):
    examples = []
    eligible_groups = {(row["split_role"], row["physical_group_id"]) for row in records}
    contributing_groups = set()
    for row in records:
        example = prepare_roi_example(detector, row, device, ignore_iou)
        if example is None:
            continue
        examples.append(example)
        if row["split_role"] == "QUERY_NORMAL" or bool((example.labels > 0).any()):
            contributing_groups.add((row["split_role"], row["physical_group_id"]))
    if not examples or contributing_groups != eligible_groups:
        raise InsufficientEpisodeAnnotations("Support 或 Query 物理组没有可用的 RoI 监督")
    return examples


def _mean_detection_loss(predictor, examples, box_coder, loss_settings, device):
    loss = torch.stack([
        roi_detection_loss(predictor, example, box_coder, loss_settings, device)
        for example in examples
    ]).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("FOMAML 的检测对齐损失不是有限值")
    return loss


def _max_weight_change(parameters, previous):
    return max((current.detach() - old).abs().max().item() for current, old in zip(parameters, previous))


def _memory_part_name(episode_id: str) -> str:
    return episode_id.replace(":", "_") + ".pt"


def _prune_old_experience_epochs(run_dir: Path, keep_epoch: int) -> list[str]:
    """只保留当前经验库目录，删除更早的 experience_epoch_*。"""
    import shutil

    kept = f"experience_epoch_{int(keep_epoch)}"
    removed = []
    for path in sorted(run_dir.glob("experience_epoch_*")):
        if not path.is_dir() or path.name == kept:
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed.append(path.name)
    return removed


def _discard_module_memory_parts(run_dir: Path, module_memory: dict) -> None:
    """训练结束后丢掉任务差卸盘；检索式留出只依赖 experience_epoch + last/best。"""
    import shutil

    for row in module_memory.values():
        if not isinstance(row, dict):
            continue
        row.pop("delta", None)
        row.pop("delta_file", None)
    parts = run_dir / "module_memory_parts"
    if parts.is_dir():
        shutil.rmtree(parts, ignore_errors=True)
    resume = run_dir / "resume.pt"
    if resume.is_file():
        resume.unlink(missing_ok=True)


def _spill_episode_delta(directory: Path, episode_id: str, row: dict, *, release: bool = True) -> None:
    """把任务差写到单独文件。release 为真时从这条记录里丢掉张量，库里仍可留着最新一份。"""
    if isinstance(row.get("delta"), dict) and not row.get("delta_file"):
        directory.mkdir(parents=True, exist_ok=True)
        name = _memory_part_name(episode_id)
        save_checkpoint(directory / name, {"delta": row["delta"]})
        row["delta_file"] = name
    if release and "delta" in row:
        row.pop("delta")
        gc.collect()


def _remember_module_delta(module_memory: dict, library: dict, family_id: str, episode_id: str, row: dict, parts: Path) -> None:
    previous = module_memory.get(episode_id)
    if isinstance(previous, dict) and previous.get("delta_file"):
        (parts / previous["delta_file"]).unlink(missing_ok=True)
    module_memory[episode_id] = row
    prior = library.get(family_id)
    library[family_id] = {"delta": row["delta"], "sketch": row["sketch"], "episode_id": episode_id}
    if prior and prior["episode_id"] != episode_id:
        _spill_episode_delta(parts, prior["episode_id"], module_memory[prior["episode_id"]])


def split_parameter_delta(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> dict:
    """按五组功能参数拆开这一步自己学到的参数差，不把借来的初始化算进这块知识。"""
    if set(before) != set(after):
        raise AssertionError("内外参数名称不一致")
    grouped = {label: {} for label in MODULE_PREFIXES}
    for name, old in before.items():
        owners = [label for label, prefixes in MODULE_PREFIXES.items() if name.startswith(prefixes)]
        if len(owners) != 1:
            raise AssertionError(f"参数没有唯一模块：{name}")
        grouped[owners[0]][name] = (after[name] - old).detach().cpu()
    if any(not block for block in grouped.values()):
        raise AssertionError("五组参数差必须同时留下")
    return grouped


class _ParameterStore:
    """三份可复用的 CPU 镜像：共享参数、内循环前、内循环后。

    参数对象始终是优化器持有的那一份，动量缓冲不会因为快照换掉。
    镜像放在 CPU，避免在 4GB 显卡上再驻留一整套模型。
    """

    def __init__(self, module):
        self.names = []
        self.params = []
        self.slots = {name: [] for name in ("primary", "before", "after")}
        for name, parameter in module.named_parameters():
            self.names.append(name)
            self.params.append(parameter)
            for slot in self.slots.values():
                # 不用锁页内存。PyTorch 2.1 的锁页分配器和 empty_cache 一起用时，会在 c10 里释放仍被拷贝占用的块。
                slot.append(torch.empty(parameter.shape, dtype=parameter.dtype, device="cpu"))

    def _sync(self) -> None:
        parameter = self.params[0] if self.params else None
        if parameter is not None and parameter.is_cuda:
            torch.cuda.synchronize(parameter.device)

    def capture(self, slot: str) -> None:
        for buffer, parameter in zip(self.slots[slot], self.params):
            buffer.copy_(parameter.detach())
        self._sync()

    def restore(self, slot: str) -> None:
        with torch.no_grad():
            for buffer, parameter in zip(self.slots[slot], self.params):
                parameter.copy_(buffer)
        self._sync()

    def slot_dict(self, slot: str) -> dict[str, torch.Tensor]:
        return {name: buffer for name, buffer in zip(self.names, self.slots[slot])}


def _capture_module(module, prefixes) -> list[tuple[torch.nn.Parameter, torch.Tensor]]:
    """只镜像当前正在评分的那一块，其它块留在显存里。"""
    saved = []
    cuda = False
    for name, parameter in module.named_parameters():
        if not name.startswith(prefixes):
            continue
        mirror = torch.empty(parameter.shape, dtype=parameter.dtype, device="cpu")
        mirror.copy_(parameter.detach())
        saved.append((parameter, mirror))
        cuda = cuda or parameter.is_cuda
    if not saved:
        raise AssertionError("模块没有可恢复的参数")
    if cuda:
        torch.cuda.synchronize(saved[0][0].device)
    return saved


def _restore_module(saved: list[tuple[torch.nn.Parameter, torch.Tensor]]) -> None:
    cuda = False
    with torch.no_grad():
        for parameter, mirror in saved:
            parameter.copy_(mirror)
            cuda = cuda or parameter.is_cuda
    if cuda:
        torch.cuda.synchronize(saved[0][0].device)


def _add_named_delta(module, delta: dict[str, torch.Tensor], *, alpha: float = 1.0) -> None:
    if not delta:
        return
    parameters = dict(module.named_parameters())
    with torch.no_grad():
        for name, value in delta.items():
            parameter = parameters[name]
            parameter.add_(value.to(device=parameter.device, dtype=parameter.dtype), alpha=alpha)


def _family_id(episode: dict) -> str:
    family = episode.get("family_id")
    product = episode.get("product_id")
    dataset, separator, product_name = str(family).partition(":")
    if not isinstance(family, str) or not separator or not dataset or product_name != product:
        raise ValueError(f"任务缺少数据集:产品 编号：{episode.get('episode_id')}")
    return family


def _views_per_group(records: list[dict], limit: int) -> list[dict]:
    """每个物理组最多保留 limit 个视角，5-shot 的组覆盖不变。缺陷组优先留有框的视角。"""
    if int(limit) < 1:
        raise ValueError("views_per_group 必须为正")
    buckets: dict[tuple, list[dict]] = {}
    for row in records:
        key = (row["split_role"], row["physical_group_id"])
        buckets.setdefault(key, []).append(row)
    selected = []
    for (role, _), rows in buckets.items():
        if role in ("SUPPORT", "QUERY_DEFECT"):
            ordered = sorted(
                rows,
                key=lambda row: (0 if row.get("annotations") else 1, str(row.get("image_id", ""))),
            )
        else:
            ordered = rows
        selected.extend(ordered[: int(limit)])
    return selected


def _usable_phase_views(detector, records, device, ignore_iou, limit: int, fallback_limit: int | None = None):
    """先按组限额取视角；不够再逐步放宽，但不一次性退回全部视角，避免 4GB 显卡被打崩。"""
    fallback_limit = int(fallback_limit or max(limit, 3))
    if fallback_limit < limit:
        raise ValueError("fallback views_per_group 不能小于主限额")
    attempts = [limit] if fallback_limit == limit else [limit, fallback_limit]
    last_error = None
    for attempt in attempts:
        candidate = _views_per_group(records, attempt)
        try:
            _phase_examples(detector, candidate, device, ignore_iou)
            return candidate
        except InsufficientEpisodeAnnotations as error:
            last_error = error
    if last_error is not None:
        raise last_error
    raise InsufficientEpisodeAnnotations("Support 或 Query 物理组没有可用的 RoI 监督")


def _memory_blocks(memory: dict) -> dict:
    nested = memory.get("delta") if isinstance(memory, dict) else None
    if isinstance(nested, dict) and any(name in MODULE_PREFIXES for name in nested):
        return nested
    return memory


def _source_sketch(memory: dict, module_name: str, delta: dict, dim: int) -> torch.Tensor:
    stored = memory.get("sketch") if isinstance(memory.get("delta"), dict) else None
    sketch = stored.get(module_name) if isinstance(stored, dict) else None
    if isinstance(sketch, torch.Tensor) and int(sketch.numel()) == dim:
        return sketch
    if isinstance(sketch, torch.Tensor) and int(sketch.numel()) != dim and not getattr(_source_sketch, "warned", False):
        _source_sketch.warned = True
        print(
            f"警告：记忆草图维度 {int(sketch.numel())} 与 --sketch-dim {dim} 不一致，已按已存 delta 重算",
            flush=True,
        )
    return module_count_sketch(delta, dim)


def _rank_sources(library: dict, family_id: str, grad_sketches: dict, sketch_dim: int = SKETCH_DIM) -> dict[str, list[str]]:
    """按 cos(-梯度草图, 参数差草图) 给 Backbone/Where 的历史源排序。同族不参与。"""
    ranking = {}
    for module_name in ("Backbone", "Where"):
        scored = []
        for source_family, memory in library.items():
            if source_family == family_id:
                continue
            blocks = _memory_blocks(memory)
            delta = blocks.get(module_name) or {}
            if not delta:
                continue
            sketch = _source_sketch(memory, module_name, delta, sketch_dim)
            scored.append((cosine_neg_grad(grad_sketches[module_name], sketch), source_family))
        scored.sort(key=lambda item: (-item[0], item[1]))
        ranking[module_name] = [source for _score, source in scored]
    return ranking


def _other_source_count(library: dict, family_id: str) -> int:
    return sum(1 for source in library if source != family_id)


def prescreen_audit_due(eligible_sources: int, prescreen_k: int, seen_large: int, every: int, audited: int, cap: int) -> bool:
    """可选源多于 2k 之后，每 every 条抽一条，最多 cap 次。"""
    if cap < 1 or prescreen_k < 1 or every < 1 or audited >= cap:
        return False
    if eligible_sources <= 2 * prescreen_k:
        return False
    return seen_large > 0 and seen_large % every == 0


def _winner_rank(order: list[str], winner: str) -> int | None:
    if winner == "Null" or winner not in order:
        return None
    return order.index(winner) + 1


def _rank_text(ranks: list[int]) -> str:
    if not ranks:
        return "rank_median=NA rank_max=NA"
    from statistics import median

    return f"rank_median={median(ranks):.1f} rank_max={max(ranks)}"


def select_meta_val_episodes(episodes: list[dict], limit: int, seed: int) -> list[dict]:
    """用固定种子抽取同一批 Meta-Val。limit 小于 1 时保留全部。"""
    ordered = sorted(episodes, key=lambda item: item["episode_id"])
    if int(limit) <= 0 or int(limit) >= len(ordered):
        return ordered
    chosen = random.Random(int(seed)).sample(ordered, int(limit))
    return sorted(chosen, key=lambda item: item["episode_id"])


def _assert_meta_val_episode(episode: dict, allowed: set[str], outer: set[str]) -> None:
    family = episode.get("family_id")
    if episode.get("bank") != "meta_val" or family in outer or family not in allowed:
        raise ValueError(f"Meta-Val 评估遇到验证名单之外的任务：{episode.get('episode_id')}")


def load_meta_val_episodes(provenance: dict, limit: int, seed: int) -> list[dict]:
    root = project_root()
    baseline = load_config(named_config_path("baseline.yaml"))
    episode_config = load_config(root / baseline["episode_config"])
    path = root / episode_config["output_dir"] / episode_config["artifacts"]["meta_validation_bank"]
    episodes = read_jsonl(path)
    if not episodes:
        raise ValueError("Meta-Val Episode 为空")
    allowed = set(provenance["meta_val_family_ids"])
    outer = set(provenance["outer_test_family_ids"])
    for episode in episodes:
        _assert_meta_val_episode(episode, allowed, outer)
    return select_meta_val_episodes(episodes, limit, seed)


def choose_val_checkpoint(rows: list[tuple[int, float]]) -> tuple[int, str]:
    """epoch 0 是 θ0。只有严格更高的 Meta-Val AP50 才换掉它。"""
    if not rows or rows[0][0] != 0:
        raise ValueError("Meta-Val 比较需要先有 epoch 0 的 θ0")
    best_epoch, best_ap = rows[0]
    for epoch, ap50 in rows[1:]:
        if ap50 > best_ap:
            best_epoch, best_ap = epoch, ap50
    return best_epoch, ("theta0_best" if best_epoch == 0 else "meta_val_ap50")


def _write_val_csv(path: Path, rows: list[dict]) -> None:
    lines = ["epoch,theta0_ap50,current_ap50,meta_val_query_loss\n"]
    for row in rows:
        lines.append(
            f"{int(row['epoch'])},{float(row['theta0_ap50']):.6f},"
            f"{float(row['current_ap50']):.6f},{float(row['meta_val_query_loss']):.6f}\n"
        )
    with _atomic_output(path) as handle:
        handle.write("".join(lines).encode("utf-8"))


def _append_val_csv(path: Path, epoch: int, theta0_ap50: float, current_ap50: float, query_loss: float) -> None:
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("epoch", "theta0_ap50", "current_ap50", "meta_val_query_loss"),
        )
        if new_file:
            writer.writeheader()
        writer.writerow({
            "epoch": epoch,
            "theta0_ap50": f"{theta0_ap50:.6f}",
            "current_ap50": f"{current_ap50:.6f}",
            "meta_val_query_loss": f"{query_loss:.6f}",
        })


def _capture_rng_state() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].copy(), numpy_state[2], numpy_state[3], numpy_state[4]),
        "torch": torch.get_rng_state().clone(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _rng_to_checkpoint(state: dict) -> dict:
    kind, keys, pos, has_gauss, cached = state["numpy"]
    return {
        "python": state["python"],
        "numpy": [kind, [int(value) for value in keys], int(pos), int(has_gauss), float(cached)],
        "torch": state["torch"],
        "cuda": state["cuda"],
    }


def _rng_from_checkpoint(payload: dict) -> dict:
    kind, keys, pos, has_gauss, cached = payload["numpy"]
    return {
        "python": payload["python"],
        "numpy": (kind, np.array(keys, dtype=np.uint32), int(pos), int(has_gauss), float(cached)),
        "torch": payload["torch"],
        "cuda": payload["cuda"],
    }


def fomaml_resume_fingerprint(provenance: dict) -> dict:
    """续训必须和断点是同一次任务、同一组开关。代码哈希不参与这道比较。"""
    settings = provenance["fomaml_settings"]
    kept = (
        "epochs", "inner_steps", "inner_lr", "outer_lr", "outer_momentum", "log_every_steps",
        "early_stopping_patience", "early_stopping_min_delta", "modules", "reuse_weight_cap",
        "where_loss_weight", "views_per_group", "fallback_views_per_group", "gradient_checkpointing",
        "prescreen_k", "prescreen_audit", "prescreen_audit_every", "sketch_dim", "val_ap50",
        "val_max_episodes", "pilot", "first_order", "grad_clip_norm", "ignore_iou",
    )
    return {
        "supervised_theta0_sha256": provenance["supervised_theta0_sha256"],
        "fomaml_episode_sha256": provenance["fomaml_episode_sha256"],
        "seed": provenance["seed"],
        "fold_id": provenance["fold_id"],
        "settings": {key: settings[key] for key in kept if key in settings},
        "loss": settings["loss"],
        "experience_bank": settings.get("experience_bank") or "",
        "data_profile": provenance.get("fomaml_data_profile") or "",
    }


def _index_module_memory(module_memory: dict) -> dict:
    indexed = {}
    for episode_id, row in module_memory.items():
        if not row.get("delta_file"):
            raise RuntimeError(f"任务差还没有写入文件：{episode_id}")
        indexed[episode_id] = {key: value for key, value in row.items() if key != "delta"}
    return indexed


def _hydrate_library(module_memory: dict, library_index: dict, parts: Path) -> dict:
    library = {}
    for family_id, episode_id in library_index.items():
        row = module_memory[episode_id]
        if not isinstance(row.get("delta"), dict):
            name = row.get("delta_file")
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError(f"断点缺少任务差文件：{episode_id}")
            row["delta"] = load_checkpoint(parts / name)["delta"]
        library[family_id] = {"delta": row["delta"], "sketch": row["sketch"], "episode_id": episode_id}
    return library


def borrow_off_ap50(detector, episodes, adapter, device, settings, seed: int) -> tuple[float, float, float]:
    """借用关闭，用留出评价同一套一步 Support 适配和 AP50。评估后恢复权重和随机数。"""
    from training.meta_transfer import _holdout_records, _score_arm, prepare_holdout_features

    if not episodes:
        raise ValueError("Meta-Val 评估没有 Episode")
    evaluation = adapter.config["evaluation"]
    defect_id = int(adapter.config["classes"]["defect"])
    snapshot = {key: value.detach().cpu().clone() for key, value in detector.state_dict().items()}
    flags = [(parameter, parameter.requires_grad) for parameter in detector.parameters()]
    was_training = detector.training
    rng = _capture_rng_state()
    started = perf_counter()
    try:
        detector.eval()
        records = _holdout_records(adapter, episodes)
        features, _stats, _frames = prepare_holdout_features(
            detector, records, device, float(settings["ignore_iou"]), int(seed), 0,
            retain_frames=True, return_frame_cache=True,
        )
        prefix = "roi_heads.box_predictor."
        head = {
            name.removeprefix(prefix): tensor.detach().cpu().clone()
            for name, tensor in detector.state_dict().items()
            if name.startswith(prefix)
        }
        adaptation = {"inner_steps": int(settings["inner_steps"]), "inner_lr": float(settings["inner_lr"])}
        arm, _queries = _score_arm(
            detector, features, episodes, records, head, evaluation, adaptation,
            settings["loss"], device, defect_id,
        )
        ap50 = float(arm["summary"]["after"]["product_macro"]["ap50"])
        query_loss = float(arm["summary"]["after"]["query_detection_loss_episode_mean"])
    finally:
        detector.load_state_dict(snapshot)
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
        detector.train(was_training)
        _restore_rng_state(rng)
    if not isfinite(ap50) or not isfinite(query_loss):
        raise FloatingPointError("Meta-Val AP50 或 Query 损失不是有限值")
    return ap50, query_loss, perf_counter() - started


def _write_val_best(
    path: Path, state: dict, provenance: dict, epoch: int, ap50: float, selection: str,
    extras: dict | None = None,
) -> None:
    copied = dict(provenance)
    copied["val_selection"] = selection
    payload = {
        "model": state, "provenance": copied, "epoch": epoch,
        "selection": selection, "meta_val_ap50": ap50,
    }
    if extras:
        payload.update(extras)
    save_checkpoint(path, payload)


def rehearse_holdout_load(supervised: Path, fomaml: Path) -> str:
    """只加载留出评价所需的权重和任务记忆，不跑检测。失败时保留已保存文件。"""
    from training.checkpoints import load_checkpoint
    from training.meta_transfer import _fold0_holdout_episodes, load_holdout, load_retrieval_holdout_state

    try:
        fomaml = Path(fomaml)
        query_best = fomaml.parent / "query_best.pt"
        if query_best.is_file() and load_checkpoint(query_best).get("experience_bank_dir"):
            load_retrieval_holdout_state(query_best)
            _fold0_holdout_episodes()
            print("留出加载演练通过（检索式）", flush=True)
            return "留出加载演练通过（检索式）"
        load_holdout(Path(supervised), fomaml)
    except Exception as error:
        message = f"留出加载演练失败：{error}"
        print(message, flush=True)
        return message
    print("留出加载演练通过", flush=True)
    return "留出加载演练通过"


def _support_grad_sketches(module, records, device, settings) -> dict[str, torch.Tensor]:
    named = [(name, parameter) for name, parameter in module.named_parameters() if parameter.requires_grad]
    _loss, grads = _mean_module_grads(module, records, device, settings)
    if len(named) != len(grads):
        raise AssertionError("梯度数量和可训练参数不一致")
    by_name = {name: grad for (name, _parameter), grad in zip(named, grads)}
    dim = int(settings.get("sketch_dim", SKETCH_DIM))
    sketches = {}
    for label, prefixes in MODULE_PREFIXES.items():
        block = {name: tensor for name, tensor in by_name.items() if name.startswith(prefixes)}
        sketches[label] = module_count_sketch(block, dim)
    del grads, by_name
    module.zero_grad(set_to_none=True)
    return sketches


def _prescreen_sketches(module, records, device, settings) -> dict[str, torch.Tensor]:
    """在 θ_t 上做一次支持集反向，取出模块梯度草图，再恢复随机数和缓冲区。"""
    devices = [device] if device.type == "cuda" else []
    buffers = [(buf, buf.detach().clone()) for _name, buf in module.named_buffers()]
    try:
        with torch.random.fork_rng(devices=devices):
            return _support_grad_sketches(module, records, device, settings)
    finally:
        with torch.no_grad():
            for buf, mirror in buffers:
                buf.copy_(mirror)
        module.zero_grad(set_to_none=True)
        module.eval()


def _fomaml_phase(trace: dict | None, phase: str, detail: str = "") -> None:
    """训练阶段日志。进程被外部停掉时，最后一行就是当时的任务和阶段。"""
    if not trace:
        return
    trace["phase"] = phase
    extra = f" {detail}" if detail else ""
    print(
        f"FOMAML 阶段 epoch={trace['epoch']} step={trace['step']} "
        f"episode={trace['episode']} phase={phase}{extra}",
        flush=True,
    )


def select_module_initialization(
    detector, library: dict, family_id: str, support_records, device, settings,
    allowed_sources: dict | None = None, timings: dict | None = None,
    trace: dict | None = None,
):
    """按块看支持集收益。收益为正才按小于 1 的权重加那一块，查询图不参与。"""
    from training.retrieval_memory import MODULES

    modules = tuple(MODULES)
    if not library:
        _fomaml_phase(trace, "挑选", "library=0")
        if timings is not None:
            for name in modules:
                timings[name] = 0.0
        return {}, {name: "Null" for name in modules}, {name: 0.0 for name in modules}
    chosen, choices, weights = {}, {}, {}
    detector.eval()
    cache = SupportScoreCache(detector, device, settings)
    try:
        with torch.no_grad():
            for module_name in modules:
                started = perf_counter()
                allowed = None if allowed_sources is None else allowed_sources.get(module_name)
                if allowed is None:
                    source_count = sum(1 for source in library if source != family_id)
                else:
                    source_count = len(allowed)
                _fomaml_phase(trace, module_name, f"sources={source_count}")
                base = cache.score(module_name, support_records)
                best_name, best_weight, best_delta = "Null", 0.0, None
                backup = _capture_module(detector, MODULE_PREFIXES[module_name])
                installed = cache.token(module_name)
                for source_family, memory in library.items():
                    if source_family == family_id:
                        continue
                    if allowed is not None and source_family not in allowed:
                        continue
                    delta = _memory_blocks(memory).get(module_name) or {}
                    if not delta or base is None:
                        continue
                    _restore_module(backup)
                    _add_named_delta(detector, delta)
                    cache.mark(module_name, ("candidate", source_family))
                    score = cache.score(module_name, support_records)
                    if score is None:
                        continue
                    weight = reuse_weight(
                        base, score, settings["reuse_weight_cap"], higher_is_better=module_name == "Where",
                    )
                    if weight > best_weight:
                        best_name, best_weight, best_delta = source_family, weight, delta
                _restore_module(backup)
                cache.mark(module_name, installed)
                choices[module_name] = best_name
                weights[module_name] = best_weight
                if best_delta is not None and best_weight > 0:
                    scaled = {name: value.detach().mul(float(best_weight)) for name, value in best_delta.items()}
                    chosen[module_name] = scaled
                    _add_named_delta(detector, scaled)
                    cache.mark(module_name, ("scaled", best_name, float(best_weight)))
                if timings is not None:
                    timings[module_name] = perf_counter() - started
    finally:
        cache.close()
    return chosen, choices, weights


def _open_modules(detector) -> int:
    """五组功能参数都参与元训练。冻结会让 Backbone 和 RPN 写不出自己的知识。"""
    detector.requires_grad_(False)
    opened = 0
    for name, parameter in detector.named_parameters():
        if any(name.startswith(prefixes) for prefixes in MODULE_PREFIXES.values() for prefix in prefixes):
            parameter.requires_grad_(True)
            opened += parameter.numel()
    if opened <= 0:
        raise AssertionError("没有可训练的模块参数")
    return opened


def _mean_module_grads(module, records, device, settings):
    """按图累积梯度。克隆参数不是叶子，不能读 .grad，要和 MAML.adapt 一样用 autograd.grad。"""
    trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
    accumulated = [torch.zeros_like(parameter) for parameter in trainable]
    total = None
    used = 0
    module.train()
    for index, record in enumerate(records, start=1):
        loss = live_detection_loss(
            module, record, device, settings["loss"], settings["where_loss_weight"],
            settings["ignore_iou"], settings["gradient_checkpointing"],
        )
        grads = torch.autograd.grad(loss, trainable, allow_unused=True)
        for slot, grad in zip(accumulated, grads):
            if grad is not None:
                slot.add_(grad.detach())
        total = loss.detach() if total is None else total + loss.detach()
        used += 1
        del loss, grads
    if used == 0 or total is None:
        raise InsufficientEpisodeAnnotations("Support 或 Query 没有可用的检测监督")
    for grad in accumulated:
        grad.div_(used)
    return total.div_(used), accumulated


def _clip_grads(grads, max_norm: float):
    norm = torch.norm(torch.stack([grad.detach().float().norm() for grad in grads]))
    if not torch.isfinite(norm):
        raise FloatingPointError("内循环梯度不是有限值")
    if float(norm) <= max_norm:
        return grads
    scale = max_norm / (float(norm) + 1e-6)
    for grad in grads:
        grad.mul_(scale)
    return grads


def _inplace_sgd(module, grads, lr: float, max_norm: float) -> None:
    """在同一份检测器上做一阶内循环，不再克隆第二套模型。"""
    grads = _clip_grads(grads, max_norm)
    trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if len(trainable) != len(grads):
        raise AssertionError("内循环梯度数量和可训练参数不一致")
    with torch.no_grad():
        for parameter, grad in zip(trainable, grads):
            parameter.add_(-lr * grad)


def _max_cpu_change(store: _ParameterStore, left: str, right: str) -> float:
    largest = None
    for old, new in zip(store.slots[left], store.slots[right]):
        peak = (new - old).abs().max()
        largest = peak if largest is None else torch.maximum(largest, peak)
    if largest is None:
        raise AssertionError("没有参数")
    value = float(largest)
    if not isfinite(value):
        raise FloatingPointError("参数变化不是有限值")
    return value


def _meta_step(
    initializer, optimizer, support_records, query_records, device, settings, store: _ParameterStore,
    trace: dict | None = None,
):
    module = initializer.module
    store.capture("before")
    support_losses = []
    _fomaml_phase(trace, "内层")
    inner_started = perf_counter()
    for _ in range(settings["inner_steps"]):
        loss, grads = _mean_module_grads(module, support_records, device, settings)
        support_losses.append(float(loss))
        _inplace_sgd(module, grads, settings["inner_lr"], settings["grad_clip_norm"])
    inner_seconds = perf_counter() - inner_started
    store.capture("after")
    module_delta = split_parameter_delta(store.slot_dict("before"), store.slot_dict("after"))
    inner_change = max(float(value.abs().max()) for block in module_delta.values() for value in block.values())
    if inner_change <= 0:
        raise AssertionError("Support 适配没有更新检测器")
    _fomaml_phase(trace, "外层")
    outer_started = perf_counter()
    query_loss, query_grads = _mean_module_grads(module, query_records, device, settings)
    store.restore("before")
    optimizer.zero_grad(set_to_none=True)
    trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if len(trainable) != len(query_grads):
        raise AssertionError("Query 梯度和共享可训练参数数量不一致")
    for parameter, grad in zip(trainable, query_grads):
        parameter.grad = grad
    grad_norm = torch.nn.utils.clip_grad_norm_(
        trainable, settings["grad_clip_norm"], error_if_nonfinite=True,
    ).item()
    optimizer.step()
    outer_seconds = perf_counter() - outer_started
    store.capture("after")
    outer_change = _max_cpu_change(store, "before", "after")
    if grad_norm <= 0 or outer_change <= 0:
        raise AssertionError("Query 外循环没有更新共享检测器")
    return {
        "support_losses": support_losses, "query_detection_loss": float(query_loss),
        "inner_max_weight_change": inner_change, "outer_max_weight_change": outer_change,
        "query_grad_norm": grad_norm, "module_delta": module_delta,
        "inner_seconds": inner_seconds, "outer_seconds": outer_seconds,
    }


def _save_metrics(path: Path, result: dict) -> None:
    with _atomic_output(path) as handle:
        handle.write((json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def _train(detector, adapter, episodes, provenance, run_dir, device, resume_state=None):
    from statistics import mean

    meta = provenance["fomaml_settings"]
    seed = provenance["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats(device)
    detector.to(device).eval()
    opened = _open_modules(detector)
    initializer = MAML(detector, lr=meta["inner_lr"], first_order=True, allow_unused=True, allow_nograd=True)
    optimizer = torch.optim.SGD(
        [parameter for parameter in initializer.parameters() if parameter.requires_grad],
        lr=meta["outer_lr"], momentum=meta["outer_momentum"],
    )
    store = _ParameterStore(initializer.module)
    result = {"complete": False, "provenance": provenance, "global_step": 0, "history": [],
              "early_stopped": False}
    result["runtime"] = {"torch": str(torch.__version__), "learn2learn": version("learn2learn"),
                         "gpu": torch.cuda.get_device_name(device), "num_workers": 0}
    result["trainable_parameters"] = opened
    metrics_path = run_dir / "metrics.json"
    trained_families = set()
    module_memory = {}
    library = {}
    memory_parts = run_dir / "module_memory_parts"
    best_loss = None
    best_state = None
    best_epoch = 0
    stall = 0
    current = None
    elapsed_base = 0.0
    prior_peak = 0.0
    cursor = {"epoch": 1, "index": 0, "pending_close": False, "status": "running"}
    if resume_state is None:
        if memory_parts.exists():
            for stale in memory_parts.glob("*.pt"):
                stale.unlink()
        print(
            f"FOMAML 开始训练，可训练参数 {opened}，每轮 {len(episodes)} 条任务，"
            f"显存已用 {torch.cuda.memory_allocated(device) / 2**20:.0f} MiB",
            flush=True,
        )
    else:
        expected = fomaml_resume_fingerprint(provenance)
        saved_fingerprint = dict(resume_state["fingerprint"])
        saved_fingerprint.setdefault("experience_bank", "")
        saved_fingerprint.setdefault("data_profile", "")
        if json.dumps(saved_fingerprint, sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise ValueError("断点的折、任务或训练开关和本次命令不一致")
        if resume_state.get("code") != provenance.get("code"):
            print("FOMAML 续训：代码与断点时不同，任务和开关一致，从上次完成的外更新继续", flush=True)
        detector.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["optimizer"])
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        _restore_rng_state(_rng_from_checkpoint(resume_state["rng"]))
        result["global_step"] = int(resume_state["global_step"])
        result["history"] = list(resume_state["history"])
        result["early_stopped"] = bool(resume_state["early_stopped"])
        if resume_state.get("prescreen_audit") is not None:
            result["prescreen_audit"] = resume_state["prescreen_audit"]
        if resume_state.get("meta_val_ap50") is not None:
            result["meta_val_ap50"] = list(resume_state["meta_val_ap50"])
        module_memory = resume_state["module_memory"]
        library = _hydrate_library(module_memory, resume_state["library_index"], memory_parts)
        trained_families = set(resume_state["trained_families"])
        best_loss = resume_state["best_loss"]
        best_epoch = int(resume_state["best_epoch"])
        stall = int(resume_state["stall"])
        current = resume_state.get("current")
        elapsed_base = float(resume_state.get("elapsed_seconds") or 0.0)
        prior_peak = float(resume_state.get("peak_cuda_allocated_mib") or 0.0)
        cursor = {
            "epoch": int(resume_state["epoch"]), "index": int(resume_state["index"]),
            "pending_close": bool(resume_state["pending_close"]), "status": resume_state["status"],
        }
        if best_loss is not None:
            best_state = load_checkpoint(run_dir / "query_best.pt")["model"]
        print(
            f"FOMAML 从断点继续 epoch={cursor['epoch']} next_index={cursor['index']} "
            f"step={result['global_step']} library={len(library)}",
            flush=True,
        )
    if int(meta.get("prescreen_audit") or 0) > 0 and "prescreen_audit" not in result:
        result["prescreen_audit"] = {
            "episodes": 0, "large_episodes": 0, "hits": {"Backbone": 0, "Where": 0},
            "ranks": {"Backbone": [], "Where": []}, "k": int(meta["prescreen_k"]),
        }
    patience = int(meta.get("early_stopping_patience", 0))
    min_delta = float(meta.get("early_stopping_min_delta", 0.0))
    log_every = max(1, int(meta["log_every_steps"]))
    val_tracker = None
    val_episodes = []
    val_csv = run_dir / "meta_val_ap50.csv"
    if meta.get("val_ap50") and cursor["status"] != "finalize":
        val_episodes = load_meta_val_episodes(
            provenance, int(meta.get("val_max_episodes") or 0), int(provenance["seed"]),
        )
        if resume_state is None:
            theta0_state = {key: value.detach().cpu().clone() for key, value in detector.state_dict().items()}
            theta0_ap50, theta0_query, eval_seconds = borrow_off_ap50(
                detector, val_episodes, adapter, device, meta, int(provenance["seed"]),
            )
            val_tracker = {
                "theta0_ap50": theta0_ap50, "best_ap50": theta0_ap50, "best_epoch": 0,
                "selection": "theta0_best", "state": theta0_state,
            }
            result["meta_val_ap50"] = [{
                "epoch": 0, "theta0_ap50": theta0_ap50, "current_ap50": theta0_ap50,
                "meta_val_query_loss": theta0_query, "eval_seconds": eval_seconds,
            }]
            _write_val_csv(val_csv, result["meta_val_ap50"])
            _write_val_best(run_dir / "val_best.pt", theta0_state, provenance, 0, theta0_ap50, "theta0_best")
            print(
                f"Meta-Val AP50 epoch=0 theta0={theta0_ap50:.6f} current={theta0_ap50:.6f} "
                f"query_loss={theta0_query:.6f} eval_seconds={eval_seconds:.1f}",
                flush=True,
            )
        else:
            saved_best = load_checkpoint(run_dir / "val_best.pt")
            theta0_ap50 = float(result["meta_val_ap50"][0]["theta0_ap50"])
            val_tracker = {
                "theta0_ap50": theta0_ap50, "best_ap50": float(saved_best["meta_val_ap50"]),
                "best_epoch": int(saved_best["epoch"]), "selection": saved_best["selection"],
                "state": saved_best["model"],
            }
    start = perf_counter()
    stopped_epoch = meta["epochs"]
    _save_metrics(metrics_path, result)
    retrieval_bank = None
    gamma_controller = None
    inner_lr_controller = None
    theta0_weights = None
    if meta.get("experience_bank"):
        from training.experience_bank import (
            GammaController, InnerLRController, _state_sha256, _state_snapshot, assert_compatible_inner_lr,
            load_experience_bank, refresh_experience_cards, retrieval_fomaml_step,
        )
        from training.retrieval_memory import load_retrieval_config

        retrieval_bank = load_experience_bank(meta["experience_bank"])
        gamma_controller = GammaController()
        inner_lr_controller = InnerLRController() if load_retrieval_config().get("learnable_inner_lr", False) else None
        if resume_state is not None:
            if not resume_state.get("gamma_state") or not resume_state.get("experience_bank_dir"):
                raise RuntimeError("检索式续训缺少 γ 或本轮经验库路径")
            gamma_controller.load_state_dict(resume_state["gamma_state"])
            retrieval_bank = load_experience_bank(resume_state["experience_bank_dir"])
            if inner_lr_controller is not None:
                if not resume_state.get("inner_lr_state"):
                    raise RuntimeError("检索式续训缺少模块学习率")
                inner_lr_controller.load_state_dict(resume_state["inner_lr_state"])
        held_theta0 = load_checkpoint(provenance["supervised_theta0"])
        theta0_weights = held_theta0["model"]
        del held_theta0

    def write_resume(epoch, index, pending_close, status="running"):
        nonlocal current, best_loss, best_epoch, stall
        for entry in library.values():
            row = module_memory[entry["episode_id"]]
            if isinstance(row.get("delta"), dict) and not row.get("delta_file"):
                _spill_episode_delta(memory_parts, entry["episode_id"], row, release=False)
        cursor["epoch"] = int(epoch)
        cursor["index"] = int(index)
        cursor["pending_close"] = bool(pending_close)
        cursor["status"] = status
        peak = prior_peak
        if device.type == "cuda":
            peak = max(peak, torch.cuda.max_memory_allocated(device) / 2**20)
        save_checkpoint(run_dir / "resume.pt", {
            "fingerprint": fomaml_resume_fingerprint(provenance),
            "code": provenance.get("code"),
            "status": status,
            "epoch": cursor["epoch"],
            "index": cursor["index"],
            "pending_close": cursor["pending_close"],
            "model": detector.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": _rng_to_checkpoint(_capture_rng_state()),
            "global_step": result["global_step"],
            "history": result["history"],
            "early_stopped": result["early_stopped"],
            "prescreen_audit": result.get("prescreen_audit"),
            "meta_val_ap50": result.get("meta_val_ap50"),
            "current": current,
            "trained_families": sorted(trained_families),
            "module_memory": _index_module_memory(module_memory),
            "library_index": {family: entry["episode_id"] for family, entry in library.items()},
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "stall": stall,
            "elapsed_seconds": elapsed_base + perf_counter() - start,
            "peak_cuda_allocated_mib": peak,
            "gamma_state": None if gamma_controller is None else gamma_controller.state_dict(),
            "inner_lr_state": None if inner_lr_controller is None else inner_lr_controller.state_dict(),
            "inner_lr_snapshot": None if inner_lr_controller is None else inner_lr_controller.current()[0],
            "experience_bank_dir": None if retrieval_bank is None else retrieval_bank["directory"],
            "experience_epoch_theta_sha256": None if retrieval_bank is None else retrieval_bank["theta_sha256"],
        })
        print(
            f"FOMAML 断点 epoch={cursor['epoch']} next_index={cursor['index']} "
            f"step={result['global_step']} pending_close={int(cursor['pending_close'])}",
            flush=True,
        )

    if resume_state is None:
        write_resume(1, 0, False)
    for epoch in range(1, meta["epochs"] + 1):
            if cursor["status"] == "finalize":
                break
            if epoch < cursor["epoch"]:
                continue
            order = list(episodes)
            random.Random(seed + epoch).shuffle(order)
            if cursor["pending_close"] and epoch == cursor["epoch"]:
                if not isinstance(current, dict) or current.get("epoch") != epoch:
                    raise RuntimeError("断点缺少本轮已完成的任务记录")
                start_index = len(order) + 1
            else:
                start_index = cursor["index"] if epoch == cursor["epoch"] else 0
                if start_index == 0:
                    current = {"epoch": epoch, "episodes": [], "skipped": []}
                elif not isinstance(current, dict) or current.get("epoch") != epoch:
                    raise RuntimeError("断点缺少本轮已完成的任务记录")
            if retrieval_bank is not None and start_index == 0:
                import shutil

                from training.experience_bank import refresh_deltas_complete

                epoch_dir = run_dir / f"experience_epoch_{epoch}"
                current_hash = _state_sha256(_state_snapshot(detector))
                if inner_lr_controller is not None:
                    values, hits = inner_lr_controller.current()
                    meta["inner_lr_snapshot"] = values
                    meta["inner_lr_source"] = "module_snapshot"
                    touched = {name: hits[name] for name in hits if hits[name]}
                    if touched:
                        print(f"FOMAML 模块学习率触及边界 epoch={epoch} {touched}", flush=True)
                resume_deltas = False
                if epoch_dir.exists() and not (epoch_dir / "module_memory.pt").is_file():
                    if refresh_deltas_complete(epoch_dir, retrieval_bank["assignments"]):
                        resume_deltas = True
                        print(
                            f"FOMAML 复用未完成刷新目录中的任务增量 epoch={epoch} dir={epoch_dir.name}",
                            flush=True,
                        )
                    else:
                        shutil.rmtree(epoch_dir)
                if (epoch_dir / "module_memory.pt").is_file():
                    retrieval_bank = load_experience_bank(epoch_dir, current_hash)
                else:
                    # 刷新峰值前把家族 Feature Δ 卸到磁盘，避免与拼卡叠加占满内存。
                    library_index = {family: entry["episode_id"] for family, entry in library.items()}
                    for entry in library.values():
                        row = module_memory[entry["episode_id"]]
                        if isinstance(row.get("delta"), dict):
                            _spill_episode_delta(memory_parts, entry["episode_id"], row, release=True)
                        entry.pop("delta", None)
                    library.clear()
                    gc.collect()
                    print(
                        f"FOMAML 按当前共享参数重算经验卡片 epoch={epoch}"
                        f"{'（续跑增量）' if resume_deltas else ''}",
                        flush=True,
                    )
                    refresh_experience_cards(
                        retrieval_bank["directory"], detector, epoch_dir, adapter, episodes, device, meta,
                        resume_deltas=resume_deltas,
                    )
                    retrieval_bank = load_experience_bank(epoch_dir, current_hash)
                    library = _hydrate_library(module_memory, library_index, memory_parts)
                    pruned = _prune_old_experience_epochs(run_dir, epoch)
                    if pruned:
                        print(
                            f"FOMAML 只保留当前经验库 experience_epoch_{epoch}，已删 {','.join(pruned)}",
                            flush=True,
                        )
            for index, episode in enumerate(order):
                if index < start_index:
                    continue
                family_id = _family_id(episode)
                store.capture("primary")
                borrowed = None
                trace = {
                    "epoch": epoch, "step": result["global_step"] + 1,
                    "episode": episode["episode_id"], "phase": "进入",
                }
                try:
                    gpu = ""
                    if device.type == "cuda":
                        gpu = f" gpu={torch.cuda.memory_allocated(device) / 2**20:.0f}MiB"
                    print(
                        f"FOMAML 进入 epoch={epoch} step={trace['step']} "
                        f"episode={episode['episode_id']} family={family_id} "
                        f"library={len(library)}{gpu}",
                        flush=True,
                    )
                    if result["global_step"] == 0:
                        print(f"FOMAML 首条任务开始：{episode['episode_id']}", flush=True)
                    _fomaml_phase(trace, "标注")
                    support_records, query_records = episode_records(adapter, episode)
                    initializer.module.eval()
                    support_records = _usable_phase_views(
                        initializer.module, support_records, device, meta["ignore_iou"],
                        meta["views_per_group"], meta.get("fallback_views_per_group", 3),
                    )
                    query_records = _usable_phase_views(
                        initializer.module, query_records, device, meta["ignore_iou"],
                        meta["views_per_group"], meta.get("fallback_views_per_group", 3),
                    )
                    if result["global_step"] == 0:
                        print(
                            f"FOMAML 首条任务标注可用，support={len(support_records)} "
                            f"query={len(query_records)} views_per_group={meta['views_per_group']}",
                            flush=True,
                        )
                    select_times = {}
                    prescreen_seconds = 0.0
                    ranking = None
                    allowed = None
                    prescreen_k = int(meta["prescreen_k"])
                    audit_n = int(meta["prescreen_audit"])
                    auditing = False
                    if retrieval_bank is not None:
                        _fomaml_phase(trace, "检索")
                        # 条件化协议：Null 进 Softmax，禁止随机整任务不借。
                        step = retrieval_fomaml_step(
                            initializer.module, optimizer, support_records, query_records, episode,
                            retrieval_bank, gamma_controller, device, meta, allow_borrow=True,
                            inner_lr_controller=inner_lr_controller,
                            theta0_state=theta0_weights, adapter=adapter,
                            writable_cache=run_dir / "feature_cache",
                            extra_caches=[run_dir / "feature_cache"],
                        )
                        prescreen_seconds = step.pop("retrieval_seconds")
                        borrowed = {}
                        choices = step["module_choices"]
                        reuse_weights = step["reuse_weights"]
                    else:
                        if prescreen_k > 0 and library:
                            if audit_n > 0:
                                others = _other_source_count(library, family_id)
                                if others > 2 * prescreen_k:
                                    result["prescreen_audit"]["large_episodes"] += 1
                                    auditing = prescreen_audit_due(
                                        others, prescreen_k, result["prescreen_audit"]["large_episodes"],
                                        int(meta["prescreen_audit_every"]), result["prescreen_audit"]["episodes"], audit_n,
                                    )
                            _fomaml_phase(trace, "预筛")
                            started = perf_counter()
                            grad_sketches = _prescreen_sketches(
                                initializer.module, support_records, device, meta,
                            )
                            ranking = _rank_sources(library, family_id, grad_sketches, int(meta["sketch_dim"]))
                            prescreen_seconds = perf_counter() - started
                            if not auditing:
                                allowed = {
                                    name: set(ranking[name][:prescreen_k]) for name in ("Backbone", "Where")
                                }
                        borrowed, choices, reuse_weights = select_module_initialization(
                            initializer.module, library, family_id, support_records, device, meta,
                            allowed, select_times, trace,
                        )
                        step = _meta_step(
                            initializer, optimizer, support_records, query_records, device, meta, store, trace,
                        )
                except InsufficientEpisodeAnnotations as error:
                    store.restore("primary")
                    print(
                        f"FOMAML 跳过 epoch={epoch} episode={episode['episode_id']} reason={error}",
                        flush=True,
                    )
                    if meta["pilot"]:
                        raise
                    current["skipped"].append({"episode_id": episode["episode_id"], "reason": str(error)})
                    continue
                except BaseException as error:
                    print(
                        f"FOMAML 异常 epoch={trace['epoch']} step={trace['step']} "
                        f"episode={trace['episode']} phase={trace['phase']} "
                        f"error={type(error).__name__}: {error}",
                        flush=True,
                    )
                    raise
                for delta in borrowed.values():
                    _add_named_delta(initializer.module, delta, alpha=-1.0)
                produced = step.pop("module_delta")
                sketches = {
                    label: module_count_sketch(block, int(meta["sketch_dim"]))
                    for label, block in produced.items()
                }
                _remember_module_delta(module_memory, library, family_id, episode["episode_id"], {
                    "product_id": episode["product_id"], "family_id": family_id,
                    "choices": choices, "reuse_weights": reuse_weights, "delta": produced,
                    "sketch": sketches,
                }, memory_parts)
                step["module_choices"] = choices
                step["reuse_weights"] = reuse_weights
                step["seconds"] = {
                    "prescreen": prescreen_seconds,
                    "backbone": select_times.get("Backbone", 0.0),
                    "where": select_times.get("Where", 0.0),
                    "roi_feature": select_times.get("RoiFeature", 0.0),
                    "box": select_times.get("Box", 0.0),
                    "class": select_times.get("Class", 0.0),
                    "inner": step.pop("inner_seconds"),
                    "outer": step.pop("outer_seconds"),
                }
                if auditing and ranking is not None:
                    audit = result["prescreen_audit"]
                    audit["episodes"] += 1
                    for name in ("Backbone", "Where"):
                        winner = choices[name]
                        hit = winner == "Null" or winner in ranking[name][:prescreen_k]
                        audit["hits"][name] += int(hit)
                        rank = _winner_rank(ranking[name], winner)
                        if rank is not None:
                            audit["ranks"][name].append(rank)
                        rate = audit["hits"][name] / audit["episodes"]
                        rank_label = "NA" if rank is None else str(rank)
                        print(
                            f"预筛核对 {name} step={result['global_step'] + 1} "
                            f"winner={winner} in_topk={hit} hit_rate={rate:.3f} "
                            f"rank={rank_label} {_rank_text(audit['ranks'][name])}",
                            flush=True,
                        )
                result["global_step"] += 1
                trained_families.add(episode["family_id"])
                current["episodes"].append({
                    "episode_id": episode["episode_id"], "support_views": len(support_records),
                    "query_views": len(query_records), **step,
                })
                if meta["pilot"] or result["global_step"] == 1 or result["global_step"] % log_every == 0:
                    from training.retrieval_memory import MODULES as _LOG_MODULES

                    brief = "/".join(
                        "Null" if choices[name] == "Null" else f"{choices[name]}@{reuse_weights[name]:.3f}"
                        for name in _LOG_MODULES
                    )
                    seconds = step["seconds"]
                    print(
                        f"FOMAML epoch={epoch} step={result['global_step']} "
                        f"query_detection={step['query_detection_loss']:.6f} init={brief} "
                        f"sec prescreen={seconds['prescreen']:.2f} backbone={seconds['backbone']:.2f} "
                        f"where={seconds['where']:.2f} roi={seconds['roi_feature']:.2f} "
                        f"box={seconds['box']:.2f} class={seconds['class']:.2f} "
                        f"inner={seconds['inner']:.2f} outer={seconds['outer']:.2f}",
                        flush=True,
                    )
                    bound_hits = step.get("inner_lr_bound_hits") or {}
                    touched = {name: bound_hits[name] for name in bound_hits if bound_hits[name]}
                    if touched:
                        print(f"FOMAML 模块学习率触及边界 step={result['global_step']} {touched}", flush=True)
                    if step.get("gamma_proxy_meaning") == "query_counterfactual_utility":
                        module_u = step.get("module_utilities") or {}
                        kinds = step.get("loo_kinds") or {}
                        loo = ",".join(
                            f"{name}={float(module_u.get(name, 0.0)):+.3f}"
                            f"/{kinds.get(name, '?')}"
                            for name in (step.get("loo_modules") or [])
                        ) or "-"
                        closed = ",".join(step.get("force_closed") or []) or "-"
                        rho_p = step.get("rho_probe") or {}
                        card_p = step.get("card_probe") or {}
                        if not rho_p:
                            rho_txt = "-"
                        elif rho_p.get("pairwise"):
                            win = rho_p.get("winner")
                            win_txt = "-" if win is None else f"{float(win):.2f}"
                            rho_txt = (
                                f"{rho_p.get('module')}@"
                                f"{float(rho_p.get('rho_a', 0.0)):.2f}/{float(rho_p.get('rho_b', 0.0)):.2f}"
                                f":Δ={float(rho_p.get('delta_u', 0.0)):+.3f}"
                                f"/ε={float(rho_p.get('eps', 0.0)):.3f}/win={win_txt}"
                            )
                        else:
                            rho_txt = (
                                f"{rho_p.get('module')}@ρ={float(rho_p.get('rho', 0.0)):.2f}"
                                f":{float(rho_p.get('utility', 0.0)):+.3f}"
                            )
                        if not card_p:
                            card_txt = "-"
                        elif card_p.get("pairwise"):
                            card_txt = (
                                f"{card_p.get('module')}#"
                                f"{int(card_p.get('index_a', -1))}/{int(card_p.get('index_b', -1))}"
                                f":Δ={float(card_p.get('delta_u', 0.0)):+.3f}"
                            )
                        else:
                            card_txt = (
                                f"{card_p.get('module')}#{int(card_p.get('index', -1))}"
                                f":{float(card_p.get('utility', 0.0)):+.3f}"
                            )
                        print(
                            f"FOMAML step={result['global_step']} "
                            f"U={step.get('counterfactual_utility', 0.0):+.4f} "
                            f"L_null={step.get('query_loss_null', 0.0):.4f} "
                            f"L_borrow={step.get('query_loss_borrow', 0.0):.4f} "
                            f"LOO[{loo}] force_close[{closed}] "
                            f"rho_probe[{rho_txt}] card_probe[{card_txt}]",
                            flush=True,
                        )
                write_resume(epoch, index + 1, index + 1 == len(order))
            if not current["episodes"]:
                raise RuntimeError(f"第 {epoch} 轮没有完成 FOMAML 外更新")
            epoch_loss = mean(row["query_detection_loss"] for row in current["episodes"])
            current["query_detection_mean"] = epoch_loss
            result["history"].append(current)
            improved = best_loss is None or epoch_loss < best_loss - min_delta
            if improved:
                best_loss = epoch_loss
                best_epoch = epoch
                stall = 0
                best_state = {key: value.detach().cpu().clone() for key, value in detector.state_dict().items()}
                best_payload = {"model": best_state, "epoch": best_epoch, "loss": best_loss}
                if gamma_controller is not None:
                    best_payload["gamma_state"] = gamma_controller.state_dict()
                    best_payload["policy_state"] = gamma_controller.state_dict()
                    best_payload["experience_bank_dir"] = retrieval_bank["directory"]
                    best_payload["theta_sha256"] = retrieval_bank["theta_sha256"]
                    best_payload["theta_version"] = retrieval_bank["theta_sha256"]
                    best_payload["bank_version"] = retrieval_bank["theta_sha256"]
                if inner_lr_controller is not None:
                    best_payload["inner_lr_state"] = inner_lr_controller.state_dict()
                    best_payload["inner_lr_snapshot"] = inner_lr_controller.current()[0]
                save_checkpoint(run_dir / "query_best.pt", best_payload)
            else:
                stall += 1
            _save_metrics(metrics_path, result)
            print(
                f"FOMAML epoch={epoch} mean_query={epoch_loss:.6f} best={best_loss:.6f} stall={stall}",
                flush=True,
            )
            if val_tracker is not None:
                ap50, query_loss, eval_seconds = borrow_off_ap50(
                    detector, val_episodes, adapter, device, meta, int(provenance["seed"]),
                )
                result["meta_val_ap50"].append({
                    "epoch": epoch, "theta0_ap50": val_tracker["theta0_ap50"], "current_ap50": ap50,
                    "meta_val_query_loss": query_loss, "eval_seconds": eval_seconds,
                })
                _write_val_csv(val_csv, result["meta_val_ap50"])
                print(
                    f"Meta-Val AP50 epoch={epoch} theta0={val_tracker['theta0_ap50']:.6f} "
                    f"current={ap50:.6f} query_loss={query_loss:.6f} eval_seconds={eval_seconds:.1f}",
                    flush=True,
                )
                if ap50 > val_tracker["best_ap50"]:
                    val_tracker["best_ap50"] = ap50
                    val_tracker["best_epoch"] = epoch
                    val_tracker["selection"] = "meta_val_ap50"
                    val_tracker["state"] = {
                        key: value.detach().cpu().clone() for key, value in detector.state_dict().items()
                    }
                    extras = {}
                    if gamma_controller is not None:
                        extras["gamma_state"] = gamma_controller.state_dict()
                        extras["policy_state"] = gamma_controller.state_dict()
                        extras["experience_bank_dir"] = retrieval_bank["directory"]
                        extras["theta_sha256"] = retrieval_bank["theta_sha256"]
                        extras["theta_version"] = retrieval_bank["theta_sha256"]
                        extras["bank_version"] = retrieval_bank["theta_sha256"]
                    if inner_lr_controller is not None:
                        extras["inner_lr_state"] = inner_lr_controller.state_dict()
                        extras["inner_lr_snapshot"] = inner_lr_controller.current()[0]
                    _write_val_best(
                        run_dir / "val_best.pt", val_tracker["state"], provenance, epoch, ap50, "meta_val_ap50",
                        extras or None,
                    )
            if patience > 0 and stall >= patience:
                result["early_stopped"] = True
                stopped_epoch = epoch
                print(f"FOMAML 早停于第 {epoch} 轮，恢复第 {best_epoch} 轮权重", flush=True)
                write_resume(epoch, len(order), False, status="finalize")
                break
            write_resume(epoch + 1, 0, False)

    if best_state is not None:
        detector.load_state_dict(best_state)
    # 覆盖口径=本轮实际任务族（检索式可只跑分组子集），不是整折 θ₀ 全族。
    expected_families = {episode["family_id"] for episode in episodes}
    if not meta["pilot"] and not result["early_stopped"] and trained_families != expected_families:
        missing = sorted(expected_families - trained_families)
        extra = sorted(trained_families - expected_families)
        raise AssertionError(f"正式 FOMAML 未覆盖本轮任务产品族 missing={missing} extra={extra}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        result["peak_cuda_allocated_mib"] = max(prior_peak, torch.cuda.max_memory_allocated(device) / 2**20)
        result["peak_cuda_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20
    else:
        result["peak_cuda_allocated_mib"] = prior_peak
        result["peak_cuda_reserved_mib"] = 0.0
    result["elapsed_seconds"] = elapsed_base + perf_counter() - start
    result["best_epoch"] = best_epoch
    result["best_query_detection_mean"] = best_loss
    provenance["fomaml_trained_family_ids"] = sorted(trained_families)
    provenance["stopped_epoch"] = stopped_epoch
    library.clear()
    gc.collect()
    from training.retrieval_memory import MODULES as _MEMORY_MODULES

    # 检索式：不把每任务 Δ 永久卸盘；非检索式仍 spill 供旧库路径。
    if retrieval_bank is not None:
        _discard_module_memory_parts(run_dir, module_memory)
        keep_epoch = stopped_epoch if stopped_epoch else best_epoch
        bank_name = Path(str(retrieval_bank.get("directory") or "")).name
        if bank_name.startswith("experience_epoch_"):
            try:
                keep_epoch = int(bank_name.rsplit("_", 1)[-1])
            except ValueError:
                pass
        _prune_old_experience_epochs(run_dir, keep_epoch)
        save_checkpoint(run_dir / "module_memory.pt", {
            "episodes": module_memory, "modules": list(_MEMORY_MODULES),
        })
    else:
        for episode_id, row in module_memory.items():
            _spill_episode_delta(memory_parts, episode_id, row)
        save_checkpoint(run_dir / "module_memory.pt", {
            "episodes": module_memory, "modules": list(_MEMORY_MODULES),
            "delta_storage": memory_parts.name,
        })
    save_checkpoint(run_dir / "last.pt", {"model": detector.state_dict(), "provenance": provenance,
                                         "epoch": best_epoch, "global_step": result["global_step"]})
    result["complete"] = True
    result["module_memory_episodes"] = len(module_memory)
    _save_metrics(metrics_path, result)
    audit = result.get("prescreen_audit")
    if audit and audit["episodes"]:
        parts = []
        for name in ("Backbone", "Where"):
            rate = audit["hits"][name] / audit["episodes"]
            parts.append(f"{name} hit_rate={rate:.3f} {_rank_text(audit['ranks'][name])}")
        print("预筛核对 " + " ".join(parts), flush=True)
    if val_tracker is not None and val_tracker["selection"] == "theta0_best":
        print("Meta-Val AP50 没有任何训练轮超过 θ0，val_best.pt 标记为 theta0_best", flush=True)
    elif val_tracker is not None:
        print(f"Meta-Val AP50 选中第 {val_tracker['best_epoch']} 轮，写入 val_best.pt", flush=True)
    rehearsal = rehearse_holdout_load(Path(provenance["supervised_theta0"]), run_dir / "last.pt")
    result["holdout_load_rehearsal"] = rehearsal
    _save_metrics(metrics_path, result)
    print(f"FOMAML 完成，显存峰值 {result['peak_cuda_allocated_mib']:.0f} MiB，权重 {run_dir / 'last.pt'}", flush=True)
    return result


def backfill_module_sketches(path: Path) -> int:
    """给已有任务记忆补上草图。delta 本身不改。"""
    payload = load_checkpoint(path)
    episodes = payload.get("episodes")
    if not isinstance(episodes, dict):
        raise ValueError(f"任务记忆没有 episodes：{path}")
    filled = 0
    for row in episodes.values():
        delta = row.get("delta")
        loaded = False
        if not isinstance(delta, dict):
            name = row.get("delta_file")
            if not isinstance(name, str):
                raise ValueError("任务记忆条目缺少 delta")
            delta = load_checkpoint(path.parent / str(payload.get("delta_storage") or "module_memory_parts") / name)["delta"]
            loaded = True
        sketch = row.get("sketch")
        if isinstance(sketch, dict) and set(sketch) == set(delta):
            continue
        row["sketch"] = {
            label: module_count_sketch(block)
            for label, block in delta.items()
            if isinstance(block, dict)
        }
        if loaded:
            del delta
        filled += 1
    if filled:
        save_checkpoint(path, payload)
    return filled


def train_theta0_fomaml(
    theta0: Path, run_name: str, pilot: bool, dry_run: bool, prescreen_k: int = 0, prescreen_audit: int = 0,
    prescreen_audit_every: int = 10, sketch_dim: int = SKETCH_DIM, val_ap50: bool = False, val_max_episodes: int = 0,
    resume: bool = False, experience_bank: Path | None = None, epochs: int | None = None,
) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_name):
        raise ValueError("run-name 只能使用字母、数字、连字符和下划线")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    theta0 = theta0.resolve(strict=True)
    theta0_hash = file_sha256(theta0)
    parent = load_checkpoint(theta0)
    if file_sha256(theta0) != theta0_hash:
        raise RuntimeError("监督 θ₀ 在加载时发生变化")
    source = parent["provenance"]
    if (source.get("mode") not in ("pilot", "fold_train_full") or source.get("theta0_training_method")
            or (not pilot and source["mode"] != "fold_train_full")):
        raise ValueError("FOMAML 需要同折监督 θ₀。正式运行须使用完成的正式折内监督权重")

    data_profile = load_fomaml_data_config()
    adapter, episodes, scope = load_training_episodes(source["fold_id"], source["train_family_ids"])
    required = ("fold_id", "fold_sha256", "theta0_train_family_ids", "delta_source_family_ids",
                "oof_target_family_ids", "meta_val_family_ids", "outer_test_family_ids",
                "data_config_sha256", "manifest_sha256", "group_split_sha256", "train_index_sha256",
                "split_manifest_sha256")
    if any(source.get(key) != scope[key] for key in required):
        raise ValueError("监督 θ₀ 的冻结折或数据来源与当前配置不一致")
    if not set(scope["train_family_ids"]) <= set(scope["theta0_train_family_ids"]):
        raise ValueError("当前数据分组超出了监督 θ₀ 允许的训练族")
    if (
        not pilot
        and data_profile["episode_source"] == "frozen_oof_v1"
        and scope["train_family_ids"] != scope["theta0_train_family_ids"]
    ):
        raise ValueError("三数据集正式 FOMAML 需要覆盖该折全部允许训练族")
    baseline_path = named_config_path("baseline.yaml")
    detector_path = named_config_path("b0_v3.yaml")
    if (source["config_sha256"] != file_sha256(baseline_path)
            or source["detector_config_sha256"] != file_sha256(detector_path)):
        raise ValueError("监督 θ₀ 的 B0 配置与当前版本不一致")
    b0_model_config(source["settings"]["model"])
    config_path = named_config_path("fomaml.yaml")
    config = load_config(config_path)
    loss = {**config["training"]["loss"], "weights": dict(config["training"]["loss"]["weights"])}
    loss.setdefault("defect_score", "margin")
    settings = {**config["training"], "loss": loss, "pilot": pilot, "first_order": True,
                "grad_clip_norm": source["settings"]["grad_clip_norm"],
                "ignore_iou": load_config(baseline_path)["evaluation"]["ignore_iou"]}
    from training.retrieval_memory import MODULES as _REQUIRED_MODULES

    if list(settings.get("modules") or []) != list(_REQUIRED_MODULES):
        raise ValueError(f"FOMAML 必须同时打开 {list(_REQUIRED_MODULES)}")
    if not 0 < float(settings["reuse_weight_cap"]) < 1:
        raise ValueError("复用权重上限必须在 (0, 1) 内，不能整段复用")
    if not isfinite(float(settings["where_loss_weight"])) or float(settings["where_loss_weight"]) < 0:
        raise ValueError("Where 损失权重必须为非负有限值")
    if int(settings.get("views_per_group", 1)) < 1:
        raise ValueError("views_per_group 必须为正")
    settings["views_per_group"] = int(settings.get("views_per_group", 1))
    settings["fallback_views_per_group"] = int(settings.get("fallback_views_per_group", max(settings["views_per_group"], 3)))
    if settings["fallback_views_per_group"] < settings["views_per_group"]:
        raise ValueError("fallback_views_per_group 不能小于 views_per_group")
    settings["gradient_checkpointing"] = bool(settings.get("gradient_checkpointing", True))
    settings["prescreen_k"] = int(prescreen_k)
    settings["prescreen_audit"] = int(prescreen_audit)
    if settings["prescreen_k"] < 0 or settings["prescreen_audit"] < 0:
        raise ValueError("prescreen-k 和 prescreen-audit 不能为负")
    if settings["prescreen_audit"] and settings["prescreen_k"] < 1:
        raise ValueError("prescreen-audit 需要同时给出正的 prescreen-k")
    settings["prescreen_audit_every"] = int(prescreen_audit_every)
    settings["sketch_dim"] = int(sketch_dim)
    settings["val_ap50"] = bool(val_ap50)
    settings["val_max_episodes"] = int(val_max_episodes)
    if settings["prescreen_audit_every"] < 1:
        raise ValueError("prescreen-audit-every 必须为正")
    if settings["sketch_dim"] < 1:
        raise ValueError("sketch-dim 必须为正")
    if settings["val_max_episodes"] < 0:
        raise ValueError("val-max-episodes 不能为负")
    from training.experience_bank import load_experience_bank
    from training.retrieval_memory import load_retrieval_config

    if experience_bank is None:
        raise ValueError("必须显式指定 --experience-bank，禁止自动选择最新经验库目录")
    bank_path = Path(experience_bank).resolve(strict=True)
    bank_payload = load_experience_bank(bank_path)
    load_retrieval_config()
    settings["experience_bank"] = str(bank_path)
    settings["bank_version"] = str(bank_payload.get("theta_sha256") or "")
    settings["theta_version"] = str(bank_payload.get("theta_sha256") or "")
    validate_detection_loss_settings(settings["loss"])
    if pilot:
        settings["epochs"] = config["pilot"]["epochs"]
        episodes = [item for item in episodes if item["dataset_id"] == config["pilot"]["dataset"]]
        episodes = sorted(episodes, key=lambda item: item["episode_id"])[:config["pilot"]["episodes"]]
    if epochs is not None:
        if int(epochs) < 1:
            raise ValueError("epochs 必须为正")
        settings["epochs"] = int(epochs)
    if (not episodes or any(settings[key] < 1 for key in ("epochs", "inner_steps", "log_every_steps"))
            or any(not isfinite(settings[key]) or settings[key] <= 0 for key in ("inner_lr", "outer_lr"))):
        raise ValueError("FOMAML 需要非空 Episode、正轮数、步数及学习率")

    code = _code_version()
    for path in ("training/theta0.py", "training/fomaml_roi.py", "data_proc/fomaml_episodes.py",
                 "scripts/train.py"):
        code["source_sha256"][path] = file_sha256(project_root() / path)
    episode_ids = [item["episode_id"] for item in episodes]
    provenance = {**source, "mode": "pilot" if pilot else "fold_train_full",
                  "theta0_training_method": "supervised_warmup_and_fomaml_modules",
                  "fomaml_settings": settings, "fomaml_config_sha256": file_sha256(config_path),
                  "fomaml_data_profile": scope["data_profile"],
                  "fomaml_data_config_sha256": file_sha256(named_config_path("fomaml_data.yaml")),
                  "fomaml_scope": scope, "fomaml_episode_ids": episode_ids,
                  "fomaml_episode_sha256": json_sha256(episodes), "code": code,
                  "supervised_theta0": str(theta0), "supervised_theta0_sha256": theta0_hash}
    run_dir = project_root() / config["output_dir"] / f"fold_{source['fold_id']}" / run_name
    summary = {"fold_id": source["fold_id"], "mode": provenance["mode"],
               "data_profile": scope["data_profile"], "episode_source": scope["episode_source"],
               "train_families": len(scope["train_family_ids"]),
               "episodes_per_epoch": len(episodes),
               "planned_outer_steps": settings["epochs"] * len(episodes), "settings": settings,
               "output": str(run_dir), "first_episode": episode_ids[0]}
    print(
        f"FOMAML 折 {source['fold_id']}，分组 {scope['data_profile']}，"
        f"每轮 {len(episodes)} 条任务，族 {len(scope['train_family_ids'])}，"
        f"经验库 {Path(settings['experience_bank']).name}",
        flush=True,
    )
    if dry_run:
        return summary
    device = _check_runtime(source["settings"])
    detector = build_faster_rcnn(source["settings"]["model"], initialize_backbone=False)
    resume_path = run_dir / "resume.pt"
    finished = run_dir / "last.pt"
    if resume:
        if finished.is_file():
            raise ValueError(f"FOMAML 已写完 last.pt，不能续训：{finished}")
        if not resume_path.is_file():
            raise FileNotFoundError(f"FOMAML 没有断点：{resume_path}")
        resume_state = load_checkpoint(resume_path)
    else:
        if resume_path.is_file() or finished.is_file():
            raise ValueError(f"已有断点或完成权重，请使用 --resume 或换运行名：{run_dir}")
        detector.load_state_dict(parent["model"], strict=True)
        resume_state = None
    del parent
    with b0_run_directory(run_dir, resume=resume):
        result = _train(detector, adapter, episodes, provenance, run_dir, device, resume_state)
    if not pilot and result.get("complete") and bool(config["training"].get("holdout_after_train")):
        from training.meta_transfer import run_holdout

        holdout_name = f"{run_name}-holdout"
        print(f"FOMAML 完成，开始留出评价 {holdout_name}", flush=True)
        result["holdout_run"] = holdout_name
        result["holdout"] = run_holdout(theta0, run_dir / "last.pt", holdout_name, "cuda")
    return result

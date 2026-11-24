"""创新点一：模块经验、效用测量和挑选。"""

from __future__ import annotations

import hashlib
import torch
from torch import nn
from pathlib import Path
from common.io import load_config, load_verified_research_splits, named_config_path
from training.checkpoints import load_checkpoint
from data_proc.detection import image_to_tensor
import json
import os
import random
import re
from time import perf_counter
import numpy as np
from common.io import (
    file_sha256,
    json_sha256,
    load_config,
    load_verified_episode_groups,
    load_verified_group_split,
    load_verified_research_splits,
    named_config_path,
    project_root,
    verify_frozen_episode_artifacts,
)
from data_proc.baseline_data import B0DetectionDataset, build_b0_index, sample_b0_epoch
from data_proc.detection import UnifiedDetectionAdapter
from models.detector import build_faster_rcnn
from training.checkpoints import _atomic_output, b0_run_directory, load_checkpoint, save_checkpoint
from training.train_b0 import _check_runtime, _code_version, _sampling_summary, _train_epoch
from math import isfinite
from common.io import (
    file_sha256, json_sha256, load_config, load_verified_episode_groups, load_verified_group_split,
    load_verified_research_splits, named_config_path, project_root, read_json, read_jsonl,
    verify_frozen_episode_artifacts,
)
from data_proc.baseline_data import build_b0_index
from data_proc.episodes import validate_episode_bank
from models.detector import b0_model_config, build_faster_rcnn
from training.fomaml_roi import validate_detection_loss_settings
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from torchvision.models.detection.transform import resize_boxes
from data_proc.detection import image_to_tensor, target_for_model
from training.fomaml_roi import MODULE_PREFIXES, ROIExample, _eligible_proposal_indices
import math
from copy import deepcopy
from torch.nn import functional as F
from common.io import load_config, named_config_path
from collections import defaultdict
from statistics import mean
from learn2learn.algorithms import MAML
from common.io import file_sha256, load_config, named_config_path, project_root
from training.checkpoints import _atomic_output, b0_run_directory
from training.evaluation import (
    _QueryMatches, _box_iou, _product_macro, _score_product, operating_point_comparison,
)
from training.theta0 import _mean_detection_loss
from training.train_b0 import _code_version
import csv
from common.io import load_config, load_verified_research_splits, named_config_path, project_root
from data_proc.baseline_data import sample_b0_epoch
from training.checkpoints import _atomic_output, load_checkpoint


# lora_adapt.py

def lora_generator(seed: int, episode_id: str) -> torch.Generator:
    digest = int.from_bytes(hashlib.sha256(episode_id.encode("utf-8")).digest()[:8], "little")
    generator = torch.Generator()
    generator.manual_seed((int(seed) + digest) % (2**63 - 1))
    return generator


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float, generator: torch.Generator):
        super().__init__()
        if rank < 1 or alpha <= 0:
            raise ValueError("LoRA 的秩和缩放必须为正")
        self.linear = linear
        self.scale = float(alpha) / float(rank)
        fan_in = linear.in_features
        weight = torch.empty(rank, fan_in)
        weight.normal_(0.0, fan_in ** -0.5, generator=generator)
        self.A = nn.Parameter(weight)
        self.B = nn.Parameter(torch.zeros(linear.out_features, rank))
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features) + (features @ self.A.T @ self.B.T) * self.scale


class LoRABoxPredictor(nn.Module):
    """包住现有分类头和框回归头。两臂按相同顺序构造时，A 的随机数一致。"""

    def __init__(self, predictor: nn.Module, rank: int, alpha: float, generator: torch.Generator):
        super().__init__()
        self.cls_score = LoRALinear(predictor.cls_score, rank, alpha, generator)
        self.bbox_pred = LoRALinear(predictor.bbox_pred, rank, alpha, generator)

    def forward(self, features: torch.Tensor):
        return self.cls_score(features), self.bbox_pred(features)

# meta_tasks.py

INNOVATION1_MODULES = ("Feature", "Where", "Box", "Class")


def load_innovation1_modules() -> list[str]:
    """完整任务固定为四块经验，少一块就不构成创新点一。"""
    config = load_config(named_config_path("utility.yaml"))
    modules = list(config.get("modules") or [])
    if modules != list(INNOVATION1_MODULES):
        raise ValueError(f"创新点一必须同时包含 {list(INNOVATION1_MODULES)}，当前是 {modules}")
    if config.get("require_all_targets") is not False:
        raise ValueError("创新点一禁止要求每个新产品都变好")
    if config.get("no_contribution") != "source_selection":
        raise ValueError("没有贡献时必须记成产品选择，不能记成模块失败")
    if config.get("chassis") != "fomaml":
        raise ValueError("创新点一底座必须是 fomaml")
    return modules


def fold_source_products(fold_id: int) -> list[str]:
    """该折允许借出经验的旧产品。留出新产品、验证产品和外层测试产品都不在这里。"""
    _, _, inner = load_verified_research_splits()
    matches = [fold for fold in inner["folds"] if fold["fold_id"] == fold_id]
    if len(matches) != 1:
        raise ValueError(f"冻结的 OOF 划分中找不到唯一折编号 {fold_id}")
    fold = matches[0]
    products = list(fold["train_product_ids"])
    if not products or len(products) != len(set(products)):
        raise AssertionError(f"折 {fold_id} 的源产品清单无效")
    held_out = set(fold["val_product_ids"])
    if set(products) & held_out:
        raise AssertionError(f"折 {fold_id} 的源产品与留出产品重叠")
    return products


def checkpoint_fold_id(path: Path) -> int:
    checkpoint = load_checkpoint(path)
    return int(checkpoint["provenance"]["fold_id"])


def utility_task_plan(fold_id: int) -> list[dict]:
    """每个旧产品各出四块经验，对应该折的留出新产品。"""
    modules = load_innovation1_modules()
    return [
        {"source": source, "modules": list(modules), "sampling_profile": "source_groups"}
        for source in fold_source_products(fold_id)
    ]


def classify_contributions(product_deltas: dict[str, float]) -> dict:
    """正的差值算对这个新产品有贡献。没有正差值只说明不该选这个来源。"""
    if any(not isinstance(value, (int, float)) for value in product_deltas.values()):
        raise ValueError("贡献差值必须是数值")
    contributed = sorted(name for name, value in product_deltas.items() if value > 0)
    not_selected = sorted(name for name, value in product_deltas.items() if value <= 0)
    return {
        "require_all_targets": False,
        "contributed_products": contributed,
        "not_selected_products": not_selected,
        "outcome": "partial_contribution" if contributed else "source_selection",
    }

# descriptors.py

@torch.no_grad()
def image_descriptor(detector, image_path: str, device: torch.device) -> torch.Tensor:
    """各层特征图做全局平均后拼接，得到这一张图的原始描述。"""
    detector.eval()
    image = image_to_tensor(image_path)
    images, _ = detector.transform([image.to(device)])
    levels = detector.backbone(images.tensors)
    parts = [level.mean(dim=(2, 3)).reshape(-1) for level in levels.values()]
    vector = torch.cat(parts).detach().cpu()
    if not torch.isfinite(vector).all() or vector.ndim != 1 or vector.numel() == 0:
        raise FloatingPointError(f"描述子不是有限向量：{image_path}")
    return vector


def average_descriptors(vectors: list[torch.Tensor]) -> list[float]:
    if not vectors:
        raise ValueError("没有可平均的图像描述")
    widths = {int(vector.numel()) for vector in vectors}
    if len(widths) != 1:
        raise ValueError("图像描述的长度不一致")
    mean = torch.stack(vectors).mean(dim=0)
    if not torch.isfinite(mean).all():
        raise FloatingPointError("平均描述子出现非有限值")
    return [float(value) for value in mean]

# transfer_delta.py

def build_parameter_manifest(model) -> dict:
    """将 B0 的全部命名参数归入四个互斥模块，保留共享层归属。"""
    modules = {label: {"prefixes": list(prefixes), "parameters": []}
               for label, prefixes in MODULE_PREFIXES.items()}
    seen = set()
    parameter_ids = set()
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if name in seen or id(parameter) in parameter_ids:
            raise ValueError(f"Δ 参数有重复名称或共享别名：{name}")
        owners = [label for label, prefixes in MODULE_PREFIXES.items() if name.startswith(prefixes)]
        if len(owners) != 1:
            raise ValueError(f"Δ 参数必须有唯一的模块归属：{name}，匹配到 {owners}")
        modules[owners[0]]["parameters"].append({
            "name": name, "shape": list(parameter.shape), "numel": parameter.numel(),
        })
        seen.add(name)
        parameter_ids.add(id(parameter))
    for label, module in modules.items():
        if not module["parameters"]:
            raise ValueError(f"Δ 参数模块为空：{label}")
        module["tensor_count"] = len(module["parameters"])
        module["scalar_count"] = sum(row["numel"] for row in module["parameters"])
    return {
        "modules": modules,
        "shared_roi_box_head_owner": "Feature",
        "frozen_buffer_names": sorted(name for name, _ in model.named_buffers()),
        "tensor_count": len(seen),
        "scalar_count": sum(row["scalar_count"] for row in modules.values()),
    }


def _prepare_source(config, source, theta_provenance):
    group_split = load_verified_group_split()
    outer, meta, inner = load_verified_research_splits()
    groups = load_verified_episode_groups()
    hashes = verify_frozen_episode_artifacts()

    fold_id = theta_provenance.get("fold_id")
    fold = next((row for row in inner["folds"] if row["fold_id"] == fold_id), None)
    if fold is None or theta_provenance.get("mode") not in ("pilot", "fold_train_full"):
        raise ValueError("Δ 需要指定冻结 OOF 折中的 θ₀ 权重")
    train_families = set(theta_provenance["train_family_ids"])
    theta_scope = set(fold["theta0_train_family_ids"])
    held_out = (set(fold["target_family_ids"]) | set(meta["meta_val_family_ids"])
                | set(outer["outer_test_family_ids"]))
    if not train_families or not train_families <= theta_scope or train_families & held_out:
        raise ValueError("θ₀ 训练产品族超出冻结折内边界")
    if theta_provenance["mode"] == "fold_train_full" and train_families != theta_scope:
        raise ValueError("正式折内 θ₀ 必须覆盖全部允许训练产品族")

    source_groups = [row for row in groups if f"{row['dataset_id']}:{row['product_id']}" == source]
    source_families = {row["family_id"] for row in source_groups}
    allowed = theta_scope & set(fold["delta_source_family_ids"]) & train_families
    if len(source_families) != 1 or not source_families <= allowed:
        raise ValueError(f"源任务不属于该 θ₀ 折实际训练且允许生成 Δ 的产品族：{source}")

    root = project_root()
    episode_config = load_config(root / config["episode_config"])
    data_config_path = root / config["data_config"]
    data_config = load_config(data_config_path)
    audit_dir = root / data_config["output_dir"]
    expected = {
        "fold_sha256": json_sha256(fold),
        "config_sha256": file_sha256(named_config_path("baseline.yaml")),
        "detector_config_sha256": file_sha256(named_config_path("b0_v3.yaml")),
        "data_config_sha256": file_sha256(data_config_path),
        "manifest_sha256": file_sha256(audit_dir / "manifest.csv"),
        "group_split_sha256": file_sha256(audit_dir / data_config["physical_group"]["artifacts"]["group_split"]),
        "split_manifest_sha256": hashes["split_artifact_hashes_sha256"],
        "episode_manifest_sha256": file_sha256(root / episode_config["output_dir"] / episode_config["artifacts"]["hashes"]),
    }
    changed = [key for key, digest in expected.items() if theta_provenance.get(key) != digest]
    if changed:
        raise ValueError(f"θ₀ 与当前冻结来源不一致：{', '.join(changed)}")
    adapter = UnifiedDetectionAdapter(config)
    index = build_b0_index(adapter, source_groups, source_families, group_split)
    return adapter, index, source_families.pop(), expected


def _isolated_delta(model, theta_state, selected):
    updates = {}
    changed = 0
    max_error = 0.0
    for name, value in model.state_dict().items():
        current = value.detach().cpu()
        original = theta_state[name]
        if name not in selected:
            if not torch.equal(current, original):
                raise AssertionError(f"模块外参数或缓冲区发生变化：{name}")
            continue
        update = current - original
        if not torch.isfinite(update).all():
            raise FloatingPointError(f"Δ 出现非有限值：{name}")
        replay = original + update
        torch.testing.assert_close(replay, current, rtol=1e-6, atol=1e-8)
        max_error = max(max_error, (replay - current).abs().max().item())
        changed += torch.count_nonzero(update).item()
        updates[name] = update
    if updates.keys() != selected or not changed:
        raise AssertionError("模块 Δ 参数不完整或训练后全部为零")
    return updates, changed, max_error


def _repeat_positions(positions: list[int], steps: int | None) -> list[int]:
    if steps is None:
        return list(positions)
    if steps < 1:
        raise ValueError("steps 必须是正整数")
    if not positions:
        raise ValueError("没有可重复的固定样本")
    repeats, remainder = divmod(steps, len(positions))
    return list(positions) * repeats + list(positions)[:remainder]


def load_delta_sampling(profile: str | None = None) -> tuple[str, dict, str]:
    """读取命名抽样配置。默认保持成对两图，产品实物组抽样必须显式指定。"""
    path = named_config_path("delta_sampling.yaml")
    spec = load_config(path)
    name = spec["profile"] if profile is None else profile
    profiles = spec.get("profiles")
    if not isinstance(profiles, dict) or name not in profiles:
        raise ValueError(f"Δ 抽样配置不存在：{name}")
    selected = dict(profiles[name])
    required = (
        "groups_per_dataset", "positive_group_fraction", "max_views_per_group",
        "allow_repeat", "paired_images",
    )
    if any(key not in selected for key in required):
        raise ValueError(f"Δ 抽样配置缺少字段：{name}")
    if not isinstance(selected["allow_repeat"], bool) or not isinstance(selected["paired_images"], bool):
        raise ValueError(f"Δ 抽样配置的开关必须是布尔值：{name}")
    return name, selected, file_sha256(path)


def schedule_delta_updates(positions: list[int], steps: int | None, allow_repeat: bool) -> list[int]:
    """成对配置可以循环同两张图。产品实物组配置只允许每个抽出的视角更新一次。"""
    if steps is not None and steps < 1:
        raise ValueError("steps 必须是正整数")
    if not allow_repeat and steps not in (None, len(positions)):
        raise ValueError("该抽样配置不循环样本，steps 必须省略或等于抽出的视角数")
    return _repeat_positions(positions, steps)


def validate_delta_rows(rows: list[dict], profile: dict) -> None:
    """成对配置锁定一张缺陷图和一张正常图；产品配置要求多个实物组且两类都出现。"""
    flags = [int(row["normal_flag"]) for row in rows]
    groups = {row["physical_group_id"] for row in rows}
    image_ids = [row["image_id"] for row in rows]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Δ 基础样本出现重复图像")
    if 0 not in flags or 1 not in flags:
        raise ValueError("Δ 样本必须同时包含缺陷图和正常图")
    if profile["paired_images"]:
        if sorted(flags) != [0, 1] or len(groups) != 2:
            raise ValueError("成对样本配置要求两个不同实物组、各一张缺陷图和正常图")
    elif len(groups) < 2:
        raise ValueError("产品实物组配置至少需要两个不同实物组")


def _train_module(model, theta_state, selected, dataset, positions, device, settings):
    model.load_state_dict(theta_state, strict=True)
    model.zero_grad(set_to_none=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in selected)
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=settings["learning_rate"], momentum=settings["momentum"],
        weight_decay=settings["weight_decay"],
    )
    started = perf_counter()
    result = _train_epoch(model, optimizer, dataset, positions, device, settings, 1, 0)
    if result["steps"] != len(positions) or result["skipped_annotations"]:
        raise AssertionError(f"Δ 训练未严格使用全部固定样本：{result}")
    result["train_seconds"] = perf_counter() - started
    updates, result["changed_scalars"], result["replay_max_abs_error"] = _isolated_delta(
        model, theta_state, selected,
    )
    return updates, result


def _write_json_atomic(path: Path, payload: dict) -> None:
    with _atomic_output(path) as output:
        output.write((json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def select_delta_modules(manifest: dict, modules: list[str] | None) -> list[str]:
    available = list(manifest["modules"])
    if modules is None:
        return available
    if not modules or len(modules) != len(set(modules)) or any(name not in available for name in modules):
        raise ValueError(f"Δ 模块必须是 {available} 的非空子集：{modules}")
    return list(modules)


def train_transfer_delta(
    theta0: Path, source: str, run_name: str, steps: int | None = None,
    sampling_profile: str | None = None, modules: list[str] | None = None, dry_run: bool = False,
) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_name):
        raise ValueError("run-name 只能使用字母、数字、连字符和下划线")
    if not re.fullmatch(r"[a-z0-9_-]+:[a-z0-9_-]+", source):
        raise ValueError("source 必须使用 dataset:product 格式")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config = load_config(named_config_path("baseline.yaml"))
    delta_config_path = named_config_path("transfer_delta.yaml")
    delta_config = load_config(delta_config_path)
    theta0 = theta0.resolve(strict=True)
    theta_hash = file_sha256(theta0)
    checkpoint = load_checkpoint(theta0)
    if file_sha256(theta0) != theta_hash:
        raise RuntimeError("θ₀ 权重在读取期间发生变化，请使用已完成的 checkpoint")
    theta_state = checkpoint["model"]
    theta_provenance = checkpoint["provenance"]
    theta_epoch = checkpoint["epoch"]
    del checkpoint

    adapter, index, family, hashes = _prepare_source(config, source, theta_provenance)
    settings = {**theta_provenance["settings"], **delta_config["training"]}
    profile_name, sampling, sampling_hash = load_delta_sampling(sampling_profile)
    if settings["batch_size"] != 1:
        raise ValueError("Δ 按单图更新，批次必须为 1")
    positions = sample_b0_epoch(index, seed=settings["seed"], epoch=1, sampling=sampling)
    schedule = schedule_delta_updates(positions, steps, sampling["allow_repeat"])
    rows = [index[position] for position in positions]
    validate_delta_rows(rows, sampling)
    if any(adapter.build_b0_record(row) is None for row in rows):
        raise ValueError("Δ 样本中有缺陷标注被过滤")

    model = build_faster_rcnn(settings["model"], initialize_backbone=False)
    model.load_state_dict(theta_state, strict=True)
    manifest = build_parameter_manifest(model)
    trained_modules = select_delta_modules(manifest, modules)
    manifest_sha = json_sha256(manifest)
    code = _code_version()
    for path in ("scripts/train.py", "training/meta_transfer.py"):
        code["source_sha256"][path] = file_sha256(project_root() / path)
    provenance = {
        **hashes, "mode": "single_source_isolation_pilot", "fold_id": theta_provenance["fold_id"],
        "source_task_id": source, "source_family_id": family, "theta0": str(theta0),
        "theta0_sha256": theta_hash, "theta0_epoch": theta_epoch,
        "theta0_actual_train_family_ids": theta_provenance["train_family_ids"],
        "theta0_allowed_train_family_ids": theta_provenance["theta0_train_family_ids"],
        "delta_config_sha256": file_sha256(delta_config_path), "parameter_manifest_sha256": manifest_sha,
        "code": code,         "training_steps_per_module": len(schedule),
        "fixed_sample_count": len(positions),
        "sampling_profile": profile_name,
        "sampling_config_sha256": sampling_hash,
        "allow_repeat": sampling["allow_repeat"],
        "unique_physical_groups": len({row["physical_group_id"] for row in rows}),
        "trained_modules": trained_modules,
        "repeated_updates": len(schedule) - len(set(schedule)),
        "update_image_ids": [index[position]["image_id"] for position in schedule],
        "source_index_sha256": json_sha256(index),
        "seed": settings["seed"], "optimizer": {key: settings[key] for key in (
            "learning_rate", "momentum", "weight_decay", "grad_clip_norm",
        )},
        "sampling": {**_sampling_summary(index, positions), "ordered_samples": rows},
    }
    result = {"provenance": provenance, "complete": False, "modules": {}}
    sampling_view = {key: value for key, value in provenance["sampling"].items() if key != "ordered_samples"}
    print(json.dumps({"fold": provenance["fold_id"], "source": source,
                      "sampling_profile": provenance["sampling_profile"],
                      "steps": provenance["training_steps_per_module"],
                      "fixed_samples": provenance["fixed_sample_count"],
                      "unique_physical_groups": provenance["unique_physical_groups"],
                      "trained_modules": provenance["trained_modules"],
                      "repeated_updates": provenance["repeated_updates"],
                      "sampling": sampling_view,
                      "groups": {label: row["tensor_count"] for label, row in manifest["modules"].items()}},
                     ensure_ascii=False), flush=True)
    fold_path = f"fold_{provenance['fold_id']}"
    run_dir = project_root() / delta_config["output_dir"] / fold_path / source.replace(":", "-") / run_name
    metrics_path = run_dir / "metrics.json"
    existing = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else None
    if existing is not None:
        if existing["provenance"].get("sampling_config_sha256") != sampling_hash:
            raise ValueError(f"已有 Δ 的抽样配置和当前不一致：{run_dir}")
        if set(existing["provenance"]["trained_modules"]) != set(trained_modules):
            raise ValueError(f"已有 Δ 的模块清单和当前不一致：{run_dir}")
        finished = {
            label for label in trained_modules
            if label in existing["modules"] and (run_dir / f"{label.lower()}.pt").is_file()
        }
        if existing.get("complete") and finished == set(trained_modules):
            print(f"Δ 已完成，跳过：{run_dir}", flush=True)
            return existing
    else:
        finished = set()
    if dry_run:
        return result

    device = _check_runtime(settings)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    model.to(device)
    dataset = B0DetectionDataset(adapter, index)
    with b0_run_directory(run_dir, resume=existing is not None):
        manifest_path = run_dir / "parameter_manifest.json"
        if existing is None:
            _write_json_atomic(manifest_path, manifest)
            _write_json_atomic(metrics_path, result)
        else:
            result = existing
            result["complete"] = False
        for label in trained_modules:
            if label in finished:
                print(f"Δ {label} {source} 已完成，跳过", flush=True)
                continue
            module = manifest["modules"][label]
            print(f"Δ {label} {source} fold={provenance['fold_id']}", flush=True)
            selected = {row["name"] for row in module["parameters"]}
            updates, metrics = _train_module(model, theta_state, selected, dataset, schedule, device, settings)
            module_path = run_dir / f"{label.lower()}.pt"
            save_checkpoint(module_path, {"module": label, "delta": updates, "provenance": provenance})
            result["modules"][label] = {**metrics, "checkpoint": module_path.name,
                                        "checkpoint_sha256": file_sha256(module_path)}
            _write_json_atomic(metrics_path, result)
            del updates
        result["complete"] = True
        _write_json_atomic(metrics_path, result)
    print(f"Δ Pilot 完成：{run_dir}", flush=True)
    return result

# holdout_inputs.py

_HEAD_PREFIX = "roi_heads.box_predictor."
_ROLE_COUNTS = {"reference": 5, "support": 5, "query_normal": 5, "query_defect": 5}


def _check_metadata(actual: dict, expected: dict, label: str) -> None:
    mismatches = [key for key, value in expected.items() if actual.get(key) != value]
    if mismatches:
        raise ValueError(f"{label} 来源或协议不一致：{', '.join(mismatches)}")


def _verified_checkpoint(path: Path, expected_hash: str | None = None) -> tuple[dict, str]:
    digest = file_sha256(path)
    if expected_hash is not None and digest != expected_hash:
        raise ValueError(f"输入权重哈希不符：{path}")
    state = load_checkpoint(path)
    if file_sha256(path) != digest:
        raise RuntimeError(f"权重在读取期间发生变化：{path}")
    return state, digest


def _completed_theta0(path: Path) -> tuple[dict, str]:
    state, digest = _verified_checkpoint(path)
    metrics = read_json(path.parent / "metrics.json")
    _check_metadata(metrics, {"complete": True, "provenance": state["provenance"],
                              "global_step": state["global_step"]}, str(path))
    if len(metrics["history"]) != state["epoch"] or state["global_step"] < 1:
        raise ValueError(f"θ₀ 训练历史和断点轮数不一致：{path}")
    return state, digest


def _holdout_episodes(episodes, groups, adapter, fold, universe, excluded) -> tuple[list[dict], dict]:
    held_out = set(fold["target_family_ids"])
    group_members = {
        (row["dataset_id"], row["product_id"], row["physical_group_id"]): row for row in groups
    }
    selected, query_ids, query_annotations, removed = [], [], [], []
    available = [action for action, enabled in zip(universe, fold["source_available_mask"]) if enabled]
    for episode in sorted(episodes, key=lambda row: row["episode_id"]):
        if episode["family_id"] not in held_out and episode["inner_target_fold"] != fold["fold_id"]:
            continue
        _check_metadata(episode, {
            "bank": "oof", "inner_target_fold": fold["fold_id"], "evaluation_stage": "inner_oof",
            "n_way": 1, "k_shot": 5, "q_defect_per_class": 5, "q_normal_groups": 5,
            "n_reference_groups": 5, "view_policy": "all_views_in_physical_group",
            "ignore_policy": "episode_external_defects_are_ignore", "source_universe_action_ids": universe,
            "source_available_mask": fold["source_available_mask"], "source_available_action_ids": available,
        }, episode["episode_id"])
        if episode["family_id"] not in held_out or len(episode["episode_classes"]) != 1:
            raise ValueError(f"Episode 不属于留出折或不是单类：{episode['episode_id']}")
        prepared = dict(episode)
        prepared["support"] = []
        used_images, used_groups = set(), set()
        label = episode["episode_classes"][0]
        for role, count in _ROLE_COUNTS.items():
            if len(episode[role]) != count:
                raise ValueError(f"Episode 的 {role} 物理组数不符合协议：{episode['episode_id']}")
            for unit in episode[role]:
                group = unit["physical_group_id"]
                key = (episode["dataset_id"], episode["product_id"], group)
                original = group_members[key]
                image_ids = unit["image_ids"]
                _check_metadata(unit, {key: original[key].split("|") if original[key] else []
                                       for key in ("image_ids", "image_paths", "mask_paths")}, str(key))
                rows = [adapter.manifest[image_id] for image_id in image_ids]
                if (
                    original["family_id"] != episode["family_id"] or group in used_groups
                    or not image_ids or len(image_ids) != len(set(image_ids)) or used_images.intersection(image_ids)
                    or any((row["dataset_id"], row["product_id"]) != key[:2] for row in rows)
                    or sorted(row["image_path"] for row in rows) != sorted(unit["image_paths"])
                    or sorted(row["mask_path"] for row in rows if row["mask_path"]) != sorted(unit["mask_paths"])
                ):
                    raise AssertionError(f"留出 Episode 的物理组或视角与冻结索引不符：{key}")
                used_images.update(image_ids)
                used_groups.add(group)
                if role.startswith("query_"):
                    query_ids.extend(image_ids)
                    query_annotations.extend(
                        (episode["episode_id"], role, group, row["image_id"], row["mask_path"], row["defect_labels"])
                        for row in rows
                    )
                if role == "support":
                    kept = [row for row in rows if row["image_id"] not in excluded]
                    labels = {part for row in kept for part in row["defect_labels"].split("|") if part}
                    if label not in labels:
                        raise ValueError(f"排除官方冲突视角后缺失 Support 缺陷组：{key}")
                    removed.extend(image_id for image_id in image_ids if image_id in excluded)
                    prepared["support"].append({
                        **unit, "image_ids": [row["image_id"] for row in kept],
                        "image_paths": sorted(row["image_path"] for row in kept),
                        "mask_paths": sorted(row["mask_path"] for row in kept if row["mask_path"]),
                        "visible_defect_labels": sorted(labels), "ignore_labels": sorted(labels - {label}),
                    })
        selected.append(prepared)
    if len(selected) != 21 or {item["family_id"] for item in selected} != held_out:
        raise ValueError("fold 0 留出集必须覆盖冻结的七个产品族和 21 条 Episode")
    return selected, {
        "episode_count": len(selected), "episode_ids": [item["episode_id"] for item in selected],
        "episode_sha256": json_sha256(selected), "unique_query_images": len(set(query_ids)),
        "query_view_occurrences": len(query_ids), "query_view_order_sha256": json_sha256(query_ids),
        "query_annotation_index_sha256": json_sha256(query_annotations),
        "support_conflict_filter": {"excluded_image_ids": sorted(set(removed)), "excluded_occurrences": len(removed)},
    }


def _checkpoint_models(supervised, fomaml, manifest_model, *, four_modules: bool):
    original, adapted = supervised["model"], fomaml["model"]
    if not (original.keys() == adapted.keys() == manifest_model.state_dict().keys()):
        raise ValueError("两份 θ₀ 的检测器权重名称与 B0-V3 不一致")
    head, changed = {}, set()
    for name, expected in manifest_model.state_dict().items():
        first, second = original[name], adapted[name]
        if any(tensor.shape != expected.shape or tensor.dtype != expected.dtype for tensor in (first, second)):
            raise ValueError(f"θ₀ 参数形状或类型与 B0-V3 不一致：{name}")
        if name.startswith(_HEAD_PREFIX):
            head[name.removeprefix(_HEAD_PREFIX)] = first
        if not torch.equal(first, second):
            changed.add(name)
    if not changed:
        raise ValueError("FOMAML 没有更新继承自监督 θ₀ 的参数")
    if four_modules:
        for name in changed:
            owners = [label for label, prefixes in MODULE_PREFIXES.items() if name.startswith(prefixes)]
            if len(owners) != 1:
                raise ValueError(f"FOMAML 更新了四个模块之外的参数：{name}")
    elif any(not name.startswith(_HEAD_PREFIX) for name in changed):
        raise ValueError("FOMAML 必须只更新继承自监督 θ₀ 的 RoI 预测头")
    return head, adapted, sorted(changed)


def _load_deltas(directory, identity, model, base_state, require_all: bool = True):
    metrics = read_json(directory / "metrics.json")
    delta_provenance = metrics["provenance"]
    _check_metadata(metrics, {"complete": True}, "Δ 生成记录")
    _check_metadata(delta_provenance, identity, "Δ 父权重和冻结数据")
    parameter_manifest = build_parameter_manifest(model)
    stored_manifest = read_json(directory / "parameter_manifest.json")
    if (stored_manifest != parameter_manifest
            or json_sha256(parameter_manifest) != delta_provenance["parameter_manifest_sha256"]):
        raise ValueError("Δ 模块参数清单与 B0-V3 不一致")
    trained = set(metrics["modules"])
    if not trained or trained - set(MODULE_PREFIXES) or (require_all and trained != set(MODULE_PREFIXES)):
        raise ValueError("Δ 模块记录不完整或越出四个互斥模块")
    deltas, hashes = {}, {}
    for label in sorted(trained):
        info = parameter_manifest["modules"][label]
        path = directory / f"{label.lower()}.pt"
        module_metrics = metrics["modules"][label]
        _check_metadata(module_metrics, {"checkpoint": path.name, "steps": delta_provenance["training_steps_per_module"]}, label)
        artifact, digest = _verified_checkpoint(path, module_metrics["checkpoint_sha256"])
        _check_metadata(artifact, {"module": label, "provenance": delta_provenance}, label)
        updates = artifact["delta"]
        if updates.keys() != {parameter["name"] for parameter in info["parameters"]}:
            raise ValueError(f"Δ {label} 模块的参数集合不完整或越界")
        for name, tensor in updates.items():
            base = base_state[name]
            if tensor.shape != base.shape or tensor.dtype != base.dtype or not torch.isfinite(tensor).all():
                raise ValueError(f"Δ 张量形状、类型或数值不合法：{name}")
        deltas[label], hashes[label] = updates, digest
    return deltas, hashes, delta_provenance


def _episode_family(episode_id: str, product_id: str) -> str:
    parts = episode_id.split(":")
    if len(parts) != 4 or parts[0] != "oof" or parts[2] != product_id:
        raise ValueError(f"任务记忆中的 Episode 编号无法对应产品：{episode_id} / {product_id}")
    return f"{parts[1]}:{parts[2]}"


def _stored_delta(row: dict, parts: Path | None) -> dict:
    delta = row.get("delta")
    if isinstance(delta, dict):
        return delta
    name = row.get("delta_file")
    if parts is None or not isinstance(name, str) or Path(name).name != name:
        raise ValueError("任务记忆缺少可用的参数差")
    payload = load_checkpoint(parts / name)
    stored = payload.get("delta")
    if not isinstance(stored, dict):
        raise ValueError(f"任务差文件不完整：{name}")
    return stored


def final_product_library(episodes: dict, episode_order: list[str], parts: Path | None = None) -> dict[str, dict]:
    """每个源产品保留最后一轮中最后一次留下的任务差。"""
    if len(episode_order) != len(set(episode_order)) or set(episode_order) != set(episodes):
        raise ValueError("任务记忆和最后一轮的 Episode 顺序不一致")
    latest = {}
    for episode_id in episode_order:
        row = episodes[episode_id]
        family = _episode_family(episode_id, row["product_id"])
        latest[family] = {
            "episode_id": episode_id,
            "product_id": row["product_id"],
            "family_id": family,
            "delta": _stored_delta(row, parts),
        }
    return latest


def _module_parameter_names(model, label: str) -> set[str]:
    """任务差只保存 named_parameters()，不含冻结归一化的缓冲区。"""
    prefixes = MODULE_PREFIXES[label]
    return {name for name, _parameter in model.named_parameters() if name.startswith(prefixes)}


def _checked_block(block, names: set[str], family: str, label: str, base_state: dict) -> dict:
    if set(block) != names:
        raise ValueError(f"{family} 的 {label} 参数集合不完整或越界")
    checked = {}
    for name, tensor in block.items():
        base = base_state[name]
        if tensor.shape != base.shape or tensor.dtype != base.dtype or not torch.isfinite(tensor).all():
            raise ValueError(f"任务记忆张量不合法：{family} {name}")
        checked[name] = tensor.detach().cpu()
    return checked


def _load_fomaml_memory(fomaml: Path, trained_episode_ids: list[str], saved_epoch: int, sources: set[str], base_state: dict, base_head: dict, model) -> dict:
    """读取与已保存 FOMAML 权重同一轮的模块任务记忆。"""
    path = fomaml.parent / "module_memory.pt"
    if not path.is_file():
        raise FileNotFoundError(f"FOMAML 运行目录缺少 module_memory.pt：{path}")
    digest = file_sha256(path)
    payload = load_checkpoint(path)
    if file_sha256(path) != digest:
        raise RuntimeError("任务记忆在读取期间发生变化")
    modules = payload.get("modules")
    if modules not in (["Box", "Class"], ["Feature", "Where", "Box", "Class"]):
        raise ValueError("任务记忆的模块必须是 Box/Class，或 Feature/Where/Box/Class")
    stored = payload.get("episodes")
    if not isinstance(stored, dict) or set(stored) != set(trained_episode_ids):
        raise ValueError("任务记忆的 Episode 和 FOMAML 训练名单不一致")
    metrics = read_json(fomaml.parent / "metrics.json")
    history = metrics.get("history") or []
    if metrics.get("early_stopped") or metrics.get("best_epoch") != saved_epoch or len(history) != saved_epoch:
        raise ValueError("任务记忆来自最后一轮，不能和早停恢复的更早权重配对")
    parts = path.parent / str(payload.get("delta_storage") or "module_memory_parts")
    library = final_product_library(
        stored, [row["episode_id"] for row in history[-1]["episodes"]],
        parts if parts.is_dir() else None,
    )
    if set(library) != sources:
        raise ValueError("任务记忆中的源产品和该折允许借出的产品族不一致")
    if modules == ["Box", "Class"]:
        reference = base_head
        expected = {
            "Box": {name for name in reference if name.startswith("bbox_pred")},
            "Class": {name for name in reference if name.startswith("cls_score")},
        }
        if set(expected) != {"Box", "Class"} or any(not names for names in expected.values()):
            raise AssertionError("检测头参数无法按 Box 和 Class 拆开")
        if set().union(*expected.values()) != set(reference):
            raise AssertionError("检测头存在 Box 和 Class 之外的参数")
        missing = ["Feature", "Where"]
    else:
        reference = base_state
        expected = {label: _module_parameter_names(model, label) for label in modules}
        if any(not names for names in expected.values()):
            raise AssertionError("检测器四块参数不完整")
        missing = []
    sources_out = {}
    for family, row in library.items():
        delta = row["delta"]
        if set(delta) != set(modules):
            raise ValueError(f"{family} 的任务记忆模块和保存名单不一致")
        blocks = {
            label: _checked_block(delta[label], expected[label], family, label, reference)
            for label in modules
        }
        sources_out[family] = {**row, "delta": blocks}
    return {
        "path": str(path), "sha256": digest, "saved_epoch": saved_epoch,
        "modules_present": list(modules), "modules_missing": missing,
        "sources": sources_out,
    }


def load_holdout(supervised: Path, fomaml: Path):
    """加载固定 fold 0 留出评价输入，在打开图像前校验范围、标签协议和权重身份。"""
    group_split = load_verified_group_split()
    outer, meta, inner = load_verified_research_splits()
    groups = load_verified_episode_groups()
    artifacts = verify_frozen_episode_artifacts()
    fold = next(item for item in inner["folds"] if item["fold_id"] == 0)
    targets = set(fold["target_family_ids"])
    sources = set(fold["theta0_train_family_ids"])
    forbidden = set(meta["meta_val_family_ids"]) | set(outer["outer_test_family_ids"])
    if (len(targets) != 7 or len(sources) != 27 or targets & sources or (targets | sources) & forbidden
            or targets != set(fold["val_family_ids"]) or sources != set(fold["train_family_ids"]) or
            sources != set(fold["delta_source_family_ids"])):
        raise ValueError("fold 0 的七个留出族和 27 个折内源族与冻结协议不一致")

    root = project_root()
    baseline_path = named_config_path("baseline.yaml")
    baseline = load_config(baseline_path)
    adapter = UnifiedDetectionAdapter(baseline)
    data_path = root / baseline["data_config"]
    data = load_config(data_path)
    audit_dir = root / data["output_dir"]
    ep_config = load_config(root / baseline["episode_config"])
    ep_dir = root / ep_config["output_dir"]
    bank = ep_config["artifacts"]["oof_bank"]
    episodes = read_jsonl(ep_dir / bank)
    validate_episode_bank(episodes, groups, outer, meta, inner)
    excluded = (set(group_split["excluded_official_train_image_ids"])
                if group_split["conflict_policy"] == "preserve_official_test" else set())
    holdout, audit = _holdout_episodes(episodes, groups, adapter, fold, inner["source_universe_action_ids"], excluded)
    hashes = {
        "fold_id": 0, "fold_sha256": json_sha256(fold), "config_sha256": file_sha256(baseline_path),
        "detector_config_sha256": file_sha256(named_config_path("b0_v3.yaml")),
        "data_config_sha256": file_sha256(data_path), "manifest_sha256": file_sha256(audit_dir / "manifest.csv"),
        "group_split_sha256": file_sha256(audit_dir / data["physical_group"]["artifacts"]["group_split"]),
        "split_manifest_sha256": artifacts["split_artifact_hashes_sha256"],
        "episode_manifest_sha256": file_sha256(ep_dir / ep_config["artifacts"]["hashes"]),
    }
    theta_scope = {
        **hashes, "mode": "fold_train_full", "train_family_ids": sorted(sources),
        "theta0_train_family_ids": sorted(sources), "delta_source_family_ids": sorted(sources),
        "oof_target_family_ids": sorted(targets), "meta_val_family_ids": sorted(meta["meta_val_family_ids"]),
        "outer_test_family_ids": sorted(outer["outer_test_family_ids"]),
        "train_index_sha256": json_sha256(build_b0_index(adapter, groups, sources, group_split)),
        "theta0_config_sha256": file_sha256(named_config_path("theta0.yaml")),
    }

    supervised = supervised.resolve(strict=True)
    fomaml = fomaml.resolve(strict=True)
    warmup, warmup_hash = _completed_theta0(supervised)
    warmup_source = warmup["provenance"]
    _check_metadata(warmup_source, {**theta_scope, "theta0_training_method": None}, "监督 θ₀")
    meta_weights, meta_hash = _completed_theta0(fomaml)
    meta_source = meta_weights["provenance"]
    method = meta_source.get("theta0_training_method")
    if method not in (
        "supervised_warmup_and_fomaml_roi_predictor",
        "supervised_warmup_and_fomaml_modules",
    ):
        raise ValueError("FOMAML 权重不是同折监督暖启动之后的元训练")
    current_fomaml_hash = file_sha256(named_config_path("fomaml.yaml"))
    if method == "supervised_warmup_and_fomaml_modules" and meta_source.get("fomaml_config_sha256") != current_fomaml_hash:
        raise ValueError("FOMAML 配置与产生该权重的配置不一致")
    _check_metadata(meta_source, {
        **theta_scope, "theta0_training_method": method,
        "supervised_theta0_sha256": warmup_hash, "fomaml_trained_family_ids": sorted(sources),
        "settings": warmup_source["settings"],
        "fomaml_config_sha256": meta_source.get("fomaml_config_sha256"),
    }, "FOMAML θ₀")
    meta_settings = meta_source["fomaml_settings"]
    adaptation = {key: meta_settings[key] for key in ("inner_steps", "inner_lr")}
    adaptation_loss = meta_settings.get("loss")
    if not isinstance(adaptation_loss, dict):
        raise ValueError("FOMAML 权重缺少检测对齐损失配置")
    validate_detection_loss_settings(adaptation_loss)
    episodes_by_id = {episode["episode_id"]: episode for episode in episodes}
    fomaml_families = {episodes_by_id[episode_id]["family_id"] for episode_id in meta_source["fomaml_episode_ids"]}
    if (
        fomaml_families != sources or not meta_settings["first_order"] or meta_settings["pilot"]
        or warmup["epoch"] != warmup_source["settings"]["epochs"] or meta_weights["epoch"] != meta_settings["epochs"]
        or adaptation["inner_steps"] < 1 or not isfinite(adaptation["inner_lr"]) or adaptation["inner_lr"] <= 0
    ):
        raise ValueError("θ₀ 权重轮数、FOMAML 源族或适配预算与正式折内协议不一致")

    model_config = b0_model_config(warmup_source["settings"]["model"])
    with torch.random.fork_rng(devices=[]):
        model = build_faster_rcnn(model_config, initialize_backbone=False)
    head, base_state, changed = _checkpoint_models(
        warmup, meta_weights, model,
        four_modules=method == "supervised_warmup_and_fomaml_modules",
    )
    meta_epoch = meta_weights["epoch"]
    del warmup, meta_weights
    library = _load_fomaml_memory(
        fomaml, meta_source["fomaml_episode_ids"], meta_epoch, sources, base_state, head, model,
    )
    for episode in holdout:
        allowed = set(episode["source_available_action_ids"])
        if not sources <= allowed:
            raise ValueError(f"留出 Episode 不能借全部源产品：{episode['episode_id']}")
    provenance = {
        **theta_scope, **audit, "episode_bank_sha256": artifacts["artifacts"][bank],
        "supervised_theta0": str(supervised), "supervised_theta0_sha256": warmup_hash,
        "fomaml_theta0": str(fomaml), "fomaml_theta0_sha256": meta_hash, "fomaml_changed_tensors": changed,
        "module_memory": library["path"], "module_memory_sha256": library["sha256"],
        "module_memory_epoch": library["saved_epoch"],
        "modules_present": library["modules_present"], "modules_missing": library["modules_missing"],
        "source_episode_ids": {
            family: row["episode_id"] for family, row in sorted(library["sources"].items())
        },
        "adaptation": adaptation,
        "adaptation_loss": adaptation_loss,
        "protocol": "fomaml_modules" if method == "supervised_warmup_and_fomaml_modules" else "fomaml_head",
    }
    provenance["support_conflict_filter"]["policy"] = group_split["conflict_policy"]
    return model_config, adapter, holdout, head, base_state, library, provenance

# holdout_features.py

@dataclass(frozen=True)
class HoldoutFrame:
    image_id: str
    proposals: torch.Tensor
    features: torch.Tensor
    valid_size: tuple[int, int]
    original_size: tuple[int, int]


@dataclass(frozen=True)
class HoldoutEpisodeFeatures:
    support: list[ROIExample]
    query: list[ROIExample]
    frames: list[HoldoutFrame]
    support_scored_ids: list[str]
    query_scored_ids: list[str]
    skipped_support: int
    skipped_query: int


def _decode_views(image_paths: dict[str, str], workers: int):
    if workers == 0:
        for image_id, path in image_paths.items():
            yield image_id, image_to_tensor(path)
        return

    views = iter(image_paths.items())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        for _, (image_id, path) in zip(range(workers), views):
            pending.append((image_id, pool.submit(image_to_tensor, path)))
        while pending:
            image_id, future = pending.popleft()
            image = future.result()
            if following := next(views, None):
                next_id, path = following
                pending.append((next_id, pool.submit(image_to_tensor, path)))
            yield image_id, image


def _sample_view(roi_heads, proposals, record, role, original_size, valid_size, ignore_iou, seed):
    ignore_annotations = record.get("ignore_annotations", [])
    if not (record["normal_flag"] or record["annotations"] or ignore_annotations):
        return None

    target = target_for_model(record)
    positive = resize_boxes(target["boxes"].to(proposals.device), original_size, valid_size)
    target = {"boxes": positive, "labels": target["labels"].to(proposals.device)}
    ignore = proposals.new_tensor([ann["bbox"] for ann in ignore_annotations]).reshape(-1, 4)
    ignore = resize_boxes(ignore, original_size, valid_size)
    eligible_index = _eligible_proposal_indices(
        proposals,
        positive,
        ignore,
        roi_heads.proposal_matcher.high_threshold,
        ignore_iou,
    )
    eligible = proposals[eligible_index]

    key = f"{seed}/{record['episode_id']}/{role}/{record['image_id']}".encode("utf-8")
    sample_seed = int.from_bytes(sha256(key).digest()[:8], "little")
    cuda_devices = [proposals.device.index] if proposals.is_cuda else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(sample_seed)
        if proposals.is_cuda:
            with torch.cuda.device(proposals.device):
                torch.cuda.manual_seed(sample_seed)
        sampled, _, labels, regression_targets = roi_heads.select_training_samples([eligible], [target])
    if not labels[0].numel() or not eligible.numel():
        return None
    regression = torch.where(labels[0][:, None] > 0, regression_targets[0], 0.0)
    return eligible_index, eligible, sampled[0], labels[0], regression, positive, ignore


def _embed_boxes(roi_heads, pyramid, boxes, valid_size):
    return roi_heads.box_head(roi_heads.box_roi_pool(pyramid, [boxes], [valid_size]))


def _embed_sampled_boxes(roi_heads, pyramid, sampled, proposals, prediction_features, valid_size):
    unique, indices = torch.unique(torch.cat(sampled), dim=0, return_inverse=True)
    if prediction_features is None or not proposals.numel():
        embeddings = _embed_boxes(roi_heads, pyramid, unique, valid_size)
    else:
        equal = (unique[:, None, :] == proposals[None, :, :]).all(dim=2)
        from_proposals = equal.any(dim=1)
        embeddings = prediction_features.new_empty((unique.shape[0], prediction_features.shape[1]))
        embeddings[from_proposals] = prediction_features[equal[from_proposals].int().argmax(dim=1)]
        if not from_proposals.all():
            embeddings[~from_proposals] = _embed_boxes(
                roi_heads, pyramid, unique[~from_proposals], valid_size
            )
    return embeddings[indices].cpu().split([boxes.shape[0] for boxes in sampled])


def _extract_view(
    detector,
    image_id,
    image,
    occurrences,
    device,
    ignore_iou,
    seed,
    reference_frame: HoldoutFrame | None = None,
    retain_frame: bool = False,
):
    original_size = tuple(image.shape[-2:])
    images, _ = detector.transform([image.to(device)])
    valid_size = reference_frame.valid_size if reference_frame is not None else tuple(images.image_sizes[0])
    pyramid = detector.backbone(images.tensors)
    if reference_frame is None:
        proposals, _ = detector.rpn(images, pyramid)
        proposal = proposals[0]
    else:
        proposal = reference_frame.proposals.to(device)
    roi_heads = detector.roi_heads
    prediction_features = _embed_boxes(roi_heads, pyramid, proposal, valid_size)
    frame = (
        HoldoutFrame(
            image_id,
            proposal.cpu(),
            prediction_features.cpu(),
            valid_size,
            original_size,
        )
        if retain_frame or any(key[1] == "query" for key, _ in occurrences)
        else None
    )

    keys, sampled_boxes, labels, regressions = [], [], [], []
    eligible_indices, eligible_boxes, target_boxes, ignore_boxes = [], [], [], []
    normal_flags = []
    for key, record in occurrences:
        sample = _sample_view(
            roi_heads,
            proposal,
            record,
            key[1],
            original_size,
            valid_size,
            ignore_iou,
            seed,
        )
        if sample is None:
            continue
        eligible_index, eligible, sampled, sample_labels, regression, positive, ignore = sample
        keys.append(key)
        eligible_indices.append(eligible_index)
        eligible_boxes.append(eligible)
        sampled_boxes.append(sampled)
        labels.append(sample_labels)
        regressions.append(regression)
        target_boxes.append(positive)
        ignore_boxes.append(ignore)
        normal_flags.append(record["normal_flag"])

    if not keys:
        return frame, {}

    sampled_features = _embed_sampled_boxes(
        roi_heads,
        pyramid,
        sampled_boxes,
        proposal,
        prediction_features,
        valid_size,
    )
    examples = {}
    for key, eligible_index, eligible, sampled, features, sample_labels, regression, positive, ignore, normal_flag in zip(
        keys,
        eligible_indices,
        eligible_boxes,
        sampled_boxes,
        sampled_features,
        labels,
        regressions,
        target_boxes,
        ignore_boxes,
        normal_flags,
    ):
        examples[key] = ROIExample(
            features,
            sample_labels.cpu(),
            regression.cpu(),
            sampled.cpu(),
            prediction_features[eligible_index].cpu(),
            eligible.cpu(),
            positive.cpu(),
            ignore.cpu(),
            bool(normal_flag),
        )
    return frame, examples


def _scored_views(records, examples, episode_index, role):
    available = [
        (record["image_id"], examples[episode_index, role, slot])
        for slot, record in enumerate(records)
        if (episode_index, role, slot) in examples
    ]
    return [image_id for image_id, _ in available], [example for _, example in available]


def _tensor_bytes(entries):
    return sum(tensor.numel() * tensor.element_size() for entry in entries for tensor in entry)


@torch.no_grad()
def prepare_holdout_features(
    detector,
    episode_records,
    device,
    ignore_iou: float,
    seed: int,
    workers: int,
    *,
    reference_frames: dict[str, HoldoutFrame] | None = None,
    retain_frames: bool = False,
    return_frame_cache: bool = False,
):
    """Run one frozen detector forward per image and retain detection candidates."""
    if detector.training or detector.rpn.training or detector.transform.training:
        raise ValueError("Holdout feature preparation requires an eval detector")
    if not episode_records or not 0 < ignore_iou <= 1 or workers not in (0, 1, 2):
        raise ValueError("Holdout features require non-empty episodes and valid settings")

    paths, requests = {}, defaultdict(list)
    for index, (support, query) in enumerate(episode_records):
        for role, records in (("support", support), ("query", query)):
            for slot, record in enumerate(records):
                image_id, path = record["image_id"], record["image_path"]
                if paths.setdefault(image_id, path) != path:
                    raise ValueError(f"Image ID maps to different paths: {image_id}")
                requests[image_id].append(((index, role, slot), record))

    if reference_frames is not None and set(paths) != set(reference_frames):
        raise ValueError("Fixed proposal evaluation requires every requested image frame")

    frames, examples = {}, {}
    backbone_forwards = 0
    total_views = len(paths)
    report_every = max(1, total_views // 10)
    for image_id, image in _decode_views(paths, workers):
        frame, supervision = _extract_view(
            detector,
            image_id,
            image,
            requests[image_id],
            device,
            ignore_iou,
            seed,
            reference_frames.get(image_id) if reference_frames is not None else None,
            retain_frames,
        )
        backbone_forwards += 1
        if backbone_forwards in (1, total_views) or backbone_forwards % report_every == 0:
            print(f"特征提取 {backbone_forwards}/{total_views}", flush=True)
        if frame is not None:
            frames[image_id] = frame
        examples.update(supervision)

    episodes = []
    for index, (support, query) in enumerate(episode_records):
        support_ids, support_examples = _scored_views(support, examples, index, "support")
        query_ids, query_examples = _scored_views(query, examples, index, "query")
        episodes.append(
            HoldoutEpisodeFeatures(
                support=support_examples,
                query=query_examples,
                frames=[frames[record["image_id"]] for record in query],
                support_scored_ids=support_ids,
                query_scored_ids=query_ids,
                skipped_support=len(support) - len(support_ids),
                skipped_query=len(query) - len(query_ids),
            )
        )
    stats = {
        "unique_raw_views_requested": len(paths),
        "backbone_forwards": backbone_forwards,
        "support_occurrences": sum(len(support) for support, _ in episode_records),
        "query_occurrences": sum(len(query) for _, query in episode_records),
        "skipped_support_occurrences": sum(episode.skipped_support for episode in episodes),
        "skipped_query_occurrences": sum(episode.skipped_query for episode in episodes),
        "cached_roi_examples": len(examples),
        "cached_query_frames": len(frames),
        "cached_tensor_bytes": _tensor_bytes(
            (frame.proposals, frame.features) for frame in frames.values()
        ) + _tensor_bytes(
            (
                example.features,
                example.labels,
                example.regression_targets,
                example.sampled_boxes,
                example.proposal_features,
                example.proposal_boxes,
                example.target_boxes,
                example.ignore_boxes,
            )
            for example in examples.values()
        ),
        "prefetch_workers": workers,
    }
    if return_frame_cache:
        return episodes, stats, frames
    return episodes, stats


def summarize_proposals(frame_cache: dict[str, HoldoutFrame], query_records, iou_threshold: float = 0.5) -> dict:
    """Summarize proposal coverage on fixed Query images."""
    from torchvision.ops import box_iou

    rows = []
    seen = set()
    for query in query_records:
        for record in query:
            if record["image_id"] in seen:
                continue
            seen.add(record["image_id"])
            frame = frame_cache[record["image_id"]]
            proposals = frame.proposals
            gt = torch.tensor(
                [ann["bbox"] for ann in record["annotations"]], dtype=torch.float32
            ).reshape(-1, 4)
            gt = resize_boxes(gt, frame.original_size, frame.valid_size)
            if gt.numel() and proposals.numel():
                best_iou = box_iou(proposals, gt).max(dim=0).values
                covered = int((best_iou >= iou_threshold).sum().item())
                mean_best_iou = float(best_iou.mean().item())
            else:
                covered = 0
                mean_best_iou = None
            rows.append({
                "image_id": record["image_id"],
                "dataset": record["dataset"],
                "product_id": record["product_id"],
                "normal_flag": bool(record["normal_flag"]),
                "proposal_count": int(proposals.shape[0]),
                "gt_count": int(gt.shape[0]),
                "gt_covered": covered,
                "mean_best_iou": mean_best_iou,
            })

    grouped = defaultdict(list)
    for row in rows:
        grouped[f"{row['dataset']}:{row['product_id']}"].append(row)

    def summary(items):
        gt_count = sum(row["gt_count"] for row in items)
        valid_iou = [row["mean_best_iou"] for row in items if row["mean_best_iou"] is not None]
        return {
            "query_images": len(items),
            "mean_proposals": float(sum(row["proposal_count"] for row in items) / max(len(items), 1)),
            "gt_boxes": gt_count,
            "gt_covered": sum(row["gt_covered"] for row in items),
            "recall_at_50": float(sum(row["gt_covered"] for row in items) / max(gt_count, 1)),
            "mean_best_iou": float(sum(valid_iou) / max(len(valid_iou), 1)),
        }

    return {
        "all": summary(rows),
        "defect": summary([row for row in rows if row["gt_count"]]),
        "normal": summary([row for row in rows if row["normal_flag"]]),
        "per_product": {key: summary(items) for key, items in sorted(grouped.items())},
    }


@torch.inference_mode()
def predict_holdout_frame(detector, predictor, frame: HoldoutFrame, device) -> dict:
    """Run post-processing on one cached RPN candidate frame."""
    if not frame.proposals.numel():
        return {
            "boxes": frame.proposals,
            "scores": frame.features.new_empty(0),
            "labels": torch.empty(0, dtype=torch.int64),
        }
    class_logits, box_regression = predictor(frame.features.to(device))
    boxes, scores, labels = detector.roi_heads.postprocess_detections(
        class_logits,
        box_regression,
        [frame.proposals.to(device)],
        [frame.valid_size],
    )
    result = detector.transform.postprocess(
        [{"boxes": boxes[0], "scores": scores[0], "labels": labels[0]}],
        [frame.valid_size],
        [frame.original_size],
    )[0]
    return {key: value.cpu() for key, value in result.items()}

# selector.py

def load_selector_config() -> dict:
    config = load_config(named_config_path("selector.yaml"))
    if config["descriptor"] != "v0":
        raise ValueError("v1 描述子尚未作为主协议，当前只实现原始特征")
    if config["primary_stage"] != "after":
        raise ValueError("挑选的主终点是预注册适应步数，对应评价里的 after")
    if int(config["stage_a_epochs"]) < 1 or int(config["stage_b_epochs"]) < 1:
        raise ValueError("选择器两个阶段的轮数都必须为正")
    if float(config["eps_loss"]) <= 0 or float(config["temperature_end"]) <= 0:
        raise ValueError("损失分母下限和最终温度必须为正")
    return config


class ModuleSelector(nn.Module):
    """score(s, m) = a(z, e_s) + b_m(z, e_s)，且各块 b 的和为 0。"""

    def __init__(self, dim: int, module_names: tuple[str, ...] = INNOVATION1_MODULES, hidden: int = 128):
        super().__init__()
        if tuple(module_names) != INNOVATION1_MODULES:
            raise ValueError(f"选择器必须同时包含 {list(INNOVATION1_MODULES)}")
        self.module_names = tuple(module_names)
        self.a_net = nn.Sequential(nn.Linear(dim * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.b_nets = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(dim * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            for name in self.module_names
        })

    def _pairs(self, target: torch.Tensor, sources: torch.Tensor) -> torch.Tensor:
        if target.ndim != 1 or sources.ndim != 2 or target.shape[0] != sources.shape[1]:
            raise ValueError("描述子维度不一致")
        repeated = target.expand(sources.shape[0], -1)
        return torch.cat((repeated, sources), dim=-1)

    def centered_module_terms(self, target: torch.Tensor, sources: torch.Tensor) -> torch.Tensor:
        pair = self._pairs(target, sources)
        raw = torch.cat([self.b_nets[name](pair) for name in self.module_names], dim=-1)
        return raw - raw.mean(dim=-1, keepdim=True)

    def source_scores(self, target: torch.Tensor, sources: torch.Tensor, module: str, *,
                      use_module_term: bool) -> torch.Tensor:
        pair = self._pairs(target, sources)
        shared = self.a_net(pair).squeeze(-1)
        if not use_module_term:
            return shared
        terms = self.centered_module_terms(target, sources)
        return shared + terms[:, self.module_names.index(module)]


def hard_select(source_scores: dict[str, float]) -> str:
    """分数不高于不借时，选择不借。并列时取名称排序靠前的来源。"""
    if not source_scores:
        return "Null"
    ordered = sorted(source_scores, key=lambda name: (-source_scores[name], name))
    best = ordered[0]
    if source_scores[best] > 0:
        return best
    return "Null"


def _present(utility: float | None) -> bool:
    return utility is not None and isinstance(utility, (int, float)) and math.isfinite(utility)


def rank_null_loss(scores: torch.Tensor, utilities: list[float | None]) -> torch.Tensor:
    """只比较某个来源和不借。效用缺失的来源不进入这一项。"""
    if len(utilities) != scores.shape[0]:
        raise ValueError("效用和来源分数数量不一致")
    terms = []
    for score, utility in zip(scores, utilities):
        if not _present(utility):
            continue
        if utility > 0:
            terms.append(F.softplus(-(score - 0)))
        elif utility < 0:
            terms.append(F.softplus(-(0 - score)))
    if not terms:
        return scores.sum() * 0
    return torch.stack(terms).mean()


def relative_cost_loss(scores: torch.Tensor, utilities: list[float | None], null_loss: float,
                       temperature: float, eps_loss: float) -> torch.Tensor:
    """相对不借的期望代价。缺失效用的来源不进入分布。"""
    if temperature <= 0 or eps_loss <= 0 or not math.isfinite(null_loss):
        raise ValueError("温度、分母下限和基线损失必须有限且合法")
    kept_scores = []
    costs = []
    scale = max(null_loss, eps_loss)
    for score, utility in zip(scores, utilities):
        if not _present(utility):
            continue
        kept_scores.append(score)
        costs.append(score.new_tensor(-float(utility) / scale))
    null_score = scores.new_zeros(())
    if not kept_scores:
        return null_score * 0
    all_scores = torch.stack([null_score, *kept_scores]) / temperature
    all_costs = torch.stack([scores.new_zeros(()), *costs])
    weights = torch.softmax(all_scores, dim=0)
    return (weights * all_costs).sum()


def _optimizer(module: ModuleSelector, config: dict) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        [
            {"params": module.a_net.parameters(), "weight_decay": 0.0},
            {"params": module.b_nets.parameters(), "weight_decay": float(config["b_weight_decay"])},
        ],
        lr=float(config["learning_rate"]),
    )


def _fit(module: ModuleSelector, episodes: list[dict], config: dict, *, stage_a: bool, stage_b: bool,
         use_module_term: bool, seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    module.train()
    optimizer = _optimizer(module, config)
    order = list(range(len(episodes)))
    stages = []
    if stage_a:
        stages.append(("a", int(config["stage_a_epochs"]), float(config["temperature_start"])))
    if stage_b:
        stages.append(("b", int(config["stage_b_epochs"]), None))
    for stage, epochs, temperature in stages:
        for epoch in range(epochs):
            if stage == "b":
                span = max(epochs - 1, 1)
                start, end = float(config["temperature_start"]), float(config["temperature_end"])
                temperature = start + (end - start) * epoch / span
            random.shuffle(order)
            for index in order:
                episode = episodes[index]
                target = torch.tensor(episode["target"], dtype=torch.float32)
                sources = torch.tensor([row["source_descriptor"] for row in episode["sources"]], dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                loss = target.new_zeros(())
                for module_name in module.module_names:
                    scores = module.source_scores(target, sources, module_name, use_module_term=use_module_term)
                    utilities = [row["modules"][module_name]["utility"] for row in episode["sources"]]
                    if stage == "a":
                        loss = loss + rank_null_loss(scores, utilities)
                    else:
                        loss = loss + relative_cost_loss(
                            scores, utilities, float(episode["null_loss"]), temperature, float(config["eps_loss"]),
                        )
                loss.backward()
                optimizer.step()
    module.eval()


def _choice_rows(module: ModuleSelector, episodes: list[dict], *, use_module_term: bool) -> list[dict]:
    rows = []
    with torch.no_grad():
        for episode in episodes:
            target = torch.tensor(episode["target"], dtype=torch.float32)
            sources = torch.tensor([row["source_descriptor"] for row in episode["sources"]], dtype=torch.float32)
            for module_name in module.module_names:
                scores = module.source_scores(target, sources, module_name, use_module_term=use_module_term)
                named = {
                    row["source"]: float(score)
                    for row, score in zip(episode["sources"], scores)
                    if _present(row["modules"][module_name]["utility"])
                }
                choice = hard_select(named)
                chosen = None if choice == "Null" else next(row for row in episode["sources"] if row["source"] == choice)
                utility = 0.0 if chosen is None else float(chosen["modules"][module_name]["utility"])
                ap_delta = None if chosen is None else chosen["modules"][module_name]["ap50_delta"]
                rows.append({
                    "episode_id": episode["episode_id"],
                    "product": episode["product"],
                    "module": module_name,
                    "choice": choice,
                    "utility": utility,
                    "ap50_delta": ap_delta,
                })
    return rows


def _mean_utility(rows: list[dict]) -> float | None:
    if not rows:
        return None
    return float(sum(row["utility"] for row in rows) / len(rows))


def _new_module(dim: int, config: dict, seed: int) -> ModuleSelector:
    torch.manual_seed(seed)
    return ModuleSelector(dim, hidden=int(config["hidden"]))


def leave_one_product_out(episodes: list[dict], config: dict | None = None) -> dict:
    """每个新产品轮流留出。挑选只用其余新产品的效用学习，不看留出产品的查询结果。"""
    config = load_selector_config() if config is None else config
    products = sorted({episode["product"] for episode in episodes})
    if len(products) < 2:
        raise ValueError("至少需要两个新产品才能做留出挑选")
    dim = len(episodes[0]["target"])
    variants = {
        "rank": {"stage_a": True, "stage_b": False, "use_module_term": True},
        "cost": {"stage_a": False, "stage_b": True, "use_module_term": True},
        "rank_cost": {"stage_a": True, "stage_b": True, "use_module_term": True},
        "shared_only": {"stage_a": True, "stage_b": True, "use_module_term": False},
    }
    reports = {name: [] for name in variants}
    for held in products:
        train, held_rows = _scale_split(
            [episode for episode in episodes if episode["product"] != held],
            [episode for episode in episodes if episode["product"] == held],
        )
        for name, variant in variants.items():
            module = _new_module(dim, config, int(config["seed"]) + products.index(held))
            _fit(module, train, config, seed=int(config["seed"]) + products.index(held), **variant)
            chosen = _choice_rows(module, held_rows, use_module_term=variant["use_module_term"])
            reports[name].extend(chosen)
    full = reports["rank_cost"]
    shared = reports["shared_only"]
    full_utility = _mean_utility(full)
    shared_utility = _mean_utility(shared)
    module_gain = None if full_utility is None or shared_utility is None else full_utility - shared_utility
    claim = bool(config["g3_requires_positive_module_gain"] and module_gain is not None and module_gain > 0)
    by_product = {}
    for row in full:
        product = by_product.setdefault(row["product"], {"contributed": [], "not_selected": []})
        label = f"{row['module']}:{row['choice']}"
        if row["choice"] != "Null" and row["ap50_delta"] is not None and row["ap50_delta"] > 0:
            product["contributed"].append(label)
        else:
            product["not_selected"].append(label)
    return {
        "require_all_targets": False,
        "variants": {name: {"mean_utility": _mean_utility(rows), "choices": rows} for name, rows in reports.items()},
        "g3_module_gain": module_gain,
        "module_level_claim": claim,
        "products": by_product,
        "null_rate": (
            sum(row["choice"] == "Null" for row in full) / len(full) if full else None
        ),
    }


def oracle_source(records: list[dict], select_ids: set[str], eval_ids: set[str]) -> dict:
    """在一批查询上选平均效用最高的来源，再到另一批查询上读这个来源。"""
    if select_ids & eval_ids:
        raise ValueError("挑选来源用的查询和评价用的查询必须分开")

    def mean_for(source: str, allowed: set[str]) -> float | None:
        values = [
            row["utility"] for row in records
            if row["source"] == source and row["episode_id"] in allowed and _present(row["utility"])
        ]
        if not values:
            return None
        return float(sum(values) / len(values))

    sources = sorted({row["source"] for row in records})
    select_means = {source: mean_for(source, select_ids) for source in sources}
    usable = {source: value for source, value in select_means.items() if value is not None}
    if not usable:
        return {"oracle_source": None, "select_mean": None, "eval_mean": None}
    chosen = sorted(usable, key=lambda source: (-usable[source], source))[0]
    return {"oracle_source": chosen, "select_mean": usable[chosen], "eval_mean": mean_for(chosen, eval_ids)}


def reliability_gate(repeat_count: int) -> dict:
    """只有一次测量时，不能做方差分解，闸门不通过。"""
    if repeat_count < 2:
        return {"passed": False, "reason": "只有一次效用测量，不能判断排序是否稳定"}
    return {"passed": None, "reason": "重复测量已有，阈值仍须由预注册的零分布确定"}


def standardize(train_vectors: list[list[float]], vectors: list[list[float]]) -> list[list[float]]:
    """标准化只用参与学习的描述子，不把留出新产品的描述子算进均值。"""
    if not train_vectors:
        raise ValueError("没有可用于标准化的描述子")
    width = len(train_vectors[0])
    means = []
    scales = []
    for index in range(width):
        column = [row[index] for row in train_vectors]
        center = sum(column) / len(column)
        variance = sum((value - center) ** 2 for value in column) / len(column)
        means.append(center)
        scales.append(math.sqrt(variance) or 1.0)
    return [[(value - means[index]) / scales[index] for index, value in enumerate(row)] for row in vectors]


def _scale_split(train: list[dict], held: list[dict]) -> tuple[list[dict], list[dict]]:
    bank = []
    for episode in train:
        bank.append(episode["target"])
        bank.extend(source["source_descriptor"] for source in episode["sources"])

    def apply(episode: dict) -> dict:
        copied = deepcopy(episode)
        copied["target"] = standardize(bank, [copied["target"]])[0]
        for source in copied["sources"]:
            source["source_descriptor"] = standardize(bank, [source["source_descriptor"]])[0]
        return copied

    return [apply(episode) for episode in train], [apply(episode) for episode in held]

# holdout_evaluation.py

HEAD_PREFIX = "roi_heads.box_predictor."


def inject_predictor_delta(base_head: dict, delta: dict, prefix: str = HEAD_PREFIX) -> dict:
    """把分类或框回归补丁加到当前检测头。随后的支持集更新从这个结果继续，不回到原头。"""
    changed = {}
    for name, update in delta.items():
        if name.startswith(prefix):
            key = name[len(prefix):]
        elif name in base_head:
            key = name
        else:
            key = ""
        if key not in base_head:
            raise KeyError(f"检测头补丁越界：{name}")
        changed[key] = base_head[key] + update
    return {**base_head, **changed}


def _query_matches(prediction: dict, record: dict, ignore_iou: float, defect_id: int) -> _QueryMatches:
    defect = prediction["labels"] == defect_id
    scores = prediction["scores"][defect].numpy()
    boxes = prediction["boxes"][defect].numpy()
    if not np.isfinite(scores).all() or not np.isfinite(boxes).all():
        raise FloatingPointError(f"留出预测含非有限值：{record['image_id']}")
    order = np.argsort(-scores, kind="stable")
    boxes, scores = boxes[order], scores[order]
    pure_normal = record["split_role"] == "QUERY_NORMAL"
    if pure_normal and record["normal_flag"] != 1:
        raise AssertionError(f"正常 Query 混入缺陷图：{record['image_id']}")
    return _QueryMatches(scores, _box_iou(boxes, record["annotations"]),
                         (_box_iou(boxes, record["ignore_annotations"]) >= ignore_iou).any(axis=1), pure_normal)


def _score_lora_arm(
    detector,
    features,
    episodes,
    records,
    head_state,
    evaluation,
    adaptation,
    loss_settings,
    device,
    defect_id,
):
    predictor = detector.roi_heads.box_predictor
    predictor.load_state_dict(head_state, strict=True)
    predictor.requires_grad_(False)
    episode_metrics = {}
    product_queries = {key: defaultdict(list) for key in ("before", "after")}
    if len(episodes) != len(records) or len(episodes) != len(features):
        raise AssertionError("Episode、标注视角和特征缓存数量不一致")
    total_episodes = len(episodes)
    report_every = max(1, total_episodes // 5)
    for episode_index, (episode, (_, query), cache) in enumerate(zip(episodes, records, features), start=1):
        episode_id = episode["episode_id"]
        dataset, product = episode["dataset_id"], episode["product_id"]
        if not cache.support or not cache.query or len(cache.frames) != len(query):
            raise ValueError(f"留出 Episode 缺少可比较的监督或预测视角：{episode_id}")
        lora = LoRABoxPredictor(
            predictor, adaptation["rank"], adaptation["alpha"], lora_generator(adaptation["seed"], episode_id),
        ).to(device)
        paired = {"episode_id": episode_id, "dataset": dataset, "product_id": product,
                  "family_id": episode["family_id"], "source_available_mask": episode["source_available_mask"],
                  "support_scored_ids": cache.support_scored_ids, "query_scored_ids": cache.query_scored_ids}
        for stage in ("before", "after"):
            if stage == "after":
                optimizer = torch.optim.SGD(
                    [parameter for parameter in lora.parameters() if parameter.requires_grad],
                    lr=adaptation["inner_lr"],
                )
                for _ in range(adaptation["inner_steps"]):
                    optimizer.zero_grad(set_to_none=True)
                    loss = _mean_detection_loss(
                        lora, cache.support, detector.roi_heads.box_coder, loss_settings, device,
                    )
                    loss.backward()
                    optimizer.step()
            with torch.no_grad():
                loss = _mean_detection_loss(
                    lora, cache.query, detector.roi_heads.box_coder, loss_settings, device,
                ).item()
                matches = []
                for record, frame in zip(query, cache.frames):
                    if record["image_id"] != frame.image_id:
                        raise AssertionError(f"缓存 Query 错位：{episode_id}")
                    output = predict_holdout_frame(detector, lora, frame, device)
                    matches.append(_query_matches(output, record, evaluation["ignore_iou"], defect_id))
            product_queries[stage][dataset, product].extend(matches)
            paired[stage] = {"query_detection_loss": loss, **_score_product(matches, evaluation)}
        episode_metrics[episode_id] = paired
        if episode_index in (1, total_episodes) or episode_index % report_every == 0:
            print(f"评分 {episode_index}/{total_episodes} {episode_id}", flush=True)
    return _finish_arm(episode_metrics, product_queries, evaluation)


def _finish_arm(episode_metrics: dict, product_queries: dict, evaluation: dict) -> tuple[dict, dict]:
    summary = {}
    for stage, queries in product_queries.items():
        scored = {key: _score_product(rows, evaluation) for key, rows in sorted(queries.items())}
        datasets = sorted({key[0] for key in scored})
        summary[stage] = {"product_macro": _product_macro(list(scored.values())),
                          "per_product": {f"{dataset}:{product}": metric
                                          for (dataset, product), metric in scored.items()},
                          "per_dataset": {dataset: _product_macro([metric for (name, _), metric in scored.items() if name == dataset])
                                          for dataset in datasets},
                          "query_detection_loss_episode_mean": mean(
                              row[stage]["query_detection_loss"] for row in episode_metrics.values()
                          )}
    return {"summary": summary, "episodes": episode_metrics}, product_queries


def _score_arm(
    detector,
    features,
    episodes,
    records,
    head_state,
    evaluation,
    adaptation,
    loss_settings,
    device,
    defect_id,
):
    if adaptation.get("adapter") == "lora":
        return _score_lora_arm(
            detector, features, episodes, records, head_state, evaluation, adaptation,
            loss_settings, device, defect_id,
        )
    predictor = detector.roi_heads.box_predictor
    predictor.load_state_dict(head_state, strict=True)
    predictor.requires_grad_(True)
    initializer = MAML(predictor, lr=adaptation["inner_lr"], first_order=True)
    episode_metrics = {}
    product_queries = {key: defaultdict(list) for key in ("before", "after")}
    if len(episodes) != len(records) or len(episodes) != len(features):
        raise AssertionError("Episode、标注视角和特征缓存数量不一致")
    total_episodes = len(episodes)
    report_every = max(1, total_episodes // 5)
    for episode_index, (episode, (_, query), cache) in enumerate(zip(episodes, records, features), start=1):
        episode_id = episode["episode_id"]
        dataset, product = episode["dataset_id"], episode["product_id"]
        if not cache.support or not cache.query or len(cache.frames) != len(query):
            raise ValueError(f"留出 Episode 缺少可比较的监督或预测视角：{episode_id}")
        learner = initializer.clone()
        paired = {"episode_id": episode_id, "dataset": dataset, "product_id": product,
                  "family_id": episode["family_id"], "source_available_mask": episode["source_available_mask"],
                  "support_scored_ids": cache.support_scored_ids, "query_scored_ids": cache.query_scored_ids}
        for stage, head in (("before", predictor), ("after", learner)):
            if stage == "after":
                for _ in range(adaptation["inner_steps"]):
                    learner.adapt(
                        _mean_detection_loss(
                            learner,
                            cache.support,
                            detector.roi_heads.box_coder,
                            loss_settings,
                            device,
                        )
                    )
            with torch.no_grad():
                loss = _mean_detection_loss(
                    head,
                    cache.query,
                    detector.roi_heads.box_coder,
                    loss_settings,
                    device,
                ).item()
                matches = []
                for record, frame in zip(query, cache.frames):
                    if record["image_id"] != frame.image_id:
                        raise AssertionError(f"缓存 Query 错位：{episode_id}")
                    output = predict_holdout_frame(detector, head, frame, device)
                    matches.append(_query_matches(output, record, evaluation["ignore_iou"], defect_id))
            product_queries[stage][dataset, product].extend(matches)
            paired[stage] = {"query_detection_loss": loss, **_score_product(matches, evaluation)}
        episode_metrics[episode_id] = paired
        if episode_index in (1, total_episodes) or episode_index % report_every == 0:
            print(f"评分 {episode_index}/{total_episodes} {episode_id}", flush=True)
    return _finish_arm(episode_metrics, product_queries, evaluation)


def _compare(reference: dict, candidate: dict) -> dict:
    a, b = reference["episodes"], candidate["episodes"]
    if a.keys() != b.keys():
        raise AssertionError("配对入口的 Episode 不一致")
    per_episode = {}
    for episode_id, base in a.items():
        transfer = b[episode_id]
        same_support = base["support_scored_ids"] == transfer["support_scored_ids"]
        same_query = base["query_scored_ids"] == transfer["query_scored_ids"]
        result = {"dataset": base["dataset"], "family_id": base["family_id"],
                  "same_support_views": same_support, "same_query_views": same_query}
        for stage in ("before", "after"):
            lm, rm = base[stage], transfer[stage]
            comparable_loss = same_query and (stage == "before" or same_support)
            result[stage] = {
                "utility": (
                    lm["query_detection_loss"] - rm["query_detection_loss"]
                    if comparable_loss else None
                ),
                "ap50_delta": rm["ap50"] - lm["ap50"],
                "normal_fp_per_image_delta": rm["normal_fp_per_image"] - lm["normal_fp_per_image"],
            }
        per_episode[episode_id] = result

    summary = {}
    for stage in ("before", "after"):
        left = reference["summary"][stage]["product_macro"]
        right = candidate["summary"][stage]["product_macro"]
        utilities = [row[stage]["utility"] for row in per_episode.values() if row[stage]["utility"] is not None]
        summary[stage] = {
            "utility_episode_mean": mean(utilities) if utilities else None,
            "utility_positive_episodes": sum(u > 0 for u in utilities),
            "utility_valid_episodes": len(utilities),
            "ap50_product_macro_delta": right["ap50"] - left["ap50"],
            "normal_fp_per_image_macro_delta": right["normal_fp_per_image"] - left["normal_fp_per_image"],
        }
    summary["adaptation_ap50_difference_of_gains"] = (
        summary["after"]["ap50_product_macro_delta"] - summary["before"]["ap50_product_macro_delta"]
    )
    return {"summary": summary, "episodes": per_episode}


@torch.no_grad()
def _verify_cached_inference(detector, record, frame, device):
    cached = predict_holdout_frame(detector, detector.roi_heads.box_predictor, frame, device)
    direct = detector([image_to_tensor(record["image_path"]).to(device)])[0]
    for key in ("boxes", "scores", "labels"):
        torch.testing.assert_close(cached[key], direct[key].cpu(), atol=1e-4, rtol=1e-5)


def _write_report(path: Path, result: dict) -> None:
    with _atomic_output(path) as handle:
        handle.write((json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def _holdout_records(adapter, episodes):
    records = []
    for episode in episodes:
        support = adapter.build_episode_records(episode, roles=("support",))
        query = adapter.build_episode_records(episode, roles=("query_normal", "query_defect"))
        for stage, rows, role in (("support", support, "SUPPORT"), ("query_defect", query, "QUERY_DEFECT")):
            available = {row["physical_group_id"] for row in rows if row["split_role"] == role and row["annotations"]}
            required = {unit["physical_group_id"] for unit in episode[stage]}
            if available != required:
                raise ValueError(f"留出 Episode 有正物理组缺少有效缺陷框：{episode['episode_id']} {stage}")
        records.append((support, query))
    return records


def _shifted_state(base_state: dict, delta: dict) -> dict:
    state = dict(base_state)
    for name, update in delta.items():
        if name not in state:
            raise KeyError(f"任务记忆参数不在检测器中：{name}")
        state[name] = state[name] + update.to(dtype=state[name].dtype, device=state[name].device)
    return state


def run_holdout(supervised: Path, fomaml: Path, run_name: str,
                device: str = "cuda", decode_workers: int | None = None) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_name):
        raise ValueError("run-name 只能使用字母、数字、连字符和下划线")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请使用 MAML 环境，或显式指定 --device cpu")
    config_path = named_config_path("holdout_eval.yaml")
    config = load_config(config_path)
    workers = config["decode_workers"] if decode_workers is None else decode_workers
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    model_config, adapter, episodes, supervised_head, base_state, library, provenance = load_holdout(
        supervised, fomaml,
    )
    utility_protocol = provenance.get("protocol") == "utility_lora"
    adaptation = provenance["adaptation"]
    if adaptation["inner_steps"] < 1 or not isfinite(adaptation["inner_lr"]) or adaptation["inner_lr"] <= 0:
        raise ValueError("FOMAML 权重中的适配预算不合法")
    evaluation = adapter.config["evaluation"]
    defect_id = adapter.config["classes"]["defect"]
    print(
        f"已核对同折权重和 {len(library['sources'])} 个源产品的任务记忆，读取 {len(episodes)} 个留出 Episode 的标注",
        flush=True,
    )
    records = _holdout_records(adapter, episodes)

    code = _code_version()
    root = project_root()
    for path in ("training/meta_transfer.py", "scripts/train.py", "training/theta0.py", "training/fomaml_roi.py"):
        code["source_sha256"][path] = file_sha256(root / path)
    run_dir = root / config["output_dir"] / f"fold_{provenance['fold_id']}" / run_name
    path = run_dir / "metrics.json"
    result = {"complete": False, "provenance": provenance, "evaluation": evaluation, "code": code,
              "config_sha256": file_sha256(config_path), "device": str(device), "decode_workers": workers,
              "arms": {}, "extraction": {}, "proposal_summary": {}, "comparison": {},
              "aggregation": {"query_detection_loss": "detection-aligned per-image loss, then episode mean",
                              "ap50": "pool predictions within product, then equal product mean",
                              "normal_fp": "pure-normal Query boxes/image at configured score threshold",
                              "proposal_recall": "RPN proposal GT coverage at IoU 0.5",
                              "common_proposal_feature": "Feature branch evaluated on Null proposal boxes",
                              "matched_recall": "per-product defect recall at the reference normal-FP rate",
                              "task_memory": "last module delta of each source product in the saved epoch"}}
    query_banks = {}
    base_head = {k[len(HEAD_PREFIX):]: v for k, v in base_state.items() if k.startswith(HEAD_PREFIX)}
    shared_heads = {} if utility_protocol else {"supervised": supervised_head}
    shared_heads["null"] = base_head
    for family, source in sorted(library["sources"].items()):
        for module in ("Box", "Class"):
            shared_heads[f"{family}/{module}"] = inject_predictor_delta(base_head, source["delta"][module])
    jobs = [("shared", shared_heads)]

    reference_frames = None
    with b0_run_directory(run_dir, resume=False):
        _write_report(path, result)
        start = perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        detector = build_faster_rcnn(model_config, initialize_backbone=False).to(device).eval().requires_grad_(False)
        for branch, heads in jobs:
            if branch != "shared":
                raise AssertionError(f"当前任务记忆不改变候选或骨干：{branch}")
            state = base_state
            detector.load_state_dict(state, strict=True)
            print(f"提取 {branch} 的 RoI 特征，{len(episodes)} 个留出 Episode", flush=True)
            branch_start = perf_counter()
            features, stats, frame_cache = prepare_holdout_features(
                detector,
                records,
                device,
                evaluation["ignore_iou"],
                config["seed"],
                workers,
                retain_frames=branch == "shared",
                return_frame_cache=True,
            )
            result["extraction"][branch] = {**stats, "seconds": perf_counter() - branch_start}
            result["proposal_summary"][branch] = summarize_proposals(
                frame_cache,
                [query for _, query in records],
                evaluation["working_iou"],
            )
            if branch == "shared":
                reference_frames = frame_cache
            _verify_cached_inference(detector, records[0][1][0], features[0].frames[0], device)
            _write_report(path, result)
            for name, head in heads.items():
                arm_start = perf_counter()
                arm, queries = _score_arm(
                    detector,
                    features,
                    episodes,
                    records,
                    head,
                    evaluation,
                    adaptation,
                    provenance["adaptation_loss"],
                    device,
                    defect_id,
                )
                query_banks[name] = queries
                result["arms"][name] = {**arm, "seconds": perf_counter() - arm_start}
                print(f"评价完成 {name}，AP50={arm['summary']['after']['product_macro']['ap50']:.6f}", flush=True)
                _write_report(path, result)
            del features

        for family, source in sorted(library["sources"].items()):
            for module_name, use_reference in (("Feature", True), ("Where", False)):
                block = source["delta"].get(module_name)
                if not block or module_name not in library["modules_present"]:
                    continue
                if use_reference and reference_frames is None:
                    raise AssertionError("Feature 评价缺少 Null 候选")
                arm_name = f"{family}/{module_name}"
                detector.load_state_dict(_shifted_state(base_state, block), strict=True)
                print(f"提取 {arm_name}", flush=True)
                branch_start = perf_counter()
                features, stats, frame_cache = prepare_holdout_features(
                    detector, records, device, evaluation["ignore_iou"], config["seed"], workers,
                    reference_frames=reference_frames if use_reference else None,
                    return_frame_cache=True,
                )
                result["extraction"][arm_name] = {**stats, "seconds": perf_counter() - branch_start}
                result["proposal_summary"][arm_name] = summarize_proposals(
                    frame_cache, [query for _, query in records], evaluation["working_iou"],
                )
                arm_start = perf_counter()
                arm, queries = _score_arm(
                    detector, features, episodes, records, base_head, evaluation, adaptation,
                    provenance["adaptation_loss"], device, defect_id,
                )
                query_banks[arm_name] = queries
                result["arms"][arm_name] = {**arm, "seconds": perf_counter() - arm_start}
                print(f"评价完成 {arm_name}，AP50={arm['summary']['after']['product_macro']['ap50']:.6f}", flush=True)
                _write_report(path, result)
                del features

        if library.get("legacy_feature_delta"):
            if reference_frames is None:
                raise AssertionError("Null branch did not retain reference proposal frames")
            fixed_state = {
                **base_state,
                **{k: base_state[k] + v for k, v in library["Feature"].items()},
            }
            detector.load_state_dict(fixed_state, strict=True)
            print("提取 Feature 的固定 Null proposal 特征", flush=True)
            fixed_start = perf_counter()
            fixed_features, fixed_stats = prepare_holdout_features(
                detector,
                records,
                device,
                evaluation["ignore_iou"],
                config["seed"],
                workers,
                reference_frames=reference_frames,
            )
            result["extraction"]["Feature_common_proposal"] = {
                **fixed_stats,
                "seconds": perf_counter() - fixed_start,
            }
            fixed_arm, fixed_queries = _score_arm(
                detector,
                fixed_features,
                episodes,
                records,
                base_head,
                evaluation,
                adaptation,
                provenance["adaptation_loss"],
                device,
                defect_id,
            )
            query_banks["Feature_common_proposal"] = fixed_queries
            result["arms"]["Feature_common_proposal"] = fixed_arm
            print(
                f"评价完成 Feature_common_proposal，AP50={fixed_arm['summary']['after']['product_macro']['ap50']:.6f}",
                flush=True,
            )
            _write_report(path, result)

        arms = result["arms"]
        if "supervised" in arms:
            result["comparison"]["fomaml_vs_supervised"] = _compare(arms["supervised"], arms["null"])
            result["comparison"]["fomaml_vs_supervised"]["operating_point"] = {
                stage: operating_point_comparison(
                    query_banks["supervised"][stage], query_banks["null"][stage], evaluation,
                )
                for stage in ("before", "after")
            }
        source_names = [name for name in arms if "/" in name]
        result["comparison"]["delta_vs_null"] = {name: _compare(arms["null"], arms[name]) for name in source_names}
        for name, comparison in result["comparison"]["delta_vs_null"].items():
            comparison["operating_point"] = {
                stage: operating_point_comparison(query_banks["null"][stage], query_banks[name][stage], evaluation)
                for stage in ("before", "after")
            }
        result["conclusion"] = {
            "protocol": provenance.get("protocol", "fomaml_head"),
            "judgment": {
                "require_all_targets": False,
                "no_contribution": "source_selection",
            },
            "modules": {name: {"sources": {}} for name in library["modules_present"]},
            "missing_modules": list(library["modules_missing"]),
        }
        if "Feature_common_proposal" in arms:
            common_comparison = _compare(arms["null"], arms["Feature_common_proposal"])
            common_comparison["operating_point"] = {
                stage: operating_point_comparison(
                    query_banks["null"][stage], query_banks["Feature_common_proposal"][stage], evaluation,
                )
                for stage in ("before", "after")
            }
            result["comparison"]["common_proposal_delta_vs_null"] = {"Feature": common_comparison}
            common_feature = common_comparison["summary"]
            common_products = common_comparison["operating_point"]["after"]
            native_feature = result["comparison"]["delta_vs_null"]["Feature"]["summary"]
            result["conclusion"]["modules"]["Feature"] = {
                "native_after_ap50_delta": native_feature["after"]["ap50_product_macro_delta"],
                "common_proposal_before_ap50_delta": common_feature["before"]["ap50_product_macro_delta"],
                "common_proposal_after_utility": common_feature["after"]["utility_episode_mean"],
                "common_proposal_after_ap50_delta": common_feature["after"]["ap50_product_macro_delta"],
                "common_proposal_adaptation_ap50_delta": common_feature["adaptation_ap50_difference_of_gains"],
                "common_proposal_after_ap50_positive_products": common_products["ap50_positive_products"],
                "common_proposal_after_ap50_product_min": common_products["ap50_product_min"],
                "common_proposal_after_ap50_product_max": common_products["ap50_product_max"],
                "common_proposal_after_matched_recall_positive_products": common_products["matched_recall_positive_products"],
                "contribution": classify_contributions({
                    name: row["ap50_delta"] for name, row in common_products["products"].items()
                }),
            }
        if "Where" in result["comparison"]["delta_vs_null"]:
            where_point = result["comparison"]["delta_vs_null"]["Where"]["operating_point"]["after"]
            result["conclusion"]["modules"]["Where"] = {
                "proposal_defect_recall_at_50": result["proposal_summary"]["Where"]["defect"]["recall_at_50"],
                "after_ap50_delta": result["comparison"]["delta_vs_null"]["Where"]["summary"]["after"]["ap50_product_macro_delta"],
                "contribution": classify_contributions({
                    name: row["ap50_delta"] for name, row in where_point["products"].items()
                }),
            }
        for name, comparison in result["comparison"]["delta_vs_null"].items():
            family, module_name = name.split("/", 1)
            if module_name not in result["conclusion"]["modules"]:
                continue
            point = comparison["operating_point"]["after"]
            entry = {
                "episode_id": library["sources"][family]["episode_id"],
                "after_ap50_delta": comparison["summary"]["after"]["ap50_product_macro_delta"],
                "after_utility": comparison["summary"]["after"]["utility_episode_mean"],
                "contribution": classify_contributions({
                    product: row["ap50_delta"] for product, row in point["products"].items()
                }),
            }
            proposal_summary = result["proposal_summary"].get(name)
            if module_name == "Where" and proposal_summary:
                entry["proposal_defect_recall_at_50"] = proposal_summary["defect"]["recall_at_50"]
            result["conclusion"]["modules"][module_name]["sources"][family] = entry
        del query_banks
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            result["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
            result["peak_cuda_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20
            result["gpu"] = torch.cuda.get_device_name(device)
        result["elapsed_seconds"] = perf_counter() - start
        result["complete"] = True
        _write_report(path, result)
    print(f"配对评价完成，结果写入 {path}", flush=True)
    return result

# innovation1.py

def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _atomic_output(path) as handle:
        handle.write((json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def holdout_run_name(run_name: str, source: str) -> str:
    name = f"{run_name}-{source.replace(':', '-')}"
    if len(name) > 80:
        raise ValueError(f"评价目录名过长：{name}")
    return name


def delta_directory(fold_id: int, source: str, run_name: str) -> Path:
    return (project_root() / "output/meta_transfer/delta" / f"fold_{fold_id}"
            / source.replace(":", "-") / run_name)


def _product_families() -> dict[str, str]:
    path = project_root() / "output/splits/product_family_manifest.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            f"{row['dataset_id']}:{row['product_id']}": row["family_id"]
            for row in csv.DictReader(handle)
        }


def utility_rows_from_holdout(payload: dict, source: str, source_family: str, universe: list[str]) -> list[dict]:
    """把一次留出评价转成逐条效用。来源不在允许集合里时不写这条，也不补 0。"""
    null_episodes = payload["arms"]["null"]["episodes"]
    compared = {
        "Feature": payload["comparison"]["common_proposal_delta_vs_null"]["Feature"]["episodes"],
    }
    for module in ("Where", "Box", "Class"):
        compared[module] = payload["comparison"]["delta_vs_null"][module]["episodes"]
    if set(compared) != set(INNOVATION1_MODULES):
        raise AssertionError("效用表缺少创新点一的某一块")
    rows = []
    for episode_id, baseline in null_episodes.items():
        mask = baseline["source_available_mask"]
        if len(mask) != len(universe):
            raise ValueError(f"Episode 的来源掩码长度和冻结名单不一致：{episode_id}")
        allowed = {action for action, flag in zip(universe, mask) if action != "Null" and int(flag) == 1}
        if source_family not in allowed:
            continue
        null_loss = baseline["after"]["query_detection_loss"]
        product = f"{baseline['dataset']}:{baseline['product_id']}"
        for module, episodes in compared.items():
            measured = episodes[episode_id]["after"]
            rows.append({
                "episode_id": episode_id,
                "product": product,
                "family_id": baseline["family_id"],
                "source": source,
                "source_family": source_family,
                "module": module,
                "utility": measured["utility"],
                "ap50_delta": measured["ap50_delta"],
                "null_loss": null_loss,
            })
    return rows


def _support_paths(adapter, episode: dict) -> list[str]:
    records = adapter.build_episode_records(episode, roles=("support",))
    paths = [record["image_path"] for record in records]
    if len(paths) != len(set(paths)) or not paths:
        raise ValueError(f"Episode 的支持图无效：{episode['episode_id']}")
    return paths


def _source_paths(provenance: dict, source: str) -> list[str]:
    config = load_config(named_config_path("baseline.yaml"))
    adapter, index, _, _ = _prepare_source(config, source, provenance)
    _, sampling, _ = load_delta_sampling("source_groups")
    positions = sample_b0_epoch(
        index, seed=int(provenance["settings"]["seed"]), epoch=1, sampling=sampling,
    )
    paths = []
    for position in positions:
        record = adapter.build_b0_record(index[position])
        if record is None:
            raise ValueError(f"源产品描述子遇到无效图像：{source}")
        paths.append(record["image_path"])
    if not paths:
        raise ValueError(f"源产品没有可用于描述的图像：{source}")
    return paths


def _describe_groups(detector, groups: dict[str, list[str]], device: torch.device) -> dict[str, list[float]]:
    cache = {}
    described = {}
    total = sum(len(paths) for paths in groups.values())
    seen = 0
    for key, paths in groups.items():
        vectors = []
        for path in paths:
            if path not in cache:
                cache[path] = image_descriptor(detector, path, device)
                seen += 1
                if seen == 1 or seen == total or seen % max(1, total // 10) == 0:
                    print(f"描述子 {seen}/{total}", flush=True)
            vectors.append(cache[path])
        described[key] = average_descriptors(vectors)
    return described


def build_selector_episodes(rows: list[dict], targets: dict[str, list[float]],
                            sources: dict[str, list[float]]) -> list[dict]:
    grouped = {}
    for row in rows:
        episode = grouped.setdefault(row["episode_id"], {
            "episode_id": row["episode_id"],
            "product": row["product"],
            "null_loss": row["null_loss"],
            "sources": {},
        })
        if episode["product"] != row["product"] or episode["null_loss"] != row["null_loss"]:
            raise AssertionError(f"同一 Episode 的基线记录不一致：{row['episode_id']}")
        source = episode["sources"].setdefault(row["source"], {
            "source": row["source"],
            "source_descriptor": sources[row["source"]],
            "modules": {},
        })
        source["modules"][row["module"]] = {"utility": row["utility"], "ap50_delta": row["ap50_delta"]}
    episodes = []
    for episode in grouped.values():
        if episode["episode_id"] not in targets:
            raise KeyError(f"缺少新产品描述子：{episode['episode_id']}")
        episode["target"] = targets[episode["episode_id"]]
        episode["sources"] = list(episode["sources"].values())
        for source in episode["sources"]:
            if set(source["modules"]) != set(INNOVATION1_MODULES):
                raise AssertionError(f"来源 {source['source']} 的四块效用不完整")
        episodes.append(episode)
    return episodes


def _oracle_report(rows: list[dict]) -> dict:
    """每个新产品内部，用一半 Episode 选来源，另一半只负责读结果。"""
    by_product = {}
    for row in rows:
        if row["module"] != "Feature":
            continue
        by_product.setdefault(row["product"], []).append(row)
    report = {}
    for product, product_rows in sorted(by_product.items()):
        episode_ids = sorted({row["episode_id"] for row in product_rows})
        if len(episode_ids) < 2:
            report[product] = {"oracle_source": None, "eval_mean": None}
            continue
        split = max(1, len(episode_ids) // 2)
        report[product] = oracle_source(
            product_rows, set(episode_ids[:split]), set(episode_ids[split:]),
        )
    return report


def train_innovation1(theta0: Path, run_name: str, device: str = "cuda") -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,40}", run_name):
        raise ValueError("run-name 过长或含有非法字符")
    theta0 = theta0.resolve(strict=True)
    provenance = load_checkpoint(theta0)["provenance"]
    fold_id = int(provenance["fold_id"])
    plan = utility_task_plan(fold_id)
    runtime = torch.device(device)
    if runtime.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请使用 MAML 环境，或显式指定 --device cpu")
    _, _, inner = load_verified_research_splits()
    fold = next(row for row in inner["folds"] if row["fold_id"] == fold_id)
    universe = list(inner["source_universe_action_ids"])
    manifest_families = _product_families()
    families = {}
    for row in plan:
        family = manifest_families[row["source"]]
        if family not in fold["theta0_train_family_ids"] or family in fold["target_family_ids"]:
            raise AssertionError(f"源产品不在该折可借出的产品族里：{row['source']}")
        families[row["source"]] = family

    for row in plan:
        print(f"训练经验 {row['source']} {row['modules']}", flush=True)
        train_transfer_delta(
            theta0, row["source"], run_name, None, row["sampling_profile"], row["modules"], False,
        )
        if runtime.type == "cuda":
            torch.cuda.empty_cache()

    rows = []
    first_delta = None
    for row in plan:
        source = row["source"]
        delta_dir = delta_directory(fold_id, source, run_name)
        eval_name = holdout_run_name(run_name, source)
        holdout_path = project_root() / "output/meta_transfer/holdout" / f"fold_{fold_id}" / eval_name / "metrics.json"
        if not (holdout_path.is_file() and _read_json(holdout_path).get("complete")):
            raise RuntimeError(
                "留出评价读取 FOMAML 目录中的 module_memory.pt。"
                "请使用 scripts/train.py holdout --supervised <监督 last.pt> --fomaml <FOMAML last.pt> --run-name <名称>"
            )
        payload = _read_json(holdout_path)
        if not payload.get("complete"):
            raise RuntimeError(f"效用测量没有完成：{holdout_path}")
        rows.extend(utility_rows_from_holdout(payload, source, families[source], universe))
        first_delta = delta_dir
        if runtime.type == "cuda":
            torch.cuda.empty_cache()
    if first_delta is None or not rows:
        raise RuntimeError("没有可学习的效用记录")

    selector_config = load_selector_config()
    run_dir = project_root() / selector_config["output_dir"] / f"fold_{fold_id}" / run_name
    descriptor_path = run_dir / "descriptors.json"
    if descriptor_path.is_file():
        described = _read_json(descriptor_path)
    else:
        raise RuntimeError(
            "描述子提取需要同折监督权重和 FOMAML last.pt。请先运行 scripts/train.py holdout"
        )

    episodes = build_selector_episodes(rows, described["targets"], described["sources"])
    print(f"学习挑选，{len(episodes)} 条新产品 Episode", flush=True)
    trained = leave_one_product_out(episodes, selector_config)
    result = {
        "complete": True,
        "fold_id": fold_id,
        "sources": [row["source"] for row in plan],
        "modules": list(INNOVATION1_MODULES),
        "primary_stage": selector_config["primary_stage"],
        "reliability": reliability_gate(1),
        "oracle": _oracle_report(rows),
        "selector": trained,
        "utility_rows": len(rows),
    }
    _write_json(run_dir / "metrics.json", result)
    print(f"创新点一训练完成：{run_dir}", flush=True)
    return result

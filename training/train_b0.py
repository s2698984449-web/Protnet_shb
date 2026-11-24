"""Meta-Train 监督训练、Meta-Val Query 评价和 epoch 级续训。"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from common.io import (
    file_sha256,
    json_sha256,
    load_config,
    load_verified_episode_groups,
    load_verified_group_split,
    load_verified_research_splits,
    named_config_path,
    project_root,
    read_jsonl,
    verify_frozen_episode_artifacts,
)
from data_proc.baseline_data import (
    B0DetectionDataset, build_b0_index, configure_detection_worker, sample_b0_epoch,
)
from data_proc.detection import UnifiedDetectionAdapter
from data_proc.episodes import validate_episode_bank
from models.detector import b0_model_config, backbone_weight_manifest, build_faster_rcnn, finite_loss_sum
from training.checkpoints import (
    b0_run_directory,
    load_checkpoint,
    restore_best_checkpoint,
    save_checkpoint,
    save_run_metrics,
)
from training.evaluation import evaluate_detector, prepare_episode_queries


def _training_settings(config: dict, pilot: bool) -> dict:
    training = dict(config["training"])
    if pilot:
        overrides = dict(config["pilot"])
        overrides["sampling"] = {**training["sampling"], **overrides["sampling"]}
        training.update(overrides)
    training["model"] = b0_model_config({**config["model"], **training["model"]})
    model = training["model"]
    if not model.get("backbone_weights") and model["trainable_backbone_layers"] is not None:
        raise ValueError("随机初始化模式的 trainable_backbone_layers 须为 null")
    if float(training["grad_clip_norm"]) <= 0:
        raise ValueError("grad_clip_norm 必须为正")
    if training["num_workers"] < 0 or training.get("prefetch_factor", 2) < 1:
        raise ValueError("num_workers 不能为负，prefetch_factor 必须为正")
    if not 0 <= model["box_score_thresh"] <= config["evaluation"]["score_threshold"]:
        raise ValueError("检测器输出阈值必须低于或等于评价工作点")
    return training


def _check_runtime(settings: dict) -> torch.device:
    expected = settings["environment"].casefold()
    prefix = Path(os.environ.get("CONDA_PREFIX", ""))
    executable = Path(sys.executable).resolve()
    # 允许已激活环境，也允许直接调用该环境的 python.exe。
    env_ok = prefix.name.casefold() == expected or expected in {part.casefold() for part in executable.parts}
    if not env_ok:
        raise RuntimeError(f"请使用 conda run -n {settings['environment']} 执行 B0")
    device = torch.device(settings["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("B0 需要可用的 CUDA PyTorch，无法启动 GPU 训练")
    return device


def _pilot_families(families: set[str], groups: list[dict], count: int) -> set[str]:
    if count < 1:
        raise ValueError("Pilot 每个数据集至少选择一个产品族")
    by_dataset: dict[str, set[str]] = {}
    for row in groups:
        if row["family_id"] in families:
            by_dataset.setdefault(row["dataset_id"], set()).add(row["family_id"])
    return {family for pool in by_dataset.values() for family in sorted(pool)[:count]}


def _prepare_data(config: dict, settings: dict, pilot: bool):
    group_split = load_verified_group_split()
    outer, meta, inner = load_verified_research_splits()
    groups = load_verified_episode_groups()
    episode_hashes = verify_frozen_episode_artifacts()

    root = project_root()
    episode_config_path = root / config["episode_config"]
    if file_sha256(episode_config_path) != episode_hashes["config_sha256"]:
        raise AssertionError("B0 的 Episode 配置与冻结版本不一致")
    episode_config = load_config(episode_config_path)
    episode_dir = root / episode_config["output_dir"]
    episode_file = episode_config["artifacts"]["meta_validation_bank"]
    if episode_file not in episode_hashes["artifacts"]:
        raise AssertionError("B0 验证 Bank 不在已校验的 Episode 哈希清单内")
    episodes = read_jsonl(episode_dir / episode_file)
    if not episodes or any(episode["bank"] != "meta_val" for episode in episodes):
        raise AssertionError("B0 评价必须使用非空的 Meta-Val Bank")
    validate_episode_bank(episodes, groups, outer, meta, inner)

    families = set(meta["meta_train_family_ids"])
    if pilot:
        families = _pilot_families(families, groups, int(settings["families_per_dataset"]))
    adapter = UnifiedDetectionAdapter(config)
    index = build_b0_index(adapter, groups, families, group_split)
    validation_ids = {
        image_id
        for episode in episodes
        for role in ("reference", "support", "query_normal", "query_defect")
        for unit in episode[role]
        for image_id in unit["image_ids"]
    }
    if validation_ids.intersection(row["image_id"] for row in index):
        raise AssertionError("B0 训练索引与 Meta-Val Episode 图像重叠")

    data_config_path = root / config["data_config"]
    data_config = load_config(data_config_path)
    audit_dir = root / data_config["output_dir"]
    group_split_file = data_config["physical_group"]["artifacts"]["group_split"]
    provenance = {
        "train_family_ids": sorted(families),
        "data_config_sha256": file_sha256(data_config_path),
        "manifest_sha256": file_sha256(audit_dir / "manifest.csv"),
        "group_split_sha256": file_sha256(audit_dir / group_split_file),
        "train_index_sha256": json_sha256(index),
        "split_manifest_sha256": episode_hashes["split_artifact_hashes_sha256"],
        "episode_manifest_sha256": file_sha256(episode_dir / episode_config["artifacts"]["hashes"]),
    }
    return adapter, index, episodes, provenance


def _code_version() -> dict:
    root = project_root()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    files = (
        "common/io.py", "data_proc/baseline_data.py", "data_proc/detection.py",
        "data_proc/detection_geometry.py", "data_proc/episodes.py", "models/detector.py", "scripts/train.py",
        "training/checkpoints.py", "training/evaluation.py", "training/train_b0.py",
    )
    return {"git_commit": commit, "source_sha256": {path: file_sha256(root / path) for path in files}}


def _rng_state() -> dict:
    name, keys, position, gaussian_flag, gaussian = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (name, keys.tolist(), position, gaussian_flag, gaussian),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    name, keys, position, gaussian_flag, gaussian = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, gaussian_flag, gaussian))
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def _collate(batch):
    images, targets, skipped, image_ids = [], [], [], []
    for image, target in batch:
        if image is None:
            skipped.append(target)
        else:
            image_ids.append(target.pop("image_id_str"))
            images.append(image)
            targets.append(target)
    return images, targets, skipped, image_ids


def _epoch_positions(index: list[dict], settings: dict, epoch: int) -> list[int]:
    positions = sample_b0_epoch(index, seed=settings["seed"], epoch=epoch, sampling=settings["sampling"])
    if not positions:
        raise ValueError("B0 未抽到任何训练样本")
    selected_families = {index[position]["family_id"] for position in positions}
    if len(selected_families) != len({row["family_id"] for row in index}):
        raise ValueError("B0 本轮抽样未覆盖全部训练产品族，请检查域配额")
    return positions


def _sampling_summary(index: list[dict], positions: list[int]) -> dict:
    labels = Counter((index[i]["dataset_id"], index[i]["normal_flag"]) for i in positions)
    return {
        "images": len(positions),
        "families": len({index[i]["family_id"] for i in positions}),
        "by_dataset": {
            dataset: {"defect_views": labels[dataset, 0], "normal_views": labels[dataset, 1]}
            for dataset in sorted({dataset for dataset, _ in labels})
        },
        "sample_order_sha256": json_sha256([index[i]["image_id"] for i in positions]),
    }


def _train_loader(dataset, positions, device, settings, *, persistent: bool = False) -> DataLoader:
    workers = settings["num_workers"]
    worker_options = {
        "prefetch_factor": settings.get("prefetch_factor", 2),
        "persistent_workers": persistent,
        "worker_init_fn": configure_detection_worker,
    } if workers else {}
    return DataLoader(
        dataset, batch_size=settings["batch_size"], sampler=positions,
        num_workers=workers, collate_fn=_collate, generator=torch.Generator(),
        pin_memory=settings.get("pin_memory", False) and torch.device(device).type == "cuda",
        **worker_options,
    )


def _train_epoch(
    model, optimizer, dataset, positions, device, settings, epoch: int, global_step: int,
    *, loader: DataLoader | None = None,
) -> dict:
    if loader is None:
        loader = _train_loader(dataset, positions, device, settings)
    loader.generator.manual_seed(settings["seed"] + epoch)
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    loss_keys = ()
    losses_sum = None
    skipped_annotations = []
    used_images = 0
    steps = 0
    for images, targets, skipped, image_ids in loader:
        skipped_annotations.extend(skipped)
        if not images:
            continue
        images = [image.to(device, non_blocking=loader.pin_memory) for image in images]
        targets = [
            {key: value.to(device, non_blocking=loader.pin_memory) for key, value in target.items()}
            for target in targets
        ]
        optimizer.zero_grad(set_to_none=True)
        losses = model(images, targets)
        try:
            total = finite_loss_sum(losses)
        except FloatingPointError as error:
            raise FloatingPointError(f"{error}；图像：{image_ids}") from error
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=float(settings["grad_clip_norm"]),
        )
        optimizer.step()
        if losses_sum is None:
            loss_keys = tuple(losses)
            losses_sum = torch.zeros(len(loss_keys) + 1, device=total.device, dtype=torch.float64)
        values = torch.stack([losses[key].detach() for key in loss_keys] + [total.detach()])
        losses_sum.add_(values, alpha=len(images))
        used_images += len(images)
        steps += 1
        if (global_step + steps) % settings["log_every_steps"] == 0:
            print(f"epoch={epoch} step={global_step + steps} loss={total.detach().item():.6f}", flush=True)
    if not steps:
        raise RuntimeError("B0 本轮训练未完成任何优化步")
    return {
        "steps": steps, "images": used_images,
        "skipped_annotations": skipped_annotations,
        "loss": dict(zip((*loss_keys, "total"), (losses_sum / used_images).tolist())),
    }


def train_b0(
    run_name: str, pilot: bool, resume: bool, dry_run: bool, stop_after: int | None,
    init_profile: str = "pretrained",
) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_name):
        raise ValueError("run-name 只能使用字母、数字、连字符和下划线")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config_file = named_config_path("baseline.yaml")
    config = load_config(config_file)
    initialization = {"profile": init_profile}
    if init_profile == "pretrained":
        profile_path = named_config_path("b0_pretrained.yaml")
        profile_config = load_config(profile_path)
        profile = profile_config["model"]
        config["training"].update(profile_config.get("training", {}))
        config["evaluation"].update(profile_config.get("evaluation", {}))
        config["training"]["model"].update(profile)
        config["training"]["model"]["backbone_checkpoint"] = str(
            project_root() / profile["backbone_checkpoint"]
        )
        initialization["profile_config_sha256"] = file_sha256(profile_path)
    elif init_profile != "scratch":
        raise ValueError(f"未知初始化方案：{init_profile}")
    detector_config_path = named_config_path("b0_v3.yaml")
    settings = _training_settings(config, pilot)
    device = _check_runtime(settings)
    if not resume:
        initialization["backbone"] = backbone_weight_manifest(settings["model"])
    end_epoch = settings["epochs"] if stop_after is None else stop_after
    if not 1 <= end_epoch <= settings["epochs"]:
        raise ValueError("stop-after 必须落在已配置的训练轮数内")

    adapter, index, episodes, data = _prepare_data(config, settings, pilot)
    first_positions = _epoch_positions(index, settings, 1)
    summary = {
        "pilot": pilot, "initialization": initialization, "train_families": len(data["train_family_ids"]),
        "train_index_images": len(index), "validation_episodes": len(episodes),
        "epoch_1_sampling": _sampling_summary(index, first_positions),
        "train_loader": {key: settings.get(key) for key in (
            "batch_size", "num_workers", "prefetch_factor", "persistent_workers", "pin_memory",
        )},
        "evaluation_batch_size": config["evaluation"]["batch_size"],
        "input_geometry": {key: settings["model"][key] for key in ("resize_mode", "min_size", "max_size")
                           if key in settings["model"]},
        "rpn_matching": settings["model"]["rpn_matching"],
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if dry_run:
        return summary

    provenance = {
        **data, "mode": "pilot" if pilot else "meta_train_full", "fold_id": None,
        "seed": settings["seed"], "config_sha256": file_sha256(config_file),
        "detector_config_sha256": file_sha256(detector_config_path),
        "initialization": initialization, "settings": settings,
        "evaluation": config["evaluation"], "code": _code_version(),
        "runtime": {"torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device)},
    }
    run_dir = project_root() / settings["output_dir"] / run_name
    with b0_run_directory(run_dir, resume):
        return _run_epochs(run_dir, resume, end_epoch, settings, provenance, adapter, index, episodes, device)


def _run_epochs(run_dir, resume, end_epoch, settings, provenance, adapter, index, episodes, device) -> dict:
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    pending_best_path = run_dir / "best.pending.pt"
    state = load_checkpoint(last_path) if resume else None
    if state is not None:
        saved_initialization = state["provenance"].get("initialization", {})
        provenance["initialization"]["backbone"] = saved_initialization.get("backbone")
    if state is not None and state["provenance"] != provenance:
        changed = [key for key in provenance if provenance[key] != state["provenance"].get(key)]
        raise ValueError(f"B0 断点的运行来源不同，不能续训：{', '.join(changed)}")

    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    model = build_faster_rcnn(settings["model"], initialize_backbone=state is None).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    print(f"trainable_parameters={sum(p.numel() for p in parameters)}", flush=True)
    optimizer = torch.optim.SGD(
        parameters, lr=settings["learning_rate"], momentum=settings["momentum"],
        weight_decay=settings["weight_decay"],
    )
    milestones = settings["lr_milestones"]
    gamma = settings["lr_gamma"]

    def learning_rate_factor(completed_epochs: int) -> float:
        return gamma ** sum(completed_epochs >= milestone for milestone in milestones)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    dataset = B0DetectionDataset(adapter, index)
    # DataLoader 每轮重新读取这个列表，持续工作者无需重新启动。
    positions = []
    loader = _train_loader(
        dataset, positions, device, settings, persistent=settings.get("persistent_workers", False),
    )
    evaluation = provenance["evaluation"]
    prepared_queries = prepare_episode_queries(episodes, adapter) if evaluation.get("cache_query_records") else None
    start_epoch, global_step, best_epoch, best_ap50 = 1, 0, None, None
    history = []
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        _restore_rng(state["rng"])
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        best_epoch, best_ap50 = state["best_epoch"], state["best_ap50"]
        history = state["history"]
        restore_best_checkpoint(run_dir, best_epoch)
        save_run_metrics(run_dir, provenance, history, best_epoch, best_ap50)
        del state

    for epoch in range(start_epoch, end_epoch + 1):
        positions[:] = _epoch_positions(index, settings, epoch)
        sampling = _sampling_summary(index, positions)
        lr = optimizer.param_groups[0]["lr"]
        train_started = perf_counter()
        train_metrics = _train_epoch(
            model, optimizer, dataset, positions, device, settings, epoch, global_step, loader=loader,
        )
        train_seconds = perf_counter() - train_started
        global_step += train_metrics["steps"]
        validation_started = perf_counter()
        validation = evaluate_detector(model, episodes, adapter, device, evaluation, prepared_queries=prepared_queries)
        validation_seconds = perf_counter() - validation_started
        ap50 = validation["product_macro"]["ap50"]
        if ap50 is None:
            raise ValueError("B0 验证集没有可用于选模的正样本框")
        is_best = best_ap50 is None or ap50 > best_ap50
        if is_best:
            best_epoch, best_ap50 = epoch, ap50
        scheduler.step()
        history.append({"epoch": epoch, "global_step": global_step, "lr": lr,
                        "sampling": sampling, "training": train_metrics, "validation": validation,
                        "train_seconds": train_seconds, "validation_seconds": validation_seconds})
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "rng": _rng_state(), "epoch": epoch,
                      "global_step": global_step, "best_epoch": best_epoch, "best_ap50": best_ap50,
                      "history": history, "provenance": provenance}
        # 提交 last 后发布 best。重启时以 last 中记录的最佳轮次为准。
        if is_best:
            save_checkpoint(pending_best_path, {"model": checkpoint["model"], "epoch": epoch,
                                                "ap50": ap50, "provenance": provenance})
        save_checkpoint(last_path, checkpoint)
        if is_best:
            os.replace(pending_best_path, best_path)
        save_run_metrics(run_dir, provenance, history, best_epoch, best_ap50)
        print(f"epoch={epoch} steps={global_step} ap50={ap50:.6f} best_epoch={best_epoch} "
              f"train={train_seconds:.1f}s validation={validation_seconds:.1f}s", flush=True)

    return {"epochs_completed": len(history), "global_step": global_step,
            "best_epoch": best_epoch, "best_ap50": best_ap50}

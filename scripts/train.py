"""训练、评价和诊断。

子命令：b0、theta0、fomaml、experience-bank、holdout、diagnose-rpn、diagnose-roi、rpn-budget。
留出评价读取 FOMAML 目录里的 module_memory.pt。
元学习按任务抽样。四块参数都更新，初始化按支持集收益加权。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.b0_diagnostic import POST_NMS_BUDGETS, diagnose_roi, diagnose_rpn, sweep_rpn_budgets
from training.meta_transfer import run_holdout
from training.experience_bank import build_experience_bank
from training.theta0 import train_theta0, train_theta0_fomaml
from training.train_b0 import train_b0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    b0 = commands.add_parser("b0", help="正式检测器训练")
    b0.add_argument("--run-name", required=True)
    b0.add_argument("--init-profile", choices=("pretrained", "scratch"), default="pretrained")
    b0.add_argument("--pilot", action="store_true")
    b0_mode = b0.add_mutually_exclusive_group()
    b0_mode.add_argument("--resume", action="store_true")
    b0_mode.add_argument("--dry-run", action="store_true")
    b0.add_argument("--stop-after", type=int)

    theta0 = commands.add_parser("theta0", help="折内监督暖启动")
    theta0.add_argument("--fold-id", required=True, type=int)
    theta0.add_argument("--run-name", required=True)
    theta0.add_argument("--pilot", action="store_true")
    theta0.add_argument("--dry-run", action="store_true")

    fomaml = commands.add_parser("fomaml", help="FOMAML 底座")
    fomaml.add_argument("--theta0", type=Path, required=True)
    fomaml.add_argument("--run-name", required=True)
    fomaml.add_argument("--epochs", type=int, default=0, help="覆盖 fomaml.yaml 的轮数，0 表示用配置")
    fomaml.add_argument("--pilot", action="store_true")
    fomaml.add_argument("--dry-run", action="store_true")
    fomaml.add_argument("--resume", action="store_true", help="从该运行名下最后一次完成的外更新继续")
    fomaml.add_argument("--prescreen-k", type=int, default=0, help="Feature 和 Where 精确评分前保留的草图候选数，0 表示全量")
    fomaml.add_argument("--prescreen-audit", type=int, default=0, help="全量核对的次数上限，0 表示不核对")
    fomaml.add_argument("--prescreen-audit-every", type=int, default=10, help="可选源超过 2k 之后，每隔多少条任务抽一条全量核对")
    fomaml.add_argument("--sketch-dim", type=int, default=4096, help="Feature 和 Where 草图维度")
    fomaml.add_argument("--val-ap50", action="store_true", help="每个 epoch 结束在 Meta-Val 上比较当前权重和 θ0 的 AP50，借用关闭")
    fomaml.add_argument("--val-max-episodes", type=int, default=0, help="Meta-Val 最多使用多少条 Episode，0 表示全部")

    bank = commands.add_parser("experience-bank", help="用冻结 θ₀ 建立检索式经验库，不跑 FOMAML 训练")
    bank.add_argument("--theta0", type=Path, required=True)
    bank.add_argument("--run-name", required=True)
    bank.add_argument("--max-families", type=int, default=0, help="只取排序后的前若干个源族，0 表示当前折全部源族")
    bank.add_argument("--tasks-per-family", type=int, default=0, help="覆盖配置里的 M，0 表示使用配置")

    holdout = commands.add_parser("holdout", help="用 FOMAML 任务记忆测量各模块在留出产品上的效用")
    holdout.add_argument("--supervised", type=Path, required=True, help="同折监督暖启动")
    holdout.add_argument("--fomaml", type=Path, required=True, help="同折 FOMAML 的 last.pt，同目录需有 module_memory.pt")
    holdout.add_argument("--run-name", required=True)
    holdout.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    holdout.add_argument("--decode-workers", type=int, choices=(0, 1))

    rpn = commands.add_parser("diagnose-rpn", help="诊断 RPN 各阶段覆盖")
    rpn.add_argument("--checkpoint", type=Path, required=True)
    rpn.add_argument("--samples", type=Path, required=True)
    rpn.add_argument("--output", type=Path, required=True)

    roi = commands.add_parser("diagnose-roi", help="诊断 RoI 分类和框回归")
    roi.add_argument("--checkpoint", type=Path, required=True)
    roi.add_argument("--output", type=Path, required=True)
    roi.add_argument("--images-per-role-per-product", type=int, default=6)
    roi.add_argument("--dataset")

    budget = commands.add_parser("rpn-budget", help="对照 RPN 候选预算")
    budget.add_argument("--checkpoint", type=Path, required=True)
    budget.add_argument("--output", type=Path, required=True)
    budget.add_argument("--budgets", type=int, nargs="+", default=POST_NMS_BUDGETS)

    args = parser.parse_args()
    if args.command == "b0":
        train_b0(args.run_name, args.pilot, args.resume, args.dry_run, args.stop_after, args.init_profile)
    elif args.command == "theta0":
        train_theta0(args.fold_id, args.run_name, args.pilot, args.dry_run)
    elif args.command == "fomaml":
        train_theta0_fomaml(
            args.theta0, args.run_name, args.pilot, args.dry_run, args.prescreen_k, args.prescreen_audit,
            args.prescreen_audit_every, args.sketch_dim, args.val_ap50, args.val_max_episodes,
            args.resume, epochs=args.epochs or None,
        )
    elif args.command == "experience-bank":
        import re

        from common.io import project_root
        from training.checkpoints import b0_run_directory, load_checkpoint
        from training.retrieval_memory import load_retrieval_config

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_name):
            raise ValueError("run-name 只能使用字母、数字、连字符和下划线")
        config = load_retrieval_config()
        parent = load_checkpoint(args.theta0)
        fold_id = parent["provenance"]["fold_id"]
        del parent
        run_dir = project_root() / config["output_dir"] / f"fold_{fold_id}" / args.run_name
        with b0_run_directory(run_dir, resume=False):
            build_experience_bank(
                args.theta0, run_dir,
                max_families=args.max_families or None,
                tasks_per_family=args.tasks_per_family or None,
            )
    elif args.command == "holdout":
        run_holdout(args.supervised, args.fomaml, args.run_name, args.device, args.decode_workers)
    elif args.command == "diagnose-rpn":
        diagnose_rpn(args.checkpoint, args.samples, args.output)
    elif args.command == "diagnose-roi":
        diagnose_roi(args.checkpoint, args.output, args.images_per_role_per_product, args.dataset)
    else:
        sweep_rpn_budgets(args.checkpoint, args.output, args.budgets)


if __name__ == "__main__":
    main()

"""数据准备。子命令：audit、annotations、split、episodes、mode-episodes。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_proc.audit import main as audit_main
from data_proc.detection import audit_annotation_pilot
from data_proc.episodes import main as episodes_main
from data_proc.split_groups import main as split_main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="审计数据与物理组")
    audit.add_argument("--verify-images", action="store_true", help="抽查跨 OK/NG 配对的单视角图像")
    commands.add_parser("annotations", help="核对标注并把 Mask 转成框")
    commands.add_parser("split", help="生成外层留出和内层折")
    commands.add_parser("episodes", help="生成固定 Episode")
    commands.add_parser("mode-episodes", help="生成缺陷模式感知 Episode 清单")
    args = parser.parse_args()
    if args.command == "audit":
        audit_main(verify_images_override=args.verify_images)
    elif args.command == "annotations":
        audit_annotation_pilot()
    elif args.command == "split":
        split_main()
    elif args.command == "mode-episodes":
        from data_proc.mode_episodes import write_mode_manifest

        report = write_mode_manifest()
        print(
            f"模式 Episode {report['episodes']} 条，产品族 {report['families']}，"
            f"缺陷模式 {report['defect_modes']}，降级 {report['degraded']}",
            flush=True,
        )
    else:
        episodes_main()


if __name__ == "__main__":
    main()

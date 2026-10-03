from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .config import AppConfig, atomic_json


def _config(path: str, context_size: int = 0) -> AppConfig:
    cfg = AppConfig.load(path)
    if context_size:
        cfg.model.text_context_size = int(context_size)
        cfg.validate()
    return cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="speech-adapter",
        description="Local/Kaggle runner for the causal Qwen-to-Mimi speech adapter",
    )
    parser.add_argument("--config", required=True, help="resolved notebook JSON configuration")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="validate configured hardware and data/model inputs")
    manifest = commands.add_parser("prepare-manifest", help="validate shards and build manifest")
    manifest.add_argument("--force", action="store_true")
    commands.add_parser("prepare-features", help="prepare indexed or recomputed Qwen features")
    commands.add_parser("structural-checks", help="run causal and CB0-gradient tests")
    commands.add_parser("local-smoke", help="run one bounded CPU forward/backward/cache check")
    commands.add_parser("codec-contract", help="verify Mimi encoder/decoder codebook compatibility")
    commands.add_parser("gate-a", help="run codec oracle diagnostics")
    commands.add_parser("gate-b", help="render alignment review panels")
    approve = commands.add_parser("approve-gate", help="record a manual gate review")
    approve.add_argument("--gate", required=True, choices=list("ABCDEF"))
    approve.add_argument("--manual-pass", required=True, choices=("yes", "no"))
    approve.add_argument("--reviewed", type=int, default=0)
    approve.add_argument("--passed", type=int, default=0)
    approve.add_argument("--note", default="")
    train = commands.add_parser("train", help="run DDP training or a qualification experiment")
    train.add_argument("--experiment", default="full")
    train.add_argument("--limit", type=int, default=0)
    train.add_argument("--epochs", type=int, default=0)
    train.add_argument("--context-size", type=int, default=0)
    train.add_argument("--init-weights", default="")
    evaluate = commands.add_parser("evaluate", help="generate four-mode audio and metrics")
    evaluate.add_argument("--limit", type=int, default=20)
    export = commands.add_parser("export", help="export and verify incremental ONNX graphs")
    export.add_argument("--parity-steps", type=int, default=100)
    export.add_argument("--context-size", type=int, default=0)
    export.add_argument("--checkpoint", default="")
    export.add_argument("--output-dir", default="")
    context = commands.add_parser("compare-context", help="select K from completed context experiments")
    context.add_argument("--small", default="E_k3")
    context.add_argument("--large", default="E_k8")
    context.add_argument("--small-k", type=int, default=3)
    context.add_argument("--large-k", type=int, default=8)
    device = commands.add_parser("import-device-report", help="validate an Android benchmark JSON")
    device.add_argument("--path", required=True)
    commands.add_parser("status", help="show gate and run status")
    package = commands.add_parser("package", help="zip final run artifacts")
    package.add_argument("--name", default="streaming_speech_artifacts")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = _config(args.config, getattr(args, "context_size", 0))
    if args.command == "doctor":
        from .diagnostics import preflight

        preflight(cfg, require_two_gpus=cfg.runtime.require_two_gpus)
    elif args.command == "prepare-manifest":
        from .data import build_manifest

        print(json.dumps(build_manifest(cfg, force=args.force), indent=2))
    elif args.command == "prepare-features":
        from .data import prepare_qwen_features

        prepare_qwen_features(cfg)
    elif args.command == "structural-checks":
        from .diagnostics import model_structural_checks

        report = model_structural_checks(cfg)
        print(json.dumps(report, indent=2))
        if not report["automatic_pass"]:
            raise SystemExit(2)
    elif args.command == "local-smoke":
        from .smoke import run_local_smoke

        report = run_local_smoke(cfg)
        print(json.dumps(report, indent=2))
        if not report["automatic_pass"]:
            raise SystemExit(2)
    elif args.command == "codec-contract":
        from .diagnostics import inspect_mimi_contract

        report = inspect_mimi_contract(cfg)
        print(json.dumps(report, indent=2))
        if not report["automatic_pass"]:
            raise SystemExit(2)
    elif args.command == "gate-a":
        from .diagnostics import run_gate_a, update_gate

        report = run_gate_a(cfg)
        update_gate(cfg, "A", report["automatic_pass"], None, note="review gates/A_codec")
        print(json.dumps(report, indent=2))
    elif args.command == "gate-b":
        from .diagnostics import run_gate_b, update_gate

        report = run_gate_b(cfg)
        update_gate(cfg, "B", report["automatic_pass"], None, note="review gates/B_alignment/index.html")
        print(json.dumps(report, indent=2))
    elif args.command == "approve-gate":
        from .diagnostics import update_gate

        path = cfg.work_dir / "gates" / "gate_report.json"
        if not path.is_file():
            raise FileNotFoundError("run the automatic gate before approving it")
        old = json.loads(path.read_text()).get("gates", {}).get(args.gate)
        if not old:
            raise RuntimeError(f"Gate {args.gate} has no automatic result")
        report = update_gate(
            cfg,
            args.gate,
            bool(old["automatic_pass"]),
            args.manual_pass == "yes",
            args.reviewed,
            args.passed,
            args.note,
        )
        print(json.dumps(report["gates"][args.gate], indent=2))
    elif args.command == "train":
        from .train import run_training

        result = run_training(
            cfg,
            limit=args.limit,
            epochs=args.epochs or None,
            experiment=args.experiment,
            init_weights=args.init_weights,
        )
        if result:
            print(json.dumps(result, indent=2))
    elif args.command == "evaluate":
        from .evaluation import generate_artifacts

        result = generate_artifacts(cfg, args.limit)
        print(json.dumps({key: value["rollout_cb0_ce"] for key, value in result.items()}, indent=2))
    elif args.command == "export":
        from .export import export_adapter

        print(json.dumps(export_adapter(
            cfg, args.parity_steps, args.checkpoint, args.output_dir
        ), indent=2))
    elif args.command == "compare-context":
        from .diagnostics import update_gate

        small_path = cfg.work_dir / "experiments" / args.small / "gate_metrics.json"
        large_path = cfg.work_dir / "experiments" / args.large / "gate_metrics.json"
        small = json.loads(small_path.read_text())
        large = json.loads(large_path.read_text())
        small_ce = float(small["metrics"]["rollout_cb0_ce"])
        large_ce = float(large["metrics"]["rollout_cb0_ce"])
        improvement = (small_ce - large_ce) / max(small_ce, 1e-12)
        selected = args.large_k if improvement >= 0.01 else args.small_k
        report = {
            "small_experiment": args.small,
            "large_experiment": args.large,
            "small_cb0_ce": small_ce,
            "large_cb0_ce": large_ce,
            "relative_improvement": improvement,
            "selected_context_size": selected,
            "test_larger_context": bool(improvement >= 0.01),
        }
        atomic_json(cfg.work_dir / "gates" / "context_selection.json", report)
        chosen_report = large if selected == args.large_k else small
        update_gate(
            cfg,
            "E",
            bool(chosen_report["automatic_pass"]),
            None,
            note=f"selected context experiment: {args.large if selected == args.large_k else args.small}",
        )
        print(json.dumps(report, indent=2))
    elif args.command == "import-device-report":
        source = json.loads(Path(args.path).read_text())
        required = (
            "p50_ms", "p95_ms", "worst_ms", "first_audio_ms", "int8_weight_bytes",
            "working_memory_bytes", "cache_bytes", "continuous_minutes", "thermal_throttling",
        )
        missing = [key for key in required if key not in source]
        if missing:
            raise ValueError(f"device report is missing fields: {missing}")
        checks = {
            "p50": float(source["p50_ms"]) < cfg.gates.max_device_p50_ms,
            "p95": float(source["p95_ms"]) < cfg.gates.max_device_p95_ms,
            "worst": float(source["worst_ms"]) <= cfg.gates.max_device_worst_ms,
            "first_audio": float(source["first_audio_ms"]) <= cfg.gates.max_first_audio_ms,
            "weights": int(source["int8_weight_bytes"]) <= cfg.gates.max_int8_weight_bytes,
            "working_memory": int(source["working_memory_bytes"]) <= cfg.gates.max_working_memory_bytes,
            "cache": int(source["cache_bytes"]) < cfg.gates.max_cache_bytes,
            "thermal": float(source["continuous_minutes"]) >= 5.0 and not bool(source["thermal_throttling"]),
        }
        report = {**source, "checks": checks, "automatic_pass": all(checks.values())}
        atomic_json(cfg.work_dir / "gates" / "device_report.json", report)
        print(json.dumps(report, indent=2))
        if not report["automatic_pass"]:
            raise SystemExit(2)
    elif args.command == "status":
        gate_path = cfg.work_dir / "gates" / "gate_report.json"
        training_path = cfg.work_dir / "training.json"
        print("gate report:")
        print(gate_path.read_text() if gate_path.is_file() else "not created")
        print("training:")
        print(training_path.read_text() if training_path.is_file() else "not started")
    elif args.command == "package":
        destination = shutil.make_archive(
            str(cfg.work_dir.parent / args.name), "zip", root_dir=cfg.work_dir
        )
        print(destination)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="v2 offline Qwen/Mimi two-stage TTS")
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke", help="dummy learning/roundtrip test without model downloads")
    smoke.add_argument("--output-dir", default="runs/v2-smoke")
    smoke.add_argument("--steps", type=int, default=160)
    dummy = commands.add_parser("make-dummy", help="write tiny NPZ fixtures and config")
    dummy.add_argument("--output-dir", default="runs/v2-dummy")
    emb = commands.add_parser("prepare-embeddings", help="extract input-table rows from a local Qwen checkpoint")
    emb.add_argument("--model-dir", required=True)
    emb.add_argument("--token-ids", required=True)
    emb.add_argument("--output-dir", required=True)
    for name in ("train", "evaluate", "inspect"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        if name in {"train", "evaluate"}:
            command.add_argument("--limit", type=int, default=0)
        if name == "train":
            command.add_argument("--phase", choices=["cb0", "acoustic"], required=True)
            command.add_argument("--init", default="")
            command.add_argument("--resume", default="")
        elif name == "evaluate":
            command.add_argument("--checkpoint", required=True)
            command.add_argument("--output", default="")
    infer = commands.add_parser("infer", help="generate eight-codebook predictions, optionally decode real Mimi")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--features", required=True, help="NPZ hidden + embeddings (or token_ids/cache), no target audio needed")
    infer.add_argument("--output-dir", required=True)
    infer.add_argument("--embedding-cache", default="")
    infer.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    infer.add_argument("--max-frames", type=int, default=0)
    infer.add_argument("--min-frames", type=int, default=0)
    infer.add_argument("--stop-threshold", type=float, default=None)
    decode = infer.add_mutually_exclusive_group()
    decode.add_argument("--mimi-weights", default="")
    decode.add_argument("--mimi-decoder-onnx", default="")
    args = parser.parse_args(argv)
    if args.command == "smoke":
        from .smoke import run_smoke
        result = run_smoke(args.output_dir, args.steps)
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise SystemExit(2)
        return
    if args.command == "make-dummy":
        from .smoke import write_dummy_dataset
        print(json.dumps(write_dummy_dataset(args.output_dir).to_dict(), indent=2))
        return
    if args.command == "prepare-embeddings":
        from .embeddings import prepare_embedding_cache
        print(json.dumps(prepare_embedding_cache(args.model_dir, args.token_ids, args.output_dir), indent=2))
        return
    if args.command == "infer":
        import numpy as np
        import torch
        from .checkpoint import load_checkpoint
        from .data import load_inference_features, move_batch
        model, cfg, payload = load_checkpoint(args.checkpoint, args.device)
        if payload["phase"] != "acoustic":
            raise ValueError("full TTS inference requires a Stage B acoustic checkpoint")
        if args.device == "cpu":
            torch.set_num_threads(cfg.train.cpu_threads)
        inputs = move_batch(load_inference_features(args.features, cfg, args.embedding_cache), torch.device(args.device))
        result = model.generate(inputs["hidden"], inputs["embeddings"], inputs["token_mask"],
            args.max_frames or cfg.train.max_generate_frames,
            args.min_frames or cfg.train.min_generate_frames,
            cfg.train.stop_threshold if args.stop_threshold is None else args.stop_threshold)
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "codes.npy", result.codes[0].cpu().numpy())
        report = {"shape": list(result.codes.shape), "termination_reason": result.termination_reason,
                  "nominal_duration_s": result.codes.shape[-1] / 12.5, "waveform_written": False}
        if args.mimi_weights or args.mimi_decoder_onnx:
            from .codec import MoshiDecoder, OnnxDecoder, write_wav
            decoder = MoshiDecoder(args.mimi_weights, args.device) if args.mimi_weights else OnnxDecoder(args.mimi_decoder_onnx)
            audio = decoder.decode(result.codes)
            write_wav(out / "speech.wav", audio)
            report.update(waveform_written=True, sample_rate=24000, samples=int(audio.size))
        (out / "inference.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return
    from .config import V2Config
    cfg = V2Config.load(args.config)
    if args.command == "train":
        from .train import run_training
        if args.limit < 0:
            raise ValueError("limit must be nonnegative")
        run_training(cfg, args.phase, args.init, args.resume, args.limit)
    elif args.command == "inspect":
        from .data import make_dataset, collate
        from .model import TwoStageTTS
        batch = collate([make_dataset(cfg, "train", 1)[0]])
        model = TwoStageTTS(cfg.model)
        print(json.dumps({"shapes": {k: list(v.shape) for k, v in batch.items() if hasattr(v, "shape")},
            "parameters": sum(p.numel() for p in model.parameters()), "mode": "offline_two_stage"}, indent=2))
    else:
        from .checkpoint import load_checkpoint
        from .data import make_dataset
        from .train import evaluate, select_device
        device = select_device(cfg)
        model, old, _ = load_checkpoint(args.checkpoint, device)
        if old.to_dict()["model"] != cfg.to_dict()["model"] or old.to_dict()["data"] != cfg.to_dict()["data"]:
            raise ValueError("evaluation config and checkpoint model/data provenance differ")
        report = evaluate(model, make_dataset(cfg, "val", args.limit), cfg, device, args.limit or None)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

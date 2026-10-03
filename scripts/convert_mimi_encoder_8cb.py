#!/usr/bin/env python3
"""Prune a fixed-32-codebook Mimi ONNX encoder to eight codebooks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import checker, helper


def _prune_to_outputs(model: Any) -> None:
    required = {output.name for output in model.graph.output}
    kept = []
    for node in reversed(model.graph.node):
        if any(output in required for output in node.output):
            kept.append(node)
            required.update(name for name in node.input if name)
    kept.reverse()
    del model.graph.node[:]
    model.graph.node.extend(kept)
    used = {name for node in kept for name in node.input}
    initializers = [value for value in model.graph.initializer if value.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    del model.graph.value_info[:]


def convert(source: Path, output: Path, codebooks: int = 8) -> dict[str, Any]:
    if codebooks < 2:
        raise ValueError("Mimi requires the semantic book plus at least one acoustic book")
    model = onnx.load(str(source))
    graph = model.graph
    if len(graph.output) != 1:
        raise RuntimeError("expected a one-output Mimi encoder")
    producers = {name: node for node in graph.node for name in node.output}
    output_value = graph.output[0]
    output_node = producers.get(output_value.name)
    if output_node is None or output_node.op_type != "Transpose":
        raise RuntimeError("could not locate Mimi encoder output transpose")
    combined = producers.get(output_node.input[0])
    if combined is None or combined.op_type != "Concat" or len(combined.input) != 2:
        raise RuntimeError("could not locate semantic/acoustic code concat")
    acoustic = next(
        (
            producers.get(name) for name in combined.input
            if producers.get(name) is not None
            and producers[name].op_type == "Concat"
            and len(producers[name].input) == 31
        ),
        None,
    )
    if acoustic is None:
        raise RuntimeError("could not locate 31-book acoustic encoder concat")
    kept_acoustic = codebooks - 1
    inputs = list(acoustic.input[:kept_acoustic])
    if len(inputs) != kept_acoustic:
        raise RuntimeError("source graph has fewer acoustic quantizers than requested")
    del acoustic.input[:]
    acoustic.input.extend(inputs)
    dims = output_value.type.tensor_type.shape.dim
    if len(dims) < 2 or dims[1].dim_value != 32:
        raise RuntimeError("source encoder output is not fixed to 32 codebooks")
    dims[1].ClearField("dim_param")
    dims[1].dim_value = codebooks
    _prune_to_outputs(model)
    helper.set_model_props(
        model,
        {
            **{item.key: item.value for item in model.metadata_props},
            "mimi_num_codebooks": str(codebooks),
            "mimi_conversion": "pruned fixed-32 residual encoder branches",
            "mimi_source": str(source),
        },
    )
    checker.check_model(model, full_check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output))
    return {
        "source": str(source),
        "output": str(output),
        "source_bytes": source.stat().st_size,
        "output_bytes": output.stat().st_size,
        "codebooks": codebooks,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
    }


def validate(source: Path, converted: Path, codebooks: int, samples: int) -> dict[str, Any]:
    import onnxruntime as ort

    rng = np.random.default_rng(42)
    audio = rng.standard_normal((1, 1, samples), dtype=np.float32) * 0.03
    original = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    reduced = ort.InferenceSession(str(converted), providers=["CPUExecutionProvider"])
    original_codes = np.asarray(
        original.run(None, {original.get_inputs()[0].name: audio})[0]
    )
    reduced_codes = np.asarray(reduced.run(None, {reduced.get_inputs()[0].name: audio})[0])
    expected = original_codes[:, :codebooks]
    exact = reduced_codes.shape == expected.shape and np.array_equal(reduced_codes, expected)
    if not exact:
        raise RuntimeError("pruned encoder does not exactly match the first source codebooks")
    return {
        "automatic_pass": True,
        "converted_input": [(item.name, item.shape, item.type) for item in reduced.get_inputs()],
        "converted_output": [(item.name, item.shape, item.type) for item in reduced.get_outputs()],
        "audio_samples": samples,
        "code_frames": int(reduced_codes.shape[-1]),
        "first_8_exact_match": exact,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codebooks", type=int, default=8)
    parser.add_argument("--test-samples", type=int, default=3840)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    conversion = convert(args.source, args.output, args.codebooks)
    validation = validate(args.source, args.output, args.codebooks, args.test_samples)
    report = {"conversion": conversion, "validation": validation}
    report_path = args.report or args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

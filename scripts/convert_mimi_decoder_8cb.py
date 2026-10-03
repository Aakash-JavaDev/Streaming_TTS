#!/usr/bin/env python3
"""Convert a fixed-32-codebook Mimi ONNX decoder into a true 8-codebook graph.

The traced 32-book graph contains one embedding branch per residual quantizer.
This tool retains the semantic quantizer plus the first seven acoustic
quantizers, rewires the acoustic sum into the shared decoder, and prunes the
remaining 24 branches. It never pads absent codebooks with token ID zero.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper


def _consumers(graph: Any) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = defaultdict(list)
    for node in graph.node:
        for name in node.input:
            result[name].append(node)
    return result


def _producer(graph: Any) -> dict[str, Any]:
    return {name: node for node in graph.node for name in node.output}


def _descendant(
    start: str, consumers: dict[str, list[Any]], op_type: str, maximum_depth: int = 6
) -> Any:
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        value, depth = queue.popleft()
        if depth > maximum_depth:
            continue
        for node in consumers.get(value, []):
            if node.op_type == op_type:
                return node
            for output in node.output:
                if output not in visited:
                    visited.add(output)
                    queue.append((output, depth + 1))
    raise RuntimeError(f"could not find {op_type} below {start!r}")


def _replace_input(node: Any, old: str, new: str) -> None:
    replaced = False
    for index, value in enumerate(node.input):
        if value == old:
            node.input[index] = new
            replaced = True
    if not replaced:
        raise RuntimeError(f"{node.name or node.op_type} does not consume {old!r}")


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
    kept_initializers = [value for value in model.graph.initializer if value.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)
    # The source graph contains inferred intermediate shapes with a fixed
    # 31-book acoustic axis. Let the checker infer those again after pruning.
    del model.graph.value_info[:]


def convert(source: Path, output: Path, codebooks: int = 8) -> dict[str, Any]:
    if codebooks < 2:
        raise ValueError("Mimi requires the semantic book plus at least one acoustic book")
    model = onnx.load(str(source))
    graph = model.graph
    if len(graph.input) != 1:
        raise RuntimeError("expected a one-input Mimi decoder")
    input_value = graph.input[0]
    dims = input_value.type.tensor_type.shape.dim
    fixed_books = dims[1].dim_value if len(dims) > 1 and dims[1].HasField("dim_value") else None
    if fixed_books != 32:
        raise RuntimeError(f"expected fixed 32-book input, found {fixed_books!r}")

    consumers = _consumers(graph)
    producers = _producer(graph)
    direct_slices = [node for node in consumers[input_value.name] if node.op_type == "Slice"]
    candidates = []
    for node in direct_slices:
        try:
            split = _descendant(node.output[0], consumers, "Split", maximum_depth=3)
        except RuntimeError:
            continue
        if len(split.output) == 31:
            candidates.append((node, split))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one 31-way acoustic quantizer split, found {len(candidates)}")
    acoustic_slice, split = candidates[0]

    kept_acoustic = codebooks - 1
    kept_split_outputs = list(split.output[:kept_acoustic])
    if len(kept_split_outputs) != kept_acoustic:
        raise RuntimeError("source graph has fewer acoustic quantizers than requested")

    ends_name = "mimi8_acoustic_slice_ends"
    split_name = "mimi8_acoustic_split_sizes"
    graph.initializer.extend(
        [
            numpy_helper.from_array(np.asarray([codebooks], dtype=np.int64), ends_name),
            numpy_helper.from_array(np.ones(kept_acoustic, dtype=np.int64), split_name),
        ]
    )
    acoustic_slice.input[2] = ends_name
    if len(split.input) >= 2:
        split.input[1] = split_name
    else:
        split.input.append(split_name)
    del split.output[:]
    split.output.extend(kept_split_outputs)
    for attribute in split.attribute:
        if attribute.name == "num_outputs":
            attribute.i = kept_acoustic

    last_transpose = _descendant(
        kept_split_outputs[-1], _consumers(graph), "Transpose", maximum_depth=4
    )
    last_sum = _descendant(
        last_transpose.output[0], _consumers(graph), "Add", maximum_depth=1
    )
    acoustic_projection = next(
        (
            node for node in graph.node
            if node.op_type == "Conv" and "output_proj_1" in node.name
        ),
        None,
    )
    if acoustic_projection is None:
        raise RuntimeError("could not locate the acoustic quantizer output projection")
    acoustic_projection.input[0] = last_sum.output[0]

    dims[1].ClearField("dim_param")
    dims[1].dim_value = codebooks
    _prune_to_outputs(model)
    helper.set_model_props(
        model,
        {
            **{item.key: item.value for item in model.metadata_props},
            "mimi_num_codebooks": str(codebooks),
            "mimi_conversion": "pruned fixed-32 residual quantizer branches",
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


def validate(source: Path, converted: Path, codebooks: int, frames: int) -> dict[str, Any]:
    import onnxruntime as ort

    converted_session = ort.InferenceSession(str(converted), providers=["CPUExecutionProvider"])
    converted_input = converted_session.get_inputs()[0]
    if converted_input.shape[1] != codebooks:
        raise RuntimeError(f"converted graph still expects {converted_input.shape[1]} books")
    rng = np.random.default_rng(42)
    codes = rng.integers(0, 2048, size=(1, codebooks, frames), dtype=np.int64)
    correct = np.asarray(
        converted_session.run(None, {converted_input.name: codes})[0], dtype=np.float32
    )
    if not correct.size or not np.isfinite(correct).all():
        raise RuntimeError("converted decoder returned empty or non-finite audio")

    source_session = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    source_input = source_session.get_inputs()[0]
    padded = np.pad(codes, ((0, 0), (0, 32 - codebooks), (0, 0)))
    wrong = np.asarray(source_session.run(None, {source_input.name: padded})[0], dtype=np.float32)
    difference = correct.reshape(-1) - wrong.reshape(-1)
    rms_correct = float(np.sqrt(np.mean(correct.astype(np.float64) ** 2)))
    rms_difference = float(np.sqrt(np.mean(difference.astype(np.float64) ** 2)))
    return {
        "automatic_pass": True,
        "converted_input": [(item.name, item.shape, item.type) for item in converted_session.get_inputs()],
        "converted_output": [(item.name, item.shape, item.type) for item in converted_session.get_outputs()],
        "test_frames": frames,
        "audio_samples": int(correct.size),
        "correct_8cb_rms": rms_correct,
        "difference_from_wrong_zero_padding_rms": rms_difference,
        "difference_relative_to_correct_rms": rms_difference / max(rms_correct, 1e-12),
        "zero_padding_is_equivalent": bool(np.allclose(correct, wrong, atol=1e-5, rtol=1e-4)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codebooks", type=int, default=8)
    parser.add_argument("--test-frames", type=int, default=2)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    conversion = convert(args.source, args.output, args.codebooks)
    validation = validate(args.source, args.output, args.codebooks, args.test_frames)
    report = {"conversion": conversion, "validation": validation}
    report_path = args.report or args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

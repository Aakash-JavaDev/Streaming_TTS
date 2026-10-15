# Streaming_TTS — v2 Qwen-conditioned Mimi TTS

The `v2` branch implements an **offline, two-stage, single-speaker TTS POC**:

1. Frozen Qwen token embeddings and cached contextual states -> gated text conditioner.
2. A small temporal Transformer generates **CB0 autoregressively** and predicts when to stop.
3. A separate Transformer predicts **CB1-CB7 in parallel**, conditioned on complete CB0 and text.
4. Assemble `[B,8,T]` and decode with the matching **frozen Mimi** checkpoint.

There is no sequential CB0->CB7 depth decoder in v2. Stage B freezes the conditioner
and CB0 generator. LLM weights and Mimi are never optimized. The implementation is
offline: bidirectional acoustic completion requires the entire CB0 sequence.
The repository name does not imply that this version supports incremental text input.
The original streaming/duration-owner pipeline is preserved under the original
`python -m speech_adapter ...` commands. **Use `python -m speech_adapter.v2 ...` for v2.**

## Install and run without downloads or real speech data

Install a PyTorch wheel appropriate for your machine. For CPU:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
bash scripts/run_v2_smoke.sh
```

The smoke command trains two tiny dummy stages and tests free-running code generation
plus checkpoint reload parity. It does **not** download Qwen/Mimi, synthesize real
speech, or establish perceptual quality. Default architecture/Qwen-width forward
shapes are tested separately with synthetic tensors.

## Your existing prepared Qwen/Mimi dataset

`configs/v2_prepared_features.yaml` reuses the existing repository's mmap dataset
through `data.legacy_config`. Point it at your actual legacy JSON config, with its
original `paths.work_dir`, `feature_mode`, seed, and split settings. The existing
feature fingerprints must match. Do not change them just to bypass validation.

If features have not been prepared, first copy/edit
`configs/legacy_dataset.example.json` as `configs/legacy_dataset.json`, then use:

```bash
python -m speech_adapter --config configs/legacy_dataset.json prepare-manifest
python -m speech_adapter --config configs/legacy_dataset.json prepare-features
python -m speech_adapter.v2 inspect --config configs/v2_prepared_features.yaml
```

`legacy_answer` reuses stored hidden states and token IDs. Its Qwen input embeddings
are extracted from local safetensors; it does not run a full Qwen forward pass.
The v2 model uses no oracle token-frame owners or duration schedule at inference.
Legacy feature preparation still validates the dataset's existing timestamps.

## Alternative: per-record NPZ dataset

Set `data.manifest` in `configs/v2_qwen15b.yaml`. Example JSONL lines (paths relative
to the manifest; train/val IDs and source recordings must be disjoint):

```json
{"record_id":"train-001","split":"train","features":"train-001.npz","text":"I enjoy playing football."}
{"record_id":"val-001","split":"val","features":"val-001.npz","text":"A separate recording."}
```

Each NPZ contains:

| Array | Shape | Type |
|---|---|---|
| `hidden` | `[L,1536]` | floating point cached Qwen states |
| `embeddings` | `[L,1536]` | frozen Qwen input embeddings |
| `codes` | `[8,T]` | integer Mimi IDs in `[0,2047]` |
| `token_ids` | `[L]` | optional; required instead of embeddings when using a cache |

The configured layout is explicit. For existing `[T,8]` arrays use
`data.codes_layout: T_K`. Padded batches have token/frame masks; no truncation or
codebook/time-axis guessing occurs. Unequal text/audio sequence lengths are expected.

### Where token embeddings come from

Use the **same Qwen checkpoint/revision/token IDs** that produced the cached states.
Extract only the input-table rows needed by the dataset:

```bash
python -m speech_adapter.v2 prepare-embeddings \
  --model-dir /path/to/Qwen2.5-1.5B-Instruct \
  --token-ids /path/to/all_used_token_ids.npy \
  --output-dir runs/qwen-input-table
```

Set `data.embedding_cache: runs/qwen-input-table` and store `token_ids` in each NPZ.
The loader will supply embeddings. An untied `lm_head` is never accepted as an input
embedding table. Cached embeddings, IDs and states must remain aligned; a new
tokenization of plain text is not assumed to reproduce the extraction sequence.
Qwen2.5-1.5B uses width 1536; adjust both configured feature widths for other models.
Record hidden layer, normalization, prompt/chat template and predictor-vs-consumed
token convention. Shapes alone cannot verify semantic provenance.

## Train Stage A, then Stage B

Using the direct NPZ config (substitute the prepared-feature config if applicable):

```bash
python -m speech_adapter.v2 train --config configs/v2_qwen15b.yaml --phase cb0 --limit 32
python -m speech_adapter.v2 train --config configs/v2_qwen15b.yaml --phase acoustic \
  --init runs/v2-qwen15b/checkpoints/cb0_best.pt --limit 32
```

Remove `--limit 32` for full data. `train.epochs` controls each invocation. Initially
use fp32 for debugging; on T4 choose `train.precision: fp16`. bf16 requires supported
hardware. Restart each phase using its last checkpoint:

```bash
python -m speech_adapter.v2 train --config configs/v2_qwen15b.yaml --phase acoustic \
  --resume runs/v2-qwen15b/checkpoints/acoustic_last.pt
```

Training writes atomic last/best checkpoints and one JSONL history per phase.
Increase `train.epochs` above the completed epoch when extending a run. Model and
data/provenance configuration must match when loading a checkpoint. Training
schedule extension changes the cosine schedule; bitwise continuation is not promised.

Optional distributed launch (CPU gloo or CUDA NCCL, one process per device):

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m speech_adapter.v2 train --config configs/v2_qwen15b.yaml --phase cb0
```

The two-process path could not be exercised in the implementation environment:
local process-group sockets were denied. CPU single-process training is tested;
CUDA/NCCL and mixed precision require validation on your training machine.

## Generate codes and optional waveform

`answer_features.npz` needs only `hidden` and `embeddings` (or `token_ids` plus the
configured embedding cache). It does not need reference audio codes or durations.

```bash
python -m speech_adapter.v2 infer \
  --checkpoint runs/v2-qwen15b/checkpoints/acoustic_best.pt \
  --features answer_features.npz --output-dir runs/generated
```

This writes `[8,T]` integer `codes.npy` and `inference.json`. `max_frames` is a
generation guard; reaching it is reported as truncation, not normal termination.
For real waveform decoding, add ONE matching backend:

```bash
# Requires the optional moshi package and matching local Mimi .safetensors.
python -m speech_adapter.v2 infer \
  --checkpoint runs/v2-qwen15b/checkpoints/acoustic_best.pt \
  --features answer_features.npz --output-dir runs/generated \
  --mimi-weights /path/to/mimi.safetensors
```

Alternatively use `--mimi-decoder-onnx /path/to/decoder_8cb.onnx` with onnxruntime.
It requires a stateless exported decoder with exactly one int64 `[B,8,T]` input;
32-codebook or multi-cache-input exports are not silently adapted. The existing
`scripts/convert_mimi_decoder_8cb.py` may help produce a compatible export. Real
codec backends are optional; fake waveform generation is never substituted.

## Validation and limitations

```bash
python -m speech_adapter.v2 evaluate --config configs/v2_qwen15b.yaml \
  --checkpoint runs/v2-qwen15b/checkpoints/acoustic_best.pt \
  --output runs/v2-qwen15b/evaluation.json
```

The built-in evaluation reports masked token losses, oracle-CB0 acoustic accuracy,
and bounded free-running termination. These are not speech-quality measurements.
Listen to and externally evaluate these three cases with the actual matching codec:

1. Ground-truth eight codebooks: verify reference reconstruction.
2. Ground-truth CB0 + predicted CB1-CB7: isolate acoustic completion.
3. Generated CB0 + predicted CB1-CB7: full TTS.

The parallel acoustic heads do not explicitly model within-frame codebook
dependencies; lower quality than a depth/iterative decoder is possible. Stage B
initially sees reference CB0, while inference uses predictions: that distribution
gap must be measured. A 32-sample overfit does not establish unseen-text quality,
adequate training-data volume, real-time speed, or memory fit on Samsung M35.

See [docs/v2_architecture.md](docs/v2_architecture.md) for exact module contracts.

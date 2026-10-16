# v2 implementation validation

Validation performed with Python 3.12, PyTorch 2.14.1+cpu and pytest 9.1.1.
No real Qwen/Mimi weights or user audio dataset were downloaded for these checks.

## Executed successfully

- `python -m pytest -q tests`: 25 tests passed, 2 skipped (local Gloo sockets
  denied; CUDA unavailable). Includes scaler recovery, finite-rank synchronized
  skip simulation, persistent-overflow diagnostics, and step-log/resume checks.
- `python -m speech_adapter.v2 smoke --output-dir runs/v2-smoke --steps 160`:
  two tiny stages learned their synthetic fixture, generated `[1,8,6]` codes,
  stopped naturally, and replayed identical codes after checkpoint reload.
- Synthetic Stage A loss: 2.98294 -> 0.000903.
- Synthetic Stage B loss: 2.86374 -> 0.000967.
- Default-width forward/backward: Qwen features 1536, temporal width 512,
  eight 2048-entry codebooks. Both phase losses and all produced gradients finite.
- Default architecture parameter counts: Stage A trainable 21,310,977;
  Stage B trainable 16,598,528; combined 37,909,505. Frozen Qwen/Mimi are excluded.
- Config parsing and source compilation; `git diff --check`.

The tests cover target-shift causality, cached/full CB0 parity, right-padding
invariance, masked losses, acoustic-target independence, phase freezing including
optimizer updates, all seven acoustic head gradients, stop and guard behavior,
checkpoint restore/resume, NPZ code layouts and feature-only inference, input-table
extraction including sharded/tied-weight aliases, and legacy shard preparation.
WAV testing checks the container format using synthetic samples, not real decoding.

## Attempted but unavailable

A two-process CPU distributed run reached process-group initialization but the
environment denied local Gloo sockets (`Operation not permitted`). Distributed
execution therefore remains unverified here. CUDA AMP/NCCL were not available.

## Not established by these results

- Real-Mimi waveform correctness/quality with your checkpoint and stored targets.
- Pronunciation, unseen-text intelligibility, voice consistency or adequate data volume.
- Incremental text-to-audio streaming (this architecture is explicitly offline).
- Real-time speed, thermal behavior or memory fit on Samsung M35.

Use the three real-data evaluations described in README before drawing those
conclusions. Synthetic loss convergence validates implementation mechanics, not
speech quality or the superiority of the parallel acoustic architecture.

# Training progress and FP16 recovery

Add the following under `train` in either v2 YAML configuration:

```yaml
train:
  step_size: 50
  amp_init_scale: 4096
  max_consecutive_overflows: 20
```

`step_size` is a logging interval, not a learning-rate scheduler or validation
interval. A step is one batch on each distributed rank. With batch_size=8 and
two GPUs, a step processes up to 16 samples. Logs appear every 50 steps within
each epoch and at its final partial interval. Set 0 for epoch-only progress.
Validation and checkpoint selection still occur at epoch boundaries.

Rank zero prints progress JSON and writes `<phase>_steps.jsonl` in `work_dir`.
`train_step` events include the global attempted batch step, epoch step, successful
optimizer updates, interval-average loss and component losses, learning rate,
pre-clipping gradient norm, current scale, skipped updates and sample throughput.
Interval losses include finite forward losses from skipped batches. Epoch logs
remain in `<phase>_history.jsonl`. Resume appends progress and restores counters;
older checkpoints without counters use the previous epoch-based step estimate.
An epoch checkpoint resumes at the next epoch, not the interrupted batch.

FP16 can produce finite forward losses but overflow during scaled backward.
The old trainer aborted before GradScaler could recover. The trainer now unscales
and clips gradients, synchronizes the finite-gradient decision across ranks, and
skips the entire optimizer update on every rank when any rank overflows. It lowers
the scale and resets its growth tracker without updating model or optimizer state.
`amp_overflow` events include affected record IDs by rank even when step logging
is disabled. Repeated failures stop after `max_consecutive_overflows`; non-finite
forward losses and non-finite FP32/BF16 gradients still fail with record diagnostics.
Investigate persistent failures rather than interpreting skipped updates as learning.

On Kaggle with two T4 GPUs, use `precision: fp16`, enable both accelerators,
and launch `python -m torch.distributed.run --standalone --nproc_per_node=2
-m speech_adapter.v2 train --config configs/v2_prepared_features.yaml --phase cb0`.
To restart after an epoch checkpoint, add `--resume <work_dir>/checkpoints/cb0_last.pt`.
The CUDA process-group initialization binds each rank's local device explicitly.


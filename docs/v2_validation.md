# v2 implementation validation

Validation performed with Python 3.12, PyTorch 2.14.1+cpu and pytest 9.1.1.
No real Qwen/Mimi weights or user audio dataset were downloaded for these checks.

## Executed successfully

- `python -m pytest -q tests`: 17 tests passed.
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

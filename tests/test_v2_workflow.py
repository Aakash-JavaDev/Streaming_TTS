import json
from pathlib import Path

import numpy as np
import pytest
import torch

from speech_adapter.v2.__main__ import main
from speech_adapter.v2.checkpoint import load_checkpoint
from speech_adapter.v2.smoke import write_dummy_dataset
from speech_adapter.v2.train import run_training


def test_two_stage_training_resume_and_features_only_cli(tmp_path):
    cfg = write_dummy_dataset(str(tmp_path))
    cfg.train.epochs = 1
    cfg.train.eval_samples = 2
    cfg.train.max_generate_frames = 8
    a = run_training(cfg, "cb0")
    assert np.isfinite(a["train_loss"])
    cp = Path(cfg.work_dir) / "checkpoints"
    initial, _, _ = load_checkpoint(cp / "cb0_last.pt")
    with pytest.raises(ValueError, match="requires"):
        run_training(cfg, "acoustic")
    b = run_training(cfg, "acoustic", init=str(cp / "cb0_last.pt"))
    assert np.isfinite(b["train_loss"])
    final, _, _ = load_checkpoint(cp / "acoustic_last.pt")
    for key, value in initial.state_dict().items():
        if not key.startswith("acoustic."):
            assert torch.equal(value, final.state_dict()[key])
    cfg.train.epochs = 2
    resumed = run_training(cfg, "acoustic", resume=str(cp / "acoustic_last.pt"))
    assert resumed["epoch"] == 1
    features = tmp_path / "inference.npz"
    with np.load(tmp_path / "dummy-43-0.npz") as f:
        np.savez(features, hidden=f["hidden"], embeddings=f["embeddings"])
    output = tmp_path / "inference"
    main(["infer", "--checkpoint", str(cp / "acoustic_last.pt"), "--features", str(features),
          "--output-dir", str(output), "--max-frames", "8"])
    codes = np.load(output / "codes.npy")
    report = json.loads((output / "inference.json").read_text())
    assert codes.shape[0] == 8 and 1 <= codes.shape[1] <= 8
    assert report["waveform_written"] is False
    assert not (output / "speech.wav").exists()


def test_decoder_contract_and_wav_container(tmp_path):
    import wave
    from speech_adapter.v2.codec import checked_audio, validate_codes, write_wav
    validate_codes(torch.zeros(1, 8, 2, dtype=torch.long))
    with pytest.raises(ValueError):
        validate_codes(torch.zeros(1, 32, 2, dtype=torch.long))
    with pytest.raises(ValueError):
        checked_audio(np.array([np.nan]))
    write_wav(tmp_path / "container_test.wav", np.zeros(1920, dtype=np.float32))
    with wave.open(str(tmp_path / "container_test.wav")) as w:
        assert (w.getframerate(), w.getnchannels(), w.getnframes()) == (24000, 1, 1920)

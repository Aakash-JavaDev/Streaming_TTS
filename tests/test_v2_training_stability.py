import datetime
import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from speech_adapter.v2.checkpoint import load_checkpoint
from speech_adapter.v2.smoke import write_dummy_dataset, tiny_config
from speech_adapter.v2.train import optimizer_step, run_training


def test_scaler_overflow_skips_then_recovers():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    opt = torch.optim.AdamW([p], lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=16, growth_interval=2)
    scaler.scale(p.sum()).backward()
    p.grad.fill_(float("inf"))
    before = p.detach().clone()
    _, updated = optimizer_step(opt, scaler, [p], 1)
    assert not updated
    assert torch.equal(before, p)
    assert not opt.state
    assert scaler.get_scale() == 8
    assert scaler.state_dict()["_growth_tracker"] == 0
    scaler.scale(p.sum()).backward()
    norm, updated = optimizer_step(opt, scaler, [p], 1)
    assert updated and norm == 1
    assert not torch.equal(before, p)
    assert torch.isfinite(p).all()


def test_fp32_nonfinite_gradients_still_fail():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    opt = torch.optim.SGD([p], lr=0.1)
    p.grad = torch.tensor([float("nan")])
    with pytest.raises(RuntimeError, match="non-finite gradient norm"):
        optimizer_step(opt, torch.amp.GradScaler("cpu", enabled=False), [p], 1)
    assert p.item() == 1


def _distributed_overflow_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2,
                            init_method="file://" + rendezvous,
                            timeout=datetime.timedelta(seconds=45))
    try:
        model = DDP(torch.nn.Linear(1, 1, bias=False))
        params = list(model.parameters())
        opt = torch.optim.AdamW(params, lr=0.1)
        scaler = torch.amp.GradScaler("cpu", init_scale=16)
        scaler.scale(model(torch.ones(1, 1)).sum()).backward()
        if rank == 1:  # Only one rank detects overflow: neither may update.
            params[0].grad.fill_(float("inf"))
        before = params[0].detach().clone()
        _, updated = optimizer_step(opt, scaler, params, 1)
        assert not updated and torch.equal(before, params[0])
        assert scaler.get_scale() == 8 and not opt.state
        scaler.scale(model(torch.ones(1, 1)).sum()).backward()
        _, updated = optimizer_step(opt, scaler, params, 1)
        assert updated
        copies = [torch.empty_like(params[0]) for _ in range(2)]
        dist.all_gather(copies, params[0].detach())
        assert torch.equal(copies[0], copies[1])
    finally:
        dist.destroy_process_group()


def test_two_rank_overflow_is_synchronized(tmp_path):
    try:
        mp.spawn(_distributed_overflow_worker,
                 args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
    except mp.ProcessRaisedException as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("environment denies local Gloo sockets")
        raise


def test_finite_local_rank_skips_when_peer_overflows(monkeypatch):
    p = torch.nn.Parameter(torch.tensor([1.0]))
    opt = torch.optim.AdamW([p], lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=16)
    scaler.scale(p.sum()).backward()
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", lambda flag, op: flag.zero_())
    norm, updated = optimizer_step(opt, scaler, [p], 1)
    assert norm == 1 and not updated
    assert p.item() == 1 and not opt.state and scaler.get_scale() == 8


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_invalid_step_interval(value):
    cfg = tiny_config()
    cfg.train.step_size = value
    with pytest.raises(ValueError, match="step_size"):
        cfg.validate()


def test_step_logs_remainder_resume_and_disable(tmp_path):
    cfg = write_dummy_dataset(str(tmp_path))
    cfg.train.batch_size = 1
    cfg.train.epochs = 1
    cfg.train.step_size = 3
    run_training(cfg, "cb0")
    work = Path(cfg.work_dir)
    path = work / "cb0_steps.jsonl"
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    assert [r["epoch_step"] for r in rows] == [3, 4]
    assert [r["window_samples"] for r in rows] == [3, 1]
    assert rows[-1]["optimizer_steps"] == 4
    assert "cb0_ce" in rows[0]["metrics"]
    cp = work / "checkpoints/cb0_last.pt"
    _, _, payload = load_checkpoint(cp)
    assert payload["extra"]["global_step"] == 4
    cfg.train.epochs = 2
    run_training(cfg, "cb0", resume=str(cp))
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    assert [r["step"] for r in rows] == [3, 4, 7, 8]
    cfg.train.step_size = 0
    run_training(cfg, "cb0")
    assert path.read_text() == ""


def test_persistent_overflows_report_records_and_stop(tmp_path, monkeypatch):
    import speech_adapter.v2.train as training
    cfg = write_dummy_dataset(str(tmp_path))
    cfg.train.batch_size = 1
    cfg.train.step_size = 0
    cfg.train.max_consecutive_overflows = 2
    monkeypatch.setattr(training, "optimizer_step", lambda *args: (float("inf"), False))
    with pytest.raises(RuntimeError, match="persistent FP16 overflow.*records_by_rank"):
        run_training(cfg, "cb0")
    events = [json.loads(s) for s in (Path(cfg.work_dir) / "cb0_steps.jsonl").read_text().splitlines()]
    assert len(events) == 2
    assert all(e["event"] == "amp_overflow" and e["records_by_rank"] for e in events)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_fp16_overflow_recovery():
    p = torch.nn.Parameter(torch.ones(1, device="cuda"))
    opt = torch.optim.AdamW([p], lr=0.1)
    scaler = torch.amp.GradScaler("cuda", init_scale=16)
    scaler.scale(p.sum()).backward()
    p.grad.fill_(float("inf"))
    _, updated = optimizer_step(opt, scaler, [p], 1)
    assert not updated and p.item() == 1 and scaler.get_scale() == 8
    scaler.scale(p.sum()).backward()
    _, updated = optimizer_step(opt, scaler, [p], 1)
    assert updated and torch.isfinite(p).all()

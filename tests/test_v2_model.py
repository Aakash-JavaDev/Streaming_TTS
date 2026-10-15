import copy

import pytest
import torch

from speech_adapter.v2.config import V2Config
from speech_adapter.v2.data import collate
from speech_adapter.v2.losses import acoustic_loss, cb0_loss
from speech_adapter.v2.model import TwoStageTTS
from speech_adapter.v2.smoke import dummy_items, tiny_config


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(2)
    torch.manual_seed(7)


def fixture():
    cfg = tiny_config()
    items = dummy_items(2)
    items[1]["codes"] = items[1]["codes"][:, :4]
    items[1]["hidden"] = items[1]["hidden"][:2]
    items[1]["embeddings"] = items[1]["embeddings"][:2]
    return cfg, TwoStageTTS(cfg.model), collate(items)


def test_shapes_and_finite_gradients_for_both_phases():
    cfg, model, batch = fixture()
    logits, stop = model(batch)
    assert logits.shape == (2, 6, 16)
    assert stop.shape == (2, 6)
    loss, _ = cb0_loss(logits, stop, batch["codes"], batch["frame_mask"])
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.conditioner.gate.weight.grad is not None
    assert all(p.grad is None for p in model.acoustic.parameters())
    model.set_phase("acoustic")
    model.train()
    logits = model(batch)
    assert logits.shape == (2, 6, 7, 16)
    loss, _ = acoustic_loss(logits, batch["codes"], batch["frame_mask"])
    loss.backward()
    assert all(p.grad is None for p in model.conditioner.parameters())
    assert all(p.grad is None for p in model.cb0.parameters())
    assert not model.conditioner.training and not model.cb0.training
    assert all(head.weight.grad is not None and head.weight.grad.norm() > 0 for head in model.acoustic.heads)
    assert all(torch.isfinite(p.grad).all() for p in model.acoustic.parameters() if p.grad is not None)


def test_cb0_shift_has_no_current_or_future_target_leakage():
    _, model, batch = fixture()
    model.eval()
    first, _ = model(batch)
    changed = copy.deepcopy(batch)
    changed["codes"][:, 0, 2:] = (changed["codes"][:, 0, 2:] + 3) % 16
    second, _ = model(changed)
    # Prediction of frame 2 cannot see target CB0[2].
    torch.testing.assert_close(first[:, :3], second[:, :3], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(first[0, 3:], second[0, 3:])


def test_acoustic_model_never_consumes_acoustic_targets():
    _, model, batch = fixture()
    model.set_phase("acoustic")
    model.eval()
    first = model(batch)
    batch["codes"][:, 1:] = (batch["codes"][:, 1:] + 7) % 16
    torch.testing.assert_close(first, model(batch), atol=0, rtol=0)


def test_padding_does_not_change_valid_predictions_or_losses():
    _, model, batch = fixture()
    model.eval()
    first, stop = model(batch)
    changed = copy.deepcopy(batch)
    changed["hidden"][1, 2:] = float("nan")
    changed["embeddings"][1, 2:] = 1e6
    changed["codes"][1, :, 4:] = 99999
    second, second_stop = model(changed)
    mask = batch["frame_mask"]
    torch.testing.assert_close(first[mask], second[mask])
    a, _ = cb0_loss(first, stop, batch["codes"], mask)
    b, _ = cb0_loss(second, second_stop, changed["codes"], mask)
    torch.testing.assert_close(a, b)
    b.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.set_phase("acoustic")
    torch.testing.assert_close(model(batch)[mask], model(changed)[mask])


def test_cached_cb0_steps_match_teacher_forced_forward():
    _, model, batch = fixture()
    model.eval()
    batch = {k: v[:1] if torch.is_tensor(v) else v for k, v in batch.items()}
    with torch.no_grad():
        z = model.condition_text(batch)
        full, full_stop = model(batch)
        caches = [None] * len(model.cb0.blocks)
        text_kvs = [b.cross_attention.prepare(z) for b in model.cb0.blocks]
        for t in range(6):
            prev = None if t == 0 else batch["codes"][:, 0, t-1]
            logits, stop, caches = model.cb0.step(prev, z, batch["token_mask"], caches, t, text_kvs)
            torch.testing.assert_close(logits, full[:, t], atol=2e-6, rtol=1e-5)
            torch.testing.assert_close(stop, full_stop[:, t], atol=2e-6, rtol=1e-5)


def test_stage_b_optimization_cannot_modify_conditioner_or_cb0():
    _, model, batch = fixture()
    model.set_phase("acoustic")
    snapshot = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith("acoustic.")}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    loss, _ = acoustic_loss(model(batch), batch["codes"], batch["frame_mask"])
    loss.backward()
    opt.step()
    for key, value in snapshot.items():
        assert torch.equal(value, model.state_dict()[key])


def test_generation_stopping_guard_and_output_contract():
    _, model, batch = fixture()
    features = [batch[k][:1] for k in ("hidden", "embeddings", "token_mask")]
    with torch.no_grad():
        model.cb0.stop_head.weight.zero_()
        model.cb0.stop_head.bias.fill_(100)
    stopped = model.generate(*features, max_frames=8, min_frames=2)
    assert stopped.codes.shape == (1, 8, 2) and stopped.termination_reason == "stop"
    assert stopped.codes.dtype == torch.long
    assert stopped.codes.min() >= 0 and stopped.codes.max() < 16
    with torch.no_grad():
        model.cb0.stop_head.bias.fill_(-100)
    guarded = model.generate(*features, max_frames=3)
    assert guarded.codes.shape == (1, 8, 3) and guarded.termination_reason == "max_frames"
    assert model.training  # generation restores caller mode


def test_context_and_token_branches_both_affect_conditioning():
    _, model, batch = fixture()
    z = model.condition_text(batch)
    hidden_change = copy.deepcopy(batch)
    hidden_change["hidden"] *= -1
    embedding_change = copy.deepcopy(batch)
    embedding_change["embeddings"] *= -1
    assert not torch.allclose(z, model.condition_text(hidden_change))
    assert not torch.allclose(z, model.condition_text(embedding_change))


def test_reject_invalid_config_and_empty_masks():
    with pytest.raises(ValueError):
        V2Config.from_dict({"model": {"width": 30}})
    _, model, batch = fixture()
    batch["token_mask"][0] = False
    with pytest.raises(ValueError, match="empty"):
        model(batch)


def test_default_qwen_dimensions_with_real_codec_vocabulary():
    cfg = V2Config()
    cfg.model.dropout = 0
    model = TwoStageTTS(cfg.model).eval()
    batch = {"hidden": torch.randn(1, 3, 1536), "embeddings": torch.randn(1, 3, 1536),
             "token_mask": torch.ones(1, 3, dtype=torch.bool),
             "codes": torch.randint(0, 2048, (1, 8, 4)), "frame_mask": torch.ones(1, 4, dtype=torch.bool)}
    with torch.no_grad():
        assert model(batch)[0].shape == (1, 4, 2048)
        model.set_phase("acoustic")
        assert model(batch).shape == (1, 4, 7, 2048)

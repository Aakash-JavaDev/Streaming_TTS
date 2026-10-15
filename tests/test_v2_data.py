import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from speech_adapter.v2.data import NPZDataset, load_inference_features, collate
from speech_adapter.v2.embeddings import embedding_rows, prepare_embedding_cache, EmbeddingCache
from speech_adapter.v2.smoke import write_dummy_dataset


def test_embedding_extraction_uses_exact_input_table_and_cache(tmp_path):
    root = tmp_path / "qwen"
    root.mkdir()
    table = torch.arange(20 * 16).reshape(20, 16).float()
    save_file({"model.embed_tokens.weight": table, "lm_head.weight": table + 1000}, root / "model.safetensors")
    (root / "config.json").write_text(json.dumps({"hidden_size": 16, "tie_word_embeddings": False}))
    ids = np.array([5, 2, 5], dtype=np.int64)
    np.testing.assert_array_equal(embedding_rows(root, ids), table[ids].numpy())
    np.save(tmp_path / "ids.npy", ids)
    prepare_embedding_cache(str(root), str(tmp_path / "ids.npy"), str(tmp_path / "cache"))
    np.testing.assert_array_equal(EmbeddingCache(tmp_path / "cache").lookup(ids), table[ids].numpy())
    with pytest.raises(ValueError, match="missing"):
        EmbeddingCache(tmp_path / "cache").lookup(np.array([7]))
    with pytest.raises(ValueError, match="outside"):
        embedding_rows(root, np.array([100]))


def test_embedding_sharded_index_and_tied_alias_only(tmp_path):
    table = torch.randn(12, 16)
    save_file({"lm_head.weight": table}, tmp_path / "part.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"lm_head.weight": "part.safetensors"}}))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"hidden_size": 16, "tie_word_embeddings": False}))
    with pytest.raises(FileNotFoundError, match="untied"):
        embedding_rows(tmp_path, np.array([1]))
    config.write_text(json.dumps({"hidden_size": 16, "tie_word_embeddings": True}))
    np.testing.assert_allclose(embedding_rows(tmp_path, np.array([1])), table[1:2].numpy(), rtol=0.002, atol=0.001)


def test_dataset_shapes_explicit_layout_and_inference_without_audio(tmp_path):
    cfg = write_dummy_dataset(str(tmp_path))
    data = NPZDataset(cfg, "train")
    batch = collate([data[0], data[1]])
    assert batch["codes"].shape == (2, 8, 6)
    sample = data[0]
    path = tmp_path / "features_only.npz"
    np.savez(path, hidden=sample["hidden"].numpy(), embeddings=sample["embeddings"].numpy())
    features = load_inference_features(str(path), cfg)
    assert set(features) == {"hidden", "embeddings", "token_mask"}
    assert features["hidden"].shape == (1, 3, 16)
    # Explicit T_K input layout must be converted exactly once.
    row = data.rows[0]
    p = tmp_path / row["features"]
    np.savez(p, hidden=sample["hidden"].numpy(), embeddings=sample["embeddings"].numpy(), codes=sample["codes"].T.numpy())
    cfg.data.codes_layout = "T_K"
    assert torch.equal(NPZDataset(cfg, "train")[0]["codes"], sample["codes"])


def test_dataset_rejects_bad_code_range_or_hidden_alignment(tmp_path):
    cfg = write_dummy_dataset(str(tmp_path))
    data = NPZDataset(cfg, "train")
    item = data[0]
    p = tmp_path / data.rows[0]["features"]
    np.savez(p, hidden=item["hidden"].numpy(), embeddings=item["embeddings"].numpy(), codes=np.full((8, 6), 50))
    with pytest.raises(ValueError, match="range"):
        NPZDataset(cfg, "train")[0]
    np.savez(p, hidden=item["hidden"].numpy(), embeddings=item["embeddings"][:2].numpy(), codes=item["codes"].numpy())
    with pytest.raises(ValueError, match="aligned"):
        NPZDataset(cfg, "train")[0]


def test_existing_shard_feature_preparation_and_v2_adapter(tmp_path):
    from speech_adapter.config import AppConfig, PathsConfig, RuntimeConfig
    from speech_adapter.data import build_manifest, prepare_qwen_features, _split
    from speech_adapter.v2.data import PreparedLegacyDataset
    from speech_adapter.v2.smoke import tiny_config
    shard = tmp_path / "shards" / "shard_0000"
    shard.mkdir(parents=True)
    qwen = tmp_path / "qwen"
    qwen.mkdir()
    save_file({"model.embed_tokens.weight": torch.randn(16, 16)}, qwen / "model.safetensors")
    (qwen / "config.json").write_text(json.dumps({"hidden_size": 16, "tie_word_embeddings": True}))
    ids = []
    for desired in ("train", "val"):
        ids.extend([str(i) for i in range(100) if _split(str(i), 42, 0.1) == desired][:4])
    n = len(ids)
    np.save(shard / "codes.npy", np.random.default_rng(1).integers(0, 16, (n*6, 8), dtype=np.int64))
    np.save(shard / "offsets.npy", np.arange(n+1)*6)
    np.save(shard / "hidden.npy", np.random.default_rng(2).normal(size=(n*3, 16)).astype(np.float16))
    np.save(shard / "hidden_offsets.npy", np.arange(n+1)*3)
    np.save(shard / "token_ids.npy", np.tile(np.array([1, 2, 3]), n))
    np.save(shard / "token_char_offsets.npy", np.tile(np.array([[0,1],[1,2],[2,3]]), (n,1)))
    words = [{"word": "abc", "start_frame": 0, "end_frame": 6}]
    for name, rows in (("ids.jsonl", ids), ("alignment.jsonl", [{"words": words}]*n),
                       ("pieces.jsonl", [["a","b","c"]]*n)):
        (shard/name).write_text("".join(json.dumps(r)+"\n" for r in rows))
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps({"id": i, "question": "q", "answer": "abc"})+"\n" for i in ids))
    legacy = AppConfig(PathsConfig(str(shard.parent), str(source), str(qwen), work_dir=str(tmp_path/"prepared")),
                       runtime=RuntimeConfig(profile="local_cpu", device="cpu", require_two_gpus=False))
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(legacy.to_dict()))
    build_manifest(legacy)
    prepare_qwen_features(legacy)
    cfg = tiny_config()
    cfg.data.legacy_config = str(legacy_path)
    item = PreparedLegacyDataset(cfg, "train")[0]
    assert item["hidden"].shape == item["embeddings"].shape == (3,16)
    assert item["codes"].shape == (8,6)
    assert "owners" not in collate([item]) and "durations" not in collate([item])

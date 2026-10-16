from pathlib import Path
import torch

from speech_adapter.v2.checkpoint import load_checkpoint
from speech_adapter.v2.data import make_dataset, collate
from speech_adapter.v2.codec import OnnxDecoder, write_wav

torch.set_num_threads(2)

model, cfg, payload = load_checkpoint(
    "/home/lazarus_25/workspace/handsfree-chatbot/Streaming_TTS/runs/v2-prepared-acoustic/checkpoints/acoustic_best.pt", "cpu"
)
if payload["phase"] != "acoustic":
    raise ValueError("Use an acoustic checkpoint")
model.eval()

item = make_dataset(cfg, "val")[0]
batch = collate([item])

decoder = OnnxDecoder("/home/lazarus_25/workspace/handsfree-chatbot/Streaming_TTS/models/kyutai-mimi-ONNX/onnx/decoder_model_8cb.onnx")
out = Path("runs/diagnostic")
out.mkdir(parents=True, exist_ok=True)

with torch.inference_mode():
    # Acoustic prediction conditioned on the recorded CB0.
    residuals = model(batch).argmax(-1).transpose(1, 2)
    oracle_cb0_codes = torch.cat(
        [batch["codes"][:, :1], residuals], dim=1
    )

    # Entire sequence generated without target audio codes.
    generated = model.generate(
        batch["hidden"],
        batch["embeddings"],
        batch["token_mask"],
        max_frames=cfg.train.max_generate_frames,
        min_frames=cfg.train.min_generate_frames,
        stop_threshold=cfg.train.stop_threshold,
    )

for name, codes in (
    ("recorded_codes", batch["codes"]),
    ("oracle_cb0", oracle_cb0_codes),
    ("fully_generated", generated.codes),
):
    audio = decoder.decode(codes)
    write_wav(out / f"{name}.wav", audio)
    print(name, "frames:", codes.shape[-1],
          "WAV seconds:", round(len(audio) / 24000, 3))

print("Text:", item["text"])
print("Generated termination:", generated.termination_reason)
(out / "text.txt").write_text(item["text"], encoding="utf-8")
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a local CPU smoke-test config")
    parser.add_argument("--data-dir", required=True, help="directory containing shard_* folders")
    parser.add_argument(
        "--source-jsonl", required=True,
        help="question/answer JSONL file or directory of part_*.jsonl files",
    )
    parser.add_argument(
        "--qwen-model", default="data/models/Qwen2.5-0.5B-Instruct",
        help="local Qwen2.5-0.5B-Instruct directory",
    )
    parser.add_argument("--output", default="configs/local_cpu_test.json")
    parser.add_argument("--work-dir", default="local_work/qwen_0.5b_cpu")
    parser.add_argument("--records", type=int, default=5000)
    args = parser.parse_args()
    if args.records < 4:
        parser.error("--records must be at least 4")

    template = Path(__file__).resolve().parents[1] / "configs" / "local_cpu_test.json"
    config = json.loads(template.read_text(encoding="utf-8"))
    config["paths"].update(
        {
            "data_dir": str(Path(args.data_dir).resolve()),
            "source_jsonl": str(Path(args.source_jsonl).resolve()),
            "qwen_model": str(Path(args.qwen_model).resolve()),
            "work_dir": str(Path(args.work_dir).resolve()),
        }
    )
    config["runtime"]["manifest_record_limit"] = args.records
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()

"""Materialize `data/<dataset>/phases/phase<k>.txt` and its manifest.

Everything for a stream lives under `data/<dataset>/`: `phases/` holds the
ICL/export corpus and its eval sets, `synth/p0<k>/` holds the self-study runs,
and `train/` symlinks to the synth artifacts. The corpus text is the phase
resource's `to_string()`, so it must stay byte-identical to the copy the eval
sets were built against.

  python examples/maintenance/data/export_phase_corpora.py --dataset quality finqa techqa
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from examples.shared.paths import ROOT
from examples.shared.synth.self_study_vllm import build_resource_config

PHASES = (1, 2, 3, 4, 5)


def export(dataset: str, tokenizer, force: bool) -> None:
    out_dir = Path(ROOT) / "data" / dataset / "phases"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict] = {}
    cumulative = 0
    for phase in PHASES:
        text_path = out_dir / f"phase{phase}.txt"
        if text_path.exists() and not force:
            text = text_path.read_text()
            action = "kept"
        else:
            text = build_resource_config(dataset, phase).instantiate().to_string()
            text_path.write_text(text)
            action = "wrote"

        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        cumulative += n_tokens

        import pyarrow.parquet as pq

        eval_path = out_dir / f"phase{phase}_eval.parquet"
        # The anchored variant names the source document in each question stem;
        # FinQA and QuALITY need it because their questions say "the" document.
        anchored_path = out_dir / f"phase{phase}_eval_anchored.parquet"
        questions = pq.ParquetFile(eval_path).metadata.num_rows if eval_path.exists() else None
        anchored = pq.ParquetFile(anchored_path).metadata.num_rows if anchored_path.exists() else None

        manifest[str(phase)] = {
            "text": str(text_path.relative_to(ROOT)),
            "eval": str(eval_path.relative_to(ROOT)) if eval_path.exists() else None,
            "eval_anchored": str(anchored_path.relative_to(ROOT)) if anchored_path.exists() else None,
            "tokens": n_tokens,
            "questions": questions,
            "questions_anchored": anchored,
            "cumulative_tokens": cumulative,
        }
        print(
            f"  {action} phase{phase}.txt  {len(text):>10,} chars  {n_tokens:>8,} tokens"
            f"  eval={'-' if questions is None else questions}"
            f"  anchored={'-' if anchored is None else anchored}"
        )

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  manifest -> {out_dir / 'manifest.json'}  (cumulative {cumulative:,} tokens)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", required=True)
    parser.add_argument(
        "--model", default="Qwen/Qwen3-4B-Instruct-2507", help="tokenizer used for token counts"
    )
    parser.add_argument("--force", action="store_true", help="rewrite phase<k>.txt if present")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    for dataset in args.dataset:
        print(f"=== {dataset} ===")
        export(dataset, tokenizer, args.force)


if __name__ == "__main__":
    main()

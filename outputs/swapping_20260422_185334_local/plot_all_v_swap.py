"""Plot per-layer V-swap log-PPL + baselines for each experiment subfolder."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PT_NAME = "per_slot_swap.pt"
META_NAME = "meta.json"


def parse_folder(name: str) -> tuple[str, str, str]:
    """Return (model_label, toks, tail) from e.g. per_token_slot_llama_toks512_..._aggregate."""
    m = re.match(r"per_token_slot_([\w]+)_toks(\d+)_(.+)$", name)
    if not m:
        return name, "", ""
    return m.group(1), m.group(2), m.group(3)


def to_float(x) -> float:
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def main() -> None:
    subdirs = sorted(p for p in ROOT.iterdir() if p.is_dir() and (p / PT_NAME).is_file())
    if not subdirs:
        raise SystemExit(f"No subfolders with {PT_NAME} under {ROOT}")

    for exp_dir in subdirs:
        data = torch.load(exp_dir / PT_NAME, map_location="cpu", weights_only=False)
        layer_series = data["log_ppl_per_slot"]
        if torch.is_tensor(layer_series):
            layer_results = layer_series.detach().cpu().numpy().ravel()
        else:
            layer_results = np.asarray(layer_series).ravel()

        base_baseline = to_float(data["base_nll"])
        donor_baseline = to_float(data["donor_nll"])

        model, toks, tail = parse_folder(exp_dir.name)
        title_core = f"{model} toks={toks}"
        if tail:
            title_core += f" | {tail.replace('_', ' ')}"

        meta_title = ""
        if (exp_dir / META_NAME).is_file():
            meta = json.loads((exp_dir / META_NAME).read_text(encoding="utf-8"))
            eval_name = meta.get("eval_name", "")
            if eval_name:
                meta_title = f"eval={eval_name}"

        fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
        layer_indices = range(len(layer_results))
        ax.plot(layer_indices, layer_results, color="#d62728", linewidth=1.5, label="per-layer V swap")
        ax.axhline(
            y=base_baseline,
            color="#333333",
            linestyle="--",
            linewidth=1.2,
            label=f"P1-10 cartridge baseline: {base_baseline:.3f}",
        )
        ax.axhline(
            y=donor_baseline,
            color="#2ca02c",
            linestyle="--",
            linewidth=1.2,
            label=f"P11-20 cartridge baseline: {donor_baseline:.3f}",
        )

        full_title = title_core
        if meta_title:
            full_title = f"{full_title}\n{meta_title}"
        ax.set_title(full_title, fontsize=11, pad=10)
        ax.set_xlabel("Slot index S (V at S replaced with donor's V)", fontsize=11)
        ax.set_ylabel("log-perplexity", fontsize=11)
        ax.grid(True, linestyle="-", alpha=0.3)
        ax.legend(fontsize=9, loc="center right", framealpha=1.0)
        fig.tight_layout()

        out = exp_dir / "v_swap_log_ppl.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()

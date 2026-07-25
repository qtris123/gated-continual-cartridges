"""Regenerate the finance granularity-x-sparsity PNGs from the summary CSVs.

The raw eval-log directories that ``plot_granularity_x_sparsity.py`` normally
scrapes are not present locally, but the per-model ``*_summary.csv`` files in
this folder already contain every perplexity number the figure needs. This
script rebuilds the ``data`` dict from those CSVs and calls the shared ``plot``
function (so the color legend defined in ``GRAN_STYLE`` is the single source of
truth), writing the PNGs back into this folder.

Run:
    python regenerate_from_summary.py
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent

# Load the canonical plotting module (holds plot() + GRAN_STYLE + constants).
_PLOT_PY = (
    HERE.parent.parent
    / "qasper"
    / "qasper-forgetting-granularity-x-sparsity"
    / "plot_granularity_x_sparsity.py"
)
_spec = importlib.util.spec_from_file_location("gxs_plot", _PLOT_PY)
gxs = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = gxs  # needed so frozen dataclasses resolve on 3.14
_spec.loader.exec_module(gxs)


TITLE_NAMES = {"llama": "Llama", "qwen": "Qwen"}


def build_data(csv_path: Path) -> dict:
    """Reconstruct plot()'s data dict from a *_summary.csv file."""
    rows = list(csv.DictReader(csv_path.open()))

    out: dict = {"per_gran": {g: {"forgetting": [], "acquisition": []} for g in gxs.GRANULARITIES}}
    base_f: dict[str, float] = {}
    base_a: dict[str, float] = {}

    # Index the per-(gran, top_k) rows so we can emit in the canonical order.
    by_key: dict[tuple[str, str], dict] = {
        (r["granularity"], r["top_k"]): r for r in rows
    }

    for gran in gxs.GRANULARITIES:
        for k in gxs.SPARSITY_LEVELS:
            r = by_key[(gran, str(k))]
            out["per_gran"][gran]["forgetting"].append(float(r["forgetting_ppl"]))
            out["per_gran"][gran]["acquisition"].append(float(r["acquisition_ppl"]))
        base_row = by_key[(gran, "baseline")]
        base_f[gran] = float(base_row["forgetting_ppl"])
        base_a[gran] = float(base_row["acquisition_ppl"])

    out["baseline_forgetting"] = sum(base_f.values()) / len(base_f)
    out["baseline_acquisition"] = sum(base_a.values()) / len(base_a)
    out["baseline_forgetting_per_gran"] = base_f
    out["baseline_acquisition_per_gran"] = base_a
    return out


def main() -> None:
    for csv_path in sorted(HERE.glob("*_granularity_x_sparsity_summary.csv")):
        slug = csv_path.name.split("_", 1)[0]
        model = SimpleNamespace(slug=slug, title_name=TITLE_NAMES.get(slug, slug.title()))
        data = build_data(csv_path)
        gxs.plot(model, data, HERE / f"{slug}_granularity_x_sparsity.png")


if __name__ == "__main__":
    main()

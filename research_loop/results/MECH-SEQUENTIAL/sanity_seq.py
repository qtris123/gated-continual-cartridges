"""MECH-SEQUENTIAL Part-A sanity check (unit level, no training run).

Proves five things about the on-policy layer-sequential re-extraction
(`AM_ONPOLICY_LAYERS` -> `onpolicy_layers`, LIT-006 / SCOUT-AM divergence #3):

1. **Off is off.** `apply_document_am_write_to_cache` with `onpolicy_layers=0`
   and NO `onpolicy_refresh_fn` reproduces the committed (`git archive HEAD`)
   implementation, run in its own interpreter, to the last bit.

2. **The hook cannot fire when the knob is off.** Passing a refresh function that
   raises, with `onpolicy_layers=0`, is bit-identical to (1) -- the function is
   never called.

3. **The plumbing itself is a no-op.** With `onpolicy_layers=1` and a refresh
   function that returns the *same* accumulator (i.e. the activations did not
   move), the write is bit-identical to (1). Any difference would mean the
   restructuring changed the math rather than the queries.

4. **The mechanism is live.** With a refresh function that returns *perturbed*
   queries, the write changes, stays finite, and `extra["onpolicy"]` records the
   expected number of refreshes at the expected layers, with the query-drift
   diagnostics (cos / rel-L2) populated. Group sizes 1 / 2 / 4 give the expected
   refresh layers. The doc-KV variant sets `doc_kv_refreshed`.

5. **It fails loud.** A refresh that returns the wrong shape, an empty layer,
   non-finite queries, the wrong batch count, or a non-tuple -- and
   `onpolicy_layers>0` with no hook at all -- all raise instead of silently
   solving against stale queries.

Usage:  python research_loop/results/MECH-SEQUENTIAL/sanity_seq.py [--cuda]
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
DEV = "cuda" if ("--cuda" in sys.argv and torch.cuda.is_available()) else "cpu"
THETA = 5000000.0
OUT: dict = {"device": DEV, "rope_theta": THETA}

N_LAYERS, N_KV, T_CART, T_DOC, NQ, TOP_T, D = 4, 2, 40, 4000, 24, 8, 128
N_BATCH = 3


def _build_inputs():
    """Deterministic stub cache / mask / accumulator / doc-KV."""
    import torch.nn as nn

    from cartridges.am.query_accum import AMQueryAccumulator
    from cartridges.sparse_cache_finetuning import GradientMask

    class TinyCache(nn.Module):
        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(7)
            self.trainable_keys = nn.ParameterList([
                nn.Parameter(torch.randn(1, N_KV, T_CART - 1, D, generator=g))
                for _ in range(N_LAYERS)])
            self.trainable_values = nn.ParameterList([
                nn.Parameter(torch.randn(1, N_KV, T_CART - 1, D, generator=g))
                for _ in range(N_LAYERS)])
            self.frozen_keys = nn.ParameterList([
                nn.Parameter(torch.randn(1, N_KV, 1, D, generator=g))
                for _ in range(N_LAYERS)])
            self.frozen_values = nn.ParameterList([
                nn.Parameter(torch.randn(1, N_KV, 1, D, generator=g))
                for _ in range(N_LAYERS)])
            self.trainable_beta = nn.ParameterList([
                nn.Parameter(torch.zeros(1, N_KV, T_CART - 1), requires_grad=False)
                for _ in range(N_LAYERS)])
            self._num_frozen_tokens = 1
            self.bias_enabled = False

        def enable_attention_bias(self, enabled=True):
            self.bias_enabled = enabled

    cache = TinyCache().to(DEV)
    mask = GradientMask(
        granularity="per_layer",
        positions_per_layer={l: torch.arange(TOP_T) for l in range(N_LAYERS)},
        n_tokens=T_CART - 1, top_t=TOP_T,
    )
    g = torch.Generator().manual_seed(99)
    acc = AMQueryAccumulator(
        granularity="per_layer", n_layers=N_LAYERS, n_kv_heads=N_KV
    )
    for l in range(N_LAYERS):
        for _ in range(N_BATCH):
            acc._queries[l].append(torch.randn(1, N_KV, NQ, D, generator=g))
    doc_kv = {
        l: (
            torch.randn(N_KV, T_DOC, D, generator=g).to(DEV),
            torch.randn(N_KV, T_DOC, D, generator=g).to(DEV),
        )
        for l in range(N_LAYERS)
    }
    return cache, mask, acc, doc_kv


def _perturbed_accumulator(acc, scale: float, seed: int):
    """A fresh accumulator whose queries have moved -- the on-policy stand-in."""
    import copy

    g = torch.Generator().manual_seed(seed)
    new = copy.copy(acc)
    new._queries = {
        l: [q + scale * torch.randn(q.shape, generator=g) for q in qs]
        for l, qs in acc._queries.items()
    }
    return new


def run(module_ft, refresh_fn=None, **cfg_kwargs) -> dict:
    torch.manual_seed(123)
    cache, mask, acc, doc_kv = _build_inputs()
    cfg_cls = getattr(module_ft, "AttentionMatchingFinetuningConfig")
    cfg = cfg_cls(
        enabled=True, top_t=TOP_T, granularity="per_layer",
        delta_weight=1e-2, ridge_lambda=1e-4, ridge_scale="spectral",
        max_queries_per_head=64, compute_update_stats=True,
        enable_beta=False, rope_theta=THETA, key_mode="freeze", **cfg_kwargs,
    )
    extra_kwargs = {} if refresh_fn is None else {"onpolicy_refresh_fn": refresh_fn}
    stats = module_ft.apply_document_am_write_to_cache(
        cache=cache, mask=mask, query_accumulator=acc, doc_kv=doc_kv,
        config=cfg, n_layers=N_LAYERS, head_dim=D, **extra_kwargs,
    )
    k_all = torch.cat([k.detach().flatten() for k in cache.trainable_keys])
    v_all = torch.cat([v.detach().flatten() for v in cache.trainable_values])
    return {
        "mean_mse": repr(float(stats.mean_mse)),
        "mse_per_layer": {
            int(a): repr(float(b)) for a, b in stats.mse_per_layer.items()
        },
        "n_queries": int(stats.n_queries),
        "k_sum": repr(float(k_all.double().sum().item())),
        "v_sum": repr(float(v_all.double().sum().item())),
        "k_absmax": repr(float(k_all.abs().max().item())),
        "v_absmax": repr(float(v_all.abs().max().item())),
        "finite": bool(
            torch.isfinite(k_all).all().item() and torch.isfinite(v_all).all().item()
        ),
        "extra_keys": sorted(stats.extra.keys()),
        "mass_on_S": {
            int(a): repr(b)
            for a, b in (stats.extra.get("ref_mass_on_S_per_layer", {}) or {}).items()
        },
        "onpolicy": stats.extra.get("onpolicy"),
        "onpolicy_events": stats.extra.get("onpolicy_events"),
    }


_COMPARE_KEYS = (
    "mean_mse", "mse_per_layer", "n_queries", "k_sum", "v_sum",
    "k_absmax", "v_absmax", "mass_on_S", "extra_keys",
)


def _same(a: dict, b: dict, keys=_COMPARE_KEYS) -> bool:
    def norm(v):
        return {str(x): y for x, y in v.items()} if isinstance(v, dict) else v

    return all(norm(a.get(k)) == norm(b.get(k)) for k in keys)


# Numerics only: the on-policy arms legitimately add the `onpolicy` /
# `onpolicy_events` diagnostic keys to `extra`, so `extra_keys` is excluded when
# asking "did the restructuring change the MATH".
_NUM_KEYS = tuple(k for k in _COMPARE_KEYS if k != "extra_keys")


def arms() -> dict:
    import cartridges.am.finetune as ft

    res: dict = {}
    _, _, acc_ref, doc_kv_ref = _build_inputs()

    def boom(layer_idx, group_size):
        raise AssertionError("refresh_fn must NOT be called when the knob is off")

    def identity(layer_idx, group_size):
        # Same object -> "the activations did not move".
        return _rebuilt_accumulator(), None

    def _rebuilt_accumulator():
        _, _, a, _ = _build_inputs()
        return a

    def perturb(layer_idx, group_size):
        return _perturbed_accumulator(_rebuilt_accumulator(), 0.25, 1000 + layer_idx), None

    def perturb_with_dockv(layer_idx, group_size):
        a = _perturbed_accumulator(_rebuilt_accumulator(), 0.25, 1000 + layer_idx)
        g = torch.Generator().manual_seed(2000 + layer_idx)
        kv = {
            l: (
                (doc_kv_ref[l][0] + 0.1 * torch.randn(doc_kv_ref[l][0].shape,
                                                      generator=g).to(DEV)),
                (doc_kv_ref[l][1] + 0.1 * torch.randn(doc_kv_ref[l][1].shape,
                                                      generator=g).to(DEV)),
            )
            for l in range(N_LAYERS)
        }
        return a, kv

    res["off"] = run(ft)
    res["off_with_hook"] = run(ft, refresh_fn=boom, onpolicy_layers=0)
    res["on_identity_g1"] = run(ft, refresh_fn=identity, onpolicy_layers=1)
    res["on_perturbed_g1"] = run(ft, refresh_fn=perturb, onpolicy_layers=1)
    res["on_perturbed_g2"] = run(ft, refresh_fn=perturb, onpolicy_layers=2)
    res["on_perturbed_g4"] = run(ft, refresh_fn=perturb, onpolicy_layers=4)
    res["on_perturbed_dockv"] = run(
        ft, refresh_fn=perturb_with_dockv, onpolicy_layers=1,
        onpolicy_refresh_doc_kv=True,
    )

    res["_checks"] = {
        "off_with_hook_bit_identical_to_off": _same(res["off"], res["off_with_hook"]),
        "on_identity_numerics_bit_identical_to_off": _same(
            res["off"], res["on_identity_g1"], _NUM_KEYS
        ),
        "on_identity_adds_only_diagnostic_keys": sorted(
            set(res["on_identity_g1"]["extra_keys"]) - set(res["off"]["extra_keys"])
        ),
        "on_perturbed_differs_from_off": not _same(
            res["off"], res["on_perturbed_g1"], _NUM_KEYS
        ),
        "on_perturbed_finite": res["on_perturbed_g1"]["finite"],
        "off_has_no_onpolicy_key": "onpolicy" not in res["off"]["extra_keys"],
        "extra_keys_off": res["off"]["extra_keys"],
        "extra_keys_on": res["on_perturbed_g1"]["extra_keys"],
        "refresh_layers_g1": (res["on_perturbed_g1"]["onpolicy"] or {}).get(
            "refresh_layers"
        ),
        "refresh_layers_g2": (res["on_perturbed_g2"]["onpolicy"] or {}).get(
            "refresh_layers"
        ),
        "refresh_layers_g4": (res["on_perturbed_g4"]["onpolicy"] or {}).get(
            "refresh_layers"
        ),
        "identity_cos_is_one": (res["on_identity_g1"]["onpolicy"] or {}).get(
            "query_cos_mean"
        ),
        "identity_rel_l2_is_zero": (res["on_identity_g1"]["onpolicy"] or {}).get(
            "query_rel_l2_max"
        ),
        "perturbed_cos_mean": (res["on_perturbed_g1"]["onpolicy"] or {}).get(
            "query_cos_mean"
        ),
        "perturbed_rel_l2_mean": (res["on_perturbed_g1"]["onpolicy"] or {}).get(
            "query_rel_l2_mean"
        ),
        "dockv_flag": (res["on_perturbed_dockv"]["onpolicy"] or {}).get(
            "doc_kv_refreshed"
        ),
        "dockv_rel_l2": [
            e.get("doc_v_rel_l2")
            for e in (res["on_perturbed_dockv"]["onpolicy_events"] or [])
        ],
    }
    return res


def fail_loud() -> dict:
    """Every corruption mode must raise, not silently use stale queries."""
    import cartridges.am.finetune as ft

    def _rebuilt():
        _, _, a, _ = _build_inputs()
        return a

    def no_hook(*_):
        raise AssertionError

    cases: dict = {}

    def check(label, refresh_fn, **cfg):
        try:
            run(ft, refresh_fn=refresh_fn, **cfg)
            cases[label] = "NO_RAISE"
        except Exception as e:  # noqa: BLE001 - this is the point
            cases[label] = f"{type(e).__name__}: {str(e)[:110]}"

    # onpolicy_layers > 0 with no hook at all
    try:
        run(ft, onpolicy_layers=1)
        cases["no_hook"] = "NO_RAISE"
    except Exception as e:  # noqa: BLE001
        cases["no_hook"] = f"{type(e).__name__}: {str(e)[:110]}"

    def wrong_shape(layer_idx, group_size):
        a = _rebuilt()
        a._queries = {
            l: [q[:, :, : NQ // 2] for q in qs] for l, qs in a._queries.items()
        }
        return a, None

    def empty_layer(layer_idx, group_size):
        a = _rebuilt()
        a._queries = {l: [] for l in a._queries}
        return a, None

    def wrong_batch_count(layer_idx, group_size):
        a = _rebuilt()
        a._queries = {l: qs[:1] for l, qs in a._queries.items()}
        return a, None

    def nonfinite(layer_idx, group_size):
        a = _rebuilt()
        for l in a._queries:
            a._queries[l][0][0, 0, 0, 0] = float("nan")
        return a, None

    def not_a_tuple(layer_idx, group_size):
        return _rebuilt()

    def bad_dockv(layer_idx, group_size):
        a = _rebuilt()
        kv = {l: (torch.randn(N_KV, 5, D).to(DEV),
                  torch.randn(N_KV, 5, D).to(DEV)) for l in range(N_LAYERS)}
        return a, kv

    check("wrong_shape", wrong_shape, onpolicy_layers=1)
    check("empty_layer", empty_layer, onpolicy_layers=1)
    check("wrong_batch_count", wrong_batch_count, onpolicy_layers=1)
    check("nonfinite", nonfinite, onpolicy_layers=1)
    check("not_a_tuple", not_a_tuple, onpolicy_layers=1)
    check("bad_dockv_shape", bad_dockv, onpolicy_layers=1,
          onpolicy_refresh_doc_kv=True)
    cases["all_raised"] = all(
        v != "NO_RAISE" for k, v in cases.items() if k != "all_raised"
    )
    return cases


def config_check() -> dict:
    from cartridges.am.finetune import AttentionMatchingFinetuningConfig as C

    c = C()
    return {
        "fields_present": [
            f in C.model_fields for f in ("onpolicy_layers", "onpolicy_refresh_doc_kv")
        ],
        "defaults": [c.onpolicy_layers, c.onpolicy_refresh_doc_kv],
        "constructs_on": [
            C(onpolicy_layers=4).onpolicy_layers,
            C(onpolicy_layers=4, onpolicy_refresh_doc_kv=True).onpolicy_refresh_doc_kv,
        ],
    }


def off_vs_head() -> dict:
    """Run the OFF arm against a `git archive HEAD` snapshot in a subprocess.

    Loading `finetune.py` twice in one process breaks pydrantic's BaseConfig
    machinery, so the committed code is exercised in its own interpreter with
    PYTHONPATH pinned at the snapshot (RUNBOOK 9c-bis).
    """
    snap = Path("/tmp/amsnap_mechseq_head")
    if snap.exists():
        shutil.rmtree(snap)
    snap.mkdir(parents=True)
    subprocess.run(
        f'git -C "{REPO}" archive HEAD cartridges | tar -x -C "{snap}"',
        shell=True, check=True,
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(snap)
    env["CARTRIDGES_DIR"] = str(snap)
    args = [sys.executable, str(Path(__file__).resolve()), "--off-only"]
    if DEV == "cuda":
        args.append("--cuda")
    out = subprocess.run(args, env=env, capture_output=True, text=True, cwd="/tmp")
    if out.returncode != 0:
        return {"error": out.stderr[-2000:]}
    head_res = json.loads(out.stdout)
    return {
        "head_import_path": head_res.get("cartridges_import_path"),
        "head": head_res["off_only"],
    }


def main() -> None:
    import cartridges

    OUT["cartridges_import_path"] = os.path.dirname(cartridges.__file__)
    if "--off-only" in sys.argv:
        import cartridges.am.finetune as ft

        OUT["off_only"] = run(ft)
        print(json.dumps(OUT, indent=2, default=str))
        return

    OUT["config"] = config_check()
    OUT["arms"] = arms()
    OUT["fail_loud"] = fail_loud()

    fh = off_vs_head()
    if "head" in fh:
        mine = OUT["arms"]["off"]
        fh["new_extra_keys"] = mine["extra_keys"]
        fh["head_extra_keys"] = fh["head"]["extra_keys"]
        fh["bit_identical"] = _same(fh["head"], mine)
    OUT["off_vs_HEAD"] = fh
    print(json.dumps(OUT, indent=2, default=str))


if __name__ == "__main__":
    main()

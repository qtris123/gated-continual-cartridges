"""MECH-KEYS Part-A sanity check (unit level, no training run).

Proves four things about the H2 fix (`AM_KEY_REPOSITION` -> `key_reposition`):

1. **The headline numeric check the orchestrator asked for.** A document key that
   wins selection is scored in `_attention_scores` against a query rotated FORWARD
   by `doc_rope_offset = T_doc`; once installed into a cartridge slot the student
   (and eval) score it with the RAW query. With `reposition=True` the installed
   key's dot product against the raw query must equal the dot product the document
   key achieved against the rotated query, i.e.
       <q, R_{-Δ} k_doc>  ==  <R_Δ q, k_doc>,   Δ = doc_rope_offset.
   Measured both at the primitive level and end-to-end through
   `rewrite_keys_on_support`, off vs on.

2. **Bit-identity when the flag is off.** `rewrite_keys_on_support` at default
   arguments reproduces the committed (pre-edit) implementation loaded from
   `git show HEAD:` into a temp module, to the last bit.

3. **The counter-rotation uses the model's own rotary base.** Doing it at the AM
   package's historical 10000.0 while the model runs 5e6 replaces one frame error
   with another -- reported as a third arm.

4. **End-to-end** `apply_document_am_write_to_cache` on a stub cache with
   `key_mode="highest_attention"`, reposition off vs on: finite everywhere, the
   `key_rewrite` diagnostics are populated, and `key_mode="freeze"` is bit-identical
   to HEAD (so no frozen-key run can change).

Usage:  python research_loop/results/MECH-KEYS/sanity_keys.py [--cuda]
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
DEV = "cuda" if ("--cuda" in sys.argv and torch.cuda.is_available()) else "cpu"
THETA = 5000000.0
OUT: dict = {"device": DEV, "rope_theta": THETA}


def load_head_module(relpath: str, name: str):
    """Import a committed source file under a private module name."""
    src = subprocess.check_output(
        ["git", "-C", str(REPO), "show", f"HEAD:{relpath}"], text=True
    )
    tmp = Path(tempfile.mkdtemp()) / f"{name}.py"
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location(name, tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def primitive_check() -> dict:
    """<q, R_{-D} k> vs <R_D q, k> for the exact operator pair used in the fix."""
    from cartridges.am.core import _apply_rope_offset_to_queries, _inv_sqrt_d
    from cartridges.am.phase1 import _rope_reposition

    D = 128
    res = {}
    for offset in (3858, 4000, 8900):
        g = torch.Generator().manual_seed(11)
        q = torch.randn(64, D, generator=g).to(DEV)
        k = torch.randn(32, D, generator=g).to(DEV)
        m = torch.arange(32, dtype=torch.float32, device=DEV)  # prefill positions
        inv = _inv_sqrt_d(D)

        # selection frame: query rotated forward by `offset`
        q_rot = _apply_rope_offset_to_queries(q, offset, D, rope_theta=THETA)
        s_sel = (q_rot @ k.T).float() * inv

        # student frame, key installed AS-IS (the H2 bug)
        s_raw = (q @ k.T).float() * inv

        # student frame, key counter-rotated to `m - offset`
        k_reb = _rope_reposition(
            k, m, m - float(offset), D, rope_theta=THETA
        )
        s_fix = (q @ k_reb.T).float() * inv

        denom = s_sel.abs().max().item()
        res[str(offset)] = {
            "max_abs_err_uncorrected": float((s_raw - s_sel).abs().max().item()),
            "max_abs_err_repositioned": float((s_fix - s_sel).abs().max().item()),
            "rel_err_uncorrected": float((s_raw - s_sel).abs().max().item() / denom),
            "rel_err_repositioned": float((s_fix - s_sel).abs().max().item() / denom),
            "corr_uncorrected": float(
                torch.corrcoef(torch.stack([s_raw.flatten(), s_sel.flatten()]))[0, 1]
            ),
            "corr_repositioned": float(
                torch.corrcoef(torch.stack([s_fix.flatten(), s_sel.flatten()]))[0, 1]
            ),
            "score_scale_maxabs": denom,
        }
        # wrong theta on the counter-rotation -> a NEW frame error (MECH-003)
        k_wrong = _rope_reposition(k, m, m - float(offset), D, rope_theta=10000.0)
        s_wrong = (q @ k_wrong.T).float() * inv
        res[str(offset)]["max_abs_err_repositioned_theta1e4"] = float(
            (s_wrong - s_sel).abs().max().item()
        )
    return res


def rewrite_check(dtype=torch.float32) -> dict:
    """End-to-end through `rewrite_keys_on_support`, off vs on."""
    from cartridges.am.core import _attention_scores, _inv_sqrt_d
    from cartridges.am import key_select as ks

    D, T_CART, T_DOC, NQ, TOP_T = 128, 512, 4000, 64, 32
    g = torch.Generator().manual_seed(5)
    keys = (torch.randn(T_CART, D, generator=g) * 0.5).to(DEV).to(dtype)
    q = (torch.randn(NQ, D, generator=g) * 0.5).to(DEV).to(dtype)
    k_doc = (torch.randn(T_DOC, D, generator=g) * 0.5).to(DEV).to(dtype)
    sel = torch.arange(TOP_T, device=DEV)
    # Give the cartridge candidates a larger norm so SOME of them survive selection
    # -- otherwise the mixed-provenance case (cartridge rows must NOT be rotated)
    # is never exercised against a 4000-row document block.
    keys = keys.clone()
    keys[sel[: TOP_T // 2]] = keys[sel[: TOP_T // 2]] * 3.0
    cand = torch.cat([keys[sel], k_doc], dim=0)
    inv = _inv_sqrt_d(D)

    # the scores the selector itself saw (selection frame)
    s_sel = _attention_scores(
        q, cand, D, doc_key_start=TOP_T, doc_rope_offset=T_DOC, rope_theta=THETA
    )
    _, _, chosen = ks.select_keys_highest_attention(
        cand, q, TOP_T, D, score_method="rms",
        doc_key_start=TOP_T, doc_rope_offset=T_DOC, rope_theta=THETA,
    )
    chosen_t = torch.tensor(chosen, device=DEV)
    from_doc = chosen_t >= TOP_T

    out = {
        "dtype": str(dtype),
        "n_selected": int(TOP_T),
        "n_from_doc": int(from_doc.sum().item()),
        "n_from_cartridge": int((~from_doc).sum().item()),
    }
    for label, repos in (("off", False), ("on", True)):
        info: dict = {}
        new_keys = ks.rewrite_keys_on_support(
            keys, sel, cand, q, mode="highest_attention", head_dim=D,
            doc_key_start=TOP_T, doc_rope_offset=T_DOC, rope_theta=THETA,
            reposition=repos, info=info,
        )
        # `out[sel[r]] = new_k[r]` -> row r of the install corresponds to candidate
        # `chosen[r]`; score it the way the STUDENT will (raw query, no offset).
        installed = new_keys[sel]
        s_stu = (q @ installed.T).float() * inv          # (NQ, TOP_T)
        s_want = s_sel[:, chosen_t].float()              # (NQ, TOP_T)
        d_doc = (s_stu[:, from_doc] - s_want[:, from_doc]).abs()
        d_cart = (s_stu[:, ~from_doc] - s_want[:, ~from_doc]).abs()
        out[label] = {
            "info": info,
            "doc_rows_max_abs_score_err": float(d_doc.max().item()) if d_doc.numel() else None,
            "doc_rows_mean_abs_score_err": float(d_doc.mean().item()) if d_doc.numel() else None,
            "cartridge_rows_max_abs_score_err": float(d_cart.max().item()) if d_cart.numel() else None,
            "score_scale_maxabs": float(s_want.abs().max().item()),
            "keys_finite": bool(torch.isfinite(new_keys).all().item()),
            "key_norm_mean": float(new_keys[sel].float().norm(dim=-1).mean().item()),
        }
        if label == "off":
            out["off"]["softmax_mass_moved"] = None
    # what the phase error does to the routing the value solve was fitted against
    return out


def bit_identity_check() -> dict:
    """`reposition=False` must reproduce the committed implementation exactly."""
    from cartridges.am import key_select as ks

    head = load_head_module("cartridges/am/key_select.py", "key_select_head")
    D, T_CART, T_DOC, NQ, TOP_T = 128, 256, 1500, 48, 16
    res = {}
    for mode in ("highest_attention", "omp"):
        g = torch.Generator().manual_seed(9)
        keys = (torch.randn(T_CART, D, generator=g) * 0.5).to(DEV)
        q = (torch.randn(NQ, D, generator=g) * 0.5).to(DEV)
        k_doc = (torch.randn(T_DOC, D, generator=g) * 0.5).to(DEV)
        sel = torch.arange(TOP_T, device=DEV)
        cand = torch.cat([keys[sel], k_doc], dim=0)
        kw = dict(
            mode=mode, head_dim=D, doc_key_start=TOP_T,
            doc_rope_offset=T_DOC, rope_theta=THETA,
        )
        # burn one lstsq call (first-call BLAS path differs by ~4e-8)
        head.rewrite_keys_on_support(keys, sel, cand, q, **kw)
        a = head.rewrite_keys_on_support(keys, sel, cand, q, **kw)
        b = ks.rewrite_keys_on_support(keys, sel, cand, q, **kw)
        c = head.rewrite_keys_on_support(keys, sel, cand, q, **kw)
        res[mode] = {
            "old_vs_old_bit_identical": bool(torch.equal(a, c)),
            "new_default_bit_identical": bool(torch.equal(a, b)),
            "max_abs_diff": float((a - b).abs().max().item()),
        }
        # and the flag ON must actually change something
        d = ks.rewrite_keys_on_support(
            keys, sel, cand, q, reposition=True, **kw
        )
        res[mode]["reposition_changes_output"] = not bool(torch.equal(b, d))
        res[mode]["reposition_max_abs_diff"] = float((b - d).abs().max().item())
    return res


def end_to_end() -> dict:
    """`apply_document_am_write_to_cache` with key_mode freeze / highattn +-repos."""
    import torch.nn as nn

    from cartridges.am.finetune import (
        AttentionMatchingFinetuningConfig,
        apply_document_am_write_to_cache,
    )
    from cartridges.am.query_accum import AMQueryAccumulator
    from cartridges.sparse_cache_finetuning import GradientMask

    head_ft = None
    N_LAYERS, N_KV, T_CART, T_DOC, NQ, TOP_T, D = 2, 2, 40, 4000, 24, 8, 128

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

    def run(module_ft, **cfg_kwargs):
        torch.manual_seed(123)
        cache = TinyCache()
        mask = GradientMask(
            granularity="per_layer",
            positions_per_layer={l: torch.arange(TOP_T) for l in range(N_LAYERS)},
            n_tokens=T_CART - 1, top_t=TOP_T,
        )
        acc = AMQueryAccumulator(
            granularity="per_layer", n_layers=N_LAYERS, n_kv_heads=N_KV
        )
        for l in range(N_LAYERS):
            acc._queries[l].append(torch.randn(1, N_KV, NQ, D))
        doc_kv = {
            l: (torch.randn(N_KV, T_DOC, D), torch.randn(N_KV, T_DOC, D))
            for l in range(N_LAYERS)
        }
        cfg_cls = getattr(module_ft, "AttentionMatchingFinetuningConfig")
        cfg = cfg_cls(
            enabled=True, top_t=TOP_T, granularity="per_layer",
            delta_weight=1e-2, ridge_lambda=1e-4, ridge_scale="spectral",
            max_queries_per_head=64, compute_update_stats=True,
            enable_beta=False, rope_theta=THETA, **cfg_kwargs,
        )
        stats = module_ft.apply_document_am_write_to_cache(
            cache=cache, mask=mask, query_accumulator=acc, doc_kv=doc_kv,
            config=cfg, n_layers=N_LAYERS, head_dim=D,
        )
        k_all = torch.cat([k.detach().flatten() for k in cache.trainable_keys])
        v_all = torch.cat([v.detach().flatten() for v in cache.trainable_values])
        return stats, k_all, v_all

    import cartridges.am.finetune as ft

    res = {}
    if "--freeze-only" in sys.argv:
        s, k_all, v_all = run(ft, key_mode="freeze")
        return {
            "freeze_only": {
                "mean_mse": repr(float(s.mean_mse)),
                "mse_per_layer": {
                    int(a): repr(float(b)) for a, b in s.mse_per_layer.items()
                },
                "k_sum": repr(float(k_all.double().sum().item())),
                "v_sum": repr(float(v_all.double().sum().item())),
                "k_absmax": repr(float(k_all.abs().max().item())),
                "v_absmax": repr(float(v_all.abs().max().item())),
                "extra_keys": sorted(s.extra.keys()),
                "mass_on_S": {
                    int(a): repr(b)
                    for a, b in (
                        s.extra.get("ref_mass_on_S_per_layer", {}) or {}
                    ).items()
                },
            }
        }
    for label, kw in (
        ("freeze", dict(key_mode="freeze")),
        ("highattn_norepos", dict(key_mode="highest_attention", key_reposition=False)),
        ("highattn_repos", dict(key_mode="highest_attention", key_reposition=True)),
        ("omp_repos", dict(key_mode="omp", key_reposition=True)),
    ):
        s, k_all, v_all = run(ft, **kw)
        res[label] = {
            "mean_mse": float(s.mean_mse),
            "finite": bool(
                torch.isfinite(k_all).all().item()
                and torch.isfinite(v_all).all().item()
            ),
            "v_absmax": float(v_all.abs().max().item()),
            "k_absmax": float(k_all.abs().max().item()),
            "mass_on_S": {
                int(a): round(b, 6)
                for a, b in (s.extra.get("ref_mass_on_S_per_layer", {}) or {}).items()
            },
            "key_rewrite": s.extra.get("key_rewrite"),
            "key_rewrite_per_layer": s.extra.get("key_rewrite_per_layer"),
        }
    return res


def freeze_vs_head() -> dict:
    """Run the freeze stub against a `git archive HEAD` snapshot in a subprocess.

    Loading `finetune.py` twice in one process breaks pydrantic's BaseConfig
    machinery, so the committed code is exercised in its own interpreter with
    PYTHONPATH pinned at the snapshot (RUNBOOK 9c-bis).
    """
    import os
    import shutil

    snap = Path("/tmp/amsnap_mechkeys_head")
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
    args = [sys.executable, str(Path(__file__).resolve()), "--freeze-only", "--e2e-only"]
    if DEV == "cuda":
        args.append("--cuda")
    out = subprocess.run(args, env=env, capture_output=True, text=True, cwd="/tmp")
    if out.returncode != 0:
        return {"error": out.stderr[-2000:]}
    head_res = json.loads(out.stdout)
    return {
        "head_import_path": head_res.get("cartridges_import_path"),
        "head": head_res["end_to_end"]["freeze_only"],
    }


def config_check() -> dict:
    from cartridges.am.finetune import AttentionMatchingFinetuningConfig as C

    return {
        "default_key_reposition": C().key_reposition,
        "field_present": "key_reposition" in C.model_fields,
        "constructs_on": C(key_mode="highest_attention", key_reposition=True).key_reposition,
        "should_fit_beta_highattn_unset": __import__(
            "cartridges.am.finetune", fromlist=["_should_fit_beta"]
        )._should_fit_beta(C(key_mode="highest_attention")),
        "should_fit_beta_highattn_off": __import__(
            "cartridges.am.finetune", fromlist=["_should_fit_beta"]
        )._should_fit_beta(C(key_mode="highest_attention", enable_beta=False)),
    }


def main() -> None:
    import cartridges
    import os as _os

    OUT["cartridges_import_path"] = _os.path.dirname(cartridges.__file__)
    if "--e2e-only" in sys.argv:
        OUT["end_to_end"] = end_to_end()
        print(json.dumps(OUT, indent=2, default=str))
        return
    OUT["primitive"] = primitive_check()
    OUT["rewrite_fp32"] = rewrite_check(torch.float32)
    OUT["rewrite_bf16"] = rewrite_check(torch.bfloat16)
    OUT["bit_identity"] = bit_identity_check()
    OUT["config"] = config_check()
    OUT["end_to_end"] = end_to_end()
    # freeze path vs the committed tree, in a separate interpreter
    fh = freeze_vs_head()
    if "head" in fh:
        mine = end_to_end_freeze_only()
        fh["new"] = mine

        def _norm(v):
            return {str(a): b for a, b in v.items()} if isinstance(v, dict) else v

        fh["bit_identical"] = all(
            _norm(fh["head"].get(k)) == _norm(mine.get(k))
            for k in ("mean_mse", "k_sum", "v_sum", "k_absmax", "v_absmax",
                      "mse_per_layer", "mass_on_S", "extra_keys")
        )
        fh["extra_keys_head"] = fh["head"]["extra_keys"]
        fh["extra_keys_new"] = mine["extra_keys"]
    OUT["freeze_vs_HEAD"] = fh
    print(json.dumps(OUT, indent=2, default=str))


def end_to_end_freeze_only() -> dict:
    sys.argv.append("--freeze-only")
    try:
        return end_to_end()["freeze_only"]
    finally:
        sys.argv.remove("--freeze-only")


if __name__ == "__main__":
    main()

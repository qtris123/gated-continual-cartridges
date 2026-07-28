"""MECH-BETA Part-A sanity check (unit level, no training run).

Three things are proved here:

1. **Bit-identity when the flags are off.** `nnls_projected_gradient` and
   `refit_beta_nnls` called with their default arguments reproduce the committed
   (pre-edit) implementation, loaded from `git show HEAD:` into a temp module,
   to the last bit on well-conditioned input.
2. **The named failure mode is real and is fixed.** A rank-deficient 64x64 Phi
   (the EXP-005b/EXP-006 shape) driven through the default `gels` warm start
   returns NaN *without raising*; with `driver='gelsd'` + the paper's box the fit
   returns finite, boxed weights.
3. **`refit_beta_nnls` end-to-end** on a rank-deficient synthetic head returns a
   finite beta inside [-box, +box], and the info dict carries the lstsq rank.

Usage:  python research_loop/results/MECH-BETA/sanity_beta.py [--cuda]
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
OUT: dict = {"device": DEV}


def load_head_version():
    """Import the committed `key_select.py` under a private module name."""
    src = subprocess.check_output(
        ["git", "-C", str(REPO), "show", "HEAD:cartridges/am/key_select.py"],
        text=True,
    )
    tmp = Path(tempfile.mkdtemp()) / "key_select_head.py"
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location("key_select_head", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    from cartridges.am import key_select as ks

    head = load_head_version()
    torch.manual_seed(0)

    # ---------------------------------------------------------- 1. bit-identity
    g = torch.Generator(device="cpu").manual_seed(1234)
    Phi = torch.rand(64, 32, generator=g).to(DEV)          # well conditioned
    tgt = torch.rand(64, generator=g).to(DEV)
    # The FIRST lstsq/SVD call in a process takes a different BLAS path and can
    # differ from later calls by ~4e-8 (verified: old-vs-old shows the same 3.7e-8
    # on call 1 and 0.0 afterwards). Burn one call before comparing.
    head.nnls_projected_gradient(Phi, tgt, n_iters=200)
    w_old0 = head.nnls_projected_gradient(Phi, tgt, n_iters=200)
    w_new = ks.nnls_projected_gradient(Phi, tgt, n_iters=200)
    w_old = head.nnls_projected_gradient(Phi, tgt, n_iters=200)
    OUT["nnls_old_vs_old_bit_identical"] = bool(torch.equal(w_old0, w_old))
    OUT["nnls_default_bit_identical"] = bool(torch.equal(w_new, w_old))
    OUT["nnls_default_max_abs_diff"] = float((w_new - w_old).abs().max().item())

    keys = torch.randn(96, 128, generator=g).to(DEV)
    qs = torch.randn(64, 128, generator=g).to(DEV)
    tlm = torch.randn(64, generator=g).to(DEV).abs() + 3.0
    sel = torch.arange(0, 32, device=DEV)
    b_new = ks.refit_beta_nnls(keys, qs, tlm, 128, selected_indices=sel)
    b_old = head.refit_beta_nnls(keys, qs, tlm, 128, selected_indices=sel)
    OUT["refit_default_bit_identical"] = bool(torch.equal(b_new, b_old))
    OUT["refit_default_max_abs_diff"] = float((b_new - b_old).abs().max().item())

    # ------------------------------------- 2. rank-deficient 64x64 (the EXP-006 shape)
    # Rank 8 out of 64: exactly the "most entries underflow toward 0 after the
    # row-shift" regime LIT-002 named as the birthplace of the NaN.
    A = torch.rand(64, 8, generator=g)
    B = torch.rand(8, 64, generator=g)
    Phi_rd = (A @ B).to(DEV)
    tgt_rd = torch.rand(64, generator=g).to(DEV)

    raised = None
    try:
        sol_gels = torch.linalg.lstsq(Phi_rd, tgt_rd)
        w_gels = sol_gels.solution
        raised = False
    except RuntimeError as e:
        w_gels, raised = None, str(e)
    OUT["gels_raised"] = raised
    OUT["gels_warm_start_finite"] = (
        None if w_gels is None else bool(torch.isfinite(w_gels).all())
    )
    OUT["gels_warm_start_n_nonfinite"] = (
        None if w_gels is None else int((~torch.isfinite(w_gels)).sum().item())
    )

    # old code path on the same input
    try:
        w_old_rd = head.nnls_projected_gradient(Phi_rd, tgt_rd, n_iters=200)
        OUT["old_path_finite"] = bool(torch.isfinite(w_old_rd).all())
        OUT["old_path_beta_min"] = float(
            torch.log(w_old_rd.clamp(min=1e-12)).min().item()
        )
        OUT["old_path_beta_max"] = float(
            torch.log(w_old_rd.clamp(min=1e-12)).max().item()
        )
    except RuntimeError as e:
        OUT["old_path_error"] = str(e)

    # new code path: gelsd + box
    info: dict = {}
    import math

    w_fix = ks.nnls_projected_gradient(
        Phi_rd, tgt_rd, n_iters=2,
        lower_bound=math.exp(-3.0), upper_bound=math.exp(3.0),
        driver="gelsd", info=info,
    )
    OUT["fixed_path_info"] = info
    OUT["fixed_path_finite"] = bool(torch.isfinite(w_fix).all())
    beta_fix = torch.log(w_fix)
    OUT["fixed_path_beta_min"] = float(beta_fix.min().item())
    OUT["fixed_path_beta_max"] = float(beta_fix.max().item())
    OUT["fixed_path_in_box"] = bool(
        (beta_fix >= -3.0 - 1e-5).all() and (beta_fix <= 3.0 + 1e-5).all()
    )

    # ------------------------------- 3. refit_beta_nnls on a rank-deficient head
    # 64 queries in an 8-dim subspace -> the score matrix (and hence Phi) is rank 8.
    sub = torch.randn(8, 128, generator=g).to(DEV)
    qs_rd = (torch.randn(64, 8, generator=g).to(DEV) @ sub)
    keys_rd = torch.randn(64, 128, generator=g).to(DEV) * 0.3
    # a teacher log-mass well above the student's (as the [C||D] teacher always is)
    tlm_rd = torch.logsumexp(
        (qs_rd @ keys_rd.T).float() * (1.0 / 128) ** 0.5, dim=-1
    ) + 1.5
    sel_rd = torch.arange(0, 32, device=DEV)

    info2: dict = {}
    beta = ks.refit_beta_nnls(
        keys_rd, qs_rd, tlm_rd, 128,
        selected_indices=sel_rd,
        n_iters=2, beta_box=3.0, nnls_driver="gelsd",
        target_mode="residual", info=info2,
    )
    OUT["refit_boxed_info"] = info2
    OUT["refit_boxed_finite"] = bool(torch.isfinite(beta).all())
    bsel = beta[sel_rd]
    OUT["refit_boxed_beta"] = {
        "min": float(bsel.min().item()),
        "median": float(bsel.median().item()),
        "max": float(bsel.max().item()),
    }
    OUT["refit_boxed_in_box"] = bool(
        (bsel >= -3.0 - 1e-5).all() and (bsel <= 3.0 + 1e-5).all()
    )
    OUT["refit_boxed_nonselected_untouched"] = bool(
        torch.equal(beta[32:], torch.zeros_like(beta[32:]))
    )

    # old path on the same rank-deficient head
    try:
        beta_old = head.refit_beta_nnls(
            keys_rd, qs_rd, tlm_rd, 128, selected_indices=sel_rd
        )
        OUT["refit_old_finite"] = bool(torch.isfinite(beta_old).all())
        OUT["refit_old_beta"] = {
            "min": float(beta_old[sel_rd].min().item()),
            "max": float(beta_old[sel_rd].max().item()),
        }
    except RuntimeError as e:
        OUT["refit_old_error"] = str(e)

    # --------------------------------------------------- 4. _should_fit_beta decoupling
    from cartridges.am.finetune import (
        AttentionMatchingFinetuningConfig as C,
        _should_fit_beta,
    )

    OUT["should_fit_beta"] = {
        "freeze_unset": _should_fit_beta(C(key_mode="freeze")),
        "freeze_on": _should_fit_beta(C(key_mode="freeze", enable_beta=True)),
        "highattn_unset": _should_fit_beta(C(key_mode="highest_attention")),
        "highattn_off": _should_fit_beta(
            C(key_mode="highest_attention", enable_beta=False)
        ),
    }
    OUT["config_defaults"] = {
        "beta_box": C().beta_box,
        "nnls_iters": C().nnls_iters,
        "nnls_driver": C().nnls_driver,
        "beta_target": C().beta_target,
    }
    OUT["config_constructs_with_flags"] = bool(
        C(beta_box=3.0, nnls_iters=2, nnls_driver="gelsd", beta_target="residual")
        .beta_box
        == 3.0
    )

    # ------------------------------------- 5. end-to-end write on a stub cache
    OUT["end_to_end"] = end_to_end()

    print(json.dumps(OUT, indent=2, default=str))


def end_to_end() -> dict:
    """`apply_document_am_write_to_cache` with beta OFF vs boxed-beta ON."""
    import torch.nn as nn

    from cartridges.am.finetune import (
        AttentionMatchingFinetuningConfig,
        apply_document_am_write_to_cache,
    )
    from cartridges.am.query_accum import AMQueryAccumulator
    from cartridges.sparse_cache_finetuning import GradientMask

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

    def run(**cfg_kwargs):
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
        cfg = AttentionMatchingFinetuningConfig(
            enabled=True, top_t=TOP_T, granularity="per_layer", key_mode="freeze",
            delta_weight=1e-2, ridge_lambda=1e-4, ridge_scale="spectral",
            max_queries_per_head=64, compute_update_stats=True,
            rope_theta=5000000.0, **cfg_kwargs,
        )
        stats = apply_document_am_write_to_cache(
            cache=cache, mask=mask, query_accumulator=acc, doc_kv=doc_kv,
            config=cfg, n_layers=N_LAYERS, head_dim=D,
        )
        beta_all = torch.cat([b.detach().flatten() for b in cache.trainable_beta])
        return stats, cache, beta_all

    s_off, c_off, b_off = run(enable_beta=False)
    s_on, c_on, b_on = run(
        enable_beta=True, beta_box=3.0, nnls_iters=2,
        nnls_driver="gelsd", beta_target="residual",
    )
    mass_off = s_off.extra.get("ref_mass_on_S_per_layer", {})
    mass_on = s_on.extra.get("ref_mass_on_S_per_layer", {})
    return {
        "off": {
            "mean_mse": s_off.mean_mse,
            "mass_on_S": {int(k): round(v, 6) for k, v in mass_off.items()},
            "beta_all_zero": bool(torch.all(b_off == 0).item()),
            "bias_enabled": c_off.bias_enabled,
            "extra_has_beta": "beta" in s_off.extra,
        },
        "on": {
            "mean_mse": s_on.mean_mse,
            "mse_finite": bool(s_on.mean_mse == s_on.mean_mse),
            "mass_on_S": {int(k): round(v, 6) for k, v in mass_on.items()},
            "bias_enabled": c_on.bias_enabled,
            "beta_finite": bool(torch.isfinite(b_on).all().item()),
            "beta_nonzero": int((b_on != 0).sum().item()),
            "beta_min": float(b_on.min().item()),
            "beta_max": float(b_on.max().item()),
            "extra_beta": s_on.extra.get("beta"),
            "extra_beta_per_layer": s_on.extra.get("beta_per_layer"),
        },
        "mass_on_S_ratio_on_over_off": {
            int(k): round(mass_on[k] / mass_off[k], 4)
            for k in mass_off if mass_off.get(k)
        },
    }


if __name__ == "__main__":
    main()

"""DIAG-NOISE: exact per-example CE recovery for cached AM checkpoints.

Replicates cartridges.train.evaluate_perplexity's arithmetic token-for-token but
accumulates the per-token CE into per-EXAMPLE (per dataset element) buckets, so the
reported token-weighted mean CE is recovered exactly as
    L = sum_e c_e / sum_e t_e
with c_e the summed CE over example e's scored tokens and t_e its token count.

No source file is edited; imports are pinned to a frozen git-archive snapshot.
"""
import json
import os
import sys
import time

SNAP = "/tmp/amsnap_DIAG-NOISE"
sys.path.insert(0, SNAP)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import cartridges
from cartridges.cache import TrainableCache
from cartridges.datasets import DataSource, LossEvalDataset
from cartridges.models import FlexQwen3ForCausalLM, HFModelConfig
from cartridges.train import CacheAndModel, _collate_first
from cartridges.utils import seed_everything

print("cartridges import path:", os.path.dirname(cartridges.__file__), flush=True)
assert os.path.dirname(cartridges.__file__) == SNAP + "/cartridges", "PIN FAILED"

REPO = "/localhome/local-triv/gated-continual-cartridges_explore"
DATA = f"{REPO}/data/qasper/eval"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
OUT = os.environ.get("PEROUT", "/tmp/diag_noise/perexample.json")

CTRL = f"{REPO}/outputs/2026-07-28-23-19-56-continual_am_sparse/508b8b0f-075a-429b-b757-2b11bfb57124"
KEYS = f"{REPO}/outputs/2026-07-28-23-37-22-continual_am_sparse/a20bc87b-e681-4122-90ed-24b3495f6849"
SEQ = f"{REPO}/outputs/2026-07-28-21-43-17-continual_am_sparse/4b0edcab-d1c5-4313-bc43-309d7ff77097"
CONB = f"{REPO}/outputs/2026-07-28-22-52-30-continual_am_sparse/03b02326-28dc-4eeb-973b-13bd100b5c3c"
ONPK = f"{REPO}/outputs/2026-07-29-02-27-29-continual_am_sparse/8299a2c7-7e6d-4016-a3f7-49d8d88f962f"

# label -> (path, published QA, published MT, provenance)
CKPTS = [
    ("phase1_k0", f"{REPO}/outputs/phase1_selfdistill_qwen512/cache_last.pt",
     2.23880672454834, 3.7825491428375244, "Phase-1 self-distilled cartridge (k=0 anchor)"),
    ("ctrl_k8", f"{CTRL}/cache-after-doc-007-e1a7f420.pt",
     1.9329710006713867, 2.453585624694824, "frozen-key control theta=5e6, k=8 (DIAG-KEYCURVE's edge point)"),
    ("ctrl_k10", f"{CTRL}/cache-after-doc-009-f77cecfb.pt",
     1.9674913883209229, 2.4083292484283447, "frozen-key control theta=5e6, k=10 = its TRUE MT optimum"),
    ("ctrl_k12", f"{CTRL}/cache-after-doc-011-cfcb19a9.pt",
     2.074141263961792, 2.4704575538635254, "frozen-key control theta=5e6, k=12 (matched-k)"),
    ("ctrl_k16", f"{CTRL}/cache-after-doc-015-af07b880.pt",
     2.159724712371826, 2.529625177383423, "frozen-key control theta=5e6, k=16 endpoint"),
    ("keys_k12", f"{KEYS}/cache-after-doc-011-cfcb19a9.pt",
     1.9560121297836304, 2.2720184326171875, "keys+reposition (MECH-005), k=12 = its optimum"),
    ("keys_k16", f"{KEYS}/cache-after-doc-015-af07b880.pt",
     2.034916400909424, 2.330503463745117, "keys+reposition (MECH-005), k=16 endpoint"),
    ("seq1e4_k8", f"{SEQ}/cache-after-doc-007-e1a7f420.pt",
     2.0471396446228027, 2.709873914718628, "value-only theta=1e4 canonical (DIAG-SEQUENCE), k=8"),
    ("seq1e4_k12", f"{SEQ}/cache-after-doc-011-cfcb19a9.pt",
     2.0421719551086426, 2.435236930847168, "value-only theta=1e4 canonical, k=12 = MT min"),
    ("seq1e4_k16", f"{SEQ}/cache-after-doc-015-af07b880.pt",
     2.1771795749664307, 2.5524158477783203, "value-only theta=1e4 canonical, k=16 endpoint"),
    ("contentB_k8", f"{CONB}/cache-after-doc-007-85188ea2.pt",
     None, 2.9978, "DIAG-CONTENT arm B (QA-topic corpus written, MT scored), k=8"),
    ("contentB_k16", f"{CONB}/cache-after-doc-015-8f0f132c.pt",
     2.8909, 3.4343, "DIAG-CONTENT arm B, k=16"),
    ("onpolkeys_k12", f"{ONPK}/cache-after-doc-011-cfcb19a9.pt",
     2.1439507007598877, 2.4483301639556885, "MECH-SEQUENTIAL on-policy + keys, k=12"),
]


ROPEA = f"{REPO}/outputs/2026-07-28-21-18-06-continual_am_sparse/f705f46c-f9ed-4340-b6a6-8dcae568ea28"
ROPEB = f"{REPO}/outputs/2026-07-28-21-23-35-continual_am_sparse/5ff66d3f-84dc-466e-b539-d2ef01a53785"
BON = f"{REPO}/outputs/2026-07-28-22-21-02-continual_am_sparse/0525f8ea-f26b-4301-b80e-2d3e77e80d90"

CKPTS = [
    ("seq1e4_step16", f"{SEQ}/cache-step16.pt", 2.1766157150268555, 2.5483615398406982, "canonical theta=1e4 run 4b0edcab, cache-step16.pt (the run's own final file)"),
    ("rope1e4_step16", f"{ROPEA}/cache-step16.pt", 2.17662, 2.54836, "DIAG-ROPE arm A cache-step16.pt"),
    ("rope5e6_step16", f"{ROPEB}/cache-step16.pt", 2.15320, 2.52961, "DIAG-ROPE arm B cache-step16.pt"),
    ("ctrl_step16", f"{CTRL}/cache-step16.pt", 2.159724712371826, 2.529625177383423, "MECH-KEYS freeze control cache-step16.pt"),
    ("keys_step16", f"{KEYS}/cache-step16.pt", 2.034916400909424, 2.330503463745117, "keys+reposition cache-step16.pt"),
    ("betaOn_step16", f"{BON}/cache-step16.pt", 2.73637, 2.81193, "MECH-BETA beta ON cache-step16.pt"),
]

# --------------------------------------------------------------------- wandb --
import wandb

wandb_run = wandb.init(
    project=os.environ.get("CARTRIDGES_WANDB_PROJECT", "SEACrowd"),
    entity=os.environ.get("CARTRIDGES_WANDB_ENTITY", "vqtri-purdue-university"),
    name=os.environ.get("RUN_NAME", "DIAG-NOISE_perexample-losses"),
    group=os.environ.get("WANDB_GROUP", "VERIFY"),
    tags=["diagnostic", "DIAG-NOISE", "eval"],
    notes="exact per-example CE recovery on 13 cached checkpoints (no training, no re-solve)",
)
print("WANDB URL:", wandb_run.url, flush=True)

# --------------------------------------------------------------------- setup --
seed_everything(42)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = HFModelConfig(
    pretrained_model_name_or_path=MODEL, model_cls=FlexQwen3ForCausalLM
).instantiate().to("cuda").to(torch.bfloat16)
print(f"model loaded in {time.time()-t0:.1f}s", flush=True)

datasets = {}
for split in ["QA", "MT"]:
    ds = LossEvalDataset(
        LossEvalDataset.Config(
            data_source=DataSource(path=f"{DATA}/qasper_eval_{split}.parquet", type="local"),
            packed_seq_length=2048,
        ),
        tokenizer=tok, seed=42,
    )
    datasets[split] = ds
    print(split, "n_elements", len(ds.elements), "n_batches", len(ds.batches), flush=True)


def eval_per_example(cam, ds):
    n_el = len(ds.elements)
    ce_sum = torch.zeros(n_el, dtype=torch.float64)
    tok_cnt = torch.zeros(n_el, dtype=torch.int64)
    p_sum = 0.0
    dl = DataLoader(ds, batch_size=1, collate_fn=_collate_first, num_workers=0)
    total_ce = 0.0
    total_n = 0
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = cam(
                    input_ids=batch.input_ids.to("cuda"),
                    seq_ids=batch.element_ids.to("cuda"),
                    position_ids=batch.position_ids.to("cuda"),
                )
                topk_pred_logprobs = F.log_softmax(outputs.logits, dim=-1)[
                    0,
                    batch.topk_token_idxs.to("cuda") - 1,
                    batch.topk_token_ids.to("cuda"),
                ]
                p = batch.topk_logprobs.to("cuda").exp()
                ce_by_token = -p * topk_pred_logprobs
            total_ce += float(ce_by_token.double().sum().item())
            total_n += int(ce_by_token.shape[0])
            p_sum += float(p.double().sum().item())

            # which element owns each scored token
            idx = batch.topk_token_idxs.to(torch.long)
            local_at_tgt = batch.element_ids[idx]
            local_at_pred = batch.element_ids[idx - 1]
            assert torch.equal(local_at_tgt, local_at_pred), "scored token crosses element boundary"
            gmap = torch.tensor(ds.batches[bi], dtype=torch.long)
            glob = gmap[local_at_tgt]
            ce_sum.index_add_(0, glob, ce_by_token.double().cpu())
            tok_cnt.index_add_(0, glob, torch.ones_like(glob))
    assert int(tok_cnt.sum().item()) == total_n
    return {
        "per_example_ce_sum": [float(x) for x in ce_sum.tolist()],
        "per_example_tokens": [int(x) for x in tok_cnt.tolist()],
        "per_example_loss": [
            (float(ce_sum[i]) / int(tok_cnt[i])) if int(tok_cnt[i]) > 0 else None
            for i in range(n_el)
        ],
        "total_ce": total_ce,
        "total_tokens": total_n,
        "mean_ce": total_ce / total_n,
        "mean_p_per_scored_token": p_sum / total_n,
    }


results = {}
for label, path, pub_qa, pub_mt, prov in CKPTS:
    if not os.path.exists(path):
        print("MISSING", label, path, flush=True)
        results[label] = {"error": "missing checkpoint", "path": path}
        continue
    t1 = time.time()
    cache = TrainableCache.from_pretrained(path, device="cuda").to("cuda").to(torch.bfloat16)
    cam = CacheAndModel(cache, model)
    entry = {"path": path, "provenance": prov, "published": {"QA": pub_qa, "MT": pub_mt}}
    for split in ["QA", "MT"]:
        r = eval_per_example(cam, datasets[split])
        pub = pub_qa if split == "QA" else pub_mt
        r["published"] = pub
        r["abs_err_vs_published"] = (abs(r["mean_ce"] - pub) if pub is not None else None)
        entry[split] = r
        print(f"{label:16s} {split}  mean_ce={r['mean_ce']:.10f}  published={pub}  "
              f"err={r['abs_err_vs_published']}", flush=True)
        wandb.log({f"diag/{label}_{split}_mean_ce": r["mean_ce"],
                   f"diag/{label}_{split}_abs_err_vs_published": r["abs_err_vs_published"] or 0.0})
    results[label] = entry
    del cam, cache
    torch.cuda.empty_cache()
    print(f"  [{label}] {time.time()-t1:.1f}s", flush=True)

payload = {
    "snapshot": SNAP,
    "cartridges_import_path": os.path.dirname(cartridges.__file__),
    "model": MODEL,
    "eval_seed": 42,
    "packed_seq_length": 2048,
    "wandb_run_url": wandb_run.url,
    "wandb_run_id": wandb_run.id,
    "batches": {s: [list(map(int, b)) for b in datasets[s].batches] for s in datasets},
    "n_elements": {s: len(datasets[s].elements) for s in datasets},
    "results": results,
}
with open(OUT, "w") as f:
    json.dump(payload, f)
print("WROTE", OUT, flush=True)
print("TOTAL_S", time.time() - t0, flush=True)
wandb.finish()

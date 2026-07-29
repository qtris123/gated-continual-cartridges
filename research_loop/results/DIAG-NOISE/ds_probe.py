import os, sys, json
sys.path.insert(0, "/tmp/amsnap_DIAG-NOISE")
from transformers import AutoTokenizer
from cartridges.datasets import DataSource, LossEvalDataset
import cartridges, os as _os
print("cartridges from:", _os.path.dirname(cartridges.__file__), flush=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
DATA="/localhome/local-triv/gated-continual-cartridges_explore/data/qasper/eval"
out={}
for split in ["MT","QA"]:
    ds = LossEvalDataset(
        LossEvalDataset.Config(data_source=DataSource(path=f"{DATA}/qasper_eval_{split}.parquet", type="local"),
                               packed_seq_length=2048),
        tokenizer=tok, seed=42)
    n_el=len(ds.elements)
    ntok=[int(len(e.topk_token_idxs)) for e in ds.elements]
    ninp=[int(len(e.input_ids)) for e in ds.elements]
    md=[e.metadata for e in ds.elements]
    print(split, "n_elements", n_el, "n_batches", len(ds.batches), "sum_scored", sum(ntok), flush=True)
    print(" batches:", ds.batches, flush=True)
    print(" md keys:", sorted(md[0].keys()) if md and isinstance(md[0],dict) else type(md[0]), flush=True)
    out[split]={"n_elements":n_el,"batches":[list(map(int,b)) for b in ds.batches],
                "scored_tokens":ntok,"input_len":ninp,
                "metadata":[{k:(v if isinstance(v,(str,int,float,bool,type(None))) else str(v)) for k,v in (m or {}).items()} for m in md]}
json.dump(out, open("/tmp/diag_noise/ds_probe.json","w"))

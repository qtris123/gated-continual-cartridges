# RoPE rebake for Attention Matching

## Structure

Introduction -> Description -> Code -> Evaluation

## The fix

After you pick which teacher keys become stored keys, and **before** you
solve values, rotate each key from the position it was prefilled at to
the position eval will treat that cache slot as. Use the model’s own
`rope_theta`. Positional encoding is only on keys and queries, not values.

The rebake rotates each key from `from_pos` to `to_pos`, so three inputs
must be right. They are independent — fixing one does not fix another:

1. Where the key came from — the true position the key was baked at during 
   prefill. For a multi-doc kv concatenation, keep prefilling at a running 
   offset so positions keep extending across appended documents. Restart 
   every doc at 0 and `from_pos` collides across docs, so the source angle 
   is wrong (Only apply to initial compaction because we don't concat kv 
   during continual compaction). Note: this technique improves my performance, 
   but I don't think it applies to your approach.
2. Where the key is going — the position you treats that slot as during later 
   stages. You rotate the key onto it so its phase matches its new location 
   in the compacted cache.
3. The base used during position fixing — must be the base the keys were
   baked with (`model.config.rope_theta`). A wrong base scatters the key
   instead of carrying it from `from_pos` to `to_pos`.

The rest of AM (slot pick, value solve, beta) is untouched. The operator
is `_rope_reposition` in `cartridges/am/core.py`.

## Why each fix is needed when compacting the Key matrices:

**Give each key its true starting position - `from_pos`.**

For a single document there is nothing extra to do — prefill positions
already match. But a cartridge concatenates many documents, and by default
each one is prefilled starting at 0, so a key selected from the second doc
carries a `from_pos` that collides with the first. The rebake then subtracts
the wrong starting angle and the key lands on a position it was never at.
Prefill each document at a running offset (`global_teacher_positions=True`)
so the concatenated teacher reads as one sequence with unique positions, and
`from_pos = teacher_pos[sel]` is the angle the key actually carries.

**Re-rotate, because the key still carries its old position.** 

RoPE is multiplied into the key at prefill, there is no position id beside it to edit. Compacting copies that vector into a shorter cache without changing its numbers, and
later stages never re-rotate those stored keys. For example, a Key at position 10 is selectively chosen and put to position 2, it would still carry the positional encoding 
at position 10 during attention score while maybe the compacted cache only has length 8. 

Rebake supplies the missing step — unrotate 10, rotate 2, i.e. one rotation by 
(2 − 10) in Hugging Face’s half-rotate convention. Values are untouched, and solved afterwards.

**While re-rotating, make sure to use the correct theta** 

Previously, we re-rotated on Qwen with theta = 1e4, while it should be 5e6, so a mismatched
base does not carry the key to row 2’s position — it scatters it somewhere unrelated, which is why the wrong-base arm below scores worse than not rebaking at all.

## How the operator works

One 2-D slice `(n_keys, head_dim)` at a time; loop heads and layers outside.

```python (cartridges/am/core.py)
def rope_reposition(keys, from_pos, to_pos, head_dim, rope_theta):
    """Move post-RoPE keys from absolute from_pos to to_pos."""
    delta = to_pos.to(torch.float32) - from_pos.to(torch.float32)
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, device=keys.device, dtype=torch.float32) / head_dim)
    )
    angle = delta[:, None] * inv_freq[None, :]
    emb = torch.cat([angle, angle], dim=-1)
    cos, sin = emb.cos().to(keys.dtype), emb.sin().to(keys.dtype)
    x1, x2 = keys[..., : head_dim // 2], keys[..., head_dim // 2 :]
    return keys * cos + torch.cat((-x2, x1), dim=-1) * sin
```

Compute cos/sin in fp32. A large position gap in bf16 eats the phase.

**On theta.** 5e6 and 1e4 are different rotation groups. Unrotating with
the wrong base does not land on the new position — it sends the key
implies for that stored key, which has caused th
iss

## What it’s worth

QASPER Phase-1 AM, Qwen3-4B-Instruct-2507, 512 slots (cartridge student):

| arm | rebake | theta | QA loss (nats) | generation |
| --- | --- | --- | ---: | --- |
| shipped | yes | 1e4 | **14.65** | `the the the…` |
| no rebake | no | — | 8.99 | poor, not collapsed |
| correct base | yes | **5e6** | 5.46 | fluent |
| + unique prefill positions | yes | **5e6** | **4.07** | fluent, question-conditioned |

Wrong-base rebake is **worse than not rebaking at all**: the installed
vector is not a RoPE key at that slot, so its logits are noise.

Same model, 8k-token docs, 1024 slots, 8 questions, on moe-am
(question-after-doc). Packing keys and moving the question are separated:

| dataset | keys at teacher pos, question at L | keys packed, question at L | keys packed, question at S |
| --- | ---: | ---: | ---: |
| QASPER | **2.57** | 4.41 | 8.03 |
| LongHealth | 1.98 | 0.59 | 3.82 |
| QuALITY | 1.89 | 0.74 | 3.88 |

Moving the question off the geometry its keys were stored in is about 3
nats on every dataset. Five-phase QASPER concatenate: native geometry
3.99 nats / 2.5% degenerate generations; dense port 9.00 / 75% loops.

Reconstruction MSE does **not** catch this (shipped 0.006 vs fixed 0.009).
Score generation or teacher-forced CE, not fit-to-teacher-attention.

## Where it lives in this repo

| file | what to read |
| --- | --- |
| `cartridges/am/core.py` | `_rope_reposition` |
| `cartridges/am/initial/compaction.py` | prefill offset, `teacher_pos`, rebake after key select (~L359, ~L485) |
| `cartridges/am/components/keys.py` | `rewrite_keys_on_support(..., reposition=True)` for continual |
| `examples/shared/am/initial_compaction.py` | `AM_ROPE_THETA` / `"model"` |

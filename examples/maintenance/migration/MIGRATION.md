# Migration / restore

The two large trees in this project are kept **off git** and published to
Hugging Face instead:

| tree | size | Hugging Face repo (dataset, public) |
|---|---:|---|
| `data/` | ~4.4 GB | [`qtris123/gated-continual-cartridges-data`](https://huggingface.co/datasets/qtris123/gated-continual-cartridges-data) |
| `outputs/caches/` | ~218 GB | [`qtris123/gated-continual-cartridges-caches`](https://huggingface.co/datasets/qtris123/gated-continual-cartridges-caches) |

Everything else (code) lives in this git repo on branch
`trivo-explore-research-work`.

## Restore on a fresh machine

```bash
git clone -b trivo-explore-research-work \
  https://github.com/faridlazuarda/gated-continual-cartridges.git
cd gated-continual-cartridges
hf auth login                                   # a token with read access
examples/maintenance/migration/reprepare.sh
```

`reprepare.sh` only restores `data/` and `outputs/caches/`. It:

1. downloads and extracts `data.tar.zst` into `data/`;
2. downloads the `caches-*.tar` shards (one per `<dataset>/<stage>`) and the
   `caches-meta/` files, extracting them into `outputs/caches/`;
3. rewrites the absolute paths baked into `index.json` / every `source.json`
   and re-points the 24 absolute symlinks from the old machine root to the new
   clone root;
4. verifies the cache count in `index.json` and reports any broken symlinks,
   checking `sha256` against the uploaded manifests.

It does **not** regenerate anything, install dependencies, or touch code.

## Upload layout (how it was published)

- `data/` was tarred with `tar --zstd` (relative symlinks preserved) to
  `data.tar.zst`, plus `data.MANIFEST.txt` (sha256).
- `outputs/caches/` was tarred **uncompressed** (tensors do not compress),
  one shard per `<dataset>/<stage>` (`caches-<dataset>-<stage>.tar`), with
  symlinks preserved. `index.json` and `README.md` were also uploaded under
  `caches-meta/`, and all shard checksums recorded in `caches.MANIFEST.txt`.

To re-publish after changes, rebuild the tars the same way and
`hf upload <repo> <file> <path> --repo-type dataset`.

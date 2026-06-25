# Slot-selection stability (Llama, Qasper) — t=32 and t=128 subset

`J̃` = median across consecutive logged snapshots of the per-component Jaccard 
between top-t selections.  Higher = more stable (snapshot tells the whole story); 
lower = snapshot is one of many.

|  granularity  | top-t |  mid-step  |  J̃ median | J̃ mean | J̃ min | J̃ max | pairs |
| ------------- | ----- | ---------- | --------- | ------ | ----- | ----- | ----- |
| global        | 32    | 270        | 1.000     | 0.964  | 0.641 | 1.000 | 18    |
| global        | 128   | 270        | 0.977     | 0.947  | 0.730 | 1.000 | 18    |
| per_layer     | 32    | 270        | 0.847     | 0.796  | 0.378 | 0.897 | 18    |
| per_layer     | 128   | 270        | 0.856     | 0.820  | 0.561 | 0.893 | 18    |
| per_head      | 32    | 270        | 0.741     | 0.689  | 0.263 | 0.782 | 18    |
| per_head      | 128   | 270        | 0.787     | 0.761  | 0.541 | 0.812 | 18    |
